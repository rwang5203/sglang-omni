# SPDX-License-Identifier: Apache-2.0
"""Serve the native SGLang media routes of a generation stage from Omni.

A pipeline whose generation stage runs native multimodal_gen declares
``native_media_factory`` on its config, usually as a partial of
``prepare_native_media_app`` that names the stage. The Omni server then
serves the native media routes, which send requests straight to the
native scheduler that the stage owns.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import socket
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from functools import partial
from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.datastructures import URLPath
from starlette.routing import BaseRoute, Match, NoMatchFound
from starlette.types import ASGIApp, Receive, Scope, Send

from sglang_omni.config import PipelineConfig

logger = logging.getLogger(__name__)

NATIVE_HOST = "127.0.0.1"
LOOPBACK_HOSTS = (NATIVE_HOST, "localhost")
# The native route families Omni serves: image generation and edits, video
# jobs with retrieval, content and deletion, and action generation.
ROUTE_PREFIXES = ("/v1/images", "/v1/videos", "/v1/actions")

NativeServerKwargs = Callable[
    [str, int, dict[str, object], list[int] | None], dict[str, object]
]


def unused_port(exclude: set[int]) -> int:
    for _ in range(100):
        with socket.socket() as sock:
            sock.bind((NATIVE_HOST, 0))
            port = sock.getsockname()[1]
        if port not in exclude:
            return port
        else:
            pass
    raise RuntimeError("Could not select distinct native runtime ports")


def prepare_native_media_app(
    config: PipelineConfig,
    *,
    port: int,
    stage_name: str,
    server_kwargs: NativeServerKwargs,
) -> Callable[[], NativeMediaFrontend]:
    """Settle the native server arguments the stage and the routes share.

    Runs before any stage starts, because the stage process reads the
    overrides written here when it builds its native runtime. Returns the
    builder of the frontend, which imports native SGLang.
    """
    stage = config.stage_named(stage_name)
    if not isinstance(stage.gpu, int):
        raise ValueError(
            f"Native media routes need stage {stage_name!r} on one GPU. "
            f"Set stages.{stage_name}.gpu to a GPU id."
        )
    else:
        pass
    kwargs = dict(getattr(stage.factory, "server_args_overrides", None) or {})
    # The native scheduler binds its request socket on this host and
    # unpickles what it receives, so it stays on loopback whatever host
    # the Omni HTTP server binds. Native SGLang serves no HTTP here.
    if kwargs.get("host", NATIVE_HOST) not in LOOPBACK_HOSTS:
        raise ValueError(
            "The native generation scheduler listens on loopback only. "
            f"Remove host from stages.{stage_name}.factory.server_args_overrides."
        )
    else:
        pass
    # Explicit strict ports make both sides address the same scheduler, and
    # startup fails if another process takes one of them.
    kwargs.setdefault("scheduler_port", unused_port({port}))
    kwargs.setdefault("master_port", unused_port({port, kwargs["scheduler_port"]}))
    output_dir = Path(getattr(stage.factory, "output_dir", None) or "outputs")
    kwargs.update(
        host=NATIVE_HOST,
        port=port,
        strict_ports=True,
        output_path=str(output_dir.resolve()),
    )
    native_kwargs = server_kwargs(
        config.model_path, stage.gpu, kwargs, stage.runtime_gpu_ids
    )
    stage.factory.server_args_overrides = kwargs
    return partial(NativeMediaFrontend.build, native_kwargs)


class NativeMediaFrontend:
    """The native media routes of one generation stage inside the Omni server.

    Omni owns the lifespan of these routes. It reuses the native scheduler
    client, warmup and video job shutdown, starts no native broker, since
    nothing in Omni sends offline jobs through one, answers 503 while native
    warmup runs, and turns a warmup failure into a startup failure.

    A request in flight fails with 503 as soon as the generation stage fails
    or a native scheduler socket closes, which happens when its process
    exits, because native SGLang before worker_failure support would wait
    for the answer forever.
    """

    def __init__(self, server_args: object) -> None:
        from sglang.multimodal_gen.runtime.entrypoints.action import api as action_api
        from sglang.multimodal_gen.runtime.entrypoints.openai import (
            image_api,
            video_api,
        )

        self.server_args = server_args
        self.ready = asyncio.Event()
        self.failure: BaseException | None = None
        self.failed = asyncio.Event()
        self.stage_failed = False
        # Last monitor event of each native scheduler endpoint.
        self.scheduler_events: dict[bytes, int] = {}
        self.unavailable_reason: str | None = None
        self.unavailable = asyncio.Event()
        # Native create_app also serves weights, memory, LoRA, rollout, mesh,
        # realtime, server info and stats routes without Omni's admin key, so
        # only the image, video and action routers are included. The action
        # websocket stays out.
        app = FastAPI()
        app.state.server_args = server_args
        app.state.prompt_enhancer = None
        app.include_router(image_api.router)
        app.include_router(video_api.router)
        if server_args.pipeline_config.supports_action_endpoint():
            actions = APIRouter()
            actions.routes.extend(
                route
                for route in action_api.router.routes
                if isinstance(route, APIRoute)
            )
            app.include_router(actions)
        else:
            pass
        self.app = app

    @classmethod
    def build(cls, server_kwargs: dict[str, object]) -> NativeMediaFrontend:
        """Import the native runtime and build the frontend off the event loop."""
        from sglang.multimodal_gen.runtime.server_args import (
            ServerArgs,
            set_global_server_args,
        )

        server_args = ServerArgs.from_kwargs(**server_kwargs)
        set_global_server_args(server_args)
        return cls(server_args)

    @property
    def warming(self) -> bool:
        return not self.ready.is_set()

    def fail(self, error: BaseException) -> None:
        if self.failure is None:
            self.failure = error
            self.failed.set()
        else:
            pass

    async def wait_failed(self) -> None:
        """Runtime watcher that raises once the native routes cannot serve."""
        await self.failed.wait()
        raise self.failure

    def mount(self, app: FastAPI, runtime_failure: asyncio.Future) -> None:
        """Serve the native media routes and run their lifespan inside Omni's."""
        runtime_failure.add_done_callback(self.on_runtime_failure)
        previous_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(parent: FastAPI) -> AsyncIterator[None]:
            async with previous_lifespan(parent):
                async with self.lifespan(runtime_failure):
                    yield

        app.router.lifespan_context = lifespan
        # Readiness reports the native warmup state, see the /health route.
        app.state.native_media = self
        app.router.routes.append(NativeMediaRoutes(self.serve))

    @asynccontextmanager
    async def lifespan(self, runtime_failure: asyncio.Future) -> AsyncIterator[None]:
        from sglang.multimodal_gen.runtime.entrypoints.openai.prompt_enhancement import (
            PromptEnhancer,
        )
        from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
            shutdown_video_jobs,
        )
        from sglang.multimodal_gen.runtime.scheduler_client import (
            async_scheduler_client,
        )
        from sglang.multimodal_gen.runtime.server_warmup import (
            should_run_synthetic_server_warmup,
        )

        initialize = async_scheduler_client.initialize
        if "worker_failure" in inspect.signature(initialize).parameters:
            # Later native SGLang settles its own requests on stage failure.
            initialize(self.server_args, worker_failure=runtime_failure)
        else:
            initialize(self.server_args)
        watch = asyncio.create_task(self.watch_scheduler())
        config = self.server_args.prompt_enhancer_config
        enhancer = None if config is None else PromptEnhancer.from_file(config)
        self.app.state.prompt_enhancer = enhancer
        if should_run_synthetic_server_warmup(self.server_args):
            warmup = asyncio.create_task(self.warm_up())
        else:
            warmup = None
            self.ready.set()
        try:
            yield
        finally:
            for task in (warmup, watch):
                if task is not None:
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                else:
                    pass
            await shutdown_video_jobs()
            if enhancer is not None:
                await enhancer.close()
            else:
                pass
            async_scheduler_client.close()

    async def warm_up(self) -> None:
        from sglang.multimodal_gen.runtime.scheduler_client import (
            async_scheduler_client,
        )
        from sglang.multimodal_gen.runtime.server_warmup import run_async_client_warmup

        try:
            # Native fails open without explicit warmup resolutions.
            await run_async_client_warmup(
                self.server_args,
                async_scheduler_client.forward,
                fail_open=self.server_args.warmup_resolutions is None,
            )
        except Exception as exc:
            logger.exception("Native server warmup failed")
            self.fail(RuntimeError(f"Native server warmup failed: {exc}"))
        else:
            self.ready.set()

    def on_runtime_failure(self, runtime_failure: asyncio.Future) -> None:
        # Serving shutdown cancels the watcher, which is no failure.
        if not runtime_failure.cancelled():
            self.stage_failed = True
            self.update_availability()
        else:
            pass

    def update_availability(self) -> None:
        import zmq

        if self.stage_failed:
            self.unavailable_reason = "The generation stage failed"
        elif zmq.EVENT_DISCONNECTED in self.scheduler_events.values():
            self.unavailable_reason = "The native generation scheduler is unreachable"
        else:
            self.unavailable_reason = None
        if self.unavailable_reason is None:
            self.unavailable.clear()
        else:
            logger.error("Native media routes unavailable: %s", self.unavailable_reason)
            self.unavailable.set()

    async def watch_scheduler(self) -> None:
        """Follow the connection to each native scheduler socket.

        A scheduler process that exits closes its socket, and the socket
        monitor reports that at once, without polling. Reconnecting, which
        only a live scheduler accepts, makes the routes available again.
        """
        import zmq
        import zmq.asyncio
        from zmq.utils.monitor import recv_monitor_message

        context = zmq.asyncio.Context()
        probe = context.socket(zmq.DEALER)
        monitor = probe.get_monitor_socket(zmq.EVENT_CONNECTED | zmq.EVENT_DISCONNECTED)
        try:
            for endpoint in self.server_args.scheduler_endpoints:
                probe.connect(endpoint)
            while True:
                event = await recv_monitor_message(monitor)
                self.scheduler_events[event["endpoint"]] = event["event"]
                self.update_availability()
        finally:
            probe.disable_monitor()
            monitor.close(linger=0)
            probe.close(linger=0)
            context.term()

    async def serve(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.unavailable.is_set():
            await self.refuse(scope, receive, send, self.unavailable_reason)
        elif self.warming:
            # Nothing waits on warmup, so warmup can never hold up shutdown.
            await self.refuse(scope, receive, send, "Native generation is warming up")
        else:
            await self.serve_while_available(scope, receive, send)

    @staticmethod
    async def refuse(scope: Scope, receive: Receive, send: Send, reason: str) -> None:
        await JSONResponse({"detail": reason}, status_code=503)(scope, receive, send)

    async def serve_while_available(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        started = False

        async def send_tracked(message: dict[str, object]) -> None:
            nonlocal started
            started = started or message["type"] == "http.response.start"
            await send(message)

        request = asyncio.ensure_future(self.app(scope, receive, send_tracked))
        unavailable = asyncio.ensure_future(self.unavailable.wait())
        try:
            await asyncio.wait(
                (request, unavailable), return_when=asyncio.FIRST_COMPLETED
            )
            reason = self.unavailable_reason or "Native generation stopped"
        finally:
            unavailable.cancel()
            if not request.done():
                request.cancel()
                with suppress(asyncio.CancelledError):
                    await request
            else:
                pass
        if not request.cancelled():
            request.result()
        elif not started:
            await self.refuse(scope, receive, send, reason)
        else:
            # The response already started, so the connection just closes.
            pass


class NativeMediaRoutes(BaseRoute):
    """Send the native media route families, and nothing else, to the native app.

    The paths keep their prefix, so the native routers match them unchanged,
    and Omni routes never depend on being registered before this one.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    def matches(self, scope: Scope) -> tuple[Match, Scope]:
        path = scope["path"] if scope["type"] == "http" else ""
        if any(
            path == prefix or path.startswith(f"{prefix}/") for prefix in ROUTE_PREFIXES
        ):
            return Match.FULL, {}
        else:
            return Match.NONE, {}

    def url_path_for(self, name: str, /, **path_params: object) -> URLPath:
        raise NoMatchFound(name, path_params)

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.app(scope, receive, send)
