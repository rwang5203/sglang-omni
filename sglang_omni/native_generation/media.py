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
        from sglang.multimodal_gen.runtime.scheduler_client import AsyncSchedulerClient
        from sglang.multimodal_gen.runtime.server_args import (
            ServerArgs,
            set_global_server_args,
        )

        if (
            "worker_failure"
            not in inspect.signature(AsyncSchedulerClient.initialize).parameters
        ):
            raise RuntimeError(
                "Native media routes require SGLang scheduler owner-failure "
                "support (worker_failure in AsyncSchedulerClient.initialize), "
                "which the installed SGLang does not provide. Without it, a "
                "request in flight when the generation worker dies waits for its "
                "client deadline. Install an SGLang build with this capability, "
                "then restart."
            )
        else:
            pass
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

        async_scheduler_client.initialize(
            self.server_args, worker_failure=runtime_failure
        )
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
            if warmup is not None:
                warmup.cancel()
                with suppress(asyncio.CancelledError):
                    await warmup
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

    async def serve(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.warming:
            # Nothing waits on warmup, so warmup can never hold up shutdown.
            response = JSONResponse(
                {"detail": "Native generation is warming up"}, status_code=503
            )
            await response(scope, receive, send)
        else:
            await self.app(scope, receive, send)


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
