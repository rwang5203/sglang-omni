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
import json
import logging
import os
import pickle
import shutil
import socket
import tempfile
import weakref
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, suppress
from functools import partial
from pathlib import Path
from urllib.parse import unquote, urlparse

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.datastructures import URLPath
from starlette.routing import BaseRoute, Match, NoMatchFound
from starlette.types import ASGIApp, Receive, Scope, Send

from sglang_omni.config import PipelineConfig
from sglang_omni.preprocessing.resource_connector import (
    MediaPolicyError,
    MultiModalResourceConnector,
    get_global_resource_connector,
)

logger = logging.getLogger(__name__)

NATIVE_HOST = "127.0.0.1"
LOOPBACK_HOSTS = (NATIVE_HOST, "localhost")
# The native route families Omni serves: image generation and edits, video
# jobs with retrieval, content and deletion, and action generation.
ROUTE_PREFIXES = ("/v1/images", "/v1/videos", "/v1/actions")
# Fields of those routes that carry a media path or URL, besides the *_path
# request fields the served model declares. Action observations use image,
# images and video.
MEDIA_REFERENCE_FIELDS = frozenset(
    {
        "image",
        "images",
        "input_reference",
        "reference_url",
        "url",
        "url[]",
        "video",
        "video_url",
    }
)
# Containers native SGLang merges into a request, as JSON objects or strings.
EXTRA_CONTAINERS = frozenset({"extra_args", "extra_body", "extra_json", "extra_params"})

RequestFields = Callable[[object], frozenset[str]]


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
    request_fields: RequestFields,
) -> Callable[[], NativeMediaFrontend]:
    """Settle the native server settings before the generation stage starts.

    Runs before any stage starts, because the stage process reads the
    overrides written here when it builds its native runtime. The stage
    saves the native server arguments it resolves to the path set here,
    and the returned builder, called once the stage started, loads them.
    Building native server arguments in the HTTP process instead would
    query the GPU and hold a CUDA context there. request_fields names the
    request fields clients may set for the served model, the same set the
    generation stage accepts.
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
    # Explicit strict ports make startup fail if another process takes one
    # of them, instead of native SGLang moving to another port.
    kwargs.setdefault("scheduler_port", unused_port({port}))
    kwargs.setdefault("master_port", unused_port({port, kwargs["scheduler_port"]}))
    # Native routes keep uploads in a temporary directory for their request
    # or video job, instead of a directory that grows with every request.
    kwargs.setdefault("input_save_path", None)
    output_dir = Path(getattr(stage.factory, "output_dir", None) or "outputs")
    kwargs.update(
        host=NATIVE_HOST,
        port=port,
        strict_ports=True,
        output_path=str(output_dir.resolve()),
    )
    stage.factory.server_args_overrides = kwargs
    # A private directory, removed once loaded or at interpreter exit.
    handoff = tempfile.mkdtemp(prefix="sglang-omni-native-args-")
    stage.factory.server_args_path = os.path.join(handoff, "server_args.pickle")
    build = partial(
        NativeMediaFrontend.build, stage.factory.server_args_path, request_fields
    )
    weakref.finalize(build, shutil.rmtree, handoff, ignore_errors=True)
    return build


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

    def __init__(self, server_args: object, media_fields: frozenset[str]) -> None:
        from sglang.multimodal_gen.runtime.entrypoints.action import api as action_api
        from sglang.multimodal_gen.runtime.entrypoints.openai import (
            image_api,
            video_api,
        )

        self.server_args = server_args
        self.media_fields = media_fields
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
        police = [Depends(self.police_request)]
        app.include_router(image_api.router, dependencies=police)
        app.include_router(video_api.router, dependencies=police)
        if server_args.pipeline_config.supports_action_endpoint():
            actions = APIRouter()
            actions.routes.extend(
                route
                for route in action_api.router.routes
                if isinstance(route, APIRoute)
            )
            app.include_router(actions, dependencies=police)
        else:
            pass
        self.app = app

    @classmethod
    def build(
        cls, server_args_path: str, request_fields: RequestFields
    ) -> NativeMediaFrontend:
        """Load the generation stage's native server arguments and build the
        frontend, off the event loop."""
        from sglang.multimodal_gen.runtime.server_args import set_global_server_args

        try:
            with open(server_args_path, "rb") as handle:
                server_args = pickle.load(handle)
        except FileNotFoundError:
            raise RuntimeError(
                "The generation stage did not save its native server arguments "
                f"to {server_args_path}. Its stage factory must accept "
                "server_args_path, as the native generation scheduler factory "
                "does."
            ) from None
        finally:
            shutil.rmtree(os.path.dirname(server_args_path), ignore_errors=True)
        set_global_server_args(server_args)
        fields = request_fields(server_args)
        return cls(server_args, frozenset(f for f in fields if f.endswith("_path")))

    async def police_request(self, request: Request) -> None:
        """Apply the server media policy before a native route reads a request.

        Fields that name server paths are refused. Without a configured policy,
        media paths and URLs pass as on other routes. With one, local paths and
        file URLs must resolve inside the allowed directory, and remote URLs are
        refused while a domain allowlist is set, because native SGLang fetches
        them itself and follows redirects without checking each hop.
        """
        content_type = request.headers.get("content-type", "")
        if request.method not in ("PATCH", "POST", "PUT"):
            return
        elif "multipart/form-data" in content_type or "urlencoded" in content_type:
            fields: dict[str, list[object]] = {}
            for name, value in (await request.form()).multi_items():
                fields.setdefault(name, []).append(value)
            body: object = fields
        elif "msgpack" in content_type:
            import msgpack

            try:
                body = msgpack.unpackb(
                    await request.body(), raw=False, strict_map_key=False
                )
            except ValueError:
                return
        else:
            try:
                body = await request.json()
            except ValueError:
                return
        try:
            self.police_fields(body)
        except RecursionError:
            # Native may still decode what nests too deeply for this check.
            raise HTTPException(
                status_code=400,
                detail="Native media request refused: it nests too deeply to check.",
            ) from None

    def police_fields(self, value: object) -> None:
        if isinstance(value, dict):
            for name, item in value.items():
                name = str(name)
                if name in EXTRA_CONTAINERS:
                    item = parse_containers(item)
                else:
                    pass
                if name.endswith("_path") and name not in self.media_fields:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Native media requests cannot set {name}, "
                        "a path the server owns.",
                    )
                elif name in self.media_fields or name in MEDIA_REFERENCE_FIELDS:
                    police_media_value(name, item)
                else:
                    pass
                self.police_fields(item)
        elif isinstance(value, list):
            for item in value:
                self.police_fields(item)
        else:
            pass

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


def parse_containers(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    elif isinstance(value, list):
        return [parse_containers(item) for item in value]
    else:
        return value


def media_references(value: object) -> Iterator[str]:
    """Each string native SGLang may read a media reference from.

    Native decodes JSON encoded request values, multipart fields once and
    model fields such as control_path once more, so a string also stands
    for every string it decodes to.
    """
    if isinstance(value, str):
        yield value
        try:
            decoded = json.loads(value)
        except ValueError:
            return
        yield from decoded_references(decoded, top=True)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from media_references(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from media_references(item)
    else:
        pass


def decoded_references(value: object, *, top: bool) -> Iterator[str]:
    """References in a value native decoded from a string.

    Native reads decoded strings as paths and turns any other decoded value,
    except a list of paths, into its text, as Cosmos3 lowering does for the
    items of control_path.
    """
    if isinstance(value, str):
        yield from media_references(value)
    elif value is not None:
        if not (top and isinstance(value, list)):
            yield str(value)
        else:
            pass
        if isinstance(value, (dict, list)):
            items = value.values() if isinstance(value, dict) else value
            for item in items:
                yield from decoded_references(item, top=False)
        else:
            pass
    else:
        pass


def police_media_value(name: str, value: object) -> None:
    """Refuse a media field value that the server media policy does not allow.

    Without a policy every reference passes, as on other routes.
    """
    connector = get_global_resource_connector()
    if (
        connector.allowed_local_media_path is None
        and not connector.allowed_media_domains
    ):
        return
    else:
        pass
    for source in media_references(value):
        police_media_reference(connector, name, source)


def police_media_reference(
    connector: MultiModalResourceConnector, name: str, source: str
) -> None:
    """Refuse a media path or URL that the server media policy does not allow.

    Follows the rules main applies to request media: a bare path or file URL
    must sit in the allowed directory. Inline media should use data URLs.
    """
    if source.startswith("data:"):
        return
    else:
        pass
    try:
        if source.startswith(("http://", "https://")):
            if connector.allowed_media_domains:
                raise MediaPolicyError(
                    "remote media is refused on native media routes while "
                    "--allowed-media-domain is set, since native SGLang fetches "
                    "it without checking each redirect. Send it as an upload or "
                    "a data URL."
                )
            else:
                pass
        elif source.startswith("file://"):
            connector.local_media_path(unquote(urlparse(source).path))
        else:
            connector.local_media_path(source)
    except MediaPolicyError as exc:
        raise HTTPException(status_code=400, detail=f"{name} refused: {exc}") from exc


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
