# SPDX-License-Identifier: Apache-2.0
"""Serve the native SGLang media routes of a generation stage from Omni.

A pipeline whose generation stage runs native multimodal_gen declares
``native_media_factory`` on its config, usually as a partial of
``prepare_native_media_app`` that names the stage. The Omni server then
serves the native media routes, which send requests straight to the
native scheduler that the stage owns.

The routes start on the first request to them, because starting them
imports native SGLang's route modules into the Omni HTTP process, which
takes seconds and hundreds of MiB. A deployment that never calls them
starts and idles as if they were absent.
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
from contextlib import AsyncExitStack, asynccontextmanager, suppress
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
# Keys of the image payloads that the native action routes decode in memory
# instead of opening a file.
INLINE_IMAGE_KEYS = frozenset({"b64_json", "base64", "data", "values"})
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
) -> NativeMediaLoader:
    """Settle the native server settings before the generation stage starts.

    Runs before any stage starts, because the stage process reads the
    overrides written here when it builds its native runtime. The stage
    saves the native server arguments it resolves to the path set here,
    and the returned loader, called once the stage started, reads them
    without importing native SGLang. Building native server arguments in
    the HTTP process instead would query the GPU and hold a CUDA context
    there. request_fields names the request fields clients may set for the
    served model, the same set the generation stage accepts.
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
    load = NativeMediaLoader(request_fields)
    stage.factory.server_args_path = load.server_args_path
    return load


class NativeMediaLoader:
    """Load the native media routes once the generation stage started.

    The stage saves its native server arguments in a private directory,
    which goes once they are loaded, when discard is called, or at
    interpreter exit. The launcher calls discard when the stages do not
    start, because a SIGTERM during startup ends the server without exit
    handlers.
    """

    def __init__(self, request_fields: RequestFields) -> None:
        handoff = tempfile.mkdtemp(prefix="sglang-omni-native-args-")
        self.server_args_path = os.path.join(handoff, "server_args.pickle")
        self.request_fields = request_fields
        self.discard = weakref.finalize(
            self, shutil.rmtree, handoff, ignore_errors=True
        )

    def __call__(self) -> NativeMediaFrontend:
        return NativeMediaFrontend.load(self.server_args_path, self.request_fields)


class NativeMediaFrontend:
    """The native media routes of one generation stage inside the Omni server.

    Omni owns the lifespan of these routes. It reuses the native scheduler
    client and video job shutdown, and starts no native broker, since nothing
    in Omni sends offline jobs through one. A native server warmup runs in
    the generation stage before it reports ready, not here.

    The routes start once, on the first request to them: native imports run
    off the event loop, then their lifespan enters, and requests that arrive
    meanwhile wait for the same start. A failed start answers 503 with its
    cause on these routes until the server restarts.

    A request in flight fails with 503 as soon as the generation stage fails
    or a native scheduler socket closes, which happens when its process
    exits, because native SGLang before worker_failure support would wait
    for the answer forever.
    """

    def __init__(self, server_args: bytes, request_fields: RequestFields) -> None:
        # The pickled native server arguments, loaded when the routes start.
        self.saved_server_args = server_args
        self.request_fields = request_fields
        self.server_args: object = None
        self.media_fields: frozenset[str] = frozenset()
        self.app: FastAPI | None = None
        self.runtime_failure: asyncio.Future | None = None
        self.start_task: asyncio.Future | None = None
        self.start_failure: str | None = None
        self.lifespans = AsyncExitStack()
        self.stage_failed = False
        # Last monitor event of each native scheduler endpoint.
        self.scheduler_events: dict[bytes, int] = {}
        self.unavailable_reason: str | None = None
        self.unavailable = asyncio.Event()

    @classmethod
    def load(
        cls, server_args_path: str, request_fields: RequestFields
    ) -> NativeMediaFrontend:
        """Read what the generation stage saved, without importing native SGLang.

        The stage saves its pickled native server arguments, which stay
        pickled until the routes start.
        """
        try:
            with open(server_args_path, "rb") as handle:
                saved = handle.read()
        except FileNotFoundError:
            raise RuntimeError(
                "The generation stage did not save its native server arguments "
                f"to {server_args_path}. Its stage factory must accept "
                "server_args_path, as the native generation scheduler factory "
                "does."
            ) from None
        finally:
            shutil.rmtree(os.path.dirname(server_args_path), ignore_errors=True)
        return cls(saved, request_fields)

    def build_app(self) -> None:
        """Load the native server arguments and routes, in a worker thread."""
        from sglang.multimodal_gen.runtime.entrypoints.action import api as action_api
        from sglang.multimodal_gen.runtime.entrypoints.openai import (
            image_api,
            video_api,
        )
        from sglang.multimodal_gen.runtime.server_args import set_global_server_args

        server_args = pickle.loads(self.saved_server_args)
        set_global_server_args(server_args)
        fields = self.request_fields(server_args)
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
        self.server_args = server_args
        self.media_fields = frozenset(f for f in fields if f.endswith("_path"))
        self.app = app

    def begin(self) -> None:
        """Start the routes, once, whatever asks first."""
        if self.start_task is None:
            self.start_task = asyncio.ensure_future(self.start())
        else:
            pass

    async def start(self) -> None:
        """Build the routes off the event loop, then enter their lifespan."""
        try:
            await asyncio.to_thread(self.build_app)
            await self.lifespans.enter_async_context(self.lifespan())
        except Exception as exc:
            logger.exception("The native media routes could not start")
            self.start_failure = (
                f"The native media routes could not start: {exc}. Fix the "
                "cause in the server log, then restart the server."
            )

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
            except RecursionError:
                raise nested_too_deeply() from None
        try:
            self.police_fields(
                body, inline_images=request.url.path.startswith("/v1/actions/")
            )
        except RecursionError:
            raise nested_too_deeply() from None

    def police_fields(self, value: object, inline_images: bool = False) -> None:
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
                    police_media_value(name, item, inline_images)
                else:
                    pass
                self.police_fields(item, inline_images)
        elif isinstance(value, list):
            for item in value:
                self.police_fields(item, inline_images)
        else:
            pass

    def mount(self, app: FastAPI, runtime_failure: asyncio.Future) -> None:
        """Serve the native media routes and end their lifespan inside Omni's."""
        self.runtime_failure = runtime_failure
        runtime_failure.add_done_callback(self.on_runtime_failure)
        previous_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(parent: FastAPI) -> AsyncIterator[None]:
            async with previous_lifespan(parent):
                try:
                    yield
                finally:
                    await self.close()

        app.router.lifespan_context = lifespan
        app.router.routes.append(NativeMediaRoutes(self.serve))

    async def close(self) -> None:
        """Stop a start still running and end the lifespan of started routes."""
        if self.start_task is not None and not self.start_task.done():
            # Its import thread runs to the end on its own.
            self.start_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.start_task
        else:
            pass
        await self.lifespans.aclose()

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        from sglang.multimodal_gen.runtime.entrypoints.openai.prompt_enhancement import (
            PromptEnhancer,
        )
        from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
            shutdown_video_jobs,
        )
        from sglang.multimodal_gen.runtime.scheduler_client import (
            async_scheduler_client,
        )

        initialize = async_scheduler_client.initialize
        if "worker_failure" in inspect.signature(initialize).parameters:
            # Later native SGLang settles its own requests on stage failure.
            initialize(self.server_args, worker_failure=self.runtime_failure)
        else:
            initialize(self.server_args)
        watch = enhancer = None
        # A failure before serving still releases what started, since the
        # server keeps running when the routes start on first use.
        try:
            watch = asyncio.create_task(self.watch_scheduler())
            config = self.server_args.prompt_enhancer_config
            enhancer = None if config is None else PromptEnhancer.from_file(config)
            self.app.state.prompt_enhancer = enhancer
            yield
        finally:
            if watch is not None:
                watch.cancel()
                with suppress(asyncio.CancelledError):
                    await watch
            else:
                pass
            await shutdown_video_jobs()
            if enhancer is not None:
                await enhancer.close()
            else:
                pass
            async_scheduler_client.close()

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
        elif self.start_failure is not None:
            await self.refuse(scope, receive, send, self.start_failure)
        else:
            self.begin()
            await self.serve_while_available(scope, receive, send)

    async def serve_started(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Requests that arrive while the routes start wait for the same start,
        # shielded, so one that stops waiting never cancels it.
        await asyncio.shield(self.start_task)
        if self.start_failure is None:
            await self.app(scope, receive, send)
        else:
            await self.refuse(scope, receive, send, self.start_failure)

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

        request = asyncio.ensure_future(
            self.serve_started(scope, receive, send_tracked)
        )
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


def nested_too_deeply() -> HTTPException:
    # Native may still decode what nests too deeply for this check.
    return HTTPException(
        status_code=400,
        detail="Native media request refused: it nests too deeply to check.",
    )


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


def media_references(
    value: object, inline_images: bool = False
) -> Iterator[tuple[str, bool]]:
    """Each string native SGLang may read a media reference from, and whether
    native can open it only as written.

    Native decodes JSON encoded request values, multipart fields once and
    model fields such as control_path once more, so a string also stands
    for every string it decodes to. JSON text never starts with a slash, a
    tilde or a scheme, so as written it can only be a relative path. With
    inline_images, image payloads that native decodes in memory hold no
    reference.
    """
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            yield value, False
            return
        yield value, True
        yield from decoded_references(decoded, top=True)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from media_references(item, inline_images)
    elif isinstance(value, dict):
        if inline_images and INLINE_IMAGE_KEYS.intersection(value):
            return
        else:
            pass
        for item in value.values():
            yield from media_references(item, inline_images)
    else:
        pass


def decoded_references(value: object, *, top: bool) -> Iterator[tuple[str, bool]]:
    """References in a value native decoded from a string.

    Native reads decoded strings as paths and turns any other decoded value,
    except a list of paths, into its text, as Cosmos3 lowering does for the
    items of control_path.
    """
    if isinstance(value, str):
        yield from media_references(value)
    elif value is not None:
        if not (top and isinstance(value, list)):
            yield str(value), True
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


def police_media_value(name: str, value: object, inline_images: bool) -> None:
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
    for source, as_written in media_references(value, inline_images):
        police_media_reference(connector, name, source, as_written)


def police_media_reference(
    connector: MultiModalResourceConnector, name: str, source: str, as_written: bool
) -> None:
    """Refuse a media path or URL that the server media policy does not allow.

    Follows the rules main applies to request media: a bare path or file URL
    must sit in the allowed directory, here in every reading native SGLang
    may open it in. Inline media should use data URLs.
    """
    remote = source.lower().startswith(("http://", "https://"))
    file_url = source[:7].lower() == "file://"
    try:
        if remote and connector.allowed_media_domains:
            raise MediaPolicyError(
                "remote media is refused on native media routes while "
                "--allowed-media-domain is set, since native SGLang fetches "
                "it without checking each redirect. Send it as an upload or "
                "a data URL."
            )
        elif connector.allowed_local_media_path is not None:
            for path in local_readings(source, as_written or remote, file_url):
                connector.local_media_path(path)
        else:
            pass
    except ValueError as exc:
        # A media policy refusal, or a path that cannot be resolved at all,
        # such as one with a NUL character.
        hint = " Send local media as a path." if file_url else ""
        raise HTTPException(
            status_code=400, detail=f"{name} refused: {exc}{hint}"
        ) from exc


def local_readings(source: str, if_present: bool, file_url: bool) -> list[str]:
    """The local paths native SGLang may open for a media reference.

    Native expands the home directory of sampling path fields, while its
    prompt enhancer opens a path as written. Its loaders open a file URL as
    written, a path relative to the working directory, while srt opens the
    path the URL names. A URL, a data URL, or text that native opens only as
    written (if_present) names a local file only when that file exists, since
    its image loader tries an existing file first. Relative paths resolve
    against the server working directory, which the stage and native worker
    processes inherit.
    """
    if if_present or source.startswith("data:"):
        return [source] if os.path.lexists(source) else []
    else:
        readings = [source]
    if source.startswith("~"):
        readings.append(os.path.join(os.getcwd(), source))
    else:
        pass
    if file_url:
        readings.append(unquote(urlparse(source).path))
    else:
        pass
    return readings


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
