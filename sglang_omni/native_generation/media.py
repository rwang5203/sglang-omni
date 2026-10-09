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
import socket
from collections.abc import Callable
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute
from starlette.datastructures import URLPath
from starlette.routing import BaseRoute, Match, NoMatchFound
from starlette.types import ASGIApp, Receive, Scope, Send

from sglang_omni.config import PipelineConfig

NATIVE_HOST = "127.0.0.1"
LOOPBACK_HOSTS = (NATIVE_HOST, "localhost")
# The native route families Omni serves: image generation and edits, video
# jobs with retrieval, content and deletion, and action generation.
# Native warmup waits until /liveness answers on this server before it runs.
ROUTE_PREFIXES = ("/v1/images", "/v1/videos", "/v1/actions", "/liveness")

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
) -> Callable[[], FastAPI]:
    """Settle the native server arguments the stage and the routes share.

    Runs before any stage starts, because the stage process reads the
    overrides written here when it builds its native runtime. Returns the
    builder of the native application, which imports native SGLang.
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
    return partial(build_native_media_app, native_kwargs)


def build_native_media_app(server_kwargs: dict[str, object]) -> FastAPI:
    """Import the native runtime and construct an app with its media routes.

    Native create_app also serves weights, memory, LoRA, rollout, mesh,
    realtime, server info and stats routes without Omni's admin key, so
    only the image, video and action routers are included. The action
    websocket stays out.
    """
    from sglang.multimodal_gen.runtime.entrypoints.action import api as action_api
    from sglang.multimodal_gen.runtime.entrypoints.http_server import lifespan, liveness
    from sglang.multimodal_gen.runtime.entrypoints.openai import image_api, video_api
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
            "Native media routes require SGLang scheduler owner-failure support "
            "(worker_failure in AsyncSchedulerClient.initialize), which the "
            "installed SGLang does not provide. Without it, a request in flight "
            "when the generation worker dies waits for its client deadline. "
            "Install an SGLang build with this capability, then restart."
        )
    else:
        pass
    native_args = ServerArgs.from_kwargs(**server_kwargs)
    set_global_server_args(native_args)
    native_app = FastAPI(lifespan=lifespan)
    native_app.state.server_args = native_args
    native_app.state.prompt_enhancer = None
    native_app.add_api_route("/liveness", liveness)
    native_app.include_router(image_api.router)
    native_app.include_router(video_api.router)
    if native_args.pipeline_config.supports_action_endpoint():
        actions = APIRouter()
        actions.routes.extend(
            route for route in action_api.router.routes if isinstance(route, APIRoute)
        )
        native_app.include_router(actions)
    else:
        pass
    return native_app


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


def mount_native_media_app(
    app: FastAPI, native_app: FastAPI, runtime_failure: asyncio.Future
) -> None:
    """Serve the native media routes and run their lifespan inside Omni's."""
    native_app.state.scheduler_failure = runtime_failure
    previous_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(parent: FastAPI):
        async with previous_lifespan(parent):
            async with native_app.router.lifespan_context(native_app):
                yield

    app.router.lifespan_context = lifespan
    # Readiness reports the native warmup state, see the /health route.
    app.state.native_media_app = native_app
    app.router.routes.append(NativeMediaRoutes(native_app))
