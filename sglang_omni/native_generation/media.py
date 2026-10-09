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

from fastapi import FastAPI

from sglang_omni.config import PipelineConfig

NATIVE_HOST = "127.0.0.1"
LOOPBACK_HOSTS = (NATIVE_HOST, "localhost")

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
    """Import the native runtime and construct its application."""
    from sglang.multimodal_gen.runtime.entrypoints.http_server import create_app
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
    return create_app(native_args)


def mount_native_media_app(
    app: FastAPI, native_app: FastAPI, runtime_failure: asyncio.Future
) -> None:
    """Preserve native routing and run its lifespan before worker shutdown."""
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
    app.mount("/", native_app)
