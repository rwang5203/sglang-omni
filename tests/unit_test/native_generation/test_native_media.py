# SPDX-License-Identifier: Apache-2.0
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from sglang_omni.native_generation import media
from sglang_omni.serve import create_app
from sglang_omni.utils.imports import import_string


class RunningClient:
    def health(self):
        return {"running": True}


@pytest.fixture
def native(monkeypatch):
    """Native runtime pieces the frontend calls, without a scheduler."""
    from sglang.multimodal_gen.runtime import (
        scheduler_client,
        server_args,
        server_warmup,
    )

    native = SimpleNamespace(events=[], warmup=None)
    args = SimpleNamespace(
        enable_metrics=False,
        prompt_enhancer_config=None,
        warmup_resolutions=None,
        pipeline_config=SimpleNamespace(
            supports_action_endpoint=lambda: True,
            supports_openpi_endpoint=lambda: True,
        ),
    )

    def initialize(self, server_args, *, worker_failure=None):
        native.events.append("initialize")

    def close(self):
        native.events.append("close")

    monkeypatch.setattr(server_args.ServerArgs, "from_kwargs", lambda **kw: args)
    monkeypatch.setattr(server_args, "set_global_server_args", lambda args: None)
    monkeypatch.setattr(scheduler_client.AsyncSchedulerClient, "initialize", initialize)
    monkeypatch.setattr(scheduler_client.AsyncSchedulerClient, "close", close)
    monkeypatch.setattr(
        server_warmup,
        "should_run_synthetic_server_warmup",
        lambda args: native.warmup is not None,
    )

    async def warm_up(args, forward, *, fail_open):
        await native.warmup()

    monkeypatch.setattr(server_warmup, "run_async_client_warmup", warm_up)
    return native


def serve(omni, frontend, check):
    """Run check(client, runtime_failure) against omni inside its lifespan."""

    async def run():
        runtime_failure = asyncio.get_running_loop().create_future()
        frontend.mount(omni, runtime_failure)
        async with omni.router.lifespan_context(omni):
            transport = httpx.ASGITransport(app=omni)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://omni"
            ) as client:
                await check(client, runtime_failure)
        runtime_failure.cancel()

    asyncio.run(run())


def test_native_routes_and_lifetime_are_composed_without_a_proxy(native):
    @asynccontextmanager
    async def omni_lifespan(app):
        native.events.append("omni_start")
        yield
        native.events.append("omni_stop")

    omni = FastAPI(lifespan=omni_lifespan)

    @omni.get("/health")
    def health():
        return {"owner": "omni"}

    frontend = media.NativeMediaFrontend.build({})

    @frontend.app.get("/v1/images/owner")
    def owner():
        return {"owner": "native"}

    async def check(client, runtime_failure):
        assert native.events == ["omni_start", "initialize"]
        assert (await client.get("/health")).json() == {"owner": "omni"}
        assert (await client.get("/v1/images/owner")).json() == {"owner": "native"}

    serve(omni, frontend, check)
    assert native.events == ["omni_start", "initialize", "close", "omni_stop"]


# Native create_app also serves these, none of them behind Omni's admin key.
NATIVE_CONTROL_ROUTES = [
    ("post", "/release_memory_occupation"),
    ("post", "/resume_memory_occupation"),
    ("post", "/get_weights_checksum"),
    ("post", "/update_weights_from_tensor_checker"),
    ("post", "/v1/set_lora"),
    ("post", "/v1/merge_lora_weights"),
    ("post", "/v1/unmerge_lora_weights"),
    ("get", "/v1/list_loras"),
    ("get", "/v1/model_info"),
    ("post", "/rollout/generate"),
    ("get", "/server_info"),
    ("get", "/stats"),
    ("get", "/models"),
    ("post", "/vertex_generate"),
    ("post", "/v1/meshes"),
]


def test_only_native_media_routes_are_served(native):
    omni = create_app(RunningClient(), model_name="native", admin_api_key="secret")

    async def check(client, runtime_failure):
        for method, path in NATIVE_CONTROL_ROUTES:
            response = await getattr(client, method)(path)
            assert response.status_code == 404, path
        assert (await client.get("/v1/videos")).json()["object"] == "list"
        response = await client.post("/v1/images/generations", json={})
        assert response.status_code == 422

    serve(omni, media.NativeMediaFrontend.build({}), check)


@pytest.mark.parametrize("warmup_fails", [False, True])
def test_native_warmup_gates_readiness_and_fails_startup(native, warmup_fails):
    finish = asyncio.Event()

    async def warmup():
        await finish.wait()
        if warmup_fails:
            raise RuntimeError("warmup request failed")
        else:
            pass

    native.warmup = warmup
    frontend = media.NativeMediaFrontend.build({})

    async def check(client, runtime_failure):
        failed = asyncio.ensure_future(frontend.wait_failed())
        health = await client.get("/health")
        assert (health.status_code, health.json()["status"]) == (503, "warming")
        # Requests answer at once instead of waiting on warmup.
        assert (await client.get("/v1/videos")).status_code == 503
        finish.set()
        if warmup_fails:
            with pytest.raises(RuntimeError, match="warmup failed"):
                await asyncio.wait_for(failed, 5)
            assert (await client.get("/v1/videos")).status_code == 503
        else:
            while frontend.warming:
                await asyncio.sleep(0)
            assert (await client.get("/health")).status_code == 200
            assert (await client.get("/v1/videos")).status_code == 200
            assert not failed.done()
            failed.cancel()

    serve(create_app(RunningClient(), model_name="native"), frontend, check)


def test_cosmos3_resolves_native_settings_before_building(monkeypatch):
    from sglang_omni.models.cosmos3.config import Cosmos3PipelineConfig

    built = []
    frontend = object()
    monkeypatch.setattr(
        media.NativeMediaFrontend,
        "build",
        lambda kwargs: built.append(kwargs) or frontend,
    )
    config = Cosmos3PipelineConfig(model_path="checkpoint")
    prepare = import_string(Cosmos3PipelineConfig.native_media_factory)
    build = prepare(config, port=19000)
    overrides = config.stages[0].factory.server_args_overrides
    # Stage processes read these overrides when they spawn, so they must be
    # resolved before the native runtime is imported or the app is built.
    # The native scheduler socket follows host, so it stays on loopback
    # whatever host the Omni HTTP server binds.
    assert overrides["host"] == "127.0.0.1" and overrides["port"] == 19000
    assert overrides["strict_ports"] is True
    assert {"scheduler_port", "master_port", "output_path"} <= set(overrides)
    assert 19000 not in {overrides["scheduler_port"], overrides["master_port"]}
    assert built == []
    assert build() is frontend
    assert built[0]["model_path"] == "checkpoint" and built[0]["base_gpu_id"] == 0
    assert built[0]["host"] == "127.0.0.1"

    config = Cosmos3PipelineConfig(model_path="checkpoint")
    config.stages[0].factory.server_args_overrides = {"host": "0.0.0.0"}
    with pytest.raises(ValueError, match="loopback"):
        prepare(config, port=19000)


def test_native_frontend_builds_before_stage_startup(monkeypatch):
    from sglang_omni.serve import launcher

    events = []

    class Config:
        native_media_factory = "model.native_media"

    class Runner:
        def __init__(self, config):
            pass

        async def start(self, timeout):
            events.append("startup")

        async def stop(self):
            events.append("stopped")

    def fail():
        events.append("frontend")
        raise RuntimeError("frontend build failed")

    monkeypatch.setattr(launcher, "MultiProcessPipelineRunner", Runner)
    monkeypatch.setattr(launcher, "find_available_port", lambda host, port: port)
    monkeypatch.setattr(launcher, "import_string", lambda path: lambda *a, **k: fail)
    with pytest.raises(RuntimeError, match="frontend build failed"):
        asyncio.run(launcher.run_server(Config()))
    # The native worker binds the frontend's strict ports, so stages start later.
    assert events == ["frontend"]
