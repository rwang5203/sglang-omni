# SPDX-License-Identifier: Apache-2.0
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang_omni.native_generation import media
from sglang_omni.native_generation.media import mount_native_media_app
from sglang_omni.utils.imports import import_string


def test_native_routes_and_lifetime_are_composed_without_a_proxy():
    events = []

    @asynccontextmanager
    async def omni_lifespan(app):
        events.append("omni_start")
        yield
        events.append("omni_stop")

    @asynccontextmanager
    async def native_lifespan(app):
        events.append("native_start")
        app.state.ready = True
        yield
        events.append("native_stop")

    omni = FastAPI(lifespan=omni_lifespan)
    native = FastAPI(lifespan=native_lifespan)

    @omni.get("/health")
    def health():
        return {"owner": "omni"}

    @native.get("/health")
    def native_health():
        return {"owner": "native"}

    @native.post("/v1/images/generations")
    def generate():
        assert native.state.ready
        return {"owner": "native"}

    mount_native_media_app(omni, native, object())
    with TestClient(omni) as client:
        assert events == ["omni_start", "native_start"]
        assert client.get("/health").json() == {"owner": "omni"}
        assert client.post("/v1/images/generations").json() == {"owner": "native"}
    assert events == ["omni_start", "native_start", "native_stop", "omni_stop"]


def test_health_reports_warming_until_the_native_warmup_finishes():
    import asyncio

    from sglang_omni.serve import create_app

    class RunningClient:
        def health(self):
            return {"running": True}

    @asynccontextmanager
    async def native_lifespan(app):
        app.state.server_warmup_done = asyncio.Event()
        yield

    native = FastAPI(lifespan=native_lifespan)
    omni = create_app(RunningClient(), model_name="native-media")
    mount_native_media_app(omni, native, object())
    with TestClient(omni) as client:
        warming = client.get("/health")
        assert warming.status_code == 503
        assert warming.json()["status"] == "warming"
        native.state.server_warmup_done.set()
        ready = client.get("/health")
        assert ready.status_code == 200
        assert ready.json()["status"] == "healthy"


def test_cosmos3_resolves_native_settings_before_building(monkeypatch):
    from sglang_omni.models.cosmos3.config import Cosmos3PipelineConfig

    built = []
    app = FastAPI()
    monkeypatch.setattr(
        media, "build_native_media_app", lambda kwargs: built.append(kwargs) or app
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
    assert build() is app
    assert built[0]["model_path"] == "checkpoint" and built[0]["base_gpu_id"] == 0
    assert built[0]["host"] == "127.0.0.1"

    config = Cosmos3PipelineConfig(model_path="checkpoint")
    config.stages[0].factory.server_args_overrides = {"host": "0.0.0.0"}
    with pytest.raises(ValueError, match="loopback"):
        prepare(config, port=19000)


def test_native_frontend_builds_before_stage_startup(monkeypatch):
    import asyncio

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
