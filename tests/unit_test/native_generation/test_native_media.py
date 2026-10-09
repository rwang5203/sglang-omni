# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import pickle
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from sglang_omni.native_generation import media
from sglang_omni.serve import create_app
from sglang_omni.utils.imports import import_string


def client_fields(server_args):
    return frozenset({"control_path", "image_path", "prompt", "video_path"})


class RunningClient:
    def health(self):
        return {"running": True}


class NativePipeline:
    def supports_action_endpoint(self):
        return True

    def supports_openpi_endpoint(self):
        return True


@pytest.fixture
def native(monkeypatch, tmp_path):
    """Native runtime pieces the frontend calls, without a scheduler."""
    from sglang.multimodal_gen.runtime import scheduler_client, server_args

    args = SimpleNamespace(
        enable_metrics=False,
        prompt_enhancer_config=None,
        scheduler_endpoints=[],
        pipeline_config=NativePipeline(),
    )
    native = SimpleNamespace(args=args, events=[])

    def build():
        # The generation stage saves its resolved arguments here.
        handoff = tmp_path / "handoff"
        handoff.mkdir()
        path = handoff / "server_args.pickle"
        path.write_bytes(pickle.dumps(native.args))
        frontend = media.NativeMediaFrontend.load(str(path), client_fields)
        assert not handoff.exists()
        return frontend

    native.build = build

    def initialize(self, server_args, *, worker_failure=None):
        native.events.append("initialize")

    def close(self):
        native.events.append("close")

    def query_devices(**kwargs):
        raise AssertionError("native server arguments built in the HTTP process")

    monkeypatch.setattr(server_args.ServerArgs, "from_kwargs", query_devices)
    # Loading the arguments and routes runs once, when the routes start.
    monkeypatch.setattr(
        server_args,
        "set_global_server_args",
        lambda args: native.events.append("build"),
    )
    monkeypatch.setattr(scheduler_client.AsyncSchedulerClient, "initialize", initialize)
    monkeypatch.setattr(scheduler_client.AsyncSchedulerClient, "close", close)
    return native


def serve(omni, frontend, check):
    """Run check(client, runtime_failure) against omni inside its lifespan."""

    async def run():
        runtime_failure = asyncio.get_running_loop().create_future()
        frontend.mount(omni, runtime_failure)
        async with omni.router.lifespan_context(omni):
            # Native handlers past the checks under test may fail without a
            # scheduler, which then answers 500.
            transport = httpx.ASGITransport(app=omni, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://omni"
            ) as client:
                await check(client, runtime_failure)
        runtime_failure.cancel()

    asyncio.run(run())


@pytest.mark.parametrize("native_requests", [0, 3])
def test_native_routes_and_lifetime_are_composed_without_a_proxy(
    native, native_requests
):
    @asynccontextmanager
    async def omni_lifespan(app):
        native.events.append("omni_start")
        yield
        native.events.append("omni_stop")

    omni = FastAPI(lifespan=omni_lifespan)

    @omni.get("/health")
    def health():
        return {"owner": "omni"}

    frontend = native.build()

    async def check(client, runtime_failure):
        assert (await client.get("/health")).json() == {"owner": "omni"}
        # Concurrent first requests wait for one start.
        listed = await asyncio.gather(
            *(client.get("/v1/videos") for _ in range(native_requests))
        )
        assert [response.status_code for response in listed] == [200] * len(listed)
        if native_requests:
            assert native.events == ["omni_start", "build", "initialize"]

            @frontend.app.get("/v1/images/owner")
            def owner():
                return {"owner": "native"}

            response = await client.get("/v1/images/owner")
            assert response.json() == {"owner": "native"}
        else:
            pass

    serve(omni, frontend, check)
    # A server whose clients never call the native routes loads no native
    # module and starts no native scheduler client.
    started = ["build", "initialize", "close"] if native_requests else []
    assert native.events == ["omni_start", *started, "omni_stop"]


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

    serve(omni, native.build(), check)


def test_failed_native_start_answers_503_until_restart(native, tmp_path):
    # A prompt enhancer config that cannot be read fails the native lifespan.
    native.args.prompt_enhancer_config = str(tmp_path / "missing-enhancer.json")
    frontend = native.build()

    async def check(client, runtime_failure):
        for _ in range(2):
            response = await client.get("/v1/videos")
            assert response.status_code == 503
            assert "missing-enhancer.json" in response.json()["detail"]
            assert "restart the server" in response.json()["detail"]
        # One start, whose scheduler client closed again, and other routes
        # keep serving.
        assert native.events == ["build", "initialize", "close"]
        assert (await client.get("/health")).status_code == 200

    serve(create_app(RunningClient(), model_name="native"), frontend, check)
    assert native.events == ["build", "initialize", "close"]


@pytest.mark.parametrize("installed", [False, True])
def test_native_worker_failure_is_used_when_installed(native, monkeypatch, installed):
    from sglang.multimodal_gen.runtime import scheduler_client

    received = []

    def legacy_initialize(self, server_args):
        received.append(None)

    def initialize(self, server_args, *, worker_failure=None):
        received.append(worker_failure)

    monkeypatch.setattr(
        scheduler_client.AsyncSchedulerClient,
        "initialize",
        initialize if installed else legacy_initialize,
    )

    async def check(client, runtime_failure):
        assert (await client.get("/v1/videos")).status_code == 200
        assert received == [runtime_failure if installed else None]

    serve(FastAPI(), native.build(), check)


@pytest.mark.parametrize("cause", ["scheduler", "stage"])
def test_in_flight_native_request_fails_when_generation_is_lost(native, cause):
    import zmq

    context = zmq.Context()
    scheduler = context.socket(zmq.ROUTER)
    port = scheduler.bind_to_random_port("tcp://127.0.0.1")
    native.args.scheduler_endpoints = [f"tcp://127.0.0.1:{port}"]
    frontend = native.build()
    entered = asyncio.Event()

    async def waiting():
        # Native SGLang waits like this for a scheduler that never answers.
        entered.set()
        await asyncio.Event().wait()

    async def check(client, runtime_failure):
        assert (await client.get("/v1/videos")).status_code == 200
        frontend.app.post("/v1/images/generations-waiting")(waiting)

        async def connected():
            while list(frontend.scheduler_events.values()) != [zmq.EVENT_CONNECTED]:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(connected(), 5)
        pending = asyncio.ensure_future(client.post("/v1/images/generations-waiting"))
        await asyncio.wait_for(entered.wait(), 5)
        if cause == "scheduler":
            scheduler.close(linger=0)
        else:
            runtime_failure.set_exception(RuntimeError("generation stage died"))
            runtime_failure.exception()
        response = await asyncio.wait_for(pending, 5)
        assert response.status_code == 503
        response = await client.post("/v1/images/generations-waiting")
        assert response.status_code == 503

    try:
        serve(FastAPI(), frontend, check)
    finally:
        scheduler.close(linger=0)
        context.term()


@pytest.mark.parametrize("configured", [False, True])
def test_native_requests_follow_the_server_media_policy(
    native, monkeypatch, tmp_path, configured
):
    import json

    from sglang_omni.preprocessing.resource_connector import MultiModalResourceConnector

    allowed = tmp_path / "media"
    allowed.mkdir()
    outside = str(tmp_path / "clip.mp4")
    inside = str(allowed / "clip.mp4")
    connector = MultiModalResourceConnector(
        allowed_local_media_path=allowed if configured else None,
        allowed_media_domains=["media.example"] if configured else None,
    )
    monkeypatch.setattr(media, "get_global_resource_connector", lambda: connector)
    remote = "https://media.example/frame.png"
    # Every request reaches a native media route. A refusal names the field.
    requests = [
        ("/v1/videos", {"json": {"prompt": "a", "output_path": "/srv"}}, True),
        ("/v1/videos", {"json": {"prompt": "a", "video_path": outside}}, configured),
        (
            "/v1/videos",
            {"json": {"prompt": "a", "video_path": "/srv/a\0b"}},
            configured,
        ),
        ("/v1/videos", {"json": {"prompt": "a", "reference_url": remote}}, configured),
        (
            "/v1/videos",
            {"json": {"prompt": "a", "reference_url": remote.upper()}},
            configured,
        ),
        (
            "/v1/videos",
            {
                "json": {
                    "prompt": "a",
                    "extra_params": json.dumps({"control_path": [outside]}),
                }
            },
            configured,
        ),
        (
            "/v1/actions/generations",
            {"json": {"input": {"observation": {"image": f"file://{outside}"}}}},
            configured,
        ),
        # Native decodes an inline action image in memory, so it is no path.
        (
            "/v1/actions/generations",
            {"json": {"input": {"observation": {"image": {"b64_json": "aGk="}}}}},
            False,
        ),
        (
            "/v1/images/edits",
            {"data": {"prompt": "a", "url": outside}},
            configured,
        ),
        (
            "/v1/images/edits",
            {"data": {"prompt": "a", "perf_dump_path": "/srv/perf.json"}},
            True,
        ),
        # A body nested past the decoder limit, which native may still decode.
        (
            "/v1/videos",
            {
                "content": "[" * 100000 + "]" * 100000,
                "headers": {"content-type": "application/json"},
            },
            True,
        ),
        # Native decodes a JSON encoded list to the allowed paths it names.
        (
            "/v1/videos",
            {"json": {"prompt": "a", "control_path": json.dumps([inside, inside])}},
            False,
        ),
        # Native decodes these JSON encoded values to the outside path.
        (
            "/v1/videos",
            {"json": {"prompt": "a", "control_path": json.dumps([outside])}},
            configured,
        ),
        (
            "/v1/videos",
            {"data": {"prompt": "a"}, "files": {"video_path": (None, f'"{outside}"')}},
            configured,
        ),
    ]

    async def check(client, runtime_failure):
        # Native resolves relative paths against the server working
        # directory, so the requests run with it outside and inside the
        # allowed directory.
        for cwd in (tmp_path, allowed):
            monkeypatch.chdir(cwd)
            for path, body, refused in requests:
                response = await client.post(path, **body)
                blocked = response.status_code == 400 and (
                    "refused" in response.text or "cannot set" in response.text
                )
                assert blocked == refused, (cwd, path, body, response.text)
            response = await client.post(
                "/v1/videos", json={"prompt": "a", "video_path": inside}
            )
            assert "refused" not in response.text
        # Native opens a file URL as written, relative to its working directory.
        monkeypatch.chdir(tmp_path)
        url = {"prompt": "a", "video_path": f"file://{allowed / 'clip.mp4'}"}
        response = await client.post("/v1/videos", json=url)
        assert ("refused" in response.text) is configured

    serve(FastAPI(), native.build(), check)


def test_cosmos3_resolves_native_settings_before_building(monkeypatch):
    from sglang_omni.models.cosmos3.config import Cosmos3PipelineConfig

    built = []
    frontend = object()
    monkeypatch.setattr(
        media.NativeMediaFrontend,
        "load",
        lambda path, request_fields: built.append(path) or frontend,
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
    assert overrides["strict_ports"] is True and overrides["input_save_path"] is None
    assert {"scheduler_port", "master_port", "output_path"} <= set(overrides)
    assert 19000 not in {overrides["scheduler_port"], overrides["master_port"]}
    # The stage saves its resolved native server arguments where the routes
    # load them, in a directory only this user can write.
    path = config.stages[0].factory.server_args_path
    assert os.stat(os.path.dirname(path)).st_mode & 0o777 == 0o700
    assert built == []
    assert build() is frontend
    assert built == [path]

    config = Cosmos3PipelineConfig(model_path="checkpoint")
    config.stages[0].factory.server_args_overrides = {"host": "0.0.0.0"}
    with pytest.raises(ValueError, match="loopback"):
        prepare(config, port=19000)


def test_native_frontend_builds_after_stage_startup(monkeypatch):
    from sglang_omni.serve import launcher

    events = []

    class Config:
        name = "native"
        native_media_factory = "model.native_media"

    class Runner:
        def __init__(self, config):
            self.coordinator = None
            self.prep = SimpleNamespace(
                placement_plan=SimpleNamespace(gpus={}), process_plan=None
            )

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
    monkeypatch.setattr(launcher, "placement_log_summary", lambda *args: {})
    with pytest.raises(RuntimeError, match="frontend build failed"):
        asyncio.run(launcher.run_server(Config()))
    # The frontend loads what the generation stage resolved while starting,
    # and a frontend failure still stops the stages.
    assert events == ["startup", "frontend", "stopped"]
