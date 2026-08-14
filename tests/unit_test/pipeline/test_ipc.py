# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import queue
import signal
import threading
from pathlib import Path
from types import FrameType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import sglang_omni.pipeline.mp_runner as mp_runner
import sglang_omni.pipeline.runtime_config as runtime_config
from sglang_omni.config.schema import EndpointsConfig, PipelineConfig, StageConfig
from sglang_omni.profiler.event_recorder import get_recorder
from tests.unit_test.fixtures.pipeline_fakes import FakeMpContext, FakeRelay


def noop_factory():
    return None


def failing_factory():
    raise RuntimeError("factory boom")


class _FakeControlPlane:
    def __init__(self, recv_endpoint: str):
        self.recv_endpoint = recv_endpoint


class _FakeStage:
    name = "preprocessing"

    def __init__(self, recv_endpoint: str):
        self.control_plane = _FakeControlPlane(recv_endpoint)

    async def run(self) -> None:
        await asyncio.Event().wait()


class _FakeCoordinator:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def run_completion_loop(self) -> None:
        await asyncio.Event().wait()

    async def stop(self) -> None:
        self.stopped = True


def _make_config(base_path: Path, *, stream_queue_maxsize: int = 256) -> PipelineConfig:
    return PipelineConfig(
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        entry_stage="preprocessing",
        stages=[
            StageConfig(
                name="preprocessing",
                process="pipeline",
                factory=f"{__name__}.noop_factory",
                terminal=True,
            )
        ],
        endpoints=EndpointsConfig(base_path=str(base_path)),
        stream_queue_maxsize=stream_queue_maxsize,
    )


@pytest.fixture(autouse=True)
def _fake_stage_relay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sglang_omni.comm.router.create_relay",
        lambda relay_type, **kwargs: FakeRelay(device=kwargs.get("device", "cpu")),
    )


def test_ipc_runtime_dir_creation_and_close_contracts(tmp_path: Path) -> None:
    """Preserves IPC runtime directory creation, uniqueness, and idempotent cleanup."""
    ipc_config = _make_config(tmp_path)

    runtime_a = runtime_config.create_ipc_runtime_dir(ipc_config)
    runtime_b = runtime_config.create_ipc_runtime_dir(ipc_config)
    assert runtime_a is not None
    assert runtime_b is not None
    assert runtime_a.path != runtime_b.path

    runtime_path = runtime_a.path
    runtime_a.close()
    runtime_a.close()
    runtime_b.close()
    assert not runtime_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_prepare_pipeline_runtime_owns_or_preserves_ipc_runtime_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserves owned IPC cleanup and caller-owned IPC directory preservation."""
    config = _make_config(tmp_path)

    def fail_allocate_endpoints(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr(runtime_config, "allocate_endpoints", fail_allocate_endpoints)

    with pytest.raises(RuntimeError, match="boom"):
        runtime_config.prepare_pipeline_runtime(config)
    assert list(tmp_path.iterdir()) == []

    caller_owned = runtime_config.create_ipc_runtime_dir(config)
    assert caller_owned is not None
    caller_path = caller_owned.path
    with pytest.raises(RuntimeError, match="boom"):
        runtime_config.prepare_pipeline_runtime(config, ipc_runtime_dir=caller_owned)
    assert caller_path.exists()
    caller_owned.close()
    assert list(tmp_path.iterdir()) == []


def test_prepare_pipeline_runtime_returns_managed_ipc_runtime_dir(
    tmp_path: Path,
) -> None:
    """Preserves managed IPC runtime directory ownership in runtime prep."""
    prep = runtime_config.prepare_pipeline_runtime(_make_config(tmp_path))
    runtime_dir = prep.runtime_dir
    assert runtime_dir is not None
    try:
        assert runtime_dir.path.exists()
        assert str(runtime_dir.path) in prep.endpoints["stage_preprocessing"]
    finally:
        runtime_dir.close()

    assert list(tmp_path.iterdir()) == []


def test_ipc_stage_groups_use_unique_endpoints_for_same_model_name(
    tmp_path: Path,
) -> None:
    """Preserves unique IPC endpoints across same-model pipeline instances."""
    config = _make_config(tmp_path)
    prep_a = runtime_config.prepare_pipeline_runtime(config)
    prep_b = runtime_config.prepare_pipeline_runtime(config)
    assert prep_a.runtime_dir is not None
    assert prep_b.runtime_dir is not None

    try:
        groups_a = mp_runner._build_stage_groups(
            config,
            FakeMpContext(),
            stages_cfg=prep_a.stages_cfg,
            name_map=prep_a.name_map,
            endpoints=prep_a.endpoints,
            placement_plan=prep_a.placement_plan,
            process_plan=prep_a.process_plan,
        )
        groups_b = mp_runner._build_stage_groups(
            config,
            FakeMpContext(),
            stages_cfg=prep_b.stages_cfg,
            name_map=prep_b.name_map,
            endpoints=prep_b.endpoints,
            placement_plan=prep_b.placement_plan,
            process_plan=prep_b.process_plan,
        )

        assert prep_a.endpoints["completion"] != prep_b.endpoints["completion"]
        assert groups_a[0].leader_endpoint != groups_b[0].leader_endpoint
    finally:
        prep_a.runtime_dir.close()
        prep_b.runtime_dir.close()

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_mp_runner_cleans_runtime_dir_on_start_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserves IPC runtime directory cleanup when runner startup fails."""

    class FailingCoordinator:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        async def start(self) -> None:
            raise RuntimeError("boom")

        async def stop(self) -> None:
            return None

    monkeypatch.setattr(mp_runner, "Coordinator", FailingCoordinator)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(tmp_path))

    with pytest.raises(RuntimeError, match="boom"):
        await runner.start()

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_mp_runner_cleans_spawned_groups_when_later_spawn_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserves spawned process cleanup if a later stage group fails to spawn."""

    class FakeProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False
            self.join_count = 0
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

        def terminate(self) -> None:
            self.terminated = True
            self._alive = False

        def kill(self) -> None:
            self.killed = True
            self._alive = False

        def join(self, timeout=None) -> None:
            del timeout
            self.join_count += 1

    class FakeGroup:
        def __init__(self, stage_name: str, *, fail_spawn: bool = False) -> None:
            self.stage_name = stage_name
            self.group_name = stage_name
            self.fail_spawn = fail_spawn
            self.process = FakeProcess() if not fail_spawn else None
            self.channels_closed = False

        @property
        def processes(self) -> list[FakeProcess]:
            return [self.process] if self.process is not None else []

        def spawn(self, ctx) -> None:
            del ctx
            if self.fail_spawn:
                raise RuntimeError(f"spawn failed for {self.stage_name}")

        async def wait_ready(self, timeout: float) -> None:
            del timeout

        def close_control_channels(self) -> None:
            self.channels_closed = True

        async def shutdown(self, join_timeout: float = 30.0) -> None:
            del join_timeout
            if self.process is not None:
                if self.process.is_alive():
                    self.process.terminate()
                self.process.join(timeout=0)
            self.close_control_channels()

    first_group = FakeGroup("preprocessing")
    second_group = FakeGroup("thinker", fail_spawn=True)
    monkeypatch.setattr(mp_runner, "Coordinator", _FakeCoordinator)
    monkeypatch.setattr(
        mp_runner,
        "_build_stage_groups",
        lambda *a, **k: [first_group, second_group],
    )

    runner = mp_runner.MultiProcessPipelineRunner(_make_config(tmp_path))
    with pytest.raises(RuntimeError, match="spawn failed"):
        await runner.start()

    assert first_group.process.terminated
    assert first_group.process.join_count >= 1
    assert first_group.channels_closed
    assert second_group.channels_closed
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_mp_runner_startup_failure_includes_child_factory_traceback(
    tmp_path: Path,
) -> None:
    config = PipelineConfig(
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        name="x",
        entry_stage="preprocessing",
        stages=[
            StageConfig(
                name="preprocessing",
                process="pipeline",
                factory=f"{__name__}.failing_factory",
                terminal=True,
            )
        ],
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
    )
    runner = mp_runner.MultiProcessPipelineRunner(config)

    with pytest.raises(RuntimeError, match="factory boom"):
        await runner.start(timeout=10.0)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_mp_runner_stop_cleans_runtime_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserves IPC runtime directory cleanup when the runner stops."""

    coordinator_settings = []
    begin_stop_calls = []

    class FakeCoordinator:
        def __init__(
            self,
            completion_endpoint: str,
            abort_endpoint: str,
            entry_stage: str,
            terminal_stages: list[str] | None = None,
            terminal_stages_resolver=None,
            max_stage_transitions: int | None = None,
            stream_queue_maxsize: int = 256,
        ) -> None:
            coordinator_settings.append(
                {
                    "max_stage_transitions": max_stage_transitions,
                    "stream_queue_maxsize": stream_queue_maxsize,
                }
            )
            del (
                abort_endpoint,
                entry_stage,
                terminal_stages,
                terminal_stages_resolver,
                max_stage_transitions,
                stream_queue_maxsize,
            )
            self.control_plane = SimpleNamespace(
                completion_endpoint=completion_endpoint
            )

        async def start(self) -> None:
            return None

        async def run_completion_loop(self) -> None:
            await asyncio.Event().wait()

        def register_stage(self, name: str, endpoint: str) -> None:
            del name, endpoint

        async def shutdown_stages(self) -> None:
            return None

        def begin_stop(self) -> None:
            begin_stop_calls.append(True)

        async def quiesce(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    class FakeGroup:
        stage_name = "preprocessing"
        leader_endpoint = "ipc://stage.sock"
        tp_size = 1
        process_count = 1
        processes: list[object] = []
        stage_control_endpoints = {"preprocessing": "ipc://stage.sock"}

        def __init__(self) -> None:
            self.shutdown_called = False

        def spawn(self, ctx) -> None:
            del ctx

        async def wait_ready(self, timeout: float) -> None:
            del timeout

        def any_dead(self) -> bool:
            return False

        def dead_summary(self) -> str:
            return "(none)"

        async def shutdown(self) -> None:
            self.shutdown_called = True

    group = FakeGroup()
    monkeypatch.setattr(mp_runner, "Coordinator", FakeCoordinator)
    monkeypatch.setattr(mp_runner, "_build_stage_groups", lambda *a, **k: [group])

    runner = mp_runner.MultiProcessPipelineRunner(
        _make_config(tmp_path, stream_queue_maxsize=17)
    )
    await runner.start()
    assert len([path for path in tmp_path.iterdir() if path.is_dir()]) == 1

    await runner.stop()

    assert group.shutdown_called
    assert coordinator_settings == [
        {"max_stage_transitions": None, "stream_queue_maxsize": 17}
    ]
    assert begin_stop_calls == [True]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_mp_runner_completion_loop_failure_fails_runtime(
    tmp_path: Path,
) -> None:
    class FailingRuntimeCoordinator:
        def __init__(self) -> None:
            self.failed_with = None
            self.begin_stop_called = False
            self.shutdown_called = False
            self.stop_called = False

        async def fail_pending_requests(self, error) -> None:
            self.failed_with = error

        def begin_stop(self) -> None:
            self.begin_stop_called = True

        async def quiesce(self) -> None:
            return None

        async def shutdown_stages(self) -> None:
            self.shutdown_called = True

        async def stop(self) -> None:
            self.stop_called = True

    async def fail_completion_loop() -> None:
        await asyncio.sleep(0)
        raise RuntimeError("completion transport failed")

    runner = mp_runner.MultiProcessPipelineRunner(_make_config(tmp_path))
    coordinator = FailingRuntimeCoordinator()
    runner._coordinator = coordinator
    runner._fatal_event = asyncio.Event()
    runner._started = True
    completion_task = asyncio.create_task(fail_completion_loop())
    runner._completion_task = completion_task
    completion_task.add_done_callback(runner._on_completion_task_done)

    with pytest.raises(RuntimeError, match="completion transport failed"):
        await asyncio.wait_for(runner.wait_failed(), timeout=1)
    for _ in range(100):
        if coordinator.stop_called:
            break
        await asyncio.sleep(0)

    assert coordinator.failed_with is not None
    assert str(coordinator.failed_with) == "completion transport failed"
    assert coordinator.begin_stop_called is True
    assert coordinator.shutdown_called is True
    assert coordinator.stop_called is True
    assert runner._completion_task is None
    assert runner._started is False


@pytest.mark.asyncio
async def test_mp_runner_rejects_completion_loop_exit_during_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExitedCoordinator(_FakeCoordinator):
        async def run_completion_loop(self) -> None:
            return None

        def register_stage(self, name: str, endpoint: str) -> None:
            del name, endpoint

    class YieldingGroup:
        group_name = "preprocessing"
        process_count = 1
        processes: list[object] = []
        stage_control_endpoints = {"preprocessing": "ipc://stage.sock"}

        def __init__(self) -> None:
            self.channels_closed = False

        def spawn(self, ctx) -> None:
            del ctx

        async def wait_ready(self, timeout: float) -> None:
            del timeout
            await asyncio.sleep(0)

        def any_dead(self) -> bool:
            return False

        def dead_summary(self) -> str:
            return "(none)"

        def close_control_channels(self) -> None:
            self.channels_closed = True

        async def shutdown(self, join_timeout: float = 30.0) -> None:
            del join_timeout
            self.close_control_channels()

    group = YieldingGroup()
    monkeypatch.setattr(mp_runner, "Coordinator", ExitedCoordinator)
    monkeypatch.setattr(mp_runner, "_build_stage_groups", lambda *a, **k: [group])
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(tmp_path))

    with pytest.raises(RuntimeError, match="exited during startup"):
        await runner.start()

    assert group.channels_closed is True
    assert runner._completion_task is None
    assert runner._coordinator is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_mp_runner_cancelled_start_cleans_owned_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StartCoordinator(_FakeCoordinator):
        def register_stage(self, name: str, endpoint: str) -> None:
            del name, endpoint

    class BlockingGroup:
        group_name = "preprocessing"
        process_count = 1
        processes: list[object] = []
        stage_control_endpoints = {"preprocessing": "ipc://stage.sock"}

        def __init__(self) -> None:
            self.wait_entered = asyncio.Event()
            self.channels_closed = False

        def spawn(self, ctx) -> None:
            del ctx

        async def wait_ready(self, timeout: float) -> None:
            del timeout
            self.wait_entered.set()
            await asyncio.Event().wait()

        def close_control_channels(self) -> None:
            self.channels_closed = True

        async def shutdown(self, join_timeout: float = 30.0) -> None:
            del join_timeout
            self.close_control_channels()

    group = BlockingGroup()
    monkeypatch.setattr(mp_runner, "Coordinator", StartCoordinator)
    monkeypatch.setattr(mp_runner, "_build_stage_groups", lambda *a, **k: [group])
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(tmp_path))

    start_task = asyncio.create_task(runner.start())
    await group.wait_entered.wait()
    start_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start_task

    assert group.channels_closed is True
    assert runner._start_task is None
    assert runner._completion_task is None
    assert runner._coordinator is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_mp_runner_stop_during_startup_joins_cancelled_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StartCoordinator(_FakeCoordinator):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.begin_stop_called = False

        def begin_stop(self) -> None:
            self.begin_stop_called = True

        def register_stage(self, name: str, endpoint: str) -> None:
            del name, endpoint

    class BlockingGroup:
        group_name = "preprocessing"
        process_count = 1
        processes: list[object] = []
        stage_control_endpoints = {"preprocessing": "ipc://stage.sock"}

        def __init__(self) -> None:
            self.wait_entered = asyncio.Event()
            self.channels_closed = False

        def spawn(self, ctx) -> None:
            del ctx

        async def wait_ready(self, timeout: float) -> None:
            del timeout
            self.wait_entered.set()
            await asyncio.Event().wait()

        def close_control_channels(self) -> None:
            self.channels_closed = True

        async def shutdown(self, join_timeout: float = 30.0) -> None:
            del join_timeout
            self.close_control_channels()

    group = BlockingGroup()
    monkeypatch.setattr(mp_runner, "Coordinator", StartCoordinator)
    monkeypatch.setattr(mp_runner, "_build_stage_groups", lambda *a, **k: [group])
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(tmp_path))

    start_task = asyncio.create_task(runner.start())
    await group.wait_entered.wait()
    coordinator = runner._coordinator
    await asyncio.wait_for(runner.stop(), timeout=1)

    with pytest.raises(asyncio.CancelledError):
        await start_task
    assert coordinator.begin_stop_called is True
    assert coordinator.stopped is True
    assert group.channels_closed is True
    assert runner._start_task is None
    assert runner._coordinator is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_mp_runner_failure_cleanup_error_cannot_wedge_failure_state(
    tmp_path: Path,
) -> None:
    class RaisingCoordinator:
        async def fail_pending_requests(self, error) -> None:
            del error
            raise RuntimeError("cleanup failed")

    runner = mp_runner.MultiProcessPipelineRunner(_make_config(tmp_path))
    runner._coordinator = RaisingCoordinator()
    runner._fatal_event = asyncio.Event()
    runner._started = True
    stop_calls = []

    async def _stop() -> None:
        stop_calls.append(True)

    runner.stop = _stop
    primary = RuntimeError("completion transport failed")
    await runner._fail_runtime(primary)

    assert runner._fatal_event.is_set()
    assert runner._fatal_error is primary
    assert stop_calls == [True]
    with pytest.raises(RuntimeError, match="completion transport failed"):
        await runner.wait_failed()

    await runner._fail_runtime(RuntimeError("later failure"))
    assert stop_calls == [True]


@pytest.mark.asyncio
async def test_mp_runner_failed_start_cleanup_preserves_primary_error(
    tmp_path: Path,
) -> None:
    class FailingCleanupGroup:
        group_name = "broken"

        def __init__(self) -> None:
            self.channels_closed = False

        async def shutdown(self, join_timeout: float = 30.0) -> None:
            del join_timeout
            raise RuntimeError("join cleanup failed")

        def close_control_channels(self) -> None:
            self.channels_closed = True

    group = FailingCleanupGroup()
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(tmp_path))
    runner._groups = [group]
    runner._ipc_runtime_dir = SimpleNamespace(closed=False)
    runner._ipc_runtime_dir.close = lambda: setattr(
        runner._ipc_runtime_dir, "closed", True
    )

    with pytest.raises(RuntimeError, match="primary startup failure"):
        try:
            raise RuntimeError("primary startup failure")
        except BaseException:
            await runner._cleanup_on_failure()
            raise

    assert group.channels_closed is True
    assert runner._groups == []
    assert runner._ipc_runtime_dir is None


@pytest.mark.asyncio
async def test_stage_group_wait_ready_prefers_queued_startup_error() -> None:
    class AliveProcess:
        exitcode = None

        @staticmethod
        def is_alive() -> bool:
            return True

    group = mp_runner.StageGroup(
        "test",
        [SimpleNamespace(stage_specs=[], process_name="pipeline")],
    )
    startup_errors: queue.Queue[str] = queue.Queue()
    startup_errors.put("RuntimeError: factory boom")
    group._processes = [AliveProcess()]
    group._ready_events = [threading.Event()]
    group._startup_error_channels = [startup_errors]

    with pytest.raises(RuntimeError, match="factory boom"):
        await group.wait_ready(timeout=0)


@pytest.mark.asyncio
async def test_stage_group_shutdown_joins_processes_concurrently() -> None:
    barrier = threading.Barrier(2)

    class FakeProcess:
        def __init__(self, name: str) -> None:
            self.name = name
            self.pid = 1
            self.met_peer = False
            self._alive = True

        def join(self, timeout=None) -> None:
            del timeout
            try:
                barrier.wait(timeout=0.5)
                self.met_peer = True
            except threading.BrokenBarrierError:
                pass
            self._alive = False

        def is_alive(self) -> bool:
            return self._alive

    group = mp_runner.StageGroup(
        "test",
        [SimpleNamespace(stage_specs=[])],
    )
    processes = [FakeProcess("a"), FakeProcess("b")]
    group._processes = processes

    await group.shutdown(join_timeout=1)

    assert all(process.met_peer for process in processes)
    assert group.processes == []


@pytest.mark.asyncio
async def test_stage_group_shutdown_terminates_after_join_failure() -> None:
    class FailingJoinProcess:
        name = "failed-join"
        pid = 1

        def __init__(self) -> None:
            self._alive = True
            self.join_count = 0
            self.terminated = False

        def join(self, timeout=None) -> None:
            del timeout
            self.join_count += 1
            if self.join_count == 1:
                raise RuntimeError("join failed")
            self._alive = False

        def is_alive(self) -> bool:
            return self._alive

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self._alive = False

    group = mp_runner.StageGroup(
        "test",
        [SimpleNamespace(stage_specs=[])],
    )
    process = FailingJoinProcess()
    group._processes = [process]

    await group.shutdown(join_timeout=1)

    assert process.terminated is True
    assert process.is_alive() is False
    assert group.processes == []


@pytest.mark.asyncio
async def test_stage_group_shutdown_cancellation_waits_for_cleanup() -> None:
    join_entered = threading.Event()
    release_join = threading.Event()

    class BlockingJoinProcess:
        name = "blocking-join"
        pid = 1

        def __init__(self) -> None:
            self._alive = True
            self.join_count = 0
            self.terminated = False

        def join(self, timeout=None) -> None:
            del timeout
            self.join_count += 1
            if self.join_count == 1:
                join_entered.set()
                release_join.wait(timeout=1)
                return
            self._alive = False

        def is_alive(self) -> bool:
            return self._alive

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self._alive = False

    group = mp_runner.StageGroup(
        "test",
        [SimpleNamespace(stage_specs=[])],
    )
    process = BlockingJoinProcess()
    group._processes = [process]

    shutdown_task = asyncio.create_task(group.shutdown(join_timeout=1))
    assert await asyncio.to_thread(join_entered.wait, 1)
    shutdown_task.cancel()
    await asyncio.sleep(0)
    assert shutdown_task.done() is False
    shutdown_task.cancel()
    await asyncio.sleep(0)
    assert shutdown_task.done() is False
    release_join.set()

    with pytest.raises(asyncio.CancelledError):
        await shutdown_task
    assert process.terminated is True
    assert process.is_alive() is False
    assert group.processes == []


async def _run_launcher_with_fake_runner(
    *,
    config: PipelineConfig,
    serve_mock: AsyncMock | None,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, FastAPI, SimpleNamespace]:
    app = FastAPI()
    profiler_calls = SimpleNamespace(starts=[], stops=[])

    from sglang_omni.serve import launcher

    runner_ref = None

    class FakeRunner:
        def __init__(self, pipeline_config: PipelineConfig) -> None:
            del pipeline_config
            nonlocal runner_ref
            self.coordinator = _FakeCoordinator()
            self.stage_control_endpoints = {
                "preprocessing": "ipc://stage_preprocessing.sock"
            }
            self.started = False
            self.stopped = False
            # launcher._run_server reads .prep.placement_plan / .process_plan
            # after start() to log the resolved topology. Provide empty stubs
            # that satisfy _placement_log_summary's attribute access.
            self.prep = SimpleNamespace(
                placement_plan=SimpleNamespace(gpus={}),
                process_plan=SimpleNamespace(
                    groups=(),
                    tp_stage_to_processes={},
                ),
            )
            runner_ref = self

        async def start(self, timeout: float) -> None:
            del timeout
            self.started = True

        async def stop(self) -> None:
            self.stopped = True

        async def wait_failed(self) -> None:
            await asyncio.Future()

    class FakeProfilerControl:
        def __init__(self, stage_control_endpoints: dict[str, str]) -> None:
            del stage_control_endpoints

        async def broadcast_start(self, **kwargs) -> None:
            profiler_calls.starts.append(kwargs)

        async def broadcast_stop(self, **kwargs) -> None:
            profiler_calls.stops.append(kwargs)

    monkeypatch.setattr(launcher, "_find_available_port", lambda host, port: port)
    monkeypatch.setattr(launcher, "MultiProcessPipelineRunner", FakeRunner)
    monkeypatch.setattr(launcher, "ProfilerControlClient", FakeProfilerControl)
    monkeypatch.setattr(launcher, "create_app", lambda *a, **k: app)
    if serve_mock is not None:
        monkeypatch.setattr(launcher.uvicorn.Server, "serve", serve_mock)

    await launcher._run_server(config, port=8000)
    assert runner_ref is not None
    return runner_ref, app, profiler_calls


@pytest.mark.asyncio
async def test_launcher_uses_runner_and_mounts_profiler_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    server_serve = AsyncMock(return_value=None)

    runner, app, profiler_calls = await _run_launcher_with_fake_runner(
        config=config,
        serve_mock=server_serve,
        monkeypatch=monkeypatch,
    )

    assert runner.started
    assert runner.stopped
    server_serve.assert_awaited_once()
    try:
        with TestClient(app) as client:
            start_resp = client.post(
                "/start_profile",
                json={
                    "enable_torch": False,
                    "event_dir": str(tmp_path / "events"),
                },
            )
            stop_resp = client.post("/stop_profile", json={})
        assert start_resp.status_code == 200
        assert stop_resp.status_code == 200
        assert profiler_calls.starts
        assert profiler_calls.starts[0]["enable_torch"] is False
        assert profiler_calls.starts[0]["event_dir"] == str(tmp_path / "events")
        assert profiler_calls.stops == [{"run_id": None}]
    finally:
        rec = get_recorder()
        if rec.is_active():
            rec.stop()


def test_start_profile_request_only_mode_does_not_require_trace_template(
    tmp_path: Path,
) -> None:
    from sglang_omni.serve import launcher

    class FakeProfilerControl:
        def __init__(self) -> None:
            self.starts: list[dict] = []

        async def broadcast_start(self, **kwargs) -> None:
            self.starts.append(kwargs)

    app = FastAPI()
    ctl = FakeProfilerControl()
    launcher._mount_profiler_routes(app, ctl, profiler_dir=None)
    event_dir = str(tmp_path / "events")

    try:
        with TestClient(app) as client:
            resp = client.post(
                "/start_profile",
                json={"enable_torch": False, "event_dir": event_dir},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["enable_torch"] is False
        assert body["trace_path_template"] == ""
        assert body["event_dir"] == event_dir
        assert ctl.starts
        assert ctl.starts[0]["enable_torch"] is False
        assert ctl.starts[0]["trace_path_template"] == ""
        assert ctl.starts[0]["event_dir"] == event_dir
    finally:
        rec = get_recorder()
        if rec.is_active():
            rec.stop()


def test_start_profile_torch_mode_still_requires_trace_template() -> None:
    from sglang_omni.serve import launcher

    class FakeProfilerControl:
        async def broadcast_start(self, **kwargs) -> None:
            raise AssertionError("start_profile should fail before broadcasting")

    app = FastAPI()
    launcher._mount_profiler_routes(app, FakeProfilerControl(), profiler_dir=None)

    with TestClient(app) as client:
        resp = client.post("/start_profile", json={"enable_torch": True})
    assert resp.status_code == 400
    assert "trace_path_template is required" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_launcher_stops_runner_when_server_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    server_serve = AsyncMock(side_effect=RuntimeError("server failed"))

    with pytest.raises(RuntimeError, match="server failed"):
        await _run_launcher_with_fake_runner(
            config=config,
            serve_mock=server_serve,
            monkeypatch=monkeypatch,
        )

    server_serve.assert_awaited_once()


@pytest.mark.asyncio
async def test_pipeline_uvicorn_server_consumes_handled_sigterm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.serve import launcher

    config = _make_config(tmp_path)
    replayed_signals: list[int] = []
    server_ref: launcher.uvicorn.Server | None = None
    original_handler = signal.getsignal(signal.SIGTERM)

    def recording_handler(sig: int, frame: FrameType | None) -> None:
        del frame
        replayed_signals.append(sig)

    async def serve_until_sigterm(
        server: launcher.uvicorn.Server,
        sockets=None,
    ) -> None:
        del sockets
        nonlocal server_ref
        server_ref = server
        signal.raise_signal(signal.SIGTERM)
        assert server.should_exit

    monkeypatch.setattr(launcher.uvicorn.Server, "_serve", serve_until_sigterm)
    signal.signal(signal.SIGTERM, recording_handler)
    try:
        runner, _, _ = await _run_launcher_with_fake_runner(
            config=config,
            serve_mock=None,
            monkeypatch=monkeypatch,
        )
        assert signal.getsignal(signal.SIGTERM) is recording_handler
    finally:
        signal.signal(signal.SIGTERM, original_handler)

    assert isinstance(server_ref, launcher._PipelineUvicornServer)
    assert runner.started
    assert runner.stopped
    assert replayed_signals == []
    assert server_ref._captured_signals == []


@pytest.mark.asyncio
async def test_launcher_preserves_runner_start_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)

    from sglang_omni.serve import launcher

    class FakeRunner:
        def __init__(self, pipeline_config: PipelineConfig) -> None:
            del pipeline_config

        async def start(self, timeout: float) -> None:
            del timeout
            raise RuntimeError("start failed")

        async def stop(self) -> None:
            raise AssertionError("launcher should not stop a runner that failed start")

    monkeypatch.setattr(launcher, "_find_available_port", lambda host, port: port)
    monkeypatch.setattr(launcher, "MultiProcessPipelineRunner", FakeRunner)

    with pytest.raises(RuntimeError, match="start failed"):
        await launcher._run_server(config, port=8000)
