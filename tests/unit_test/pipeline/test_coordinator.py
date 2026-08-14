# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import gc
from dataclasses import replace

import pytest

from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import CompleteMessage, OmniRequest, StreamMessage
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane


class _BlockingAbortControlPlane(RecordingCoordinatorControlPlane):
    def __init__(self) -> None:
        super().__init__()
        self.abort_started = asyncio.Event()
        self.release_abort = asyncio.Event()

    async def broadcast_abort(self, msg) -> None:
        self.aborts.append(msg)
        self.abort_started.set()
        await self.release_abort.wait()


class _BlockingSubmitControlPlane(RecordingCoordinatorControlPlane):
    def __init__(self) -> None:
        super().__init__()
        self.submit_started = asyncio.Event()
        self.submit_cancelled = False

    async def submit_to_stage(self, stage, endpoint, msg) -> None:
        await super().submit_to_stage(stage, endpoint, msg)
        self.submit_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.submit_cancelled = True
            raise


class _BlockingAdminControlPlane(RecordingCoordinatorControlPlane):
    def __init__(self) -> None:
        super().__init__()
        self.admin_started = asyncio.Event()
        self.admin_cancelled = False

    async def send_admin(self, stage, endpoint, msg) -> None:
        await super().send_admin(stage, endpoint, msg)
        self.admin_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.admin_cancelled = True
            raise


async def _wait_for_abort(
    coordinator: Coordinator,
    control_plane: RecordingCoordinatorControlPlane,
    request_id: str,
) -> None:
    execution_id = coordinator._execution_ids[request_id]
    for _ in range(100):
        if any(msg.request_id == execution_id for msg in control_plane.aborts):
            return
        await asyncio.sleep(0)
    raise AssertionError(f"abort was not broadcast for {request_id}")


async def _handle_completion(_coordinator: Coordinator, msg: CompleteMessage) -> None:
    execution_id = _coordinator._execution_ids[msg.request_id]
    await _coordinator._handle_completion(replace(msg, request_id=execution_id))


async def _handle_stream(_coordinator: Coordinator, msg: StreamMessage) -> None:
    execution_id = _coordinator._execution_ids[msg.request_id]
    await _coordinator._handle_stream(replace(msg, request_id=execution_id))


def _last_execution_id(
    control_plane: RecordingCoordinatorControlPlane,
) -> str:
    return control_plane.submitted[-1][2].request_id


@pytest.mark.parametrize(
    ("max_stage_transitions", "expected_trace"),
    [(None, ()), (4, ("entry",))],
)
def test_coordinator_initializes_route_trace_only_for_feedback(
    max_stage_transitions, expected_trace
) -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="entry",
            max_stage_transitions=max_stage_transitions,
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("entry", "inproc://entry")

        await coordinator._submit_request("req", "hello")

        wire_message = control_plane.submitted[0][2]
        payload = wire_message.data
        assert payload.route_trace == expected_trace
        assert payload.public_request_id == "req"
        assert wire_message.request_id == payload.request_id
        assert wire_message.request_id != "req"
        assert coordinator._execution_ids["req"] == wire_message.request_id
        serialized = payload.to_dict()
        route_keys = {"route_transitions", "route_trace", "stream_positions"}
        if max_stage_transitions is None:
            assert route_keys.isdisjoint(serialized)
        else:
            assert route_keys <= serialized.keys()

    asyncio.run(_run())


def test_coordinator_constructor_enforces_runtime_bounds() -> None:
    for field, values in (
        ("max_stage_transitions", (True, 0, -1, 10_001, 1.5, "7")),
        ("stream_queue_maxsize", (True, 0, -1, 100_001, 1.5, "7")),
    ):
        for value in values:
            with pytest.raises(ValueError):
                Coordinator(
                    "inproc://complete",
                    "inproc://abort",
                    entry_stage="entry",
                    **{field: value},
                )


def test_coordinator_multi_terminal_completion_and_abort_contracts() -> None:
    """Preserves multi-terminal completion and abort cancellation semantics."""

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request("req-1", {"text": "hello"})
        await _handle_completion(
            coordinator, CompleteMessage("req-1", "decode", True, result={"text": "hi"})
        )
        assert not coordinator._completion_futures["req-1"].done()
        await _handle_completion(
            coordinator,
            CompleteMessage("req-1", "code2wav", True, result={"audio": "ok"}),
        )
        assert coordinator._completion_futures["req-1"].result() == {
            "decode": {"text": "hi"},
            "code2wav": {"audio": "ok"},
        }

        await coordinator._submit_request("req-2", "hello")
        future = coordinator._completion_futures["req-2"]
        execution_id = _last_execution_id(control_plane)
        assert await coordinator.abort("req-2") is True
        assert control_plane.aborts[0].request_id == execution_id
        with pytest.raises(asyncio.CancelledError):
            await future

    asyncio.run(_run())


def test_coordinator_resolves_active_terminal_subset_per_request() -> None:
    async def _run() -> None:
        def terminal_stages(request: OmniRequest) -> list[str]:
            assert isinstance(request, OmniRequest)
            if request.metadata.get("audio"):
                return ["decode", "code2wav"]
            return ["decode"]

        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
            terminal_stages_resolver=terminal_stages,
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request(
            "text-req",
            OmniRequest(inputs="hello", metadata={"audio": False}),
        )
        await _handle_completion(
            coordinator,
            CompleteMessage("text-req", "decode", True, result={"text": "hi"}),
        )
        assert coordinator._completion_futures["text-req"].result() == {"text": "hi"}

        await coordinator._submit_request("raw-text-req", "hello")
        await _handle_completion(
            coordinator,
            CompleteMessage("raw-text-req", "decode", True, result={"text": "raw"}),
        )
        assert coordinator._completion_futures["raw-text-req"].result() == {
            "text": "raw"
        }

        await coordinator._submit_request(
            "audio-req",
            OmniRequest(inputs="hello", metadata={"audio": True}),
        )
        await _handle_completion(
            coordinator,
            CompleteMessage("audio-req", "decode", True, result={"text": "hi"}),
        )
        assert not coordinator._completion_futures["audio-req"].done()
        await _handle_completion(
            coordinator,
            CompleteMessage(
                "audio-req",
                "code2wav",
                True,
                result={"audio": "ok"},
            ),
        )
        assert coordinator._completion_futures["audio-req"].result() == {
            "decode": {"text": "hi"},
            "code2wav": {"audio": "ok"},
        }

    asyncio.run(_run())


def test_coordinator_rejects_invalid_resolved_terminal_subset() -> None:
    async def _run() -> None:
        for resolved, error in (
            ([], "no terminal stages"),
            (["decode", "missing"], "outside the static terminal stages"),
            ("decode", "must return a sequence"),
        ):
            coordinator = Coordinator(
                "inproc://complete",
                "inproc://abort",
                entry_stage="preprocess",
                terminal_stages=["decode", "code2wav"],
                terminal_stages_resolver=lambda request, resolved=resolved: resolved,
            )
            coordinator.control_plane = RecordingCoordinatorControlPlane()
            coordinator.register_stage("preprocess", "inproc://preprocess")

            with pytest.raises(ValueError, match=error):
                await coordinator._submit_request("req-1", OmniRequest(inputs="hello"))
            assert coordinator._requests == {}
            assert coordinator.control_plane.submitted == []

    asyncio.run(_run())


def test_coordinator_stream_cleans_queue_when_terminal_resolver_rejects() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
            terminal_stages_resolver=lambda request: [],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        stream = coordinator.stream("req-1", OmniRequest(inputs="hello"))
        with pytest.raises(ValueError, match="no terminal stages"):
            await stream.__anext__()
        await stream.aclose()

        assert coordinator._stream_queues == {}
        assert coordinator._completion_futures == {}
        assert coordinator.control_plane.submitted == []

    asyncio.run(_run())


def test_coordinator_stream_uses_request_terminal_subset_after_cleanup() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
            terminal_stages_resolver=lambda request: ["decode"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        events = []

        async def _consume() -> None:
            async for event in coordinator.stream("req-1", OmniRequest(inputs="hello")):
                events.append(event)

        task = asyncio.create_task(_consume())
        for _ in range(10):
            if "req-1" in coordinator._requests:
                break
            await asyncio.sleep(0)
        await _handle_completion(
            coordinator, CompleteMessage("req-1", "decode", True, result={"text": "hi"})
        )
        await asyncio.wait_for(task, timeout=1)

        assert [event.from_stage for event in events] == ["decode"]

    asyncio.run(_run())


def test_coordinator_stream_received_event_pairs_terminal_chunk(monkeypatch) -> None:
    events: list[dict] = []
    monkeypatch.setattr(
        "sglang_omni.pipeline.coordinator._emit_event",
        lambda **kwargs: events.append(kwargs),
    )

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")
        queue: asyncio.Queue = asyncio.Queue()
        await coordinator._submit_request("req-1", "hello", stream_queue=queue)

        await _handle_stream(
            coordinator,
            StreamMessage(
                request_id="req-1",
                from_stage="decode",
                chunk={"text": "hi"},
                modality="text",
                chunk_id=1,
            ),
        )

        routed = queue.get_nowait()
        assert routed.chunk_id == 1

    asyncio.run(_run())

    receive_events = [
        event
        for event in events
        if event["event_name"] == "stage_stream_chunk_received"
    ]
    assert len(receive_events) == 1
    assert receive_events[0]["stage"] == "coordinator"
    admission = next(
        event for event in events if event["event_name"] == "request_admission"
    )
    assert receive_events[0]["request_id"] == admission["request_id"]
    assert receive_events[0]["request_id"] != "req-1"
    assert receive_events[0]["metadata"] == {
        "from_stage": "decode",
        "chunk_id": 1,
        "modality": "text",
        "public_request_id": "req-1",
    }


def test_slow_stream_consumer_is_failed_without_blocking_other_requests() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="decode",
            terminal_stages=["decode"],
            stream_queue_maxsize=2,
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("decode", "inproc://decode")

        stream = coordinator.stream("slow", "hello")
        first_chunk = asyncio.create_task(anext(stream))
        for _ in range(100):
            if "slow" in coordinator._stream_queues:
                break
            await asyncio.sleep(0)
        await _handle_stream(
            coordinator,
            StreamMessage(
                request_id="slow",
                from_stage="decode",
                chunk={"text": "0"},
                modality="text",
                chunk_id=0,
            ),
        )
        assert (await first_chunk).chunk_id == 0

        queue = coordinator._stream_queues["slow"]
        for chunk_id in (1, 2, 3):
            await _handle_stream(
                coordinator,
                StreamMessage(
                    request_id="slow",
                    from_stage="decode",
                    chunk={"text": str(chunk_id)},
                    modality="text",
                    chunk_id=chunk_id,
                ),
            )

        buffered = [queue.get_nowait() for _ in range(queue.qsize())]
        assert queue.maxsize == 3
        assert [msg.chunk_id for msg in buffered[:-1]] == [1, 2]
        terminal = buffered[-1]
        assert isinstance(terminal, CompleteMessage)
        assert terminal.success is False
        assert terminal.error == "client stream buffer capacity exceeded: 2"
        assert "slow" not in coordinator._requests
        await _wait_for_abort(coordinator, control_plane, "slow")
        assert [msg.request_id for msg in control_plane.aborts] == [
            _last_execution_id(control_plane)
        ]

        await coordinator._submit_request("healthy", "hello")
        healthy_future = coordinator._completion_futures["healthy"]
        await _handle_completion(
            coordinator, CompleteMessage("healthy", "decode", True, result="ok")
        )
        assert await healthy_future == "ok"

        await stream.aclose()
        assert coordinator._stream_queues == {}
        assert "slow" not in coordinator._completion_futures

    asyncio.run(_run())


def test_stream_terminal_capacity_is_separate_and_completions_are_deduplicated() -> (
    None
):
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="entry",
            terminal_stages=["a", "b", "c"],
            stream_queue_maxsize=1,
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("entry", "inproc://entry")

        stream = coordinator.stream("req", "hello")
        first_event = asyncio.create_task(anext(stream))
        for _ in range(100):
            if "req" in coordinator._stream_queues:
                break
            await asyncio.sleep(0)
        queue = coordinator._stream_queues["req"]
        assert queue.maxsize == 4

        await _handle_stream(
            coordinator,
            StreamMessage("req", "entry", {"text": "0"}, modality="text", chunk_id=0),
        )
        assert (await first_event).chunk_id == 0
        await _handle_stream(
            coordinator,
            StreamMessage("req", "entry", {"text": "1"}, modality="text", chunk_id=1),
        )
        await _handle_completion(coordinator, CompleteMessage("req", "a", True, 1))
        await _handle_completion(coordinator, CompleteMessage("req", "a", True, 99))
        await _handle_completion(coordinator, CompleteMessage("req", "b", True, 2))
        await _handle_completion(coordinator, CompleteMessage("req", "c", True, 3))

        remaining = []
        while True:
            try:
                remaining.append(await asyncio.wait_for(anext(stream), timeout=1))
            except StopAsyncIteration:
                break

        assert [type(msg) for msg in remaining] == [
            StreamMessage,
            CompleteMessage,
            CompleteMessage,
            CompleteMessage,
        ]
        assert [msg.from_stage for msg in remaining[1:]] == ["a", "b", "c"]
        assert coordinator._requests == {}
        assert coordinator._stream_queues == {}
        assert coordinator._stream_pending_chunks == {}
        assert control_plane.aborts == []

    asyncio.run(_run())


def test_fail_pending_requests_cannot_block_on_a_full_stream_buffer() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="entry",
            terminal_stages=["a", "b", "c"],
            stream_queue_maxsize=1,
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("entry", "inproc://entry")

        stream = coordinator.stream("req", "hello")
        first_event = asyncio.create_task(anext(stream))
        for _ in range(100):
            if "req" in coordinator._stream_queues:
                break
            await asyncio.sleep(0)
        await _handle_stream(
            coordinator,
            StreamMessage("req", "entry", {"text": "0"}, modality="text", chunk_id=0),
        )
        await first_event
        await _handle_stream(
            coordinator,
            StreamMessage("req", "entry", {"text": "1"}, modality="text", chunk_id=1),
        )
        await _handle_completion(coordinator, CompleteMessage("req", "a", True, 1))
        await _handle_completion(coordinator, CompleteMessage("req", "b", True, 2))

        await asyncio.wait_for(
            coordinator.fail_pending_requests(RuntimeError("stage died")), timeout=1
        )
        assert coordinator._requests == {}

        observed = []
        with pytest.raises(RuntimeError, match="stage died"):
            async for event in stream:
                observed.append(event)
        assert [event.from_stage for event in observed[1:]] == ["a", "b"]
        assert coordinator._stream_queues == {}
        assert coordinator._stream_pending_chunks == {}

    asyncio.run(_run())


@pytest.mark.parametrize("abort_first", [False, True])
def test_stream_overflow_and_client_abort_have_one_terminal_and_broadcast(
    abort_first: bool,
) -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="entry",
            terminal_stages=["done"],
            stream_queue_maxsize=1,
        )
        control_plane = _BlockingAbortControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("entry", "inproc://entry")

        stream = coordinator.stream("req", "hello")
        first_event = asyncio.create_task(anext(stream))
        for _ in range(100):
            if "req" in coordinator._stream_queues:
                break
            await asyncio.sleep(0)
        await _handle_stream(
            coordinator,
            StreamMessage("req", "entry", {"text": "0"}, modality="text", chunk_id=0),
        )
        await first_event
        await _handle_stream(
            coordinator,
            StreamMessage("req", "entry", {"text": "1"}, modality="text", chunk_id=1),
        )

        if abort_first:
            client_abort = asyncio.create_task(coordinator.abort("req"))
            await control_plane.abort_started.wait()
            await _handle_stream(
                coordinator,
                StreamMessage(
                    "req", "entry", {"text": "2"}, modality="text", chunk_id=2
                ),
            )
            expected_abort_result = True
            broadcast_task = client_abort
        else:
            await _handle_stream(
                coordinator,
                StreamMessage(
                    "req", "entry", {"text": "2"}, modality="text", chunk_id=2
                ),
            )
            await control_plane.abort_started.wait()
            assert await asyncio.wait_for(coordinator.abort("req"), timeout=1) is False
            expected_abort_result = None
            broadcast_task = coordinator._abort_broadcast_tasks["req"]

        queued = list(coordinator._stream_queues["req"]._queue)
        assert sum(isinstance(msg, CompleteMessage) for msg in queued) == 1
        assert len(control_plane.aborts) == 1
        control_plane.release_abort.set()
        result = await asyncio.wait_for(asyncio.shield(broadcast_task), timeout=1)
        if expected_abort_result is not None:
            assert result is expected_abort_result
        await asyncio.sleep(0)
        assert coordinator._abort_tasks == {}
        assert coordinator._abort_broadcast_tasks == {}
        await stream.aclose()

    asyncio.run(_run())


def test_stop_cancels_and_joins_background_abort_broadcasts() -> None:
    class BlockingAbortControlPlane(RecordingCoordinatorControlPlane):
        def __init__(self) -> None:
            super().__init__()
            self.abort_started = asyncio.Event()
            self.abort_cancelled = False

        async def broadcast_abort(self, msg) -> None:
            self.aborts.append(msg)
            self.abort_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.abort_cancelled = True
                raise

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="entry",
            terminal_stages=["done"],
        )
        control_plane = BlockingAbortControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("entry", "inproc://entry")
        await coordinator.start()
        stream_queue: asyncio.Queue = asyncio.Queue(maxsize=2)
        await coordinator._submit_request("req", "hello", stream_queue=stream_queue)

        await _handle_completion(
            coordinator, CompleteMessage("req", "done", False, error="failed")
        )
        await control_plane.abort_started.wait()
        task = coordinator._abort_broadcast_tasks["req"]
        await asyncio.wait_for(coordinator.stop(), timeout=1)

        assert task.done()
        assert control_plane.abort_cancelled is True
        assert control_plane.closed is True
        assert coordinator._abort_tasks == {}
        assert coordinator._abort_broadcast_tasks == {}

    asyncio.run(_run())


def test_stream_handles_completion_delivered_inside_submit() -> None:
    class CompleteInsideSubmitControlPlane(RecordingCoordinatorControlPlane):
        coordinator: Coordinator

        async def submit_to_stage(self, stage, endpoint, msg) -> None:
            await super().submit_to_stage(stage, endpoint, msg)
            await self.coordinator._handle_completion(
                CompleteMessage(
                    request_id=msg.request_id,
                    from_stage="decode",
                    success=True,
                    result="ok",
                )
            )

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="entry",
            terminal_stages=["decode", "code2wav"],
            terminal_stages_resolver=lambda _request: ["decode"],
        )
        control_plane = CompleteInsideSubmitControlPlane()
        control_plane.coordinator = coordinator
        coordinator.control_plane = control_plane
        coordinator.register_stage("entry", "inproc://entry")

        async def _consume() -> list[CompleteMessage | StreamMessage]:
            return [event async for event in coordinator.stream("public-id", "hello")]

        events = await asyncio.wait_for(
            _consume(),
            timeout=1,
        )

        assert len(events) == 1
        assert events[0].request_id == "public-id"
        assert events[0].from_stage == "decode"
        assert events[0].result == "ok"
        assert coordinator._requests == {}
        assert coordinator._completion_futures == {}
        assert coordinator._stream_queues == {}
        assert coordinator._execution_ids == {}
        assert coordinator._request_ids_by_execution == {}

    asyncio.run(_run())


def test_stop_fails_active_callers_and_rejects_late_admission() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="entry",
            terminal_stages=["done"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("entry", "inproc://entry")
        await coordinator.start()

        submit_task = asyncio.create_task(coordinator.submit("nonstream", "hello"))
        stream = coordinator.stream("stream", "hello")
        stream_task = asyncio.create_task(anext(stream))
        for _ in range(100):
            if set(coordinator._requests) == {"nonstream", "stream"}:
                break
            await asyncio.sleep(0)
        execution_ids = set(coordinator._execution_ids.values())

        await asyncio.wait_for(coordinator.stop(), timeout=1)

        with pytest.raises(RuntimeError, match="Coordinator stopped"):
            await submit_task
        with pytest.raises(RuntimeError, match="Coordinator stopped"):
            await stream_task
        await stream.aclose()
        await asyncio.sleep(0)

        assert {msg.request_id for msg in control_plane.aborts} == execution_ids
        assert coordinator._requests == {}
        assert coordinator._completion_futures == {}
        assert coordinator._stream_queues == {}
        assert coordinator._stream_pending_chunks == {}
        assert coordinator._execution_ids == {}
        assert coordinator._request_ids_by_execution == {}
        assert coordinator._abort_tasks == {}
        assert coordinator._abort_broadcast_tasks == {}
        assert control_plane.closed is True

        submitted_count = len(control_plane.submitted)
        with pytest.raises(RuntimeError, match="stopping or stopped"):
            await coordinator.submit("after-stop", "hello")
        late_stream = coordinator.stream("after-stop-stream", "hello")
        with pytest.raises(RuntimeError, match="stopping or stopped"):
            await anext(late_stream)
        await late_stream.aclose()
        assert await coordinator.abort("after-stop") is False
        assert len(control_plane.submitted) == submitted_count

    asyncio.run(_run())


def test_stop_gates_admission_while_abort_tasks_are_draining() -> None:
    class CancellationDelayedControlPlane(RecordingCoordinatorControlPlane):
        def __init__(self) -> None:
            super().__init__()
            self.cancel_started = asyncio.Event()
            self.release_cancel = asyncio.Event()

        async def broadcast_abort(self, msg) -> None:
            self.aborts.append(msg)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancel_started.set()
                await self.release_cancel.wait()
                raise

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="entry",
            terminal_stages=["done"],
        )
        control_plane = CancellationDelayedControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("entry", "inproc://entry")
        await coordinator.start()

        submit_task = asyncio.create_task(coordinator.submit("active", "hello"))
        for _ in range(100):
            if "active" in coordinator._requests:
                break
            await asyncio.sleep(0)

        stop_task = asyncio.create_task(coordinator.stop())
        await control_plane.cancel_started.wait()
        assert coordinator._stopping is True
        assert stop_task.done() is False
        assert await coordinator.abort("active") is False
        with pytest.raises(RuntimeError, match="stopping or stopped"):
            await coordinator.submit("late", "hello")
        with pytest.raises(RuntimeError, match="Coordinator stopped"):
            await submit_task

        control_plane.release_cancel.set()
        await asyncio.wait_for(stop_task, timeout=1)
        assert coordinator._abort_tasks == {}
        assert coordinator._abort_broadcast_tasks == {}
        assert coordinator._execution_ids == {}
        assert coordinator._request_ids_by_execution == {}
        assert control_plane.closed is True

    asyncio.run(_run())


def test_stop_cancels_and_joins_inflight_entry_stage_send() -> None:
    async def _run() -> None:
        loop = asyncio.get_running_loop()
        unhandled: list[dict] = []
        loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="entry",
            terminal_stages=["done"],
        )
        control_plane = _BlockingSubmitControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("entry", "inproc://entry")
        await coordinator.start()

        request_task = asyncio.create_task(coordinator.submit("blocked", "hello"))
        await control_plane.submit_started.wait()
        execution_id = coordinator._execution_ids["blocked"]

        await asyncio.wait_for(coordinator.stop(), timeout=1)

        with pytest.raises(RuntimeError, match="Coordinator stopped"):
            await request_task
        assert control_plane.submit_cancelled is True
        assert [msg.request_id for msg in control_plane.aborts] == [execution_id]
        assert coordinator._submission_tasks == {}
        assert coordinator._requests == {}
        assert coordinator._completion_futures == {}
        assert coordinator._execution_ids == {}
        assert coordinator._request_ids_by_execution == {}
        gc.collect()
        await asyncio.sleep(0)
        assert unhandled == []
        loop.set_exception_handler(None)

    asyncio.run(_run())


def test_submit_cancelled_after_stop_gate_still_broadcasts_cleanup() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="entry",
            terminal_stages=["done"],
        )
        control_plane = _BlockingSubmitControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("entry", "inproc://entry")
        await coordinator.start()

        request_task = asyncio.create_task(coordinator.submit("raced", "hello"))
        await control_plane.submit_started.wait()
        execution_id = coordinator._execution_ids["raced"]
        coordinator.begin_stop()
        request_task.cancel()

        with pytest.raises(RuntimeError, match="Coordinator stopped"):
            await request_task
        await asyncio.wait_for(coordinator.stop(), timeout=1)

        assert [msg.request_id for msg in control_plane.aborts] == [execution_id]
        assert coordinator._execution_ids == {}
        assert coordinator._request_ids_by_execution == {}

    asyncio.run(_run())


def test_stop_cancels_and_joins_inflight_admin_send() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="entry",
        )
        control_plane = _BlockingAdminControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("entry", "inproc://entry")
        await coordinator.start()

        admin_task = asyncio.create_task(coordinator.admin("model_info", timeout_s=300))
        await control_plane.admin_started.wait()
        assert coordinator._admin_ops

        await asyncio.wait_for(coordinator.stop(), timeout=1)

        with pytest.raises(RuntimeError, match="Coordinator stopped"):
            await admin_task
        assert control_plane.admin_cancelled is True
        assert coordinator._admin_ops == {}
        assert coordinator._admin_tasks == set()

    asyncio.run(_run())


def test_fatal_error_aborts_and_reaches_blocked_submitter() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="entry",
            terminal_stages=["done"],
        )
        control_plane = _BlockingSubmitControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("entry", "inproc://entry")
        await coordinator.start()

        request_task = asyncio.create_task(coordinator.submit("blocked", "hello"))
        await control_plane.submit_started.wait()
        execution_id = coordinator._execution_ids["blocked"]

        await coordinator.fail_pending_requests(
            RuntimeError("completion transport failed")
        )
        await asyncio.wait_for(coordinator.stop(), timeout=1)

        with pytest.raises(RuntimeError, match="completion transport failed"):
            await request_task
        assert [msg.request_id for msg in control_plane.aborts] == [execution_id]
        assert coordinator._requests == {}
        assert coordinator._completion_futures == {}
        assert coordinator._execution_ids == {}

    asyncio.run(_run())


def test_fatal_error_reaches_blocked_admin_operation() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="entry",
        )
        control_plane = _BlockingAdminControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("entry", "inproc://entry")
        await coordinator.start()

        admin_task = asyncio.create_task(coordinator.admin("model_info", timeout_s=300))
        await control_plane.admin_started.wait()

        await coordinator.fail_pending_requests(
            RuntimeError("completion transport failed")
        )
        await asyncio.wait_for(coordinator.stop(), timeout=1)

        with pytest.raises(RuntimeError, match="completion transport failed"):
            await admin_task
        assert coordinator._admin_ops == {}
        assert coordinator._admin_tasks == set()

    asyncio.run(_run())


def test_fail_pending_requests_aborts_dispatched_stage_work() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="entry",
            terminal_stages=["done"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("entry", "inproc://entry")
        await coordinator.start()

        request_task = asyncio.create_task(coordinator.submit("active", "hello"))
        for _ in range(100):
            if "active" in coordinator._requests:
                break
            await asyncio.sleep(0)
        execution_id = coordinator._execution_ids["active"]

        await coordinator.fail_pending_requests(RuntimeError("stage died"))
        with pytest.raises(RuntimeError, match="stage died"):
            await request_task
        for _ in range(100):
            if control_plane.aborts:
                break
            await asyncio.sleep(0)

        assert [msg.request_id for msg in control_plane.aborts] == [execution_id]
        assert coordinator._requests == {}

        await coordinator.stop()

    asyncio.run(_run())


def test_shutdown_stage_signals_are_parallel_and_bounded() -> None:
    class BlockingShutdownControlPlane(RecordingCoordinatorControlPlane):
        def __init__(self) -> None:
            super().__init__()
            self.started: set[str] = set()
            self.cancelled: set[str] = set()

        async def send_shutdown(self, stage, endpoint) -> None:
            del endpoint
            self.started.add(stage)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.add(stage)
                raise

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="a",
        )
        control_plane = BlockingShutdownControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("a", "inproc://a")
        coordinator.register_stage("b", "inproc://b")

        await coordinator.shutdown_stages(timeout_s=0.01)

        assert control_plane.started == {"a", "b"}
        assert control_plane.cancelled == {"a", "b"}

        for invalid in (True, "1"):
            with pytest.raises(TypeError, match="positive number"):
                await coordinator.shutdown_stages(timeout_s=invalid)
        for invalid in (0, float("inf"), float("nan")):
            with pytest.raises(ValueError, match="positive"):
                await coordinator.shutdown_stages(timeout_s=invalid)

    asyncio.run(_run())


def test_stop_racing_control_plane_start_cannot_restart_coordinator() -> None:
    class BlockingStartControlPlane(RecordingCoordinatorControlPlane):
        def __init__(self) -> None:
            super().__init__()
            self.start_entered = asyncio.Event()
            self.release_start = asyncio.Event()

        async def start(self) -> None:
            self.start_entered.set()
            await self.release_start.wait()

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="entry",
        )
        control_plane = BlockingStartControlPlane()
        coordinator.control_plane = control_plane

        start_task = asyncio.create_task(coordinator.start())
        await control_plane.start_entered.wait()
        await coordinator.stop()
        control_plane.release_start.set()

        with pytest.raises(RuntimeError, match="stopped during startup"):
            await start_task
        assert coordinator._closed is True
        assert coordinator._running is False
        assert control_plane.closed is True

    asyncio.run(_run())


def test_control_plane_start_failure_closes_partial_resources() -> None:
    class FailingStartControlPlane(RecordingCoordinatorControlPlane):
        async def start(self) -> None:
            raise RuntimeError("bind failed")

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="entry",
        )
        control_plane = FailingStartControlPlane()
        coordinator.control_plane = control_plane

        with pytest.raises(RuntimeError, match="bind failed"):
            await coordinator.start()

        assert coordinator._running is False
        assert control_plane.closed is True

    asyncio.run(_run())


def test_stream_message_round_trips_terminal_chunk_id() -> None:
    msg = StreamMessage(
        request_id="req-1",
        from_stage="decode",
        chunk={"text": "hi"},
        modality="text",
        chunk_id=3,
    )

    round_trip = StreamMessage.from_dict(msg.to_dict())

    assert round_trip.chunk_id == 3
    assert round_trip.modality == "text"


def test_coordinator_failure_completion_fails_fast_and_cleans_state() -> None:
    """Preserves fail-fast behavior and cleanup after any terminal failure."""

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request("req-1", "hello")
        future = coordinator._completion_futures["req-1"]
        await _handle_completion(
            coordinator, CompleteMessage("req-1", "decode", True, result={"text": "hi"})
        )
        assert coordinator._partial_results["req-1"] == {"decode": {"text": "hi"}}

        await _handle_completion(
            coordinator, CompleteMessage("req-1", "code2wav", False, error="boom")
        )

        with pytest.raises(RuntimeError, match="boom"):
            await future
        assert "req-1" not in coordinator._requests
        assert "req-1" not in coordinator._partial_results
        await _wait_for_abort(coordinator, control_plane, "req-1")
        assert control_plane.aborts[-1].request_id == _last_execution_id(control_plane)

    asyncio.run(_run())


def test_coordinator_fail_pending_requests_resolves_waiters() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request("req-1", "hello")
        future = coordinator._completion_futures["req-1"]

        await coordinator.fail_pending_requests(RuntimeError("stage died"))

        with pytest.raises(RuntimeError, match="stage died"):
            await future
        assert coordinator._requests == {}
        assert coordinator._partial_results == {}

    asyncio.run(_run())


async def _drive_stream_until_registered(coordinator: Coordinator, request_id: str):
    """Start consuming a stream and return (task, error_sink, future) once the
    request's completion future has been created."""
    error_sink: list[str] = []

    async def _consume() -> None:
        try:
            async for _msg in coordinator.stream(request_id, "hello"):
                pass
        except RuntimeError as exc:
            error_sink.append(str(exc))

    task = asyncio.create_task(_consume())
    for _ in range(100):
        if request_id in coordinator._completion_futures:
            break
        await asyncio.sleep(0)
    future = coordinator._completion_futures[request_id]
    return task, error_sink, future


def test_coordinator_stream_early_close_aborts_and_cleans_state() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        stream = coordinator.stream("req-1", OmniRequest(inputs="hello"))
        first_chunk = asyncio.create_task(anext(stream))
        for _ in range(100):
            if "req-1" in coordinator._stream_queues:
                break
            await asyncio.sleep(0)
        await _handle_stream(
            coordinator,
            StreamMessage(
                request_id="req-1",
                from_stage="decode",
                chunk={"text": "hello"},
                modality="text",
            ),
        )
        await first_chunk
        execution_id = _last_execution_id(control_plane)
        await stream.aclose()

        assert [msg.request_id for msg in control_plane.aborts] == [execution_id]
        assert "req-1" not in coordinator._requests
        assert "req-1" not in coordinator._stream_queues
        assert "req-1" not in coordinator._completion_futures

    asyncio.run(_run())


def test_stream_close_after_one_terminal_aborts_remaining_terminal_work() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        stream = coordinator.stream("req-1", "hello")
        first_terminal = asyncio.create_task(anext(stream))
        for _ in range(100):
            if "req-1" in coordinator._requests:
                break
            await asyncio.sleep(0)
        await _handle_completion(
            coordinator,
            CompleteMessage("req-1", "decode", True, result={"text": "done"}),
        )
        assert (await first_terminal).from_stage == "decode"
        assert coordinator._partial_results["req-1"] == {"decode": {"text": "done"}}

        execution_id = _last_execution_id(control_plane)
        await stream.aclose()

        assert [msg.request_id for msg in control_plane.aborts] == [execution_id]
        assert coordinator._requests == {}
        assert coordinator._partial_results == {}
        assert coordinator._completion_futures == {}
        assert coordinator._stream_queues == {}

    asyncio.run(_run())


def test_coordinator_stream_natural_completion_does_not_abort() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        async def _consume() -> list[CompleteMessage | StreamMessage]:
            return [
                message
                async for message in coordinator.stream(
                    "req-1", OmniRequest(inputs="hello")
                )
            ]

        task = asyncio.create_task(_consume())
        for _ in range(100):
            if "req-1" in coordinator._requests:
                break
            await asyncio.sleep(0)
        await _handle_completion(
            coordinator,
            CompleteMessage(
                request_id="req-1",
                from_stage="decode",
                success=True,
                result={"text": "hello"},
            ),
        )
        messages = await task

        assert len(messages) == 1
        assert control_plane.aborts == []
        assert "req-1" not in coordinator._requests
        assert "req-1" not in coordinator._stream_queues
        assert "req-1" not in coordinator._completion_futures

    asyncio.run(_run())


def test_duplicate_stream_preserves_existing_non_stream_request() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request("req-1", "original")
        original_request = coordinator._requests["req-1"]
        original_future = coordinator._completion_futures["req-1"]

        duplicate = coordinator.stream("req-1", "duplicate")
        with pytest.raises(ValueError, match="already exists"):
            await anext(duplicate)

        assert coordinator._requests["req-1"] is original_request
        assert coordinator._completion_futures["req-1"] is original_future
        assert "req-1" not in coordinator._stream_queues
        assert control_plane.aborts == []

        assert await coordinator.abort("req-1") is True
        with pytest.raises(asyncio.CancelledError):
            await original_future

    asyncio.run(_run())


def test_completed_stream_allows_request_id_reuse_after_owner_closes() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        stream = coordinator.stream("req-1", "original")
        terminal_event = asyncio.create_task(anext(stream))
        for _ in range(100):
            if "req-1" in coordinator._requests:
                break
            await asyncio.sleep(0)
        await _handle_completion(
            coordinator,
            CompleteMessage("req-1", "decode", True, result={"text": "done"}),
        )
        assert (await terminal_event).result == {"text": "done"}

        old_future = coordinator._completion_futures["req-1"]
        old_queue = coordinator._stream_queues["req-1"]
        assert "req-1" not in coordinator._requests

        with pytest.raises(ValueError, match="already exists"):
            await coordinator._submit_request("req-1", "replacement")
        assert coordinator._completion_futures["req-1"] is old_future
        assert coordinator._stream_queues["req-1"] is old_queue

        await stream.aclose()
        assert "req-1" not in coordinator._completion_futures
        assert "req-1" not in coordinator._stream_queues
        await coordinator._submit_request("req-1", "replacement")
        assert coordinator._requests["req-1"].request_id == "req-1"

    asyncio.run(_run())


def test_stream_abort_reserves_request_id_while_broadcast_is_in_flight() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        control_plane = _BlockingAbortControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        stream = coordinator.stream("req-1", "original")
        first_chunk = asyncio.create_task(anext(stream))
        for _ in range(100):
            if "req-1" in coordinator._requests:
                break
            await asyncio.sleep(0)
        await _handle_stream(
            coordinator,
            StreamMessage(
                request_id="req-1",
                from_stage="decode",
                chunk={"text": "partial"},
                modality="text",
            ),
        )
        await first_chunk

        close_task = asyncio.create_task(stream.aclose())
        await control_plane.abort_started.wait()

        await _handle_completion(
            coordinator,
            CompleteMessage("req-1", "decode", True, result={"text": "done"}),
        )
        assert "req-1" not in coordinator._requests
        assert "req-1" in coordinator._abort_tasks

        with pytest.raises(ValueError, match="already exists"):
            await coordinator._submit_request("req-1", "replacement")

        control_plane.release_abort.set()
        await close_task
        assert coordinator._abort_tasks == {}
        assert coordinator._completion_futures == {}
        assert coordinator._stream_queues == {}

    asyncio.run(_run())


def test_stream_cancellation_is_preserved_after_abort_cleanup() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        next_event = asyncio.create_task(anext(coordinator.stream("req-1", "hello")))
        for _ in range(100):
            if "req-1" in coordinator._requests:
                break
            await asyncio.sleep(0)

        execution_id = _last_execution_id(control_plane)
        next_event.cancel()
        with pytest.raises(asyncio.CancelledError):
            await next_event

        assert [msg.request_id for msg in control_plane.aborts] == [execution_id]
        assert coordinator._requests == {}
        assert coordinator._completion_futures == {}
        assert coordinator._stream_queues == {}
        assert coordinator._abort_tasks == {}

    asyncio.run(_run())


def test_coordinator_stream_abort_failure_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingAbortControlPlane(RecordingCoordinatorControlPlane):
        async def broadcast_abort(self, msg) -> None:
            self.aborts.append(msg)
            raise RuntimeError("abort transport unavailable")

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        control_plane = FailingAbortControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        stream = coordinator.stream("req-1", OmniRequest(inputs="hello"))
        first_chunk = asyncio.create_task(anext(stream))
        for _ in range(100):
            if "req-1" in coordinator._stream_queues:
                break
            await asyncio.sleep(0)
        await _handle_stream(
            coordinator,
            StreamMessage(
                request_id="req-1",
                from_stage="decode",
                chunk={"text": "hello"},
                modality="text",
            ),
        )
        await first_chunk
        await stream.aclose()

        assert "req-1" not in coordinator._requests
        assert "req-1" not in coordinator._stream_queues
        assert "req-1" not in coordinator._completion_futures
        assert "req-1" not in coordinator._stream_pending_chunks
        assert coordinator._abort_tasks == {}

    with caplog.at_level("WARNING"):
        asyncio.run(_run())
    assert "Failed to abort request req-1" in caplog.text


def test_coordinator_stream_abort_cancels_future_without_unretrieved_exception() -> (
    None
):
    """Aborting a streaming request cancels its completion future instead of
    setting an exception no one retrieves, so the event loop never reports a
    'Future exception was never retrieved' error."""

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        loop = asyncio.get_running_loop()
        handler_contexts: list = []
        loop.set_exception_handler(
            lambda _loop, context: handler_contexts.append(context)
        )

        task, error_sink, future = await _drive_stream_until_registered(
            coordinator, "req-1"
        )

        assert await coordinator.abort("req-1") is True
        await asyncio.wait_for(task, timeout=1)

        # Stream terminated via its queue; the future is cancelled rather than
        # carrying an un-retrieved exception.
        assert error_sink == ["aborted"]
        assert future.cancelled() is True
        assert "req-1" not in coordinator._completion_futures

        # Dropping the future must not trip the loop's exception handler.
        del future
        gc.collect()
        assert not any(
            "never retrieved" in str(ctx.get("message", "")) for ctx in handler_contexts
        )

    asyncio.run(_run())


def test_coordinator_stream_fail_pending_requests_cancels_future() -> None:
    """A coordinator failure reaches the stream without leaving an exception
    on its unused completion future."""

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        loop = asyncio.get_running_loop()
        handler_contexts: list = []
        loop.set_exception_handler(
            lambda _loop, context: handler_contexts.append(context)
        )

        task, error_sink, future = await _drive_stream_until_registered(
            coordinator, "req-1"
        )

        await coordinator.fail_pending_requests(RuntimeError("stage died"))
        await asyncio.wait_for(task, timeout=1)

        assert error_sink == ["stage died"]
        assert future.cancelled() is True
        assert "req-1" not in coordinator._completion_futures

        del future
        gc.collect()
        assert not any(
            "never retrieved" in str(ctx.get("message", "")) for ctx in handler_contexts
        )

    asyncio.run(_run())


def test_coordinator_stream_stage_failure_cancels_future() -> None:
    """A stage failure on a streaming request cancels the completion future
    (which the stream consumer never awaits) rather than setting an exception
    that would be reported as never retrieved."""

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        task, error_sink, future = await _drive_stream_until_registered(
            coordinator, "req-1"
        )

        await _handle_completion(
            coordinator, CompleteMessage("req-1", "decode", False, error="boom")
        )
        await asyncio.wait_for(task, timeout=1)

        assert error_sink == ["boom"]
        assert future.cancelled() is True
        assert "req-1" not in coordinator._completion_futures

    asyncio.run(_run())


def test_nonstream_cancellation_aborts_and_releases_request() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="entry",
            terminal_stages=["done"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("entry", "inproc://entry")

        task = asyncio.create_task(coordinator.submit("public-id", "hello"))
        for _ in range(100):
            if "public-id" in coordinator._requests:
                break
            await asyncio.sleep(0)
        execution_id = coordinator._execution_ids["public-id"]

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        for _ in range(100):
            if not coordinator._request_id_is_reserved("public-id"):
                break
            await asyncio.sleep(0)

        assert [msg.request_id for msg in control_plane.aborts] == [execution_id]
        assert coordinator._requests == {}
        assert coordinator._completion_futures == {}
        assert coordinator._execution_ids == {}
        assert coordinator._request_ids_by_execution == {}
        assert coordinator._abort_tasks == {}

    asyncio.run(_run())


def test_ambiguous_submit_failure_cannot_cross_talk_after_id_reuse() -> None:
    class AcceptedThenRaisedControlPlane(RecordingCoordinatorControlPlane):
        async def submit_to_stage(self, stage, endpoint, msg) -> None:
            await super().submit_to_stage(stage, endpoint, msg)
            if len(self.submitted) == 1:
                raise RuntimeError("accepted then transport failed")

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="entry",
            terminal_stages=["done"],
        )
        control_plane = AcceptedThenRaisedControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("entry", "inproc://entry")

        with pytest.raises(RuntimeError, match="accepted then transport failed"):
            await coordinator.submit("same", "old")
        old_execution_id = _last_execution_id(control_plane)
        for _ in range(100):
            if not coordinator._request_id_is_reserved("same"):
                break
            await asyncio.sleep(0)

        replacement = asyncio.create_task(coordinator.submit("same", "new"))
        for _ in range(100):
            if len(control_plane.submitted) == 2:
                break
            await asyncio.sleep(0)
        new_execution_id = _last_execution_id(control_plane)
        assert new_execution_id != old_execution_id

        await coordinator._handle_completion(
            CompleteMessage(old_execution_id, "done", True, "stale")
        )
        await asyncio.sleep(0)
        assert replacement.done() is False

        await coordinator._handle_completion(
            CompleteMessage(new_execution_id, "done", True, "fresh")
        )
        assert await asyncio.wait_for(replacement, timeout=1) == "fresh"
        assert [msg.request_id for msg in control_plane.aborts] == [old_execution_id]
        assert coordinator._execution_ids == {}
        assert coordinator._request_ids_by_execution == {}

    asyncio.run(_run())


def test_late_completion_after_abort_cannot_resolve_reused_id() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="entry",
            terminal_stages=["done"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("entry", "inproc://entry")

        old_request = asyncio.create_task(coordinator.submit("same", "old"))
        for _ in range(100):
            if "same" in coordinator._requests:
                break
            await asyncio.sleep(0)
        old_execution_id = coordinator._execution_ids["same"]
        assert await coordinator.abort("same") is True
        with pytest.raises(asyncio.CancelledError):
            await old_request
        for _ in range(100):
            if not coordinator._request_id_is_reserved("same"):
                break
            await asyncio.sleep(0)

        replacement = asyncio.create_task(coordinator.submit("same", "new"))
        for _ in range(100):
            if "same" in coordinator._requests:
                break
            await asyncio.sleep(0)
        new_execution_id = coordinator._execution_ids["same"]
        assert new_execution_id != old_execution_id

        await coordinator._handle_completion(
            CompleteMessage(old_execution_id, "done", True, "late")
        )
        await asyncio.sleep(0)
        assert replacement.done() is False

        await coordinator._handle_completion(
            CompleteMessage(new_execution_id, "done", True, "fresh")
        )
        assert await asyncio.wait_for(replacement, timeout=1) == "fresh"
        assert coordinator._execution_ids == {}
        assert coordinator._request_ids_by_execution == {}

    asyncio.run(_run())
