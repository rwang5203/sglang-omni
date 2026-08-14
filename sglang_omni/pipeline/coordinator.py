# SPDX-License-Identifier: Apache-2.0
"""Coordinator for managing the multi-stage pipeline."""

import asyncio
import logging
import math
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator

from sglang_omni.pipeline.control_plane import CoordinatorControlPlane
from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.proto import (
    AbortMessage,
    AdminMessage,
    AdminOperation,
    AdminResult,
    AdminResultMessage,
    CompleteMessage,
    OmniRequest,
    RequestInfo,
    RequestState,
    StageInfo,
    StagePayload,
    StreamMessage,
    SubmitMessage,
    is_update_action,
)

logger = logging.getLogger(__name__)

_ABORT_DELIVERY_TIMEOUT_S = 0.1


@dataclass
class _AdminPendingOperation:
    expected_stages: set[str]
    action: str
    results: dict[str, AdminResult] = field(default_factory=dict)
    future: asyncio.Future | None = None


class Coordinator:
    """Central coordinator for the multi-stage pipeline.

    Responsibilities:
    - Register stages
    - Submit requests to entry stage
    - Track request state
    - Handle completions
    - Broadcast abort signals
    """

    def __init__(
        self,
        completion_endpoint: str,
        abort_endpoint: str,
        entry_stage: str,
        terminal_stages: list[str] | None = None,
        terminal_stages_resolver: (
            Callable[[OmniRequest], list[str] | None] | None
        ) = None,
        max_stage_transitions: int | None = None,
        stream_queue_maxsize: int = 256,
    ):
        """Initialize coordinator.

        Args:
            completion_endpoint: ZMQ endpoint to receive completions
            abort_endpoint: ZMQ endpoint for abort broadcasts
            entry_stage: Name of the entry stage for new requests
            terminal_stages: Terminal stage names. When multiple are given,
                the coordinator waits for all to complete before resolving.
        """
        if max_stage_transitions is not None and (
            isinstance(max_stage_transitions, bool)
            or not isinstance(max_stage_transitions, int)
            or not 1 <= max_stage_transitions <= 10_000
        ):
            raise ValueError(
                "max_stage_transitions must be None or an integer from 1 to 10000"
            )
        if (
            isinstance(stream_queue_maxsize, bool)
            or not isinstance(stream_queue_maxsize, int)
            or not 1 <= stream_queue_maxsize <= 100_000
        ):
            raise ValueError("stream_queue_maxsize must be an integer from 1 to 100000")
        self.entry_stage = entry_stage
        self._max_stage_transitions = max_stage_transitions
        self._stream_queue_maxsize = stream_queue_maxsize
        self._terminal_stages: set[str] = (
            set(terminal_stages) if terminal_stages else set()
        )
        self._terminal_stages_resolver = terminal_stages_resolver
        self._partial_results: dict[str, dict[str, Any]] = {}

        # Control plane
        self.control_plane = CoordinatorControlPlane(
            completion_endpoint=completion_endpoint,
            abort_endpoint=abort_endpoint,
        )

        # Stage registry
        self._stages: dict[str, StageInfo] = {}

        # Request tracking
        self._requests: dict[str, RequestInfo] = {}
        self._completion_futures: dict[str, asyncio.Future] = {}
        self._stream_queues: dict[
            str, asyncio.Queue[CompleteMessage | StreamMessage]
        ] = {}
        self._stream_pending_chunks: dict[str, int] = {}
        # Stage traffic uses a fresh opaque ID for every admission. Public IDs
        # may be reused after cleanup without accepting delayed events from an
        # earlier execution.
        self._execution_ids: dict[str, str] = {}
        self._request_ids_by_execution: dict[str, str] = {}
        self._abort_scheduled_executions: set[str] = set()
        self._submission_tasks: dict[str, asyncio.Task[Any]] = {}
        # Abort messages carry only the request ID. A strongly held task keeps
        # local admission closed and lets the broadcast survive caller cancellation.
        self._abort_tasks: dict[str, asyncio.Task[bool]] = {}
        self._abort_broadcast_tasks: dict[str, asyncio.Task[bool]] = {}
        self._admin_ops: dict[str, _AdminPendingOperation] = {}
        self._admin_tasks: set[asyncio.Task[Any]] = set()
        self._admin_lock = asyncio.Lock()

        # State
        self._running = False
        self._stopping = False
        self._closed = False
        self._quiesce_task: asyncio.Task[None] | None = None
        self._stop_task: asyncio.Task[None] | None = None
        self._fatal_error: str | None = None

    def register_stage(self, name: str, endpoint: str) -> None:
        """Register a stage.

        Args:
            name: Stage name
            endpoint: ZMQ endpoint for the stage
        """
        self._stages[name] = StageInfo(name=name, control_endpoint=endpoint)
        logger.info("Coordinator registered stage: %s at %s", name, endpoint)

    async def start(self) -> None:
        """Start the coordinator."""
        if self._stopping or self._closed:
            raise RuntimeError("Coordinator is stopping or stopped")
        try:
            await self.control_plane.start()
        except BaseException:
            self.control_plane.close()
            raise
        if self._stopping or self._closed:
            self.control_plane.close()
            raise RuntimeError("Coordinator stopped during startup")
        self._running = True
        logger.info("Coordinator started")

    async def stop(self) -> None:
        """Stop the coordinator."""
        if self._closed:
            return
        stop_task = self._stop_task
        if stop_task is None:
            self.begin_stop()
            stop_task = asyncio.create_task(self._run_stop(), name="coordinator-stop")
            self._stop_task = stop_task
        await asyncio.shield(stop_task)

    def begin_stop(self) -> None:
        """Reject new operations while the owner shuts down stage processes."""
        if self._closed:
            return
        self._stopping = True
        self._running = False

    async def quiesce(self) -> None:
        """Settle owned operations while leaving control sockets available."""
        self.begin_stop()
        quiesce_task = self._quiesce_task
        if quiesce_task is None:
            quiesce_task = asyncio.create_task(
                self._run_quiesce(), name="coordinator-quiesce"
            )
            self._quiesce_task = quiesce_task
        await asyncio.shield(quiesce_task)

    async def _run_quiesce(self) -> None:
        stop_error = self._fatal_error or "Coordinator stopped"
        active_executions = [
            (request_id, self._execution_ids.get(request_id))
            for request_id in self._requests
        ]
        self._fail_active_requests(stop_error)
        self._fail_admin_operations(stop_error)
        for request_id, execution_id in active_executions:
            if execution_id is not None:
                self._schedule_abort_broadcast(
                    request_id,
                    execution_id=execution_id,
                )

        # Entry-stage and admin sends are coordinator-owned operations. Cancel
        # and join them before closing sockets so stop cannot return while a
        # caller remains stranded inside the transport.
        current_task = asyncio.current_task()
        operation_tasks = set(self._submission_tasks.values()) | set(self._admin_tasks)
        operation_tasks.discard(current_task)
        for task in operation_tasks:
            task.cancel()
        if operation_tasks:
            await asyncio.gather(*operation_tasks, return_exceptions=True)

    async def _run_stop(self) -> None:
        await self.quiesce()
        current_task = asyncio.current_task()

        # Give best-effort abort sends a bounded delivery window, then cancel
        # and join every owned task before the transport is closed. New client
        # work is gated, while cleanup racing with quiesce may still register an
        # abort task and will be included by the drain-to-quiescence loop.
        await asyncio.sleep(0)
        while True:
            abort_tasks = set(self._abort_tasks.values()) | set(
                self._abort_broadcast_tasks.values()
            )
            abort_tasks.discard(current_task)
            if not abort_tasks:
                break
            _, pending = await asyncio.wait(
                abort_tasks,
                timeout=_ABORT_DELIVERY_TIMEOUT_S,
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            await asyncio.sleep(0)
        self._abort_tasks.clear()
        self._abort_broadcast_tasks.clear()
        self._submission_tasks.clear()
        self._admin_tasks.clear()
        self._requests.clear()
        self._partial_results.clear()
        self._completion_futures.clear()
        self._stream_queues.clear()
        self._stream_pending_chunks.clear()
        self._execution_ids.clear()
        self._request_ids_by_execution.clear()
        self._abort_scheduled_executions.clear()
        self.control_plane.close()
        self._closed = True
        logger.info("Coordinator stopped")

    async def fail_pending_requests(self, error: BaseException | str) -> None:
        """Fail all requests currently owned by the coordinator."""
        self._running = False
        message = self._fatal_error or str(error)
        self._fatal_error = message
        active_executions = [
            (request_id, self._execution_ids.get(request_id))
            for request_id in self._requests
        ]
        self._fail_active_requests(message)
        self._fail_admin_operations(message)
        for request_id, execution_id in active_executions:
            if execution_id is not None:
                self._schedule_abort_broadcast(
                    request_id,
                    execution_id=execution_id,
                )

    def _fail_active_requests(self, message: str) -> None:
        for request_id, info in list(self._requests.items()):
            info.state = RequestState.FAILED
            info.error = message
            self._reject_completion_future(request_id, RuntimeError(message))
            self._put_stream_terminal(
                request_id,
                CompleteMessage(
                    request_id=request_id,
                    from_stage="coordinator",
                    success=False,
                    error=message,
                ),
            )
        self._requests.clear()
        self._partial_results.clear()

    def _fail_admin_operations(self, message: str) -> None:
        for pending in self._admin_ops.values():
            future = pending.future
            if future is not None and not future.done():
                future.set_exception(RuntimeError(message))
        self._admin_ops.clear()

    async def shutdown_stages(self, timeout_s: float = 5.0) -> None:
        """Send shutdown signal to all registered stages."""
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise TypeError("timeout_s must be a positive number")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

        async def _send(name: str, info: StageInfo) -> None:
            try:
                await asyncio.wait_for(
                    self.control_plane.send_shutdown(name, info.control_endpoint),
                    timeout=float(timeout_s),
                )
                logger.info("Sent shutdown to stage: %s", name)
            except asyncio.TimeoutError:
                logger.warning(
                    "Timed out sending shutdown to stage %s after %.3fs",
                    name,
                    timeout_s,
                )
            except Exception as e:
                logger.warning("Failed to send shutdown to stage %s: %s", name, e)

        await asyncio.gather(
            *(_send(name, info) for name, info in self._stages.items())
        )

    async def admin(
        self,
        action: str,
        payload: dict[str, Any] | None = None,
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        """Run an administrative operation against one or more stages."""
        if not self._running:
            raise RuntimeError("Coordinator is not running")

        owner_task = asyncio.current_task()
        if owner_task is None:
            raise RuntimeError("Coordinator admin operation requires an asyncio task")
        self._admin_tasks.add(owner_task)

        try:
            target_stages = self._resolve_admin_stages(stages)
            if not target_stages:
                raise ValueError("No stages registered for admin operation")

            op_id = str(uuid.uuid4())
            loop = asyncio.get_running_loop()
            pending = _AdminPendingOperation(
                expected_stages=set(target_stages),
                action=action,
                future=loop.create_future(),
            )
            operation = AdminOperation(
                op_id=op_id,
                action=action,
                payload=dict(payload or {}),
                target_stages=list(target_stages),
                timeout_s=timeout_s,
            )

            async with self._admin_lock:
                self._admin_ops[op_id] = pending
                try:
                    for stage_name in target_stages:
                        info = self._stages[stage_name]
                        await self.control_plane.send_admin(
                            stage_name,
                            info.control_endpoint,
                            AdminMessage(operation=operation),
                        )

                    assert pending.future is not None
                    results = await asyncio.wait_for(pending.future, timeout=timeout_s)
                finally:
                    self._admin_ops.pop(op_id, None)
                    if pending.future is not None:
                        if not pending.future.done():
                            pending.future.cancel()
                        elif not pending.future.cancelled():
                            pending.future.exception()

            return self._aggregate_admin_results(
                op_id=op_id,
                action=action,
                results=list(results.values()),
            )
        except asyncio.CancelledError:
            if self._stopping or self._closed:
                raise RuntimeError(self._fatal_error or "Coordinator stopped") from None
            raise
        finally:
            self._admin_tasks.discard(owner_task)

    async def model_info(
        self,
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "model_info",
            stages=stages,
            timeout_s=timeout_s,
        )

    async def pause_generation(
        self,
        payload: dict[str, Any] | None = None,
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "pause_generation",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def continue_generation(
        self,
        payload: dict[str, Any] | None = None,
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "continue_generation",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def update_weights_from_disk(
        self,
        payload: dict[str, Any],
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "update_weights_from_disk",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def init_weights_update_group(
        self,
        payload: dict[str, Any],
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "init_weights_update_group",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def destroy_weights_update_group(
        self,
        payload: dict[str, Any],
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "destroy_weights_update_group",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def update_weights_from_distributed(
        self,
        payload: dict[str, Any],
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "update_weights_from_distributed",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def weights_checker(
        self,
        payload: dict[str, Any] | None = None,
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "weights_checker",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def submit(self, request_id: str, request: OmniRequest | Any) -> Any:
        """Submit a request to the pipeline and wait for completion."""
        await self._submit_request(request_id, request)

        future = self._completion_futures[request_id]
        try:
            result = await future
            return result
        except asyncio.CancelledError:
            if request_id in self._requests:
                try:
                    await asyncio.shield(self.abort(request_id))
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.warning(
                        "Failed to abort cancelled request %s",
                        request_id,
                        exc_info=True,
                    )
            raise
        finally:
            self._completion_futures.pop(request_id, None)
            self._release_execution_id_if_unowned(request_id)

    async def stream(
        self, request_id: str, request: OmniRequest | Any
    ) -> AsyncIterator[CompleteMessage | StreamMessage]:
        """Submit a request and yield stream events until completion."""
        # Terminal messages have separate capacity and do not consume the chunk
        # quota. Successful multi-terminal messages are de-duplicated by stage.
        terminal_capacity = max(1, len(self._terminal_stages))
        queue: asyncio.Queue[CompleteMessage | StreamMessage] = asyncio.Queue(
            maxsize=self._stream_queue_maxsize + terminal_capacity
        )

        try:
            expected_terminal_stages = await self._submit_request(
                request_id, request, stream_queue=queue
            )

            completed_stages: set[str] = set()
            while True:
                msg = await queue.get()
                if isinstance(msg, CompleteMessage):
                    if not msg.success:
                        raise RuntimeError(msg.error or "Unknown error")
                    yield msg
                    completed_stages.add(msg.from_stage)
                    if (
                        not expected_terminal_stages
                        or completed_stages >= expected_terminal_stages
                    ):
                        return
                else:
                    pending_chunks = self._stream_pending_chunks.get(request_id, 0)
                    if pending_chunks > 0:
                        self._stream_pending_chunks[request_id] = pending_chunks - 1
                    yield msg
        finally:
            if self._stream_queues.get(request_id) is queue:
                try:
                    if request_id in self._requests:
                        try:
                            await self.abort(request_id)
                        except Exception:
                            # The coordinator-owned abort task logs its own failure.
                            # Do not replace the exception already leaving the stream.
                            pass
                finally:
                    if self._stream_queues.get(request_id) is queue:
                        self._stream_queues.pop(request_id, None)
                        self._completion_futures.pop(request_id, None)
                        self._stream_pending_chunks.pop(request_id, None)
                        self._release_execution_id_if_unowned(request_id)

    async def _submit_request(
        self,
        request_id: str,
        request: OmniRequest | Any,
        *,
        stream_queue: asyncio.Queue[CompleteMessage | StreamMessage] | None = None,
    ) -> set[str]:
        """Submit a request without waiting for completion."""
        if self._stopping or self._closed:
            raise RuntimeError("Coordinator is stopping or stopped")
        if self._fatal_error is not None:
            raise RuntimeError(self._fatal_error)
        if self._request_id_is_reserved(request_id):
            raise ValueError(f"Request {request_id} already exists")

        if self.entry_stage not in self._stages:
            raise ValueError(f"Entry stage {self.entry_stage} not registered")

        if not isinstance(request, OmniRequest):
            request = OmniRequest(inputs=request)

        terminal_stages = self._resolve_terminal_stages(request)
        execution_id = uuid.uuid4().hex
        while execution_id in self._request_ids_by_execution:
            execution_id = uuid.uuid4().hex
        self._execution_ids[request_id] = execution_id
        self._request_ids_by_execution[execution_id] = request_id

        # Track request
        self._requests[request_id] = RequestInfo(
            request_id=request_id,
            state=RequestState.PENDING,
            current_stage=self.entry_stage,
            terminal_stages=terminal_stages,
        )

        # Create future for completion
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._completion_futures[request_id] = future
        if stream_queue is not None:
            self._stream_queues[request_id] = stream_queue
            self._stream_pending_chunks[request_id] = 0

        payload = StagePayload(
            request_id=execution_id,
            request=request,
            data={"raw_inputs": request.inputs},
            public_request_id=request_id,
            route_trace=(
                (self.entry_stage,) if self._max_stage_transitions is not None else ()
            ),
        )

        _emit_event(
            request_id=execution_id,
            stage="coordinator",
            event_name="request_admission",
            metadata={
                "entry_stage": self.entry_stage,
                "public_request_id": request_id,
            },
        )

        # Submit to entry stage
        entry_info = self._stages[self.entry_stage]
        owner_task = asyncio.current_task()
        if owner_task is None:
            raise RuntimeError("Coordinator submission requires an asyncio task")
        self._submission_tasks[request_id] = owner_task
        try:
            await self.control_plane.submit_to_stage(
                self.entry_stage,
                entry_info.control_endpoint,
                SubmitMessage(request_id=execution_id, data=payload),
            )
        except BaseException as exc:
            self._requests.pop(request_id, None)
            self._partial_results.pop(request_id, None)
            self._stream_queues.pop(request_id, None)
            self._stream_pending_chunks.pop(request_id, None)
            failed_future = self._completion_futures.pop(request_id, None)
            if failed_future is not None:
                if not failed_future.done():
                    failed_future.cancel()
                elif not failed_future.cancelled():
                    failed_future.exception()
            self._schedule_abort_broadcast(
                request_id,
                execution_id=execution_id,
            )
            if isinstance(exc, asyncio.CancelledError) and (
                self._stopping or self._closed
            ):
                raise RuntimeError(self._fatal_error or "Coordinator stopped") from None
            raise
        finally:
            if self._submission_tasks.get(request_id) is owner_task:
                self._submission_tasks.pop(request_id, None)
            self._release_execution_id_if_unowned(request_id)

        # Update state
        info = self._requests.get(request_id)
        if info is not None:
            info.state = RequestState.RUNNING

        logger.info(
            "Coordinator submitted req=%s to %s at %s",
            request_id,
            self.entry_stage,
            entry_info.control_endpoint,
        )
        return set(terminal_stages)

    def _request_id_is_reserved(self, request_id: str) -> bool:
        """Return whether any coordinator owner still holds this request ID."""
        return (
            request_id in self._requests
            or request_id in self._completion_futures
            or request_id in self._stream_queues
            or request_id in self._abort_tasks
            or request_id in self._abort_broadcast_tasks
            or request_id in self._submission_tasks
            or request_id in self._execution_ids
        )

    def _release_execution_id_if_unowned(self, request_id: str) -> None:
        if (
            request_id in self._requests
            or request_id in self._completion_futures
            or request_id in self._stream_queues
            or request_id in self._abort_tasks
            or request_id in self._abort_broadcast_tasks
            or request_id in self._submission_tasks
        ):
            return
        execution_id = self._execution_ids.pop(request_id, None)
        if execution_id is not None:
            self._request_ids_by_execution.pop(execution_id, None)
            self._abort_scheduled_executions.discard(execution_id)

    def _resolve_execution_id(self, execution_id: str) -> str | None:
        request_id = self._request_ids_by_execution.get(execution_id)
        if request_id is None or self._execution_ids.get(request_id) != execution_id:
            return None
        return request_id

    def _reject_completion_future(
        self,
        request_id: str,
        exc: BaseException,
    ) -> None:
        # Note: (Akazaakane) Non-streaming callers await the completion future,
        # so errors must be propagated with set_exception(). Streaming callers
        # receive errors through the stream queue and never await that future;
        # cancel it instead to avoid "Future exception was never retrieved".
        future = self._completion_futures.get(request_id)
        if future is None or future.done():
            return
        if request_id in self._stream_queues:
            future.cancel()
        else:
            future.set_exception(exc)

    async def abort(self, request_id: str) -> bool:
        """Abort a request.

        Args:
            request_id: Request to abort

        Returns:
            True if aborted, False if not found
        """
        if self._stopping or self._closed:
            return False
        abort_task = self._abort_tasks.get(request_id)
        if abort_task is not None:
            return await asyncio.shield(abort_task)

        info = self._requests.get(request_id)
        if info is None:
            return False

        if info.state in (
            RequestState.COMPLETED,
            RequestState.FAILED,
            RequestState.ABORTED,
        ):
            return False

        execution_id = self._execution_ids.get(request_id)
        if execution_id is None:
            return False
        self._requests.pop(request_id, None)
        info.state = RequestState.ABORTED
        self._partial_results.pop(request_id, None)
        self._reject_completion_future(
            request_id, asyncio.CancelledError(f"Request {request_id} aborted")
        )
        self._put_stream_terminal(
            request_id,
            CompleteMessage(
                request_id=request_id,
                from_stage="coordinator",
                success=False,
                error="aborted",
            ),
        )
        abort_task = asyncio.create_task(
            self._run_abort(request_id, execution_id),
            name=f"coordinator-abort-{request_id}",
        )
        self._abort_scheduled_executions.add(execution_id)
        self._abort_tasks[request_id] = abort_task
        abort_task.add_done_callback(
            lambda done, rid=request_id: self._on_abort_task_done(rid, done)
        )
        return await asyncio.shield(abort_task)

    async def _run_abort(
        self,
        request_id: str,
        execution_id: str,
    ) -> bool:
        await self._broadcast_abort_safely(request_id, execution_id)

        logger.info("Coordinator aborted req=%s", request_id)
        return True

    async def _broadcast_abort_safely(self, request_id: str, execution_id: str) -> bool:
        try:
            await self.control_plane.broadcast_abort(
                AbortMessage(request_id=execution_id)
            )
        except Exception:
            logger.warning("Failed to abort request %s", request_id, exc_info=True)
            return False
        return True

    def _schedule_abort_broadcast(
        self,
        request_id: str,
        *,
        execution_id: str | None = None,
    ) -> None:
        # Admission is already closed while stopping, but cleanup paths that
        # race with quiesce must still be able to abort ambiguously sent work.
        if self._closed:
            return
        if request_id in self._abort_tasks or request_id in self._abort_broadcast_tasks:
            return
        if execution_id is None:
            execution_id = self._execution_ids.get(request_id)
        if execution_id is None:
            return
        if execution_id in self._abort_scheduled_executions:
            return
        self._abort_scheduled_executions.add(execution_id)
        task = asyncio.create_task(
            self._broadcast_abort_safely(request_id, execution_id),
            name=f"coordinator-abort-{request_id}",
        )
        self._abort_broadcast_tasks[request_id] = task
        task.add_done_callback(
            lambda done, rid=request_id: self._on_abort_broadcast_task_done(rid, done)
        )

    def _put_stream_terminal(
        self,
        request_id: str,
        msg: CompleteMessage,
    ) -> None:
        queue = self._stream_queues.get(request_id)
        if queue is None:
            return
        queue.put_nowait(msg)

    def _on_abort_task_done(
        self,
        request_id: str,
        task: asyncio.Task[bool],
    ) -> None:
        if self._abort_tasks.get(request_id) is task:
            self._abort_tasks.pop(request_id, None)
        self._release_execution_id_if_unowned(request_id)
        if task.cancelled():
            if self._running:
                logger.warning(
                    "Coordinator abort task cancelled for req=%s", request_id
                )
            return
        exc = task.exception()
        if exc is not None:
            logger.warning(
                "Failed to abort request %s",
                request_id,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    def _on_abort_broadcast_task_done(
        self,
        request_id: str,
        task: asyncio.Task[bool],
    ) -> None:
        if self._abort_broadcast_tasks.get(request_id) is task:
            self._abort_broadcast_tasks.pop(request_id, None)
        self._release_execution_id_if_unowned(request_id)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning(
                "Failed to broadcast abort for request %s",
                request_id,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    async def run_completion_loop(self) -> None:
        """Run the completion receiving loop.

        This should be run as a background task.
        """
        try:
            while self._running:
                msg = await self.control_plane.recv_event()
                if isinstance(msg, StreamMessage):
                    await self._handle_stream(msg)
                elif isinstance(msg, AdminResultMessage):
                    self._handle_admin_result(msg.result)
                else:
                    await self._handle_completion(msg)
        except asyncio.CancelledError:
            logger.info("Coordinator completion loop cancelled")
        except Exception as e:
            logger.error("Coordinator completion loop error: %s", e)
            raise

    async def _handle_completion(self, msg: CompleteMessage) -> None:
        """Handle a completion message from a stage."""
        execution_id = msg.request_id
        request_id = self._resolve_execution_id(execution_id)
        if request_id is None:
            logger.warning(
                "Coordinator ignored completion for unknown execution=%s",
                execution_id,
            )
            return
        msg = replace(msg, request_id=request_id)
        logger.debug(
            "Coordinator received completion: req=%s from %s success=%s",
            request_id,
            msg.from_stage,
            msg.success,
        )
        _emit_event(
            request_id=execution_id,
            stage="coordinator",
            event_name="terminal_response",
            metadata={
                "from_stage": msg.from_stage,
                "success": msg.success,
                "public_request_id": request_id,
            },
        )

        if request_id not in self._requests:
            logger.warning(
                "Coordinator received completion for unknown req=%s", request_id
            )
            return

        info = self._requests[request_id]

        # Fail-fast: any terminal failure -> fail entire request
        if not msg.success:
            self._requests.pop(request_id, None)
            info.state = RequestState.FAILED
            info.error = msg.error
            self._partial_results.pop(request_id, None)
            self._reject_completion_future(
                request_id, RuntimeError(msg.error or "Unknown error")
            )
            self._put_stream_terminal(request_id, msg)
            self._schedule_abort_broadcast(request_id)
            return

        expected_terminal_stages = self._expected_terminal_stages(request_id)
        if expected_terminal_stages and msg.from_stage not in expected_terminal_stages:
            logger.debug(
                "Coordinator ignoring completion from inactive terminal: "
                "req=%s stage=%s expected=%s",
                request_id,
                msg.from_stage,
                sorted(expected_terminal_stages),
            )
            return

        # Single active terminal (original behavior) or no terminal_stages configured
        if len(expected_terminal_stages) <= 1:
            info.state = RequestState.COMPLETED
            info.result = msg.result
            if request_id in self._completion_futures:
                future = self._completion_futures[request_id]
                if not future.done():
                    future.set_result(msg.result)
            self._put_stream_terminal(request_id, msg)
            self._requests.pop(request_id, None)
            return

        # Multi-terminal: collect partial results
        partials = self._partial_results.setdefault(request_id, {})
        if msg.from_stage in partials:
            logger.warning(
                "Coordinator ignored duplicate terminal completion: req=%s stage=%s",
                request_id,
                msg.from_stage,
            )
            return
        partials[msg.from_stage] = msg.result

        # Forward stream completion per-stage
        self._put_stream_terminal(request_id, msg)

        if set(partials) < expected_terminal_stages:
            return  # still waiting

        # All terminal stages done -> merge and resolve
        merged = dict(partials)
        self._partial_results.pop(request_id)
        info.state = RequestState.COMPLETED
        info.result = merged

        if request_id in self._completion_futures:
            future = self._completion_futures[request_id]
            if not future.done():
                future.set_result(merged)
        self._requests.pop(request_id, None)

    async def _handle_stream(self, msg: StreamMessage) -> None:
        """Handle a stream chunk from a stage."""
        execution_id = msg.request_id
        request_id = self._resolve_execution_id(execution_id)
        if request_id is None:
            return
        msg = replace(msg, request_id=request_id)
        if request_id not in self._stream_queues or request_id not in self._requests:
            return
        _emit_event(
            request_id=execution_id,
            stage="coordinator",
            event_name="coordinator_stream_received",
            metadata={
                "from_stage": msg.from_stage,
                "chunk_id": msg.chunk_id,
                "modality": msg.modality,
                "public_request_id": request_id,
            },
        )
        _emit_event(
            request_id=execution_id,
            stage="coordinator",
            event_name="stage_stream_chunk_received",
            metadata={
                "from_stage": msg.from_stage,
                "chunk_id": msg.chunk_id,
                "modality": msg.modality,
                "public_request_id": request_id,
            },
        )
        queue = self._stream_queues[request_id]
        pending_chunks = self._stream_pending_chunks.get(request_id, 0)
        if pending_chunks >= self._stream_queue_maxsize:
            await self._handle_completion(
                CompleteMessage(
                    request_id=execution_id,
                    from_stage="coordinator",
                    success=False,
                    error=(
                        "client stream buffer capacity exceeded: "
                        f"{self._stream_queue_maxsize}"
                    ),
                )
            )
            return
        queue.put_nowait(msg)
        self._stream_pending_chunks[request_id] = pending_chunks + 1

    def _handle_admin_result(self, result: AdminResult) -> None:
        pending = self._admin_ops.get(result.op_id)
        if pending is None:
            logger.warning(
                "Coordinator received admin result for unknown op=%s stage=%s",
                result.op_id,
                result.stage,
            )
            return
        pending.results[result.stage] = result
        if (
            pending.future is not None
            and pending.results.keys() >= pending.expected_stages
        ):
            if not pending.future.done():
                pending.future.set_result(dict(pending.results))

    def _resolve_admin_stages(self, stages: Sequence[str] | None) -> list[str]:
        if stages is None:
            return sorted(self._stages)
        resolved = list(stages)
        unknown = sorted(set(resolved) - set(self._stages))
        if unknown:
            raise ValueError(f"Unknown admin target stage(s): {unknown}")
        return resolved

    def _aggregate_admin_results(
        self,
        *,
        op_id: str,
        action: str,
        results: list[AdminResult],
    ) -> dict[str, Any]:
        updated_results = [
            item
            for item in results
            if not item.data.get("skipped") and not item.data.get("unsupported")
        ]
        if is_update_action(action):
            success = bool(updated_results) and all(
                item.success for item in updated_results
            )
        else:
            success = all(item.success for item in results)

        errors = [item.error for item in results if item.error]
        if success:
            message = "ok"
        elif errors:
            message = "; ".join(errors)
        else:
            message = "admin operation did not complete successfully"

        return {
            "op_id": op_id,
            "action": action,
            "success": success,
            "message": message,
            "results": [item.to_dict() for item in results],
        }

    def get_request_info(self, request_id: str) -> RequestInfo | None:
        """Get info about a request."""
        return self._requests.get(request_id)

    def _resolve_terminal_stages(self, request: OmniRequest) -> set[str]:
        if self._terminal_stages_resolver is None:
            return set(self._terminal_stages)
        resolved = self._terminal_stages_resolver(request)
        if resolved is None:
            return set(self._terminal_stages)
        if isinstance(resolved, str) or not isinstance(resolved, Sequence):
            raise ValueError(
                "terminal_stages_resolver must return a sequence of terminal "
                "stage names or None"
            )
        if not all(isinstance(stage, str) for stage in resolved):
            raise ValueError(
                "terminal_stages_resolver must return terminal stage names"
            )
        resolved_stages = set(resolved)
        if not resolved_stages:
            raise ValueError("terminal_stages_resolver returned no terminal stages")
        unknown = resolved_stages - self._terminal_stages
        if unknown:
            raise ValueError(
                "terminal_stages_resolver returned stages outside the static "
                f"terminal stages: {sorted(unknown)}. Allowed terminal stages: "
                f"{sorted(self._terminal_stages)}"
            )
        return resolved_stages

    def _expected_terminal_stages(self, request_id: str) -> set[str]:
        info = self._requests.get(request_id)
        if info is None or info.terminal_stages is None:
            return set(self._terminal_stages)
        return info.terminal_stages

    def health(self) -> dict[str, Any]:
        """Return health status."""
        state_counts = {}
        for info in self._requests.values():
            state = info.state.value
            state_counts[state] = state_counts.get(state, 0) + 1

        return {
            "running": self._running,
            "stages": list(self._stages.keys()),
            "entry_stage": self.entry_stage,
            "total_requests": len(self._requests),
            "pending_completions": len(self._completion_futures),
            "request_states": state_counts,
        }


async def run_coordinator(
    completion_endpoint: str,
    abort_endpoint: str,
    entry_stage: str,
    stages: dict[str, str],  # name -> endpoint
    terminal_stages: list[str] | None = None,
    terminal_stages_resolver: Callable[[OmniRequest], list[str] | None] | None = None,
    max_stage_transitions: int | None = None,
    stream_queue_maxsize: int = 256,
) -> Coordinator:
    """Create and start a coordinator.

    Args:
        completion_endpoint: ZMQ endpoint to receive completions
        abort_endpoint: ZMQ endpoint for abort broadcasts
        entry_stage: Name of the entry stage
        stages: Dict of stage_name -> stage_endpoint
        terminal_stages: Optional list of terminal stage names for multi-terminal merge
        max_stage_transitions: Optional feedback-pipeline hop bound. When set,
            initializes runtime-owned route state at the entry stage.
        stream_queue_maxsize: Maximum buffered stream chunks per client request.

    Returns:
        Started Coordinator instance
    """
    coordinator = Coordinator(
        completion_endpoint=completion_endpoint,
        abort_endpoint=abort_endpoint,
        entry_stage=entry_stage,
        terminal_stages=terminal_stages,
        terminal_stages_resolver=terminal_stages_resolver,
        max_stage_transitions=max_stage_transitions,
        stream_queue_maxsize=stream_queue_maxsize,
    )

    # Register stages
    for name, endpoint in stages.items():
        coordinator.register_stage(name, endpoint)

    # Start
    await coordinator.start()

    return coordinator
