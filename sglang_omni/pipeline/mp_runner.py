# SPDX-License-Identifier: Apache-2.0
"""Multi-process pipeline runner.

The runner owns the single serving path. It can start one OS process containing
multiple non-TP stages, multiple OS processes on the same GPU, and the existing
one-process-per-rank TP topology.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import socket
from typing import Any

from sglang_omni.config.placement import (
    StagePlacementPlan,
    resolve_gpu_stage_names,
    resolve_stage_gpu_ids,
)
from sglang_omni.config.runtime import (
    resolve_stage_factory_arg_defaults,
    resolve_stage_static_factory_args,
)
from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.config.topology import ProcessTopologyPlan
from sglang_omni.pipeline import Coordinator
from sglang_omni.pipeline.runtime_config import (
    IpcRuntimeDir,
    PipelineRuntimePrep,
    build_comm_config,
    prepare_pipeline_runtime,
)
from sglang_omni.pipeline.stage_workers import (
    StageGroup,
    StageLaunchConfig,
    StageWorkerProcessSpec,
)
from sglang_omni.utils.imports import import_string

logger = logging.getLogger(__name__)


def _build_stage_groups(
    config: PipelineConfig,
    ctx: multiprocessing.context.BaseContext | None = None,
    *,
    stages_cfg: list[StageConfig],
    name_map: dict[str, str],
    endpoints: dict[str, str],
    placement_plan: StagePlacementPlan,
    process_plan: ProcessTopologyPlan,
) -> list[StageGroup]:
    """Build lifecycle groups from prepared endpoints and process topology.

    The caller owns endpoint allocation and IPC runtime-dir lifecycle. This
    helper only converts prepared runtime state into subprocess specs.
    """
    if ctx is None:
        ctx = multiprocessing.get_context("spawn")

    stage_endpoints = {s.name: endpoints[f"stage_{s.name}"] for s in stages_cfg}
    rank_endpoints = {
        stage.name: tuple(
            endpoints[f"comm_{stage.name}_rank{tp_rank}"]
            for tp_rank in range(stage.tp_size)
        )
        for stage in stages_cfg
    }
    stream_receivers: set[str] = set()
    for scfg in stages_cfg:
        for target in scfg.stream_to:
            stream_receivers.add(target)
    stage_cfg_by_name = {stage.name: stage for stage in stages_cfg}

    nccl_port_counter = _NcclPortAllocator()

    # GPU-resident stages, shared by every stage so the transport router can
    # decide GPU vs host transport per edge from static placement alone. Include the
    # pre-fusion aliases so lookups work whether an edge names a stage by its
    # raw or canonical (fused) name.
    gpu_canonical = resolve_gpu_stage_names(placement_plan)
    gpu_stage_names = set(gpu_canonical)
    for raw_name, canonical_name in name_map.items():
        if canonical_name in gpu_canonical:
            gpu_stage_names.add(raw_name)
    stage_gpu_ids = {
        name: placement.gpu_ids for name, placement in placement_plan.stages.items()
    }
    for raw_name, canonical_name in name_map.items():
        if canonical_name in stage_gpu_ids:
            stage_gpu_ids[raw_name] = stage_gpu_ids[canonical_name]

    single_stage_specs: dict[str, StageLaunchConfig] = {}
    tp_groups: list[StageGroup] = []
    for stage_cfg in stages_cfg:
        tp_size = stage_cfg.tp_size
        gpu_ids = resolve_stage_gpu_ids(placement_plan, stage_cfg)
        nccl_port = nccl_port_counter.allocate() if tp_size > 1 else None

        same_process_targets = _resolve_same_process_targets(
            stage_cfg,
            stage_cfg_by_name,
            name_map,
            process_plan,
        )

        # Avoid importing stage factories in the parent process. The child
        # injects signature-dependent args after importing the factory it must
        # construct anyway.
        base_factory_args = resolve_stage_static_factory_args(stage_cfg, config)

        stage_kwargs = dict(
            stage_name=stage_cfg.name,
            factory=stage_cfg.factory,
            next_stages=stage_cfg.next,
            route_fn=stage_cfg.route_fn,
            is_terminal=stage_cfg.terminal,
            max_stage_transitions=config.max_stage_transitions,
            env_defaults={**dict(config.env_defaults), **stage_cfg.env},
            wait_for=stage_cfg.wait_for,
            wait_for_fn=stage_cfg.wait_for_fn,
            merge_fn=stage_cfg.merge_fn,
            project_payload={
                name_map.get(target, target): dotted_path
                for target, dotted_path in stage_cfg.project_payload.items()
            },
            coordinator_endpoint=endpoints["completion"],
            abort_endpoint=endpoints["abort"],
            stage_endpoints=stage_endpoints,
            rank_endpoints=rank_endpoints,
            stream_targets=list(stage_cfg.stream_to),
            stream_done_to_fn=stage_cfg.stream_done_to_fn,
            gpu_stage_names=gpu_stage_names,
            stage_gpu_ids=stage_gpu_ids,
            same_process_targets=same_process_targets,
            is_stream_receiver=stage_cfg.name in stream_receivers,
            can_accept_stream_before_payload=stage_cfg.can_accept_stream_before_payload,
            disable_direct_cuda_ipc_payload=stage_cfg.disable_direct_cuda_ipc_payload,
            name_map=name_map,
        )
        if tp_size == 1:
            single_stage_specs[stage_cfg.name] = _build_single_stage_spec(
                stage_cfg=stage_cfg,
                config=config,
                gpu_id=gpu_ids[0],
                recv_endpoint=stage_endpoints[stage_cfg.name],
                base_factory_args=base_factory_args,
                stage_kwargs=stage_kwargs,
            )
        else:
            specs = _build_tp_stage_specs(
                ctx=ctx,
                stage_cfg=stage_cfg,
                config=config,
                gpu_ids=gpu_ids,
                nccl_port=nccl_port,
                recv_endpoint=stage_endpoints[stage_cfg.name],
                base_factory_args=base_factory_args,
                stage_kwargs=stage_kwargs,
            )
            process_specs = [
                StageWorkerProcessSpec(
                    process_name=process_plan.tp_stage_to_processes[stage_cfg.name][
                        spec.tp_rank
                    ],
                    stage_specs=[spec],
                )
                for spec in specs
            ]
            tp_groups.append(StageGroup(stage_cfg.name, process_specs))

    groups: list[StageGroup] = []
    for group in process_plan.groups:
        groups.append(
            StageGroup(
                group.name,
                [
                    StageWorkerProcessSpec(
                        process_name=group.name,
                        stage_specs=[
                            single_stage_specs[stage_name]
                            for stage_name in group.stage_names
                        ],
                    )
                ],
            )
        )
    groups.extend(tp_groups)
    _attach_process_memory_fraction_defaults(groups)

    return groups


def _attach_process_memory_fraction_defaults(groups: list[StageGroup]) -> None:
    """Expose the per-GPU process budget loaded through each stage.

    Stage resource fractions remain component budgets for placement. A process
    constructs its stages in ``stage_specs`` order, so a stage that profiles
    process-scoped memory must include earlier stages but not reserve memory
    assigned to stages that have not been constructed yet.
    """

    for group in groups:
        for process_spec in group.process_specs:
            by_gpu: dict[int, list[StageLaunchConfig]] = {}
            for stage_spec in process_spec.stage_specs:
                if stage_spec.gpu_id is not None:
                    by_gpu.setdefault(int(stage_spec.gpu_id), []).append(stage_spec)

            for stage_specs in by_gpu.values():
                fractions = [
                    stage.factory_arg_defaults.get("total_gpu_memory_fraction")
                    for stage in stage_specs
                ]
                if any(fraction is None for fraction in fractions):
                    continue
                process_loaded_fraction = 0.0
                for stage_spec, fraction in zip(stage_specs, fractions, strict=True):
                    process_loaded_fraction += float(fraction)
                    stage_spec.factory_arg_defaults[
                        "process_total_gpu_memory_fraction"
                    ] = process_loaded_fraction


def _resolve_same_process_targets(
    stage_cfg: StageConfig,
    stage_cfg_by_name: dict[str, StageConfig],
    name_map: dict[str, str],
    process_plan: ProcessTopologyPlan,
) -> set[str]:
    if stage_cfg.tp_size > 1:
        return set()
    source_process = process_plan.stage_to_process.get(stage_cfg.name)
    if source_process is None:
        return set()

    raw_targets: list[str] = []
    if stage_cfg.next is not None:
        raw_targets.extend(
            [stage_cfg.next] if isinstance(stage_cfg.next, str) else stage_cfg.next
        )
    raw_targets.extend(stage_cfg.stream_to)

    same_process_targets: set[str] = set()
    for raw_target in raw_targets:
        target = name_map.get(raw_target, raw_target)
        target_cfg = stage_cfg_by_name.get(target)
        if target_cfg is None or target_cfg.tp_size > 1:
            continue
        if process_plan.stage_to_process.get(target) == source_process:
            same_process_targets.add(target)
    return same_process_targets


def _build_single_stage_spec(
    *,
    stage_cfg: StageConfig,
    config: PipelineConfig,
    gpu_id: int | None,
    recv_endpoint: str,
    base_factory_args: dict[str, Any],
    stage_kwargs: dict[str, Any],
) -> StageLaunchConfig:
    factory_args = dict(base_factory_args)
    comm_config = _resolve_comm_config(stage_cfg, gpu_id=gpu_id)
    return StageLaunchConfig(
        role="single",
        tp_rank=0,
        tp_size=1,
        placement_gpu_id=gpu_id,
        gpu_id=gpu_id,
        nccl_port=None,
        factory_args=factory_args,
        factory_arg_defaults=resolve_stage_factory_arg_defaults(
            stage_cfg, config, gpu_id=gpu_id
        ),
        comm_config=comm_config,
        recv_endpoint=recv_endpoint,
        **stage_kwargs,
    )


def _build_tp_stage_specs(
    *,
    ctx: multiprocessing.context.BaseContext,
    stage_cfg: StageConfig,
    config: PipelineConfig,
    gpu_ids: list[int | None],
    nccl_port: int | None,
    recv_endpoint: str,
    base_factory_args: dict[str, Any],
    stage_kwargs: dict[str, Any],
) -> list[StageLaunchConfig]:
    follower_work_queues = [ctx.Queue() for _ in range(stage_cfg.tp_size - 1)]
    follower_abort_queues = [ctx.Queue() for _ in range(stage_cfg.tp_size - 1)]
    follower_admin_result_queues = [ctx.Queue() for _ in range(stage_cfg.tp_size - 1)]
    specs: list[StageLaunchConfig] = []

    for tp_rank in range(stage_cfg.tp_size):
        gpu_id = gpu_ids[tp_rank] if tp_rank < len(gpu_ids) else gpu_ids[0]
        if gpu_id is None:
            raise ValueError(f"TP stage {stage_cfg.name!r} requires GPU placement")
        factory_args = dict(base_factory_args)
        factory_args["tp_rank"] = tp_rank
        factory_args["tp_size"] = stage_cfg.tp_size
        factory_args["nccl_port"] = nccl_port

        comm_config = _resolve_comm_config(stage_cfg, gpu_id=gpu_id)

        if tp_rank == 0:
            specs.append(
                StageLaunchConfig(
                    role="leader",
                    tp_rank=tp_rank,
                    tp_size=stage_cfg.tp_size,
                    placement_gpu_id=gpu_id,
                    gpu_id=gpu_id,
                    nccl_port=nccl_port,
                    factory_args=factory_args,
                    factory_arg_defaults=resolve_stage_factory_arg_defaults(
                        stage_cfg, config, gpu_id=gpu_id
                    ),
                    comm_config=comm_config,
                    recv_endpoint=recv_endpoint,
                    follower_work_queues=follower_work_queues,
                    follower_abort_queues=follower_abort_queues,
                    follower_admin_result_queues=follower_admin_result_queues,
                    **stage_kwargs,
                )
            )
            continue

        idx = tp_rank - 1
        specs.append(
            StageLaunchConfig(
                role="follower",
                tp_rank=tp_rank,
                tp_size=stage_cfg.tp_size,
                placement_gpu_id=gpu_id,
                gpu_id=gpu_id,
                nccl_port=nccl_port,
                factory_args=factory_args,
                factory_arg_defaults=resolve_stage_factory_arg_defaults(
                    stage_cfg, config, gpu_id=gpu_id
                ),
                comm_config=comm_config,
                recv_endpoint="",
                internal_work_queue=follower_work_queues[idx],
                internal_abort_queue=follower_abort_queues[idx],
                internal_admin_result_queue=follower_admin_result_queues[idx],
                **stage_kwargs,
            )
        )

    return specs


def _resolve_comm_config(
    stage_cfg: StageConfig,
    *,
    gpu_id: int | None,
) -> dict[str, Any]:
    """Build stage-local communication options from placement."""
    comm_config = build_comm_config(stage_cfg)
    if stage_cfg.gpu is not None:
        comm_config["gpu_id"] = gpu_id
    return comm_config


class _NcclPortAllocator:
    """Allocate unique NCCL ports for per-stage TP groups."""

    def __init__(self, base_port: int = 29500):
        self._next = base_port

    def allocate(self) -> int:
        """Return an available port, incrementing the counter."""
        while True:
            port = self._next
            self._next += 1
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", port))
                    return port
            except OSError:
                continue


class MultiProcessPipelineRunner:

    def __init__(self, config: PipelineConfig):
        self._config = config
        self._coordinator: Coordinator | None = None
        self._ipc_runtime_dir: IpcRuntimeDir | None = None
        self._groups: list[StageGroup] = []
        self._completion_task: asyncio.Task | None = None
        self._completion_failure_task: asyncio.Task | None = None
        self._start_task: asyncio.Task[Any] | None = None
        self._stop_task: asyncio.Task | None = None
        self._monitor_task: asyncio.Task | None = None
        self._fatal_event: asyncio.Event | None = None
        self._fatal_error: BaseException | None = None
        self._prep: PipelineRuntimePrep | None = None
        self._started = False
        self._stop_requested = False

    @property
    def coordinator(self) -> Coordinator:
        if self._coordinator is None:
            raise RuntimeError("Runner not started")
        return self._coordinator

    @property
    def prep(self) -> PipelineRuntimePrep:
        """Return the resolved runtime prep (placement plan, process plan,
        endpoints, fused stages). Valid only after :meth:`start`."""
        if self._prep is None:
            raise RuntimeError("Runner not started")
        return self._prep

    @property
    def stage_control_endpoints(self) -> dict[str, str]:
        if not self._started:
            raise RuntimeError("Runner not started")
        endpoints: dict[str, str] = {}
        for group in self._groups:
            endpoints.update(group.stage_control_endpoints)
        return endpoints

    async def start(self, timeout: float = 120.0) -> None:
        if self._started:
            raise RuntimeError("Already started")
        if self._start_task is not None and not self._start_task.done():
            raise RuntimeError("Runner is starting")
        if self._stop_task is not None and not self._stop_task.done():
            raise RuntimeError("Runner is stopping")
        start_task = asyncio.current_task()
        if start_task is None:
            raise RuntimeError("Runner start requires an asyncio task")
        self._start_task = start_task
        self._stop_task = None
        self._stop_requested = False

        try:
            ctx = multiprocessing.get_context("spawn")
            self._fatal_event = asyncio.Event()
            self._fatal_error = None
            prep = prepare_pipeline_runtime(
                self._config,
                ipc_runtime_dir=self._ipc_runtime_dir,
            )
            self._prep = prep
            self._ipc_runtime_dir = prep.runtime_dir
            groups = _build_stage_groups(
                self._config,
                ctx,
                stages_cfg=prep.stages_cfg,
                name_map=prep.name_map,
                endpoints=prep.endpoints,
                placement_plan=prep.placement_plan,
                process_plan=prep.process_plan,
            )

            terminal_stages_resolver = (
                import_string(self._config.terminal_stages_fn)
                if self._config.terminal_stages_fn
                else None
            )
            self._coordinator = Coordinator(
                completion_endpoint=prep.endpoints["completion"],
                abort_endpoint=prep.endpoints["abort"],
                entry_stage=prep.entry_stage,
                terminal_stages=self._config.terminal_stages or None,
                terminal_stages_resolver=terminal_stages_resolver,
                max_stage_transitions=self._config.max_stage_transitions,
                stream_queue_maxsize=self._config.stream_queue_maxsize,
            )
            await self._coordinator.start()
            self._completion_task = asyncio.create_task(
                self._coordinator.run_completion_loop()
            )
            self._completion_task.add_done_callback(self._on_completion_task_done)

            self._groups = groups
            if self._config.env_defaults:
                env_names = ", ".join(sorted(self._config.env_defaults))
                logger.info(f"Configured stage process env defaults: {env_names}")
            for group in self._groups:
                group.spawn(ctx)

            await asyncio.gather(*(g.wait_ready(timeout) for g in self._groups))

            if self._stop_requested:
                raise RuntimeError("Runner stopped during startup")

            for group in self._groups:
                if group.any_dead():
                    raise RuntimeError(
                        f"Stage process(es) died during startup: "
                        f"{group.dead_summary()}"
                    )

            if self._completion_task.done():
                try:
                    self._completion_task.result()
                except asyncio.CancelledError as exc:
                    raise RuntimeError(
                        "Coordinator completion loop stopped during startup"
                    ) from exc
                raise RuntimeError("Coordinator completion loop exited during startup")

            for group in self._groups:
                for stage_name, endpoint in group.stage_control_endpoints.items():
                    self._coordinator.register_stage(stage_name, endpoint)

            self._started = True
            self._monitor_task = asyncio.create_task(self._monitor_children())

            total_stages = sum(
                len(group.stage_control_endpoints) for group in self._groups
            )
            total_procs = sum(g.process_count for g in self._groups)
            logger.info(
                "MultiProcessPipelineRunner started: %d stage(s), %d process(es)",
                total_stages,
                total_procs,
            )

        except BaseException:
            await self._cleanup_on_failure()
            raise
        finally:
            if self._start_task is start_task:
                self._start_task = None

    async def _monitor_children(self) -> None:
        while self._started:
            for group in self._groups:
                if group.any_dead():
                    error = RuntimeError(
                        f"Dead stage process(es) detected: {group.dead_summary()}"
                    )
                    logger.error("%s", error)
                    await self._fail_runtime(error)
                    return
            await asyncio.sleep(5.0)

    async def _fail_runtime(self, error: BaseException) -> None:
        if self._fatal_error is not None:
            return
        self._fatal_error = error
        if self._fatal_event is not None:
            self._fatal_event.set()
        if self._coordinator is not None:
            try:
                await self._coordinator.fail_pending_requests(error)
            except BaseException:
                logger.exception("Failed to settle requests after runtime failure")
        try:
            await asyncio.shield(self.stop())
        except BaseException:
            logger.exception("Failed to stop pipeline after runtime failure")

    def _on_completion_task_done(self, task: asyncio.Task) -> None:
        if task.cancelled() or not self._started:
            return
        error = task.exception()
        if error is None:
            error = RuntimeError("Coordinator completion loop exited unexpectedly")
        if (
            self._completion_failure_task is not None
            and not self._completion_failure_task.done()
        ):
            return
        failure_task = asyncio.create_task(
            self._fail_runtime(error),
            name="pipeline-completion-failure",
        )
        self._completion_failure_task = failure_task
        failure_task.add_done_callback(self._on_completion_failure_done)

    def _on_completion_failure_done(self, task: asyncio.Task) -> None:
        if self._completion_failure_task is task:
            self._completion_failure_task = None
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "Pipeline completion failure cleanup failed",
                exc_info=(type(error), error, error.__traceback__),
            )

    async def wait_failed(self) -> None:
        if self._fatal_event is None:
            raise RuntimeError("Runner not started")
        await self._fatal_event.wait()
        if self._fatal_error is not None:
            raise self._fatal_error
        raise RuntimeError("Pipeline runtime failed")

    async def _cancel_completion_task(self) -> None:
        if self._completion_task is None:
            return
        if not self._completion_task.done():
            self._completion_task.cancel()
        await asyncio.gather(self._completion_task, return_exceptions=True)
        self._completion_task = None

    def _close_runtime_dir(self) -> None:
        if self._ipc_runtime_dir is None:
            return
        self._ipc_runtime_dir.close()
        self._ipc_runtime_dir = None

    async def stop(self) -> None:
        stop_task = self._stop_task
        if stop_task is None:
            start_task = self._start_task
            if not self._started and start_task is None:
                return
            self._stop_requested = True
            self._started = False
            if self._coordinator is not None:
                self._coordinator.begin_stop()
            stop_task = asyncio.create_task(
                self._run_stop(start_task), name="pipeline-runner-stop"
            )
            self._stop_task = stop_task
        await asyncio.shield(stop_task)

    async def _run_stop(self, start_task: asyncio.Task[Any] | None) -> None:
        current_task = asyncio.current_task()
        if (
            start_task is not None
            and start_task is not current_task
            and not start_task.done()
        ):
            start_task.cancel()
            await asyncio.gather(start_task, return_exceptions=True)

        if self._coordinator is None:
            self._close_runtime_dir()
            return

        await self._coordinator.quiesce()

        if self._monitor_task is not None:
            current = asyncio.current_task()
            if current != self._monitor_task:
                self._monitor_task.cancel()
            self._monitor_task = None

        # Send shutdown to stages via coordinator
        try:
            await self._coordinator.shutdown_stages()
        except Exception as e:
            logger.warning("shutdown_stages error: %s", e)

        # Shutdown all groups
        await asyncio.gather(
            *(g.shutdown() for g in self._groups),
            return_exceptions=True,
        )

        await self._cancel_completion_task()

        await self._coordinator.stop()
        self._groups.clear()
        self._coordinator = None

        self._close_runtime_dir()

    async def _cleanup_on_failure(self) -> None:
        """Best-effort cleanup after a failed start()."""

        async def _cleanup() -> None:
            self._started = False
            groups = tuple(self._groups)
            try:
                results = await asyncio.gather(
                    *(group.shutdown(join_timeout=0) for group in groups),
                    return_exceptions=True,
                )
                for group, result in zip(groups, results):
                    if isinstance(result, BaseException):
                        logger.warning(
                            "Failed to clean stage group %s after startup error: %s",
                            group.group_name,
                            result,
                        )
                        try:
                            group.close_control_channels()
                        except Exception:
                            logger.warning(
                                "Failed to close stage group %s control channels",
                                group.group_name,
                                exc_info=True,
                            )
            finally:
                self._groups.clear()

            try:
                await self._cancel_completion_task()
            except BaseException:
                logger.warning(
                    "Failed to stop completion task after startup error",
                    exc_info=True,
                )

            coordinator = self._coordinator
            self._coordinator = None
            if coordinator is not None:
                try:
                    await coordinator.stop()
                except BaseException:
                    logger.warning(
                        "Failed to stop coordinator after startup error",
                        exc_info=True,
                    )

            try:
                self._close_runtime_dir()
            except BaseException:
                logger.warning(
                    "Failed to close runtime directory after startup error",
                    exc_info=True,
                )

        cleanup_task = asyncio.create_task(
            _cleanup(), name="pipeline-start-failure-cleanup"
        )
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                continue
        if not cleanup_task.cancelled():
            cleanup_error = cleanup_task.exception()
            if cleanup_error is not None:
                logger.warning(
                    "Pipeline startup cleanup failed",
                    exc_info=(
                        type(cleanup_error),
                        cleanup_error,
                        cleanup_error.__traceback__,
                    ),
                )
