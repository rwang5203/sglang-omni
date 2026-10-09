# SPDX-License-Identifier: Apache-2.0
"""Cooperative cancellation for native multimodal_gen generation run by Omni.

Native SGLang loads this module in every multimodal_gen process through the
``sglang.multimodal_gen.plugins`` entry point of the sglang-omni package.
``register`` adds hooks only when an Omni native generation stage named a
handshake directory in ACTIVATION_ENV before it started its native workers,
so every other multimodal_gen user in the environment is unaffected.

The stage arms a request by creating ARMED_MARKER in the request's output
directory and cancels it by creating CANCELLED_MARKER beside it. Inside the
native worker, each pipeline stage call and each step of a progress loop
checks the cancelled marker of an armed request and raises
GenerationCancelled, which native error handling turns into a failed output
for that request only.

Keep module scope limited to the standard library: native loads plugins in
each worker before its runtime modules are imported.
"""

from __future__ import annotations

import inspect
import logging
import os
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# The entry point name in the sglang.multimodal_gen.plugins group.
PLUGIN_NAME = "omni_generation_cancellation"
ACTIVATION_ENV = "SGLANG_OMNI_NATIVE_CANCELLATION_DIR"
ARMED_MARKER = ".sglang-omni-armed"
CANCELLED_MARKER = ".sglang-omni-cancelled"

# Hook targets, pinned to SGLang 0.5.21.
STAGE_CALL_TARGET = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base.PipelineStage.__call__"
)
PROGRESS_TARGET = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base."
    "PipelineStage.progress_bar"
)
WORKER_TARGET = (
    "sglang.multimodal_gen.runtime.managers.gpu_worker.run_scheduler_process"
)


class GenerationCancelled(RuntimeError):
    """Raised inside native generation for a request that Omni cancelled."""


class CancellationState:
    """Per process switch, enabled only after the worker handshake."""

    enabled = False


def handshake_path(directory: str, rank: int) -> str:
    return os.path.join(directory, f"active-rank{rank}")


@contextmanager
def activation_environment(directory: str | None) -> Generator[None, None, None]:
    """Arm the native workers started inside this block, then restore."""
    if directory is None:
        yield
        return
    else:
        pass
    previous = os.environ.get(ACTIVATION_ENV)
    os.environ[ACTIVATION_ENV] = directory
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(ACTIVATION_ENV, None)
        else:
            os.environ[ACTIVATION_ENV] = previous


def cancelled_marker(batch: object) -> str | None:
    """The cancelled marker path of an armed request, else None."""
    directory = getattr(batch, "output_path", None)
    if not isinstance(directory, str):
        return None
    elif not os.path.exists(os.path.join(directory, ARMED_MARKER)):
        return None
    else:
        return os.path.join(directory, CANCELLED_MARKER)


def raise_if_cancelled(marker: str) -> None:
    if os.path.exists(marker):
        raise GenerationCancelled("Generation cancelled by its Omni stage")
    else:
        pass


def checked_steps(iterable: Iterable[object], marker: str) -> Iterator[object]:
    for item in iterable:
        raise_if_cancelled(marker)
        yield item


def before_stage_call(
    stage: object, batch: object = None, *args: object, **kwargs: object
) -> None:
    """Stop a cancelled request at a pipeline stage boundary."""
    if CancellationState.enabled:
        marker = cancelled_marker(batch)
        if marker is not None:
            raise_if_cancelled(marker)
        else:
            pass
    else:
        pass


def around_progress_bar(
    original: Callable[..., object], stage: object, *args: object, **kwargs: object
) -> object:
    """Check an armed request's cancelled marker before every loop step."""
    marker = (
        cancelled_marker(kwargs.get("batch")) if CancellationState.enabled else None
    )
    if marker is None:
        return original(stage, *args, **kwargs)
    else:
        pass
    raise_if_cancelled(marker)
    iterable = args[0] if args else kwargs.get("iterable")
    if iterable is not None:
        total = args[1] if len(args) > 1 else kwargs.get("total")
        if total is None and hasattr(iterable, "__len__"):
            kwargs["total"] = len(iterable)
        else:
            pass
        steps = checked_steps(iterable, marker)
        if args:
            args = (steps, *args[1:])
        else:
            kwargs["iterable"] = steps
        return original(stage, *args, **kwargs)
    else:
        pass
    bar = original(stage, *args, **kwargs)
    update = bar.update

    def checked_update(n: float = 1) -> object:
        raise_if_cancelled(marker)
        return update(n)

    try:
        bar.update = checked_update
    except AttributeError:
        pass
    return bar


def announce_worker(*args: object, **kwargs: object) -> None:
    """Enable the hooks and write the handshake once they are in place.

    Never raises, so a mismatch leaves native startup untouched and the stage
    falls back to waiting for aborted work.
    """
    try:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
            PipelineStage,
        )
        from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry

        directory = os.environ.get(ACTIVATION_ENV)
        rank = kwargs.get("rank")
        patched = (
            HookRegistry._patched
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        applied = {STAGE_CALL_TARGET, PROGRESS_TARGET} <= patched
        parameters = inspect.signature(PipelineStage.progress_bar).parameters
        # Independent checks on several ranks could disagree at a collective.
        single_rank = getattr(kwargs.get("server_args"), "num_gpus", None) == 1
        if directory and isinstance(rank, int) and applied and "batch" in parameters:
            CancellationState.enabled = single_rank
        else:
            pass
        if CancellationState.enabled:
            with open(handshake_path(directory, rank), "w"):
                pass
        else:
            logger.warning(
                "Omni cooperative cancellation is off in this native worker "
                "(hooks applied=%s, single rank=%s)",
                applied,
                single_rank,
            )
    except Exception:
        CancellationState.enabled = False
        logger.exception("Omni cooperative cancellation could not start")


def register() -> None:
    """Entry point callback of the sglang.multimodal_gen.plugins group."""
    if not os.environ.get(ACTIVATION_ENV):
        return
    else:
        pass
    from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

    HookRegistry.register(STAGE_CALL_TARGET, before_stage_call, HookType.BEFORE)
    HookRegistry.register(PROGRESS_TARGET, around_progress_bar, HookType.AROUND)
    HookRegistry.register(WORKER_TARGET, announce_worker, HookType.BEFORE)
