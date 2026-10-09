# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from sglang_omni.config import StageConfig
from sglang_omni.config.placement import build_stage_placement_plan
from sglang_omni.models.cosmos3.config import (
    Cosmos3PipelineConfig,
    Cosmos3ReasonerPipelineConfig,
    Variants,
)
from sglang_omni.models.cosmos3.stages import native_server_kwargs
from sglang_omni.pipeline import runtime_config
from sglang_omni.pipeline.mp_runner import build_stage_groups
from sglang_omni.pipeline.replicas import validate_device_assignment
from sglang_omni.scheduling.native_chat_scheduler import native_chat_server_kwargs


def config(devices=(1, 3)):
    cfg = Cosmos3PipelineConfig(model_path="checkpoint")
    cfg.stages[0].gpu = devices[0]
    cfg.stages[0].runtime_gpu_ids = list(devices)
    return cfg


def test_native_workers_reserve_all_gpus_but_use_one_omni_process(monkeypatch):
    from sglang_omni.utils import cpu

    monkeypatch.setattr(runtime_config, "visible_device_count", lambda: 4)
    monkeypatch.setattr(cpu, "effective_cpu_count", lambda: 10)
    cfg = config()
    prep = runtime_config.prepare_pipeline_runtime(cfg)
    try:
        groups = build_stage_groups(
            cfg,
            stages_cfg=prep.stages_cfg,
            endpoints=prep.endpoints,
            placement_plan=prep.placement_plan,
            process_plan=prep.process_plan,
        )
        assert set(prep.placement_plan.gpus) == {1, 3}
        assert len(groups) == 1
        assert len(groups[0].specs) == 1
        spec = groups[0].specs[0]
        assert spec.tp_size == 1
        assert spec.factory_kwargs["runtime_gpu_ids"] == [1, 3]
        # Each of the two native workers gets its own share, not the plan's.
        assert spec.env_defaults["OMP_NUM_THREADS"] == "5"
    finally:
        prep.runtime_dir.close()


def test_native_gpu_budget_includes_child_devices():
    cfg = config()
    cfg.stages[0].gpu_memory_fraction = 0.6
    cfg.stages.append(
        StageConfig(
            name="other",
            factory_path=cfg.stages[0].factory_path,
            gpu=3,
            gpu_memory_fraction=0.6,
            terminal=True,
        )
    )
    with pytest.raises(ValueError, match="exceeds placement limit"):
        build_stage_placement_plan(cfg)


def test_native_gpu_range_is_validated_before_spawn():
    with pytest.raises(ValueError, match="GPU id 3 out of range"):
        validate_device_assignment(config().stages, device_count=2)


def test_native_parallel_options_use_the_declared_gpu_group():
    gen = native_server_kwargs("checkpoint", 1, {"tp_size": 2}, [1, 3])
    assert gen["gpu_ids"] == [1, 3]
    assert gen["num_gpus"] == gen["tp_size"] == 2
    reasoner = native_chat_server_kwargs("checkpoint", 1, None, [1, 3])
    assert reasoner["base_gpu_id"] == 1
    assert reasoner["gpu_id_step"] == 2
    assert reasoner["tp_size"] == 2


@pytest.mark.parametrize("devices", [[], [1, 1], [2, 3], [1, -1]])
def test_invalid_native_gpu_lists_are_rejected(devices):
    cfg = config()
    cfg.stages[0].runtime_gpu_ids = devices
    with pytest.raises(ValueError):
        build_stage_placement_plan(cfg)


def test_native_runtime_cannot_share_a_process_with_another_stage(monkeypatch):
    monkeypatch.setattr(runtime_config, "visible_device_count", lambda: 4)
    cfg = config()
    cfg.stages.append(
        StageConfig(
            name="other",
            process="generation",
            factory_path=cfg.stages[0].factory_path,
            terminal=True,
        )
    )
    with pytest.raises(ValueError, match="own its OS process"):
        runtime_config.prepare_pipeline_runtime(cfg)


@pytest.mark.parametrize("devices", [None, [3, 5]])
def test_generation_factory_passes_resolved_device_to_native(
    monkeypatch, tmp_path, devices
):
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    from sglang_omni.models.cosmos3.stages import create_generation_scheduler
    from sglang_omni.utils import device as device_utils

    monkeypatch.setattr(
        device_utils,
        "resolve_concrete_device",
        lambda device, index: SimpleNamespace(index=3),
    )
    monkeypatch.setattr(DiffGenerator, "supports_cancellation", True, raising=False)

    def startup(**kwargs):
        if devices is None:
            assert kwargs["base_gpu_id"] == 3
        else:
            assert kwargs["gpu_ids"] == [3, 5]
        raise RuntimeError("native startup reached")

    monkeypatch.setattr(ServerArgs, "from_kwargs", startup)
    with pytest.raises(RuntimeError, match="native startup reached"):
        create_generation_scheduler(
            "checkpoint",
            device=None,
            gpu_id=None,
            runtime_gpu_ids=devices,
            output_dir=str(tmp_path),
        )


def test_generation_factory_rejects_cpu_before_native_startup(monkeypatch):
    from sglang_omni.models.cosmos3.stages import create_generation_scheduler
    from sglang_omni.utils import device as device_utils

    monkeypatch.setattr(
        device_utils,
        "resolve_concrete_device",
        lambda device, index: SimpleNamespace(index=None),
    )
    with pytest.raises(ValueError, match="indexed accelerator"):
        create_generation_scheduler("checkpoint", device="cpu")


@pytest.mark.parametrize(
    "case", ["native", "native_off", "plugin", "plugin_missing", "two_gpus"]
)
def test_generation_factory_selects_cooperative_cancellation(
    monkeypatch, tmp_path, caplog, case
):
    import logging
    import os

    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    from sglang_omni.models.cosmos3 import stages
    from sglang_omni.native_generation import cancellation
    from sglang_omni.utils import device as device_utils

    monkeypatch.setattr(
        device_utils,
        "resolve_concrete_device",
        lambda device, index: SimpleNamespace(index=0),
    )
    if case.startswith("native"):
        monkeypatch.setattr(DiffGenerator, "supports_cancellation", True, raising=False)
    else:
        monkeypatch.delattr(DiffGenerator, "supports_cancellation", raising=False)
    monkeypatch.delenv(cancellation.ACTIVATION_ENV, raising=False)
    activations = []

    def start_workers(args):
        # Stands in for a native worker that ran the plugin's handshake.
        directory = os.environ.get(cancellation.ACTIVATION_ENV)
        activations.append(directory)
        if case == "plugin":
            with open(cancellation.handshake_path(directory, 0), "x"):
                pass
        else:
            pass
        return SimpleNamespace(
            shutdown=lambda: None,
            local_scheduler_process=[],
            server_args=None,
            supports_cancellation=case == "native",
        )

    monkeypatch.setattr(ServerArgs, "from_kwargs", lambda **kwargs: kwargs)
    monkeypatch.setattr(DiffGenerator, "from_server_args", start_workers)
    monkeypatch.setattr(stages, "model_request_fields", lambda server_args: frozenset())
    with caplog.at_level(logging.WARNING, logger="sglang_omni.models.cosmos3.stages"):
        scheduler = stages.create_generation_scheduler(
            "checkpoint",
            output_dir=str(tmp_path),
            runtime_gpu_ids=[0, 1] if case == "two_gpus" else None,
        )
    expected = {"native": "native", "plugin": "plugin"}.get(case)
    assert scheduler.cancellation_mode == expected
    assert ("keeps the generation stage busy" in caplog.text) is (expected is None)
    # Only a single GPU stage without native cancellation arms its workers,
    # and the handshake directory and the variable do not outlive startup.
    armed = case in ("plugin", "plugin_missing")
    assert (activations[0] is not None) is armed
    assert activations[0] is None or not os.path.exists(activations[0])
    assert cancellation.ACTIVATION_ENV not in os.environ


def test_reasoner_variant_uses_native_child_process_ownership():
    config = Cosmos3ReasonerPipelineConfig(model_path="checkpoint")
    assert config.native_media_stage is None
    assert config.stages[0].allow_child_processes
    assert config.stages[0].name == "reasoner"
    assert Variants["text"] is Cosmos3ReasonerPipelineConfig


def test_reasoner_is_an_engine_stage():
    from sglang_omni.cli.serve import engine_stage_names

    assert engine_stage_names(Cosmos3ReasonerPipelineConfig(model_path="m")) == [
        "reasoner"
    ]
