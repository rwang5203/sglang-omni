# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 generation configuration backed by SGLang multimodal_gen."""

from collections.abc import Mapping
from typing import ClassVar

from sglang_omni.config import FactoryArgs, PipelineConfig, StageConfig
from sglang_omni.pipeline.umm import (
    CONTROLLER_FACTORY_PATH,
    OUTPUT_FACTORY_PATH,
    ROUTE_FN_PATH,
    STREAM_DONE_FN_PATH,
    UMMPipelineConfig,
)


class Cosmos3PipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "Cosmos3ForConditionalGeneration"
    native_media_stage: ClassVar[str] = "generation"
    native_media_factory_path: ClassVar[str] = (
        "sglang_omni.models.cosmos3.media.prepare_native_media_app"
    )

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="generation",
            process="generation",
            factory_path="sglang_omni.models.cosmos3.stages.create_generation_scheduler",
            allow_child_processes=True,
            factory=FactoryArgs(),
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = Cosmos3PipelineConfig


class Cosmos3ReasonerPipelineConfig(Cosmos3PipelineConfig):
    native_media_stage: ClassVar[str | None] = None
    stages: list[StageConfig] = [
        StageConfig(
            name="reasoner",
            process="reasoner",
            factory_path="sglang_omni.models.cosmos3.reasoner.create_reasoner_scheduler",
            allow_child_processes=True,
            factory=FactoryArgs(),
            gpu=0,
            terminal=True,
        )
    ]


class Cosmos3UMMPipelineConfig(UMMPipelineConfig, Cosmos3PipelineConfig):
    """Interleaved understanding and generation over native SGLang stages.

    The orchestrator runs the shared UMM controller with the Cosmos3 adapter.
    Its phase reason runs on the native Reasoner and its phase generate on
    native generation, which returns inline media to the orchestrator.
    """

    native_media_stage: ClassVar[str | None] = None
    native_media_factory_path: ClassVar[str | None] = None
    entry_stage: str = "orchestrator"
    stages: list[StageConfig] = [
        StageConfig(
            name="orchestrator",
            process="orchestrator",
            factory_path=CONTROLLER_FACTORY_PATH,
            factory=FactoryArgs(
                adapter="sglang_omni.models.cosmos3.umm.Cosmos3UMMAdapter",
                phases={"reason": "reasoner", "generate": "generation"},
                # Eight generation turns, each followed by a Reasoner turn,
                # after the first Reasoner turn. Each turn accepts at most
                # one output.
                max_dispatches=17,
                max_outputs=17,
                max_active=32,
                max_retained_bytes=64 * 1024 * 1024,
                timeout_s=3600.0,
            ),
            next=["reasoner", "generation", "output"],
            route_fn=ROUTE_FN_PATH,
            stream_to=["output"],
            stream_done_to_fn=STREAM_DONE_FN_PATH,
        ),
        StageConfig(
            name="output",
            process="orchestrator",
            factory_path=OUTPUT_FACTORY_PATH,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
        StageConfig(
            name="reasoner",
            process="reasoner",
            factory_path="sglang_omni.models.cosmos3.reasoner.create_reasoner_scheduler",
            allow_child_processes=True,
            factory=FactoryArgs(),
            gpu=1,
            next="orchestrator",
        ),
        StageConfig(
            name="generation",
            process="generation",
            factory_path="sglang_omni.models.cosmos3.stages.create_generation_scheduler",
            allow_child_processes=True,
            factory=FactoryArgs(),
            gpu=0,
            next="orchestrator",
        ),
    ]

    def stage_factory_kwargs(self, stage_name: str) -> Mapping[str, object]:
        if stage_name == "generation":
            # The orchestrator keeps generated media after the stage releases
            # its request directory, so the stage returns the bytes inline.
            return {"inline_media": True}
        else:
            return super().stage_factory_kwargs(stage_name)


Variants = {
    "text": Cosmos3ReasonerPipelineConfig,
    "generation": Cosmos3PipelineConfig,
    "umm": Cosmos3UMMPipelineConfig,
}
