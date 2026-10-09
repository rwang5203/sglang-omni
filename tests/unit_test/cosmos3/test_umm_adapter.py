# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the Cosmos3 UMM adapter, without model execution."""

from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest, SamplingParams
from sglang_omni.models.cosmos3 import stages as cosmos3_stages
from sglang_omni.models.cosmos3.reasoner import build_chat_fields
from sglang_omni.models.cosmos3.stages import (
    INLINE_MEDIA_LIMIT_BYTES,
    NativeGenerationScheduler,
    build_sampling_params,
    data_url,
)
from sglang_omni.models.cosmos3.umm import Cosmos3UMMAdapter
from sglang_omni.pipeline.umm import (
    CONTINUATION_METADATA_KEY,
    UMMController,
    UMMLimits,
    route_umm,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.serve.openai_errors import generation_error_status_code

PHASES = {"reason": "reasoner", "generate": "generation"}


def controller(**bounds: int) -> UMMController:
    limits = {
        "max_dispatches": 17,
        "max_outputs": 17,
        "max_active": 32,
        "max_retained_bytes": 64 * 1024 * 1024,
        "timeout_s": 3600.0,
        **bounds,
    }
    return UMMController(
        Cosmos3UMMAdapter("checkpoint"),
        limits=UMMLimits(**limits),
        phases=PHASES,
        output_stage="output",
    )


def submit(umm: UMMController, inputs="Draw a cube.", params=None, rid="r"):
    request = OmniRequest(inputs, params or {})
    return umm.advance(StagePayload(rid, request, request.inputs))


def reply(dispatched: StagePayload, data: object) -> StagePayload:
    return StagePayload(dispatched.request_id, dispatched.request, data)


def reasoner_reply(kind="generate", text="", modality="image", prompt="A blue cube"):
    generation = None if kind == "final" else {"modality": modality, "prompt": prompt}
    return {
        "text": json.dumps({"kind": kind, "text": text, "generation": generation}),
        "finish_reason": "stop",
    }


def media(content=b"image fixture", kind="image"):
    mime = "image/png" if kind == "image" else "video/mp4"
    return {
        "media": [
            {
                "kind": kind,
                "mime_type": mime,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "url": data_url(mime, content),
            }
        ]
    }


def rejection(umm: UMMController) -> str:
    (message,) = [umm.outbox.get_nowait() for _ in range(umm.outbox.qsize())]
    assert message.type == "error"
    return str(message.data)


@pytest.mark.parametrize(
    "inputs",
    [
        {"messages": [{"role": "user"}, 1]},
        {"messages": [{"role": "tool", "content": "result"}]},
    ],
)
def test_invalid_chat_history_is_a_client_error_before_any_dispatch(inputs):
    umm = controller()
    original = deepcopy(inputs)
    assert submit(umm, inputs) is None
    assert generation_error_status_code(RuntimeError(rejection(umm))) == 400
    assert not umm.contexts and not umm.adapter.histories
    assert inputs == original
    dispatched = submit(umm, "hello", rid="healthy")
    assert route_umm("healthy", dispatched) == "reasoner"


def test_native_chat_contract_and_user_options_survive_internal_decision():
    umm = controller()
    params = {
        "stream": True,
        "stage_params": {
            "reasoner": {
                "stream": True,
                "response_format": {"type": "text"},
                "temperature": 0.25,
                "max_new_tokens": 4096,
            }
        },
    }
    inputs = {
        "messages": [{"role": "user", "content": "Make a blue cube"}],
        "images": [data_url("image/png", bytes(3))],
    }
    original = deepcopy(params)
    internal = submit(umm, inputs, params)
    fields = build_chat_fields(
        internal,
        "cosmos3",
        {
            "stream",
            "response_format",
            "temperature",
            "max_tokens",
            "chat_template_kwargs",
            "n",
        },
    )
    assert fields["stream"] is False
    assert fields["response_format"]["json_schema"]["strict"]
    assert fields["chat_template_kwargs"]["enable_thinking"] is False
    assert fields["temperature"] == 0.25
    assert fields["max_tokens"] == 4096
    assert fields["messages"][1]["content"][1]["image_url"]["url"].endswith("AAAA")
    assert params == original


def decision_text(kind, text, generation):
    return {"text": json.dumps({"kind": kind, "text": text, "generation": generation})}


@pytest.mark.parametrize(
    "response",
    [
        {"text": "Here is an image prompt: A blue cube"},
        {"text": '```json\n{"kind":"final","text":"done","generation":null}\n```'},
        {"text": '{"kind":"final","kind":"final","text":"done","generation":null}'},
        {
            "text": '{"kind":"final","text":"done","generation":null}',
            "finish_reason": "length",
        },
        {"text": None},
        decision_text("final", "x", {}),
        decision_text("generate", "", {"modality": "audio", "prompt": "x"}),
        decision_text("generate", "", {"modality": "image", "prompt": "x", "y": 1}),
    ],
)
def test_unstructured_or_malformed_decisions_fail_as_server_errors(response):
    umm = controller()
    dispatched = submit(umm)
    with pytest.raises(ValueError) as rejected:
        umm.advance(reply(dispatched, response))
    assert generation_error_status_code(rejected.value) == 500
    assert not umm.contexts and not umm.adapter.histories


def generate_turn(umm, decision=None, params=None, inputs="task"):
    dispatched = submit(umm, inputs, params)
    return umm.advance(reply(dispatched, decision or reasoner_reply()))


def test_generation_decision_controls_prompt_and_modality_with_native_tuning(tmp_path):
    params = {
        "stream": True,
        "diffusion": {"prompt": "stale", "seed": 0, "num_frames": 33},
        "stage_params": {
            "generation": {
                "prompt": "also stale",
                "guidance_scale": 0.0,
                "num_outputs_per_prompt": 9,
            }
        },
    }
    original = deepcopy(params)
    generation = generate_turn(controller(), params=params)
    assert route_umm("r", generation) == "generation"
    native = build_sampling_params(generation, str(tmp_path))
    assert native["prompt"] == "A blue cube"
    assert native["num_frames"] == 1
    assert native["num_outputs_per_prompt"] == 1
    assert native["seed"] == 0
    assert native["guidance_scale"] == 0.0
    assert params == original


@pytest.mark.parametrize(
    "diffusion, frames",
    [({}, 33), ({"num_frames": 65}, 65), ({"num_frames": 1}, None)],
)
def test_video_decisions_use_native_frame_options(diffusion, frames):
    umm = controller()
    video = reasoner_reply(modality="video")
    if frames is None:
        assert generate_turn(umm, video, {"diffusion": diffusion}) is None
        assert "more than one frame" in rejection(umm)
    else:
        generation = generate_turn(umm, video, {"diffusion": diffusion})
        assert generation.request.params["diffusion"]["num_frames"] == frames


@pytest.mark.parametrize(
    "change",
    [
        {"path": "/foreign.png"},
        {"kind": "audio"},
        {"kind": "video"},
        {"mime_type": "image/jpeg"},
        {"size_bytes": True},
        {"size_bytes": INLINE_MEDIA_LIMIT_BYTES + 1},
        {"sha256": "0" * 64},
        {"url": "file:///foreign.png"},
        {"url": data_url("image/png", b"") + "????"},
        None,
    ],
)
def test_generated_media_must_be_one_owned_item_of_the_chosen_modality(change):
    umm = controller()
    generation = generate_turn(umm)
    value = media()
    if change is None:
        value["media"].append(deepcopy(value["media"][0]))
    else:
        value["media"][0].update(change)
    with pytest.raises(ValueError):
        umm.advance(reply(generation, value))
    assert not umm.contexts and not umm.adapter.histories


def test_last_generation_turn_allows_only_a_final_decision():
    umm = controller(max_dispatches=3)
    first = submit(umm, "Draw a cube, then describe it.")
    generation = umm.advance(reply(first, reasoner_reply()))
    last = umm.advance(reply(generation, media()))
    schemas = [
        payload.request.params["response_format"]["json_schema"]["schema"]
        for payload in (first, last)
    ]
    assert [schema["properties"]["kind"]["enum"] for schema in schemas] == [
        ["final", "generate"],
        ["final"],
    ]
    assert schemas[1]["properties"]["generation"] == {"type": "null"}
    final = umm.advance(reply(last, reasoner_reply("final", "A cube.")))
    assert route_umm("r", final) == "output"
    parsed = Client.default_result_builder("r", final.data)
    assert [segment.kind for segment in parsed.segments] == ["image", "text"]
    assert parsed.text == "A cube."
    assert not umm.contexts and not umm.adapter.histories


class SavedGenerator:
    local_scheduler_process = None

    def __init__(self, content=b"png fixture", foreign=None, symlink=False):
        self.content, self.foreign, self.symlink = content, foreign, symlink

    def generate(self, *, sampling_params_kwargs):
        target = Path(sampling_params_kwargs["output_path"]) / "result.png"
        if self.symlink:
            target.symlink_to(self.foreign)
        elif self.foreign is None:
            target.write_bytes(self.content)
        else:
            target = self.foreign
        return [
            SimpleNamespace(
                output_file_path=str(target),
                size=(16, 16),
                prompt="cube",
                generation_time=0,
                peak_memory_mb=0,
                metrics={},
            )
        ]

    def shutdown(self):
        pass


def inline_scheduler(tmp_path, generator=None):
    return NativeGenerationScheduler(
        generator or SavedGenerator(), str(tmp_path), inline_media=True
    )


def test_inline_output_is_owned_hashed_and_released_before_forwarding(tmp_path):
    keep = tmp_path / "keep"
    keep.write_text("keep")
    scheduler = inline_scheduler(tmp_path)
    result = scheduler.generate(StagePayload("r", OmniRequest("cube"), None))
    assert list(tmp_path.iterdir()) == [keep]
    assert not scheduler.native_requests
    assert scheduler.claim_result(result, terminal=False)
    item = result.data["media"][0]
    assert "path" not in item
    assert item["sha256"] == hashlib.sha256(b"png fixture").hexdigest()
    assert base64.b64decode(item["url"].split(",", 1)[1]) == b"png fixture"


@pytest.mark.parametrize("symlink", [False, True])
def test_inline_generation_rejects_foreign_files_without_removing_them(
    tmp_path, symlink
):
    foreign = tmp_path / "foreign.png"
    foreign.write_bytes(b"retain")
    scheduler = inline_scheduler(
        tmp_path, SavedGenerator(foreign=foreign, symlink=symlink)
    )
    with pytest.raises(ValueError, match="outside its owned request directory"):
        scheduler.generate(StagePayload("r", OmniRequest("cube"), None))
    assert list(tmp_path.iterdir()) == [foreign]
    assert foreign.read_bytes() == b"retain"
    assert not scheduler.native_requests


@pytest.mark.parametrize("content", [b"1234", b""])
def test_inline_byte_bounds_fail_and_release_only_owned_directory(
    tmp_path, monkeypatch, content
):
    monkeypatch.setattr(cosmos3_stages, "INLINE_MEDIA_LIMIT_BYTES", 3)
    scheduler = inline_scheduler(tmp_path, SavedGenerator(content))
    with pytest.raises(ValueError, match="byte limit"):
        scheduler.generate(StagePayload("r", OmniRequest("cube"), None))
    assert list(tmp_path.iterdir()) == []
    assert not scheduler.native_requests


def test_umm_variant_wires_the_controller_and_inline_generation():
    from sglang_omni.config.runtime import resolve_stage_typed_kwargs
    from sglang_omni.models.cosmos3.config import Cosmos3UMMPipelineConfig, Variants
    from sglang_omni.pipeline.umm import create_umm_controller
    from tests.unit_test.pipeline.helpers import build_compiled_process_topology

    config = Cosmos3UMMPipelineConfig(model_path="checkpoint")
    build_compiled_process_topology(config)
    assert Variants["umm"] is Cosmos3UMMPipelineConfig
    assert config.stage_factory_kwargs("generation") == {"inline_media": True}
    args = {
        **resolve_stage_typed_kwargs(config.stage_named("orchestrator")),
        **config.stage_factory_kwargs("orchestrator"),
    }
    umm = create_umm_controller("checkpoint", **args)
    assert isinstance(umm.adapter, Cosmos3UMMAdapter)
    assert config.entry_stage_admission_limit() == umm.limits.max_active
    umm.stop()


@pytest.mark.parametrize("kind", ["image", "video"])
def test_completed_receipt_binds_model_prompt_and_resubmits_history(kind):
    umm = controller()
    prompt = 'A sign reading "你好" on a cube.\nUse the requested lighting.'
    generation = generate_turn(umm, reasoner_reply(modality=kind, prompt=prompt))
    generated = media(b"completed result", kind=kind)
    original = deepcopy(generated)
    resumed = umm.advance(reply(generation, generated))
    assert route_umm("r", resumed) == "reasoner"
    messages = resumed.request.inputs["messages"]
    receipt = messages[-1]
    text = receipt["content"][0]["text"]
    status = json.loads(text.split("Result receipt: ", 1)[1].split("\n", 1)[0])
    assert status == {
        "status": "completed",
        "modality": kind,
        "requested_prompt": prompt,
    }
    assert receipt["content"][1] == {
        "type": f"{kind}_url",
        f"{kind}_url": {"url": generated["media"][0]["url"]},
    }
    assert json.loads(messages[-2]["content"])["generation"]["prompt"] == prompt
    assert generated == original


@pytest.mark.parametrize("seed", [0, 7])
def test_umm_generation_preserves_sdk_seed(seed):
    request = Client.build_omni_request(
        GenerateRequest(prompt="draw a lake", sampling=SamplingParams(seed=seed))
    )
    generation = generate_turn(
        controller(), params=request.params, inputs=request.inputs
    )
    native = build_sampling_params(generation, "outputs")
    assert native["seed"] == seed


@pytest.mark.parametrize(
    "diffusion, message",
    [
        ({"action_mode": "policy"}, "action-only"),
        ({"action_mode": " POLICY "}, "action-only"),
        ({"action_mode": "forward_dynamics"}, "forward_dynamics requires a video"),
    ],
)
def test_action_modes_reject_before_visual_generation(diffusion, message):
    umm = controller()
    assert generate_turn(umm, params={"diffusion": diffusion}) is None
    error = rejection(umm)
    assert message in error
    assert generation_error_status_code(RuntimeError(error)) == 400


def test_forward_dynamics_video_keeps_the_known_actions():
    options = {"action_mode": "forward_dynamics", "action": [[0.0] * 8] * 4}
    generation = generate_turn(
        controller(), reasoner_reply(modality="video"), {"diffusion": options}
    )
    assert generation.request.params["diffusion"]["action"] == options["action"]
    assert generation.request.params["diffusion"]["num_frames"] > 1
    assert CONTINUATION_METADATA_KEY in generation.request.metadata


@pytest.mark.parametrize(
    "params",
    [
        {"stage_params": {"reasoner": []}},
        {"chat_template_kwargs": "invalid"},
        {"diffusion": []},
    ],
)
def test_user_option_rejections_are_client_errors(params):
    umm = controller()
    dispatched = submit(umm, "hello", params)
    if dispatched is not None:
        assert umm.advance(reply(dispatched, reasoner_reply())) is None
    else:
        pass
    assert generation_error_status_code(RuntimeError(rejection(umm))) == 400
    assert not umm.contexts and not umm.adapter.histories
