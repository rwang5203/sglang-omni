# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 decisions and native media re-ingestion for the shared UMM loop."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass, field

from sglang_omni.models.cosmos3.reasoner import build_chat_messages
from sglang_omni.models.cosmos3.stages import (
    INLINE_MEDIA_LIMIT_BYTES,
    data_url,
    resolve_generation_options,
)
from sglang_omni.pipeline.umm import (
    CloseReason,
    UMMContext,
    UMMOutput,
    UMMRequestError,
    UMMStep,
)
from sglang_omni.proto import OmniRequest

# Adapter phase names, mapped to stages by the orchestrator's factory.phases.
REASON = "reason"
GENERATE = "generate"

MAX_TEXT_CHARS = 8192
MAX_PROMPT_CHARS = 4096

# A string with a character outside JSON insignificant whitespace. Native
# grammar decoding and parse_decision both read whitespace as ASCII, so every
# decision the schema admits is one the adapter accepts.
NONBLANK_PATTERN = "^[\\s\\S]*\\S[\\s\\S]*$"
NONBLANK = re.compile(r"\S", re.ASCII)

FINAL_DECISION = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["final"]},
        "text": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_TEXT_CHARS,
            "pattern": NONBLANK_PATTERN,
        },
        "generation": {"type": "null"},
    },
    "required": ["kind", "text", "generation"],
    "additionalProperties": False,
}

GENERATE_DECISION = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["generate"]},
        "text": {"type": "string", "maxLength": MAX_TEXT_CHARS},
        "generation": {
            "type": "object",
            "properties": {
                "modality": {"type": "string", "enum": ["image", "video"]},
                "prompt": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_PROMPT_CHARS,
                    "pattern": NONBLANK_PATTERN,
                },
            },
            "required": ["modality", "prompt"],
            "additionalProperties": False,
        },
    },
    "required": ["kind", "text", "generation"],
    "additionalProperties": False,
}

SYSTEM_INSTRUCTION = (
    "You can understand images and videos and request image or video generation. "
    "Respond only with a JSON object matching the supplied schema. "
    "Use kind generate when media is needed, with a precise generation prompt and "
    "modality image or video. The text field is an optional message for the user. "
    "After a generation request, the generated media will be returned to you. "
    "Inspect that media and continue the user's task. You may request another "
    "generation when needed. Use kind final, generation null and a nonempty text "
    "answer only when the task is complete. Do not claim to have generated media "
    "before receiving the generation result."
)

MEDIA_TYPES = {
    "image": ("image/png", "image/jpeg", "image/webp"),
    "video": ("video/mp4",),
}


@dataclass(frozen=True)
class Decision:
    kind: str
    text: str
    generation: dict[str, str] | None


@dataclass
class Cosmos3History:
    """Chat history one request resubmits to the Reasoner, and its bytes."""

    messages: list[dict[str, object]] = field(default_factory=list)
    retained_bytes: int = 0
    decision: Decision | None = None


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Cosmos3 decision contains duplicate JSON keys")
        else:
            pass
        result[key] = value
    return result


def parse_decision(value: object) -> Decision:
    if not isinstance(value, dict) or set(value) != {"kind", "text", "generation"}:
        raise ValueError("Cosmos3 requires a structured final or generate decision")
    else:
        pass
    kind, text, generation = value["kind"], value["text"], value["generation"]
    if not isinstance(text, str) or len(text) > MAX_TEXT_CHARS:
        raise ValueError("Cosmos3 decision text exceeds its string limit")
    elif kind == "final":
        if generation is not None or NONBLANK.search(text) is None:
            raise ValueError("Final Cosmos3 decisions require text and no generation")
        else:
            return Decision(kind, text, None)
    elif kind != "generate":
        raise ValueError("Unknown Cosmos3 decision kind")
    elif not isinstance(generation, dict) or set(generation) != {"modality", "prompt"}:
        raise ValueError("Cosmos3 generation requires only modality and prompt")
    elif generation["modality"] not in MEDIA_TYPES:
        raise ValueError("Cosmos3 interleaving currently supports images and videos")
    else:
        prompt = generation["prompt"]
        if (
            not isinstance(prompt, str)
            or NONBLANK.search(prompt) is None
            or len(prompt) > MAX_PROMPT_CHARS
        ):
            raise ValueError("Cosmos3 generation requires a bounded nonempty prompt")
        else:
            return Decision(
                kind, text, {"modality": generation["modality"], "prompt": prompt}
            )


def interpret_reasoner(result: object) -> Decision:
    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
        raise ValueError("Cosmos3 Reasoner did not return a structured text decision")
    elif result.get("finish_reason") not in (None, "stop"):
        raise ValueError("Cosmos3 Reasoner decision was truncated or interrupted")
    else:
        pass
    try:
        parsed = json.loads(result["text"], object_pairs_hook=unique_object)
    except (ValueError, RecursionError) as exc:
        raise ValueError("Cosmos3 Reasoner returned an invalid JSON decision") from exc
    return parse_decision(parsed)


def generated_media(result: object) -> dict[str, object]:
    """The one owned inline item of a generation turn, with verified content."""
    if not isinstance(result, dict) or not isinstance(result.get("media"), list):
        raise ValueError("Cosmos3 generation did not return media")
    elif len(result["media"]) != 1:
        raise ValueError(
            "Cosmos3 interleaving requires one generated media item per turn"
        )
    else:
        pass
    item = result["media"][0]
    if not isinstance(item, dict) or "path" in item:
        raise ValueError("Cosmos3 interleaving requires owned inline media")
    else:
        pass
    kind, mime, size = item.get("kind"), item.get("mime_type"), item.get("size_bytes")
    if not isinstance(kind, str) or mime not in MEDIA_TYPES.get(kind, ()):
        raise ValueError("Unsupported inline Cosmos3 media type")
    elif type(size) is not int or not 0 < size <= INLINE_MEDIA_LIMIT_BYTES:
        raise ValueError("Cosmos3 inline media exceeds its byte limit")
    else:
        pass
    url = item.get("url")
    prefix = data_url(mime, b"")
    if not isinstance(url, str) or not url.startswith(prefix):
        raise ValueError("Cosmos3 media must be a typed data URL")
    elif len(url) - len(prefix) > 4 * ((INLINE_MEDIA_LIMIT_BYTES + 2) // 3):
        raise ValueError("Cosmos3 inline media exceeds its encoded byte limit")
    else:
        pass
    try:
        content = base64.b64decode(url[len(prefix) :], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Malformed Cosmos3 inline media") from exc
    if len(content) != size or hashlib.sha256(content).hexdigest() != item.get(
        "sha256"
    ):
        raise ValueError("Cosmos3 inline media identity does not match its content")
    else:
        return item


def encoded_size(message: dict[str, object]) -> int:
    return len(json.dumps(message, ensure_ascii=False).encode("utf-8"))


class Cosmos3UMMAdapter:
    """Translate Cosmos3 decisions while native runtimes keep model execution.

    Each request alternates between the Reasoner (phase reason), which answers
    with a structured decision, and native generation (phase generate), whose
    media returns to the Reasoner in the next turn.
    """

    def __init__(self, model_path: str) -> None:
        del model_path
        self.histories: dict[str, Cosmos3History] = {}

    def start(self, context: UMMContext) -> UMMStep:
        try:
            messages = build_chat_messages(context.request.inputs)
        except ValueError as exc:
            raise UMMRequestError(str(exc)) from exc
        if any(
            message.get("role") not in ("system", "user", "assistant")
            for message in messages
        ):
            raise UMMRequestError(
                "Cosmos3 interleaving accepts native user and assistant chat history"
            )
        elif messages[0].get("role") == "system" and isinstance(
            messages[0].get("content"), str
        ):
            messages[0]["content"] = (
                SYSTEM_INSTRUCTION + "\n\n" + messages[0]["content"]
            )
        else:
            messages.insert(0, {"role": "system", "content": SYSTEM_INSTRUCTION})
        history = Cosmos3History()
        self.histories[context.execution_id] = history
        for message in messages:
            self.remember(history, message)
        return self.reason(context, history, ())

    @staticmethod
    def remember(history: Cosmos3History, message: dict[str, object]) -> None:
        history.messages.append(message)
        history.retained_bytes += encoded_size(message)

    def reason(
        self,
        context: UMMContext,
        history: Cosmos3History,
        outputs: tuple[UMMOutput, ...],
    ) -> UMMStep:
        params = deepcopy(context.request.params)
        # Flatten native reasoner overrides first so the internal decision
        # contract cannot be replaced by a stage-specific stream or schema.
        for key in ("stage_sampling", "stage_params"):
            value = params.pop(key, {})
            if not isinstance(value, dict):
                raise UMMRequestError(f"{key} must be a mapping")
            else:
                pass
            reasoner = value.get("reasoner", {})
            if not isinstance(reasoner, dict):
                raise UMMRequestError("Reasoner stage parameters must be a mapping")
            else:
                pass
            params.update(reasoner)
        template = params.get("chat_template_kwargs") or {}
        if not isinstance(template, dict):
            raise UMMRequestError("chat_template_kwargs must be a mapping")
        else:
            pass
        # A generation and the Reasoner turn that inspects it need two more
        # dispatches after this one, so with fewer left only final remains.
        if context.remaining_dispatches < 3:
            schema = deepcopy(FINAL_DECISION)
        else:
            schema = {"anyOf": [deepcopy(FINAL_DECISION), deepcopy(GENERATE_DECISION)]}
        params.update(
            stream=False,
            n=1,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "cosmos3_umm_decision",
                    "strict": True,
                    "schema": schema,
                },
            },
            chat_template_kwargs={**template, "enable_thinking": False},
        )
        params.setdefault("temperature", 0.0)
        params.setdefault("max_tokens", 2048)
        request = OmniRequest(
            {"messages": list(history.messages)},
            params,
            deepcopy(context.request.metadata),
        )
        return UMMStep(REASON, request, outputs)

    def advance(self, context: UMMContext, phase: str, result: object) -> UMMStep:
        history = self.histories[context.execution_id]
        if phase == REASON:
            decision = interpret_reasoner(result)
            outputs = (UMMOutput("text", decision.text),) if decision.text else ()
            if decision.kind == "final":
                return UMMStep(outputs=outputs)
            else:
                history.decision = decision
                request = self.generation_request(context, decision)
                return UMMStep(GENERATE, request, outputs)
        elif phase == GENERATE and history.decision is not None:
            decision = history.decision
            item = generated_media(result)
            if item["kind"] != decision.generation["modality"]:
                raise ValueError(
                    "Generated Cosmos3 modality differs from the model decision"
                )
            else:
                pass
            self.remember(history, self.decision_message(decision))
            self.remember(history, self.receipt(decision, item))
            history.decision = None
            return self.reason(context, history, (UMMOutput(item["kind"], item),))
        else:
            raise ValueError(f"Cosmos3 adapter received an unexpected phase {phase!r}")

    def usage(self, context: UMMContext) -> int:
        history = self.histories.get(context.execution_id)
        return 0 if history is None else history.retained_bytes

    def close(self, context: UMMContext, reason: CloseReason) -> None:
        self.histories.pop(context.execution_id, None)

    @staticmethod
    def generation_request(context: UMMContext, decision: Decision) -> OmniRequest:
        try:
            options = resolve_generation_options(deepcopy(context.request.params))
        except ValueError as exc:
            raise UMMRequestError(str(exc)) from exc
        action_mode = options.get("action_mode")
        if action_mode is not None and (
            not isinstance(action_mode, str)
            or action_mode.strip().lower() != "forward_dynamics"
        ):
            raise UMMRequestError(
                "Cosmos3 visual interleaving cannot return action-only results"
            )
        else:
            pass
        options["prompt"] = decision.generation["prompt"]
        options["num_outputs_per_prompt"] = 1
        if decision.generation["modality"] == "image":
            if action_mode is not None:
                raise UMMRequestError(
                    "Cosmos3 forward_dynamics requires a video decision"
                )
            else:
                options["num_frames"] = 1
        else:
            options.setdefault("num_frames", 33)
            if type(options["num_frames"]) is not int or options["num_frames"] <= 1:
                raise UMMRequestError("Video interleaving requires more than one frame")
            else:
                pass
        return OmniRequest(
            {"prompt": decision.generation["prompt"]},
            {"diffusion": options, "stream": False},
            deepcopy(context.request.metadata),
        )

    @staticmethod
    def decision_message(decision: Decision) -> dict[str, object]:
        content = json.dumps(
            {
                "kind": decision.kind,
                "text": decision.text,
                "generation": decision.generation,
            },
            ensure_ascii=False,
        )
        return {"role": "assistant", "content": content}

    @staticmethod
    def receipt(decision: Decision, item: dict[str, object]) -> dict[str, object]:
        status = json.dumps(
            {
                "status": "completed",
                "modality": item["kind"],
                "requested_prompt": decision.generation["prompt"],
            },
            ensure_ascii=False,
        )
        text = (
            "Generation execution completed. Result receipt: "
            + status
            + "\nThe attached media is the result of your preceding generation "
            "request. Inspect the received results against the original user's "
            "task. If the task is complete, choose kind final with generation "
            "null and your answer in text. Otherwise request the next needed "
            "generation. Repeat a completed generation only when the user's "
            "task or your assessment of its result calls for a revision."
        )
        part = f"{item['kind']}_url"
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": part, part: {"url": item["url"]}},
            ],
        }
