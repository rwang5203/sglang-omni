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
    check_media_source,
    data_url,
    media_policy_configured,
    resolve_generation_options,
)
from sglang_omni.pipeline.umm import (
    CloseReason,
    UMMContext,
    UMMOutput,
    UMMRequestError,
    UMMStep,
)
from sglang_omni.preprocessing.resource_connector import get_global_resource_connector
from sglang_omni.proto import (
    EXPLICIT_GENERATION_PARAMS_KEY,
    EXPLICIT_STAGE_SAMPLING_PARAMS_KEY,
    OmniRequest,
)

# Adapter phase names, mapped to stages by the orchestrator's factory.phases.
REASON = "reason"
GENERATE = "generate"
# The stage name clients use for Reasoner overrides in stage_sampling and
# stage_params.
REASONER_STAGE = "reasoner"

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

# Decisions sample greedily unless the client sets a sampling field, and only
# the decision schema bounds their length: a client length or stop setting
# would truncate the JSON and fail the request.
DECISION_SAMPLING = {"temperature": 0.0}
SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty")
DECISION_OWNED_FIELDS = frozenset(
    {
        "chat_template_kwargs",
        "ignore_eos",
        "max_completion_tokens",
        "max_new_tokens",
        "max_tokens",
        "n",
        "response_format",
        "stop",
        "stop_token_ids",
        "stream",
    }
)
# Interleaved results carry no logprobs or rollout artifacts.
ROLLOUT_FIELDS = (
    "return_logprob",
    "return_omni_rollout",
    "return_routed_experts",
    "return_indexer_topk",
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
    """Chat history one request resubmits to the Reasoner, and its bytes.

    A receipt refers to its generated media instead of copying it, since the
    controller already retains that media as an output. media_urls holds the
    message index of each receipt and the data URL that output carries.
    """

    reasons_left: int
    generations_left: int
    sampling: dict[str, object]
    options: dict[str, object]
    messages: list[dict[str, object]] = field(default_factory=list)
    media_urls: list[tuple[int, str]] = field(default_factory=list)
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
        return {
            "kind": kind,
            "mime_type": mime,
            "size_bytes": size,
            "sha256": item["sha256"],
            "url": url,
        }


def stage_mapping(params: dict[str, object], key: str) -> dict[str, object]:
    """The Reasoner entry of a per-stage mapping, which must be a mapping."""
    value = params.get(key)
    if value is None:
        return {}
    elif not isinstance(value, dict):
        raise UMMRequestError(f"{key} must be a mapping")
    else:
        reasoner = value.get("reasoner", {})
        if not isinstance(reasoner, dict):
            raise UMMRequestError(f"{key}.reasoner must be a mapping")
        else:
            return reasoner


def decision_sampling(request: OmniRequest) -> dict[str, object]:
    """Native sampling fields for every decision of one request.

    The adapter's defaults apply to fields the client did not set. The client
    sets fields explicitly at the top level, through stage_sampling.reasoner,
    or as native fields in stage_params.reasoner.
    """
    params, metadata = request.params, request.metadata
    requested = [name for name in ROLLOUT_FIELDS if params.get(name)]
    if requested:
        raise UMMRequestError(
            "Cosmos3 interleaving returns no logprobs or rollout artifacts. "
            f"Leave {', '.join(requested)} unset or false"
        )
    else:
        pass
    explicit = set(metadata.get(EXPLICIT_GENERATION_PARAMS_KEY) or ())
    sampling: dict[str, object] = dict(DECISION_SAMPLING)
    sampling.update(
        {name: params[name] for name in SAMPLING_FIELDS if name in explicit}
    )
    if params.get("seed") is not None:
        sampling["seed"] = params["seed"]
    else:
        pass
    stage_sampling = stage_mapping(params, "stage_sampling")
    explicit_stage = (metadata.get(EXPLICIT_STAGE_SAMPLING_PARAMS_KEY) or {}).get(
        REASONER_STAGE
    )
    for name, value in stage_sampling.items():
        if name in DECISION_OWNED_FIELDS or (
            explicit_stage is not None and name not in explicit_stage
        ):
            continue
        else:
            sampling[name] = value
    reasoner = stage_mapping(params, "stage_params")
    for name, value in reasoner.items():
        if name not in DECISION_OWNED_FIELDS:
            sampling[name] = value
        else:
            pass
    template = reasoner.get("chat_template_kwargs", params.get("chat_template_kwargs"))
    if template is not None and not isinstance(template, dict):
        raise UMMRequestError("chat_template_kwargs must be a mapping")
    else:
        sampling["chat_template_kwargs"] = {
            **(template or {}),
            "enable_thinking": False,
        }
    return sampling


def generation_options(request: OmniRequest) -> dict[str, object]:
    """The client's native generation options, checked before any turn runs."""
    try:
        options = resolve_generation_options(deepcopy(request.params))
    except ValueError as exc:
        raise UMMRequestError(str(exc)) from exc
    action_mode = options.get("action_mode")
    frames = options.get("num_frames")
    if action_mode is not None and (
        not isinstance(action_mode, str)
        or action_mode.strip().lower() != "forward_dynamics"
    ):
        raise UMMRequestError(
            "Cosmos3 visual interleaving cannot return action-only results"
        )
    elif frames is not None and (type(frames) is not int or frames <= 1):
        # Image decisions always generate one frame, so a client frame count
        # applies to videos only.
        raise UMMRequestError("Video interleaving requires more than one frame")
    else:
        return options


def check_media_policy(
    messages: list[dict[str, object]], options: dict[str, object]
) -> None:
    """Refuse client media the server media policy refuses, before any turn.

    Chat media parts and generation path options are checked without being
    read. The stages that read them apply the same policy again.
    """
    connector = get_global_resource_connector()
    if not media_policy_configured(connector):
        return
    else:
        pass
    sources: list[tuple[str, object]] = []
    for message in messages:
        content = message.get("content")
        for part in content if isinstance(content, list) else []:
            kind = part.get("type") if isinstance(part, dict) else None
            if isinstance(kind, str) and kind.endswith("_url"):
                reference = part.get(kind)
                url = reference.get("url") if isinstance(reference, dict) else None
                sources.append((kind, url))
            else:
                pass
    for name, value in options.items():
        if name.endswith("_path"):
            items = value if isinstance(value, list) else [value]
            sources.extend((name, item) for item in items)
        else:
            pass
    for name, source in sources:
        try:
            check_media_source(connector, name, source)
        except ValueError as exc:
            raise UMMRequestError(
                f"{name} refused by the server media policy: {exc}"
            ) from exc


def encoded_size(message: dict[str, object]) -> int:
    return len(json.dumps(message, ensure_ascii=False).encode("utf-8"))


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


def receipt(decision: Decision, kind: str) -> dict[str, object]:
    """The user turn that returns a generation, its media URL left empty."""
    status = json.dumps(
        {
            "status": "completed",
            "modality": kind,
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
    part = f"{kind}_url"
    return {
        "role": "user",
        "content": [{"type": "text", "text": text}, {"type": part, part: {"url": ""}}],
    }


def worst_case_bytes() -> tuple[int, int]:
    """Most bytes one Reasoner turn and one generation turn can add.

    A Reasoner turn adds its text output, counted in UTF-8 at up to four bytes
    a character, and its decision in the history, where a control character
    costs the most: seven bytes once the decision and then the message are
    JSON encoded. A generation turn adds its media output, encoded the way the
    controller counts outputs, and its receipt.
    """
    worst = Decision(
        "generate",
        "\x01" * MAX_TEXT_CHARS,
        {"modality": "video", "prompt": "\x01" * MAX_PROMPT_CHARS},
    )
    reason = 4 * MAX_TEXT_CHARS + encoded_size(decision_message(worst))
    mime = max((m for mimes in MEDIA_TYPES.values() for m in mimes), key=len)
    item = {
        "kind": "video",
        "mime_type": mime,
        "size_bytes": INLINE_MEDIA_LIMIT_BYTES,
        "sha256": "0" * 64,
        "url": data_url(mime, b""),
    }
    # Base64 characters need no JSON escaping.
    media = encoded_size(item) + 4 * ((INLINE_MEDIA_LIMIT_BYTES + 2) // 3)
    return reason, media + encoded_size(receipt(worst, "video"))


WORST_CASE_BYTES = worst_case_bytes()


def max_model_bytes(max_dispatches: int) -> int:
    """Most bytes a request's model turns can retain within max_dispatches.

    The first dispatch is a Reasoner turn, and each generation is followed by
    a Reasoner turn that inspects it.
    """
    generations = (max_dispatches - 1) // 2
    reason, generation = WORST_CASE_BYTES
    return (generations + 1) * reason + generations * generation


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
        options = generation_options(context.request)
        check_media_policy(messages, options)
        generations = (context.remaining_dispatches - 1) // 2
        history = Cosmos3History(
            reasons_left=generations + 1,
            generations_left=generations,
            sampling=decision_sampling(context.request),
            options=options,
        )
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
        # A generation and the Reasoner turn that inspects it need two more
        # dispatches after this one, so with fewer left only final remains.
        if context.remaining_dispatches < 3:
            schema = deepcopy(FINAL_DECISION)
        else:
            generate = deepcopy(GENERATE_DECISION)
            if history.options.get("action_mode") is not None:
                # Known actions condition a video, so only videos remain.
                modality = generate["properties"]["generation"]["properties"]
                modality["modality"]["enum"] = ["video"]
            else:
                pass
            schema = {"anyOf": [deepcopy(FINAL_DECISION), generate]}
        params = {
            **deepcopy(history.sampling),
            "stream": False,
            "n": 1,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "cosmos3_umm_decision",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        messages = list(history.messages)
        for index, url in history.media_urls:
            message = messages[index]
            text, media = message["content"]
            part = media["type"]
            messages[index] = {
                **message,
                "content": [text, {"type": part, part: {"url": url}}],
            }
        return UMMStep(REASON, OmniRequest({"messages": messages}, params), outputs)

    def advance(self, context: UMMContext, phase: str, result: object) -> UMMStep:
        history = self.histories[context.execution_id]
        if phase == REASON:
            decision = interpret_reasoner(result)
            history.reasons_left -= 1
            outputs = (UMMOutput("text", decision.text),) if decision.text else ()
            if decision.kind == "final":
                return UMMStep(outputs=outputs)
            elif (
                history.options.get("action_mode") is not None
                and decision.generation["modality"] != "video"
            ):
                raise ValueError("Cosmos3 forward_dynamics requires a video decision")
            else:
                history.decision = decision
                self.remember(history, decision_message(decision))
                request = self.generation_request(history, decision)
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
            history.generations_left -= 1
            history.media_urls.append((len(history.messages), item["url"]))
            self.remember(history, receipt(decision, item["kind"]))
            history.decision = None
            return self.reason(context, history, (UMMOutput(item["kind"], item),))
        else:
            raise ValueError(f"Cosmos3 adapter received an unexpected phase {phase!r}")

    def usage(self, context: UMMContext) -> int:
        """Retained history bytes plus the most the remaining turns can add.

        The controller checks this against max_retained_bytes at start, where
        a breach is the client's input, so a request it admits can never
        exceed the bound through model output.
        """
        history = self.histories.get(context.execution_id)
        if history is None:
            return 0
        else:
            reason, generation = WORST_CASE_BYTES
            return (
                history.retained_bytes
                + history.reasons_left * reason
                + history.generations_left * generation
            )

    def close(self, context: UMMContext, reason: CloseReason) -> None:
        self.histories.pop(context.execution_id, None)

    @staticmethod
    def generation_request(history: Cosmos3History, decision: Decision) -> OmniRequest:
        options = deepcopy(history.options)
        options["prompt"] = decision.generation["prompt"]
        options["num_outputs_per_prompt"] = 1
        if decision.generation["modality"] == "image":
            options["num_frames"] = 1
        else:
            options.setdefault("num_frames", 33)
        return OmniRequest(
            {"prompt": decision.generation["prompt"]},
            {"diffusion": options, "stream": False},
        )
