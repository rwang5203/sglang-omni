# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 decisions and native media re-ingestion for the shared UMM loop."""

from __future__ import annotations

import base64
import binascii
import hashlib
import itertools
import json
import math
import re
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field

from sglang_omni.admission import QueueFullError
from sglang_omni.models.cosmos3.stages import (
    COMMON_REQUEST_FIELDS,
    DEADLINE_METADATA_KEY,
    INLINE_MEDIA_LIMIT_BYTES,
    check_media_source,
    data_url,
    media_policy_configured,
    resolve_generation_options,
)
from sglang_omni.pipeline.umm import (
    CloseReason,
    MonotonicClock,
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
    StagePayload,
)
from sglang_omni.scheduling.native_chat_scheduler import build_chat_fields

# Adapter phase names, mapped to stages by the orchestrator's factory.phases.
REASON = "reason"
GENERATE = "generate"
# The stage name clients use for Reasoner overrides in stage_sampling and
# stage_params.
REASONER_STAGE = "reasoner"

# Server limits on the decision text and the generation prompt. The retained
# byte reservation plans for them, so longer model output fails the request
# as a server error. The schema leaves free text strings without pattern or
# length keywords, because native grammar decoding then admits no JSON escape
# in them, which would cut answers at their first newline or quote.
MAX_TEXT_CHARS = 8192
MAX_PROMPT_CHARS = 4096

# Native generation refuses a prompt without a character outside ASCII
# whitespace, so such a prompt returns to the Reasoner as a receipt.
NONBLANK = re.compile(r"\S", re.ASCII)

FINAL_DECISION = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["final"]},
        "text": {"type": "string"},
        "generation": {"type": "null"},
    },
    "required": ["kind", "text", "generation"],
    "additionalProperties": False,
}

GENERATE_DECISION = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["generate"]},
        "text": {"type": "string"},
        "generation": {
            "type": "object",
            "properties": {
                "modality": {"type": "string", "enum": ["image", "video"]},
                "prompt": {"type": "string"},
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

# The fields native Cosmos3 declares for its image and video routes, which
# the generation stage adds to COMMON_REQUEST_FIELDS through native SGLang.
# The orchestrator does not import native SGLang, so they are listed here,
# and a unit test pins them to the native declaration.
COSMOS3_REQUEST_FIELDS = frozenset(
    {
        "action",
        "action_fps",
        "action_mode",
        "action_normalization",
        "action_view_point",
        "condition_frame_indexes",
        "condition_video_keep",
        "control_guidance",
        "control_guidance_interval",
        "control_hint",
        "control_path",
        "domain_id",
        "domain_name",
        "guidance_interval",
        "max_frames",
        "num_conditional_frames",
        "num_first_chunk_conditional_frames",
        "num_video_frames_per_chunk",
        "raw_action_dim",
        "share_vision_temporal_positions",
        "show_control_condition",
        "show_input",
        "sound_duration",
        "use_duration_template",
        "use_guardrails",
        "use_resolution_template",
        "use_system_prompt",
    }
)


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
    dispatched_at: float = 0.0
    # The deadline and the admission index, which order generation turns.
    order: tuple[float, int] = (math.inf, 0)


class GenerationBacklogError(QueueFullError):
    """A request that admitted generation work cannot finish in time (HTTP 503)."""

    def __init__(self, detail: str) -> None:
        RuntimeError.__init__(self, f"{QueueFullError.MESSAGE} {detail}")


class RecentMaximum:
    """The largest value recorded within a trailing time window.

    Keeps only values that no later and larger value dominates, so each
    record and read takes amortized constant time.
    """

    def __init__(self) -> None:
        self.entries: deque[tuple[float, float]] = deque()

    def record(self, now: float, value: object) -> None:
        if type(value) not in (int, float) or not 0 <= value < math.inf:
            return
        else:
            pass
        while self.entries and self.entries[-1][1] <= value:
            self.entries.pop()
        self.entries.append((now, value))

    def maximum(self, now: float, window_s: float) -> float | None:
        while self.entries and self.entries[0][0] <= now - window_s:
            self.entries.popleft()
        return self.entries[0][1] if self.entries else None


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
    if not isinstance(text, str):
        raise ValueError("Cosmos3 decision text must be a string")
    elif len(text) > MAX_TEXT_CHARS:
        raise ValueError(
            f"Cosmos3 decision text exceeds the server limit of {MAX_TEXT_CHARS} "
            "characters"
        )
    elif kind == "final":
        if generation is not None:
            raise ValueError("Final Cosmos3 decisions carry no generation")
        else:
            return Decision(kind, text, None)
    elif kind != "generate":
        raise ValueError("Unknown Cosmos3 decision kind")
    elif not isinstance(generation, dict) or set(generation) != {"modality", "prompt"}:
        raise ValueError("Cosmos3 generation requires only modality and prompt")
    elif generation["modality"] not in MEDIA_TYPES:
        raise ValueError("Cosmos3 interleaving currently supports images and videos")
    elif not isinstance(generation["prompt"], str):
        raise ValueError("Cosmos3 generation prompt must be a string")
    elif len(generation["prompt"]) > MAX_PROMPT_CHARS:
        raise ValueError(
            "Cosmos3 generation prompt exceeds the server limit of "
            f"{MAX_PROMPT_CHARS} characters"
        )
    else:
        return Decision(
            kind,
            text,
            {"modality": generation["modality"], "prompt": generation["prompt"]},
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
    refused = sorted(set(options) - COMMON_REQUEST_FIELDS - COSMOS3_REQUEST_FIELDS)
    action_mode = options.get("action_mode")
    frames = options.get("num_frames")
    if refused:
        # The generation stage would refuse them at the first generation.
        raise UMMRequestError(
            f"Cosmos3 generation cannot set {', '.join(refused)}. Clients may "
            "set the fields that the native image and video routes accept, and "
            "the server owns paths it writes or reads on its own"
        )
    elif action_mode is not None and (
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


def check_media_policy(options: dict[str, object]) -> None:
    """Refuse generation media paths the server media policy refuses, before
    any turn.

    The paths are checked without being read, and the generation stage
    applies the same policy when it reads them. The Reasoner stage applies
    the policy to chat media before native SGLang loads any.
    """
    connector = get_global_resource_connector()
    if not media_policy_configured(connector):
        return
    else:
        pass
    sources: list[tuple[str, object]] = []
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


# The Reasoner's next turn learns how its generation request ended. Only a
# completed generation attaches media.
RECEIPTS = {
    "completed": (
        "Generation execution completed",
        "The attached media is the result of your preceding generation "
        "request. Inspect the received results against the original user's "
        "task. If the task is complete, choose kind final with generation "
        "null and your answer in text. Otherwise request the next needed "
        "generation. Repeat a completed generation only when the user's "
        "task or your assessment of its result calls for a revision.",
    ),
    "not_executed": (
        "Generation was not executed because its prompt was blank",
        "No media is attached. Request the generation again with a precise "
        "nonblank prompt, or choose kind final with generation null and your "
        "answer in text.",
    ),
    "no_media": (
        "Generation finished without media",
        "Native generation returned no media for this prompt, for example "
        "because a safety guardrail blocked it. Explain this to the user with "
        "kind final and generation null, or request a revised generation.",
    ),
}


def receipt(decision: Decision, status: str) -> dict[str, object]:
    """The user turn that returns a generation outcome. A completed one
    attaches its media with the URL left empty."""
    summary, guidance = RECEIPTS[status]
    kind = decision.generation["modality"]
    record = json.dumps(
        {
            "status": status,
            "modality": kind,
            "requested_prompt": decision.generation["prompt"],
        },
        ensure_ascii=False,
    )
    content: list[dict[str, object]] = [
        {"type": "text", "text": f"{summary}. Result receipt: {record}\n{guidance}"}
    ]
    if status == "completed":
        part = f"{kind}_url"
        content.append({"type": part, part: {"url": ""}})
    else:
        pass
    return {"role": "user", "content": content}


def worst_case_bytes() -> tuple[int, int]:
    """Most bytes one Reasoner turn and one generation turn can add, for
    answers and prompts within MAX_TEXT_CHARS and MAX_PROMPT_CHARS.

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
    return reason, media + max(encoded_size(receipt(worst, s)) for s in RECEIPTS)


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

    def __init__(
        self, model_path: str, *, clock: MonotonicClock = time.monotonic
    ) -> None:
        """clock must be the controller's clock, which sets each deadline."""
        del model_path
        self.histories: dict[str, Cosmos3History] = {}
        self.clock = clock
        # Measured within the last timeout_s, the deadline every request has.
        self.window_s = math.inf
        self.generation_times = RecentMaximum()
        self.reason_times = RecentMaximum()
        self.admissions = itertools.count()

    def admit(self, context: UMMContext, now: float, generations: int) -> None:
        """Refuse a request whose planned generations cannot finish in time.

        The generation stage serves one request at a time, earliest deadline
        first, so this request waits at most for the generations every active
        request may still run, plus one younger request's running generation
        per own generation. Each takes the longest service time measured
        recently, and each own Reasoner turn the longest recent Reasoner turn.
        Before any generation was measured the request is admitted, and the
        deadline order keeps overload with the newest requests, which
        generation_fits stops from generating more than time allows.
        """
        service = self.generation_times.maximum(now, self.window_s)
        if service is None or generations <= 0:
            return
        else:
            pass
        reasoning = self.reason_times.maximum(now, self.window_s) or 0.0
        admitted = sum(h.generations_left for h in tuple(self.histories.values()))
        finish = (
            now + service * (admitted + 2 * generations) + reasoning * (generations + 1)
        )
        if finish > context.deadline:
            raise GenerationBacklogError(
                f"Admitted Cosmos3 work needs about {math.ceil(finish - now)} s "
                f"for this request at {service:.1f} s per generation, beyond "
                "the request deadline. Retry later"
            )
        else:
            pass

    def generation_fits(
        self, context: UMMContext, history: Cosmos3History, now: float
    ) -> bool:
        """Whether one more generation and the turn that inspects it can end
        before the deadline, behind the generations older requests may run.

        Applies the admission estimate turn by turn, so a request that an
        estimate measured later shows short of time answers with what it has.
        """
        service = self.generation_times.maximum(now, self.window_s)
        if service is None:
            return True
        else:
            pass
        reasoning = self.reason_times.maximum(now, self.window_s) or 0.0
        older = sum(
            other.generations_left
            for other in tuple(self.histories.values())
            if other.order < history.order
        )
        return now + 2 * reasoning + service * (older + 2) <= context.deadline

    def start(self, context: UMMContext) -> UMMStep:
        try:
            # The chat history in native form, as the Reasoner stage builds it.
            request = OmniRequest(context.request.inputs, {}, context.request.metadata)
            payload = StagePayload(context.execution_id, request, None)
            messages = build_chat_fields(payload, "", set(), REASONER_STAGE)["messages"]
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
        check_media_policy(options)
        generations = (context.remaining_dispatches - 1) // 2
        now = self.clock()
        self.window_s = context.deadline - now
        self.admit(context, now, generations)
        history = Cosmos3History(
            reasons_left=generations + 1,
            generations_left=generations,
            sampling=decision_sampling(context.request),
            options=options,
            order=(context.deadline, next(self.admissions)),
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
        # Every earlier generate decision used one of the planned generations,
        # and the generation must also end in time.
        if (
            context.remaining_dispatches < 3
            or history.generations_left <= 0
            or not self.generation_fits(context, history, self.clock())
        ):
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
        # Every sampling field here is intended, so the Reasoner stage keeps the
        # ones equal to its placeholders too.
        metadata = {EXPLICIT_GENERATION_PARAMS_KEY: sorted(history.sampling)}
        request = OmniRequest({"messages": messages}, params, metadata)
        history.dispatched_at = self.clock()
        return UMMStep(REASON, request, outputs)

    def advance(self, context: UMMContext, phase: str, result: object) -> UMMStep:
        history = self.histories[context.execution_id]
        now = self.clock()
        if phase == REASON:
            self.reason_times.record(now, now - history.dispatched_at)
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
            elif NONBLANK.search(decision.generation["prompt"]) is None:
                # Native generation refuses a blank prompt, so the Reasoner
                # learns that nothing ran and decides again.
                history.generations_left -= 1
                self.remember(history, decision_message(decision))
                self.remember(history, receipt(decision, "not_executed"))
                return self.reason(context, history, outputs)
            else:
                history.decision = decision
                self.remember(history, decision_message(decision))
                request = self.generation_request(history, decision)
                return UMMStep(GENERATE, request, outputs)
        elif phase == GENERATE and history.decision is not None:
            decision = history.decision
            if isinstance(result, dict):
                self.generation_times.record(now, result.get("service_s"))
            else:
                pass
            if isinstance(result, dict) and result.get("media") == []:
                # A generation native refused, such as a guardrail block, goes
                # back to the Reasoner, which can explain it or revise.
                history.generations_left -= 1
                self.remember(history, receipt(decision, "no_media"))
                history.decision = None
                return self.reason(context, history, ())
            else:
                pass
            item = generated_media(result)
            if item["kind"] != decision.generation["modality"]:
                raise ValueError(
                    "Generated Cosmos3 modality differs from the model decision"
                )
            else:
                pass
            history.generations_left -= 1
            history.media_urls.append((len(history.messages), item["url"]))
            self.remember(history, receipt(decision, "completed"))
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
            # The generation stage serves the oldest request first.
            {DEADLINE_METADATA_KEY: list(history.order)},
        )
