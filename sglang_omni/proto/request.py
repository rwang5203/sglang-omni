# SPDX-License-Identifier: Apache-2.0
"""Request state and tracking."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RequestState(Enum):
    """State of a request in the pipeline."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class RequestInfo:
    """Tracking info for a request in the coordinator."""

    request_id: str
    state: RequestState = RequestState.PENDING
    current_stage: str | None = None
    terminal_stages: set[str] | None = None
    result: Any = None
    error: str | None = None


EXPLICIT_GENERATION_PARAMS_KEY = "explicit_generation_params"


@dataclass
class OmniRequest:
    """User-facing request with inputs and parameters."""

    inputs: Any
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "_type": "OmniRequest",
            "inputs": self.inputs,
            "params": self.params,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OmniRequest":
        return cls(
            inputs=data.get("inputs"),
            params=data.get("params", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class StagePayload:
    """Payload passed between stages with request context."""

    request_id: str
    request: OmniRequest
    data: Any
    # Runtime-owned routing state. Model stages may replace ``data`` but the
    # Stage shell carries these fields across hops. They bound explicitly
    # configured feedback graphs without trusting request metadata.
    route_transitions: int = 0
    route_trace: tuple[str, ...] = ()
    stream_positions: dict[str, dict[str, int]] = field(default_factory=dict)
    # Caller-visible ID. ``request_id`` may be a runtime execution ID used to
    # reject delayed traffic after the caller reuses an ID. Keep this after the
    # existing positional fields for constructor compatibility.
    public_request_id: str | None = None
    # Scheduler-local stream ingress state. These fields intentionally stay
    # out of to_dict(); they are rebuilt by the receiving scheduler and never
    # form part of the inter-stage wire contract.
    prefetched_chunks: list[Any] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
    prefetched_stream_done: bool = field(
        default=False, init=False, repr=False, compare=False
    )

    def to_dict(self) -> dict[str, Any]:
        serialized = {
            "_type": "StagePayload",
            "request_id": self.request_id,
            "request": self.request.to_dict(),
            "data": self.data,
        }
        if self.public_request_id is not None:
            serialized["public_request_id"] = self.public_request_id
        if (
            type(self.route_transitions) is not int
            or self.route_transitions != 0
            or self.route_trace != ()
            or self.stream_positions != {}
        ):
            serialized.update(
                {
                    "route_transitions": self.route_transitions,
                    "route_trace": list(self.route_trace),
                    "stream_positions": {
                        source: dict(targets)
                        for source, targets in self.stream_positions.items()
                    },
                }
            )
        return serialized

    def validated_route_state(
        self,
    ) -> tuple[int, tuple[str, ...], dict[str, dict[str, int]]]:
        transitions = self.route_transitions
        if isinstance(transitions, bool) or not isinstance(transitions, int):
            raise TypeError("route_transitions must be an integer")
        if transitions < 0:
            raise ValueError("route_transitions must be non-negative")
        trace = self.route_trace
        if not isinstance(trace, tuple) or not all(
            isinstance(stage, str) and stage for stage in trace
        ):
            raise TypeError("route_trace must contain non-empty stage names")
        if len(trace) > 16:
            raise ValueError("route_trace cannot contain more than 16 stage names")
        if trace and len(trace) != min(transitions + 1, 16):
            raise ValueError("route_trace is inconsistent with route_transitions")
        if not trace and transitions:
            raise ValueError("nonzero route_transitions requires a route_trace")

        if not isinstance(self.stream_positions, dict):
            raise TypeError("stream_positions must be an object")
        if len(self.stream_positions) > 128:
            raise ValueError("stream_positions cannot contain more than 128 sources")
        positions: dict[str, dict[str, int]] = {}
        entry_count = 0
        for source, raw_targets in self.stream_positions.items():
            if not isinstance(source, str) or not source:
                raise TypeError("stream_positions source names must be non-empty")
            if not isinstance(raw_targets, dict):
                raise TypeError("stream_positions targets must be an object")
            if not raw_targets:
                raise ValueError("stream_positions target maps cannot be empty")
            targets: dict[str, int] = {}
            for target, position in raw_targets.items():
                if not isinstance(target, str) or not target:
                    raise TypeError("stream_positions target names must be non-empty")
                if (
                    isinstance(position, bool)
                    or not isinstance(position, int)
                    or position < 0
                ):
                    raise TypeError(
                        "stream_positions values must be non-negative integers"
                    )
                targets[target] = position
                entry_count += 1
            positions[source] = targets
        if entry_count > 128:
            raise ValueError("stream_positions cannot contain more than 128 edges")
        return transitions, trace, positions

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StagePayload":
        request = data.get("request", {})
        if isinstance(request, dict) and request.get("_type") == "OmniRequest":
            request_obj = OmniRequest.from_dict(request)
        else:
            request_obj = OmniRequest.from_dict(request)
        raw_trace = data.get("route_trace", ())
        if not isinstance(raw_trace, (list, tuple)):
            raise TypeError("route_trace must contain non-empty stage names")
        public_request_id = data.get("public_request_id")
        if public_request_id is not None and (
            not isinstance(public_request_id, str) or not public_request_id
        ):
            raise TypeError("public_request_id must be a non-empty string or None")
        payload = cls(
            request_id=data.get("request_id", ""),
            request=request_obj,
            data=data.get("data"),
            public_request_id=public_request_id,
            route_transitions=data.get("route_transitions", 0),
            route_trace=tuple(raw_trace),
            stream_positions=data.get("stream_positions", {}),
        )
        _, _, payload.stream_positions = payload.validated_route_state()
        return payload
