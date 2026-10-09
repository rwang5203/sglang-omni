# SPDX-License-Identifier: Apache-2.0
"""Adapt Omni payloads to the native SGLang generation API."""

from __future__ import annotations

import logging
import multiprocessing
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

from sglang_omni.preprocessing.resource_connector import (
    MediaPolicyError,
    get_global_resource_connector,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

logger = logging.getLogger(__name__)

REQUEST_ERROR = "Native generation request"

# SamplingParams fields that native SGLang's image and video routes accept from
# clients (ImageGenerationsRequest and VideoGenerationsRequest in the
# multimodal_gen OpenAI protocol). Server paths such as output_path,
# prompt_path, perf_dump_path and model paths, profiling switches and raw
# diffusers kwargs stay with the server. Each model adds the fields it declares
# for those routes, see model_request_fields.
COMMON_REQUEST_FIELDS = frozenset(
    {
        "attention_backend_override",
        "cache_dit_params",
        "cfg_gate_step",
        "enable_cache_dit",
        "enable_frame_interpolation",
        "enable_teacache",
        "enable_upscaling",
        "flow_shift",
        "fps",
        "frame_interpolation_exp",
        "frame_interpolation_scale",
        "generator_device",
        "guidance_scale",
        "guidance_scale_2",
        "height",
        "image_path",
        "max_sequence_length",
        "negative_prompt",
        "num_frames",
        "num_inference_steps",
        "num_outputs_per_prompt",
        "output_compression",
        "output_quality",
        "progressive_delta",
        "progressive_levels",
        "progressive_mode",
        "prompt",
        "quality",
        "seed",
        "skip_softmax_params",
        "task_type",
        "true_cfg_scale",
        "upscaling_scale",
        "video_path",
        "width",
    }
)

# Native image and video routes clamp n to this many outputs per prompt.
# Native SGLang keeps the bound inline, so it is restated here.
NATIVE_MAX_OUTPUTS_PER_PROMPT = 10


def model_request_fields(server_args: object) -> frozenset[str]:
    """Fields the served model declares for native image and video requests."""
    from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
        get_sampling_request_extra_fields,
        resolve_sampling_params_cls,
    )

    sampling_params_cls = resolve_sampling_params_cls(server_args)
    return get_sampling_request_extra_fields(
        sampling_params_cls, "image"
    ) | get_sampling_request_extra_fields(sampling_params_cls, "video")


def resolve_generation_options(
    request_params: dict[str, Any], inputs: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Lower common SDK seeds and native overrides with one precedence rule."""
    params = {}
    if request_params.get("seed") is not None:
        params["seed"] = request_params["seed"]
    else:
        pass
    params.update(inputs or {})
    stage_sampling = request_params.get("stage_sampling")
    if stage_sampling is not None:
        if not isinstance(stage_sampling, dict):
            raise ValueError("stage_sampling must be a mapping")
        else:
            pass
        generation = stage_sampling.get("generation", {})
        if not isinstance(generation, dict):
            raise ValueError("Generation stage sampling must be a mapping")
        else:
            pass
        if generation.get("seed") is not None:
            params["seed"] = generation["seed"]
        else:
            pass
    else:
        pass
    overrides = request_params.get("diffusion", {})
    if not isinstance(overrides, dict):
        raise ValueError("diffusion parameters must be a mapping")
    else:
        pass
    params.update(overrides)
    stage_params = request_params.get("stage_params")
    if stage_params is not None:
        if not isinstance(stage_params, dict):
            raise ValueError("stage_params must be a mapping")
        else:
            pass
        generation = stage_params.get("generation", {})
        if not isinstance(generation, dict):
            raise ValueError("Generation stage parameters must be a mapping")
        else:
            pass
        params.update(generation)
    else:
        pass
    return params


def build_sampling_params(
    payload: StagePayload,
    output_dir: str,
    request_fields: frozenset[str] = COMMON_REQUEST_FIELDS,
) -> dict[str, Any]:
    """Lower a request to native sampling fields that clients may set."""
    inputs = payload.request.inputs
    if isinstance(inputs, str):
        inputs = {"prompt": inputs}
    elif not isinstance(inputs, dict):
        raise ValueError("Generation inputs must be a prompt or native sampling fields")
    else:
        pass
    params = resolve_generation_options(payload.request.params, inputs)
    refused = sorted(set(params) - request_fields)
    if refused:
        raise ValueError(
            f"{REQUEST_ERROR} cannot set {', '.join(refused)}. Clients may set "
            "the fields that the native image and video routes accept for this "
            "model, and the server owns paths it writes or reads on its own."
        )
    else:
        pass
    if not isinstance(params.get("prompt"), str) or not params["prompt"].strip():
        raise ValueError("Generation requires a nonempty prompt")
    else:
        pass
    if payload.request.params.get("stream"):
        raise ValueError("Use the native realtime video API for streaming generation")
    else:
        pass
    outputs = params.get("num_outputs_per_prompt", 1)
    if (
        not isinstance(outputs, int)
        or isinstance(outputs, bool)
        or not 1 <= outputs <= NATIVE_MAX_OUTPUTS_PER_PROMPT
    ):
        raise ValueError(
            f"{REQUEST_ERROR} num_outputs_per_prompt must be an integer from 1 "
            f"to {NATIVE_MAX_OUTPUTS_PER_PROMPT}, as on the native routes"
        )
    else:
        pass
    params.update(
        save_output=True,
        return_file_paths_only=True,
        output_path=output_dir,
        output_file_name=uuid4().hex,
    )
    return params


def police_media_inputs(
    params: dict[str, Any], request_fields: frozenset[str], directory: Path
) -> list[Path]:
    """Apply the server media policy to client media paths and URLs.

    Without a configured policy the native loaders read them as before. With
    one, local paths and file URLs must sit in the allowed directory, and
    http(s) URLs are fetched through the connector, which checks the domain on
    every redirect, into the request directory. Returns the fetched files.
    """
    connector = get_global_resource_connector()
    if (
        connector.allowed_local_media_path is None
        and not connector.allowed_media_domains
    ):
        return []
    else:
        pass
    fetched: list[Path] = []

    def police(name: str, source: object) -> str:
        if not isinstance(source, str):
            raise ValueError(f"{REQUEST_ERROR} {name} must be a path or URL")
        elif source.startswith("data:"):
            return source
        elif source.startswith(("http://", "https://")):
            data, _ = connector.load_http_bytes(
                source, timeout=connector.connection.timeout, max_bytes=None
            )
            target = (
                directory / f"input_{uuid4().hex}{Path(urlparse(source).path).suffix}"
            )
            target.write_bytes(data)
            fetched.append(target)
            return str(target)
        elif source.startswith("file://"):
            return str(connector.local_media_path(unquote(urlparse(source).path)))
        else:
            return str(connector.local_media_path(source))

    for name in sorted(params):
        if name not in request_fields or not name.endswith("_path"):
            continue
        else:
            pass
        value = params[name]
        try:
            if isinstance(value, list):
                params[name] = [police(name, item) for item in value]
            else:
                params[name] = police(name, value)
        except MediaPolicyError as exc:
            raise ValueError(f"{REQUEST_ERROR} {name} refused: {exc}") from exc
    return fetched


class NativeGenerationScheduler(SimpleScheduler):
    """Keep scheduler and pipeline execution in the native SGLang runtime."""

    def __init__(
        self,
        generator: object,
        output_dir: str,
        request_fields: frozenset[str] = COMMON_REQUEST_FIELDS,
    ):
        self.generator = generator
        self.output_dir = output_dir
        self.request_fields = request_fields
        self.native_requests: dict[str, GenerationRequest] = {}
        super().__init__(
            self.generate,
            abort_callback=self.cancel_native,
            shutdown_callback=generator.shutdown,
        )
        self.requires_tp_work_fanout = False

    def next_message(self):
        for process in self.generator.local_scheduler_process or []:
            if not process.is_alive():
                raise RuntimeError(
                    f"Native generation worker {process.pid} exited "
                    f"with code {process.exitcode}"
                )
            else:
                pass
        return super().next_message()

    @staticmethod
    def remove_request_directory(directory: Path) -> None:
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            pass

    def cancel_native(self, request_id: str) -> None:
        directory = None
        with self.abort_lock:
            request = self.native_requests.get(request_id)
            if request is not None:
                request.cancelled.set()
                if not request.running and not request.delivering:
                    self.native_requests.pop(request_id)
                    directory = request.directory
                else:
                    pass
            else:
                pass
        if directory is not None:
            self.remove_request_directory(directory)
        else:
            pass

    def emit_result(self, request_id, result, outbox) -> None:
        directory = None
        with self.abort_lock:
            request = self.native_requests.get(request_id)
            if request_id in self.aborted:
                self.aborted.discard(request_id)
                if request is not None and request.result is result:
                    self.native_requests.pop(request_id)
                    directory = request.directory
                else:
                    pass
            else:
                super().emit_result(request_id, result, outbox)
        if directory is not None:
            self.remove_request_directory(directory)
        else:
            pass

    def claim_result(self, result: StagePayload, *, terminal: bool) -> bool:
        if not terminal:
            raise ValueError(
                "Saved native media requires terminal delivery. "
                "Use inline media for inter-stage routing"
            )
        else:
            pass
        with self.abort_lock:
            request = self.native_requests.get(result.request_id)
            if request is None or request.result is not result:
                return False
            else:
                pass
            request.delivering = True
            return True

    def release_result(self, result: StagePayload, *, delivered: bool) -> None:
        directory = None
        with self.abort_lock:
            request = self.native_requests.get(result.request_id)
            if request is not None and request.result is result:
                self.native_requests.pop(result.request_id)
                if not delivered:
                    directory = request.directory
                else:
                    pass
            else:
                pass
        if directory is not None:
            self.remove_request_directory(directory)
        else:
            pass

    def stop(self) -> None:
        directories = []
        with self.abort_lock:
            for request_id, request in list(self.native_requests.items()):
                request.cancelled.set()
                if not request.running and not request.delivering:
                    self.native_requests.pop(request_id)
                    directories.append(request.directory)
                else:
                    pass
        try:
            super().stop()
        finally:
            for directory in directories:
                self.remove_request_directory(directory)

    def generate(self, payload: StagePayload) -> StagePayload:
        params = build_sampling_params(payload, self.output_dir, self.request_fields)
        with self.abort_lock:
            if payload.request_id in self.aborted:
                return payload
            else:
                pass
            request = GenerationRequest(
                threading.Event(),
                Path(tempfile.mkdtemp(prefix="request_", dir=self.output_dir)),
            )
            self.native_requests[payload.request_id] = request
        params["output_path"] = str(request.directory)
        kwargs = {"sampling_params_kwargs": params}
        if (
            getattr(self.generator, "supports_cancellation", False)
            and params.get("num_outputs_per_prompt", 1) == 1
        ):
            kwargs["cancellation_event"] = request.cancelled
        else:
            pass
        succeeded = False
        try:
            fetched = police_media_inputs(
                params, self.request_fields, request.directory
            )
            results = self.generator.generate(**kwargs)
            for path in fetched:
                path.unlink(missing_ok=True)
            if results is None:
                raise RuntimeError(
                    "Native generation failed without producing an output"
                )
            else:
                pass
            if not isinstance(results, list):
                results = [results]
            else:
                pass
            if not results or any(not result.output_file_path for result in results):
                raise RuntimeError("Native generation returned no saved media")
            else:
                pass
            payload.data = {
                "media": [
                    {
                        "path": result.output_file_path,
                        "size": result.size,
                        "prompt": result.prompt,
                        "generation_time": result.generation_time,
                        "peak_memory_mb": result.peak_memory_mb,
                        "metrics": result.metrics,
                    }
                    for result in results
                ],
                "finish_reason": "stop",
            }
            succeeded = True
            return payload
        finally:
            with self.abort_lock:
                request.running = False
                request.result = payload if succeeded else None
                cleanup = not succeeded or request.cancelled.is_set()
                if cleanup and self.native_requests.get(payload.request_id) is request:
                    self.native_requests.pop(payload.request_id)
                else:
                    pass
            if cleanup:
                self.remove_request_directory(request.directory)
            else:
                pass


@dataclass
class GenerationRequest:
    cancelled: threading.Event
    directory: Path
    running: bool = True
    delivering: bool = False
    result: StagePayload | None = None


def native_server_kwargs(
    model_path: str,
    gpu_id: int,
    overrides: dict[str, Any] | None,
    runtime_gpu_ids: list[int] | None = None,
) -> dict[str, Any]:
    kwargs = dict(overrides or {})
    if kwargs.get("scheduler_rpc_timeout") is not None:
        raise ValueError(
            "Native generation cannot use scheduler RPC deadlines before work settles"
        )
    else:
        pass
    devices = list(runtime_gpu_ids) if runtime_gpu_ids is not None else [gpu_id]
    if not devices or devices[0] != gpu_id or len(set(devices)) != len(devices):
        raise ValueError("Native GPU placement must start with the stage GPU")
    else:
        pass
    if any(gpu < 0 for gpu in devices):
        raise ValueError("Native GPU ids must be nonnegative")
    else:
        pass
    if kwargs.get("num_gpus", len(devices)) != len(devices):
        raise ValueError("Native GPU count must match Omni placement")
    else:
        pass
    if kwargs.get("nnodes", 1) != 1:
        raise ValueError("This stage owns GPUs on one host")
    else:
        pass
    if any(key in kwargs for key in ("gpu_ids", "base_gpu_id", "model_path")):
        raise ValueError("Set the generation model and GPUs through Omni placement")
    else:
        pass
    kwargs.update(model_path=model_path, num_gpus=len(devices))
    if runtime_gpu_ids is None:
        kwargs["base_gpu_id"] = gpu_id
    else:
        kwargs["gpu_ids"] = devices
    return kwargs


def create_generation_scheduler(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    output_dir: str = "outputs",
    runtime_gpu_ids: list[int] | None = None,
    server_args_overrides: dict[str, Any] | None = None,
) -> NativeGenerationScheduler:
    from sglang_omni.utils.device import resolve_concrete_device

    concrete_device = resolve_concrete_device(device, gpu_id)
    if concrete_device.index is None:
        raise ValueError("Native Cosmos3 execution requires an indexed accelerator")
    else:
        pass
    gpu_id = concrete_device.index
    if multiprocessing.current_process().daemon:
        raise RuntimeError("Native generation requires allow_child_processes=true")
    else:
        pass
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    # Only the cancellable native path watches worker liveness. Without it a
    # worker death leaves the request waiting forever, so refuse builds that
    # lack the capability. Configs whose instance reports False and requests
    # with more than one output per prompt still take the plain path.
    # note (Richard Wang): some supported configs report False, so check the class.
    if not hasattr(DiffGenerator, "supports_cancellation"):
        raise RuntimeError(
            "Cosmos3 generation requires SGLang cooperative generation "
            "cancellation (DiffGenerator.supports_cancellation), which the "
            "installed SGLang does not provide. Without it, an aborted request "
            "cannot stop its native generation, and a request in flight when "
            "the generation worker dies waits indefinitely. Install an SGLang "
            "build with this capability, then restart."
        )
    else:
        pass
    kwargs = native_server_kwargs(
        model_path, gpu_id, server_args_overrides, runtime_gpu_ids
    )
    output_dir = str(Path(output_dir).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    generator = DiffGenerator.from_server_args(ServerArgs.from_kwargs(**kwargs))
    if not generator.supports_cancellation:
        logger.warning(
            "Cooperative generation cancellation is unavailable for this native "
            "configuration, for example with more than one native GPU. An "
            "aborted request keeps the generation stage busy until its native "
            "generation finishes, and a request in flight when the generation "
            "worker dies waits indefinitely."
        )
    else:
        pass
    try:
        return NativeGenerationScheduler(
            generator,
            output_dir,
            COMMON_REQUEST_FIELDS | model_request_fields(generator.server_args),
        )
    except BaseException:
        generator.shutdown()
        raise
