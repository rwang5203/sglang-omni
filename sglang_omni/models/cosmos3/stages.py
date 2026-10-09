# SPDX-License-Identifier: Apache-2.0
"""Adapt Omni payloads to the native SGLang generation API."""

from __future__ import annotations

import logging
import multiprocessing
import multiprocessing.connection
import os
import queue
import shutil
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
from uuid import uuid4

from sglang_omni.native_generation import cancellation
from sglang_omni.preprocessing.resource_connector import (
    MediaPolicyError,
    get_global_resource_connector,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
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
    request_params: dict[str, object], inputs: dict[str, object] | None = None
) -> dict[str, object]:
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
) -> dict[str, object]:
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
    params: dict[str, object], request_fields: frozenset[str], directory: Path
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
    """Keep scheduler and pipeline execution in the native SGLang runtime.

    Each request saves its media in its own directory under output_dir.
    Delivered media belongs to the receiver of the completion, and the stage
    never removes it. The stage removes a request directory when the request
    fails, is aborted, or is dropped before the coordinator accepts its
    completion. An abort observed during or after delivery also removes it,
    because the coordinator aborts only requests it has not completed, so it
    drops that completion.
    """

    # The same retention the scheduler keeps for abort tombstones, which
    # outlasts any completion the coordinator has yet to read.
    retained_deliveries = 10000

    def __init__(
        self,
        generator: object,
        output_dir: str,
        request_fields: frozenset[str] = COMMON_REQUEST_FIELDS,
        *,
        cancellation_mode: str | None = None,
    ):
        """``cancellation_mode`` is "native" when the native API cancels,
        "plugin" when the Omni plugin runs in the native worker, else None."""
        self.generator = generator
        self.output_dir = output_dir
        self.request_fields = request_fields
        self.cancellation_mode = cancellation_mode
        self.native_requests: dict[str, GenerationRequest] = {}
        self.delivered: OrderedDict[str, Path] = OrderedDict()
        self.workers = list(generator.local_scheduler_process or [])
        self.worker_failure: RuntimeError | None = None
        self.closing = False
        super().__init__(
            self.generate,
            abort_callback=self.cancel_native,
            shutdown_callback=generator.shutdown,
        )
        self.requires_tp_work_fanout = False
        if self.workers:
            threading.Thread(
                target=self.watch_workers, name="native-worker-watch", daemon=True
            ).start()
        else:
            pass

    @staticmethod
    def worker_exit(process: multiprocessing.process.BaseProcess) -> RuntimeError:
        return RuntimeError(
            f"Native generation worker {process.pid} exited "
            f"with code {process.exitcode}"
        )

    def watch_workers(self) -> None:
        """Record a native worker exit while idle, without polling."""
        sentinels = {process.sentinel: process for process in self.workers}
        ready = multiprocessing.connection.wait(list(sentinels))
        if not self.closing and self.worker_failure is None:
            self.worker_failure = self.worker_exit(sentinels[ready[0]])
            logger.error("%s, stopping the generation stage", self.worker_failure)
        else:
            pass

    def next_message(self) -> IncomingMessage | None:
        if self.worker_failure is not None:
            raise self.worker_failure
        else:
            pass
        return super().next_message()

    def run_native(self, kwargs: dict[str, object]) -> object:
        """Run native generation on a helper thread and wait for it or a worker exit.

        Native generation blocks until its scheduler answers, which never
        happens once a worker died, so waiting on the worker sentinels fails the
        request at once. The stage then stops on its next scheduler iteration.
        """
        reader, writer = multiprocessing.Pipe(duplex=False)
        outcome: dict[str, object] = {}

        def run() -> None:
            try:
                outcome["result"] = self.generator.generate(**kwargs)
            except BaseException as exc:
                outcome["error"] = exc
            finally:
                writer.close()

        threading.Thread(target=run, name="native-generation", daemon=True).start()
        sentinels = {process.sentinel: process for process in self.workers}
        try:
            ready = multiprocessing.connection.wait([reader, *sentinels])
        finally:
            reader.close()
        if reader not in ready:
            failure = self.worker_exit(sentinels[ready[0]])
            if self.worker_failure is None:
                self.worker_failure = failure
            else:
                pass
            raise failure
        elif "error" in outcome:
            raise outcome["error"]
        else:
            return outcome.get("result")

    def signal_native_cancellation(self, directory: Path) -> None:
        try:
            (directory / cancellation.CANCELLED_MARKER).touch()
        except FileNotFoundError:
            pass

    @staticmethod
    def remove_request_directory(directory: Path) -> None:
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            pass

    def cancel_native(self, request_id: str) -> None:
        directory = None
        signal = None
        with self.abort_lock:
            request = self.native_requests.get(request_id)
            if request is not None:
                request.aborted = True
                request.cancelled.set()
                if not request.running and not request.delivering:
                    self.native_requests.pop(request_id)
                    directory = request.directory
                elif request.running and self.cancellation_mode == "plugin":
                    signal = request.directory
                else:
                    pass
            else:
                directory = self.delivered.pop(request_id, None)
        if signal is not None:
            self.signal_native_cancellation(signal)
        else:
            pass
        if directory is not None:
            self.remove_request_directory(directory)
        else:
            pass

    def emit_result(
        self,
        request_id: str,
        result: object,
        outbox: queue.Queue[OutgoingMessage],
    ) -> None:
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

    def claim_result(self, result: object, *, terminal: bool) -> bool:
        if not isinstance(result, StagePayload):
            return False
        elif not terminal:
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

    def release_result(self, result: object, *, delivered: bool) -> None:
        if not isinstance(result, StagePayload):
            return
        else:
            pass
        directory = None
        with self.abort_lock:
            request = self.native_requests.get(result.request_id)
            if request is not None and request.result is result:
                self.native_requests.pop(result.request_id)
                if delivered and not request.aborted:
                    self.delivered[result.request_id] = request.directory
                    while len(self.delivered) > self.retained_deliveries:
                        self.delivered.popitem(last=False)
                else:
                    directory = request.directory
            else:
                pass
        if directory is not None:
            self.remove_request_directory(directory)
        else:
            pass

    def stop(self) -> None:
        directories = []
        signals = []
        self.closing = True
        with self.abort_lock:
            for request_id, request in list(self.native_requests.items()):
                request.cancelled.set()
                if not request.running and not request.delivering:
                    self.native_requests.pop(request_id)
                    directories.append(request.directory)
                elif request.running and self.cancellation_mode == "plugin":
                    signals.append(request.directory)
                else:
                    pass
        for directory in signals:
            self.signal_native_cancellation(directory)
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
            # A reused ID means the coordinator completed its earlier request.
            self.delivered.pop(payload.request_id, None)
            self.native_requests[payload.request_id] = request
        params["output_path"] = str(request.directory)
        kwargs = {"sampling_params_kwargs": params}
        markers = []
        if (
            self.cancellation_mode == "native"
            and params.get("num_outputs_per_prompt", 1) == 1
        ):
            kwargs["cancellation_event"] = request.cancelled
        elif self.cancellation_mode == "plugin":
            markers = [
                request.directory / cancellation.ARMED_MARKER,
                request.directory / cancellation.CANCELLED_MARKER,
            ]
        else:
            pass
        succeeded = False
        try:
            if markers:
                markers[0].touch()
            else:
                pass
            fetched = police_media_inputs(
                params, self.request_fields, request.directory
            )
            results = self.run_native(kwargs)
            for path in [*fetched, *markers]:
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
    aborted: bool = False
    result: StagePayload | None = None


def native_server_kwargs(
    model_path: str,
    gpu_id: int,
    overrides: dict[str, object] | None,
    runtime_gpu_ids: list[int] | None = None,
) -> dict[str, object]:
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
    server_args_overrides: dict[str, object] | None = None,
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

    kwargs = native_server_kwargs(
        model_path, gpu_id, server_args_overrides, runtime_gpu_ids
    )
    output_dir = str(Path(output_dir).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # A later SGLang that cancels natively takes precedence over the plugin.
    native_api = hasattr(DiffGenerator, "supports_cancellation")
    handshake = (
        tempfile.mkdtemp(prefix="sglang-omni-native-")
        if not native_api and kwargs["num_gpus"] == 1
        else None
    )
    try:
        with cancellation.activation_environment(handshake):
            generator = DiffGenerator.from_server_args(ServerArgs.from_kwargs(**kwargs))
        if native_api:
            mode = "native" if generator.supports_cancellation else None
        elif handshake is not None and os.path.exists(
            cancellation.handshake_path(handshake, 0)
        ):
            mode = "plugin"
        else:
            mode = None
    finally:
        if handshake is not None:
            shutil.rmtree(handshake, ignore_errors=True)
        else:
            pass
    if mode is None:
        logger.warning(
            "%s An aborted request keeps the generation stage busy until its "
            "native generation finishes.",
            degraded_cancellation_reason(native_api, kwargs["num_gpus"]),
        )
    else:
        pass
    try:
        return NativeGenerationScheduler(
            generator,
            output_dir,
            COMMON_REQUEST_FIELDS | model_request_fields(generator.server_args),
            cancellation_mode=mode,
        )
    except BaseException:
        generator.shutdown()
        raise


def degraded_cancellation_reason(native_api: bool, num_gpus: object) -> str:
    if native_api:
        return "This native configuration reports no cooperative cancellation."
    elif num_gpus != 1:
        return (
            "Cooperative generation cancellation covers one native GPU, and "
            f"this stage runs {num_gpus}."
        )
    else:
        return (
            "Cooperative generation cancellation did not start in the native "
            "worker. Install sglang-omni as a package so native SGLang finds "
            "its sglang.multimodal_gen.plugins entry point, and list "
            f"{cancellation.PLUGIN_NAME} in SGLANG_PLUGINS when that variable is set."
        )
