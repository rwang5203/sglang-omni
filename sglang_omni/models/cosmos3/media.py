# SPDX-License-Identifier: Apache-2.0
"""Serve the native media routes of the Cosmos3 generation stage."""

from functools import partial

from sglang_omni.models.cosmos3.stages import native_server_kwargs
from sglang_omni.native_generation import media

prepare_native_media_app = partial(
    media.prepare_native_media_app,
    stage_name="generation",
    server_kwargs=native_server_kwargs,
)
