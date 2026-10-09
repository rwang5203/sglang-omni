# SPDX-License-Identifier: Apache-2.0
"""Serve the native media routes of the Cosmos3 generation stage."""

from functools import partial

from sglang_omni.native_generation import media


def request_fields(server_args: object) -> frozenset[str]:
    """The same client settable fields as the generation stage."""
    # Imported when the routes start, which keeps the stage module out of
    # the HTTP process of a deployment that never calls them.
    from sglang_omni.models.cosmos3.stages import (
        COMMON_REQUEST_FIELDS,
        model_request_fields,
    )

    return COMMON_REQUEST_FIELDS | model_request_fields(server_args)


prepare_native_media_app = partial(
    media.prepare_native_media_app,
    stage_name="generation",
    request_fields=request_fields,
)
