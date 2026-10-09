# SPDX-License-Identifier: Apache-2.0
"""Refuse a native frontend that cannot observe its scheduler owner."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from sglang_omni.native_generation.media import build_native_media_app


@pytest.mark.parametrize("supported", [False, True])
def test_native_owner_failure_contract_is_required(monkeypatch, supported):
    from sglang.multimodal_gen.runtime import scheduler_client, server_args
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    def legacy_initialize(self, server_args):
        pass

    def supported_initialize(self, server_args, *, worker_failure=None):
        pass

    monkeypatch.setattr(server_args, "set_global_server_args", lambda args: None)
    built = []
    pipeline = SimpleNamespace(supports_action_endpoint=lambda: False)
    monkeypatch.setattr(
        scheduler_client.AsyncSchedulerClient,
        "initialize",
        supported_initialize if supported else legacy_initialize,
    )
    monkeypatch.setattr(
        ServerArgs,
        "from_kwargs",
        lambda **kwargs: built.append(kwargs)
        or SimpleNamespace(pipeline_config=pipeline),
    )
    if supported:
        assert isinstance(build_native_media_app({}), FastAPI)
        assert len(built) == 1
    else:
        with pytest.raises(RuntimeError, match="worker_failure"):
            build_native_media_app({})
        assert built == []
