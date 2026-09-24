# SPDX-License-Identifier: Apache-2.0
"""Talker helpers run on CPU through the real ModelRunner constructor."""

from types import SimpleNamespace

import torch

from sglang_omni.model_runner import base as model_runner_base
from sglang_omni.models.nemotron_voicechat import talker_model_runner
from sglang_omni.models.nemotron_voicechat.talker_model_runner import (
    NemotronVoiceChatTalkerModelRunner,
)

PAD_ID = 0
EOS_ID = 1
CODEBOOK_SIZE = 16


class _FakeTokenizer:
    pad_token_id = PAD_ID
    eos_token_id = EOS_ID
    tokens = {PAD_ID: "<pad>", EOS_ID: "<eos>", 2: "x", 3: "y", 4: "xy"}

    def get_vocab(self):
        return {token: index for index, token in self.tokens.items()}

    def convert_ids_to_tokens(self, token_id):
        return self.tokens.get(token_id)


class _FakeTalker:
    num_quantizers = 3
    bos_emb = torch.zeros(4)

    def embed_codes(self, codes):
        return torch.zeros(codes.shape[0], 4)

    def embed_subword(self, ids, chars, lengths, mask=None):
        self.subword_mask = mask
        return torch.zeros(ids.shape[0], 4)

    def gated_fusion_audio_text(self, audio, text):
        return audio + text


def _runner(monkeypatch) -> NemotronVoiceChatTalkerModelRunner:
    monkeypatch.setattr(
        talker_model_runner,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda name, **kwargs: _FakeTokenizer()),
    )
    # note (Richard Wang): current_platform.get_device raises without a visible GPU.
    monkeypatch.setattr(
        model_runner_base,
        "current_platform",
        SimpleNamespace(get_device=lambda gpu_id: torch.device("cpu")),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            nemotron_speech={
                "tokenizer_name": "fake",
                "tts_config": {"exponent": 1.0},
                "inference_top_p_or_k": 0.9,
                "inference_noise_scale": 0.0,
                "inference_force_speech_silence_on_eos": True,
                "codec_config": {"codebook_size": CODEBOOK_SIZE},
            }
        ),
        fusion_buffer=torch.zeros(2, 4),
        talker=_FakeTalker(),
        audio_prompt_latent=torch.zeros(3, 4),
        codec_silence_tokens=torch.zeros(3, dtype=torch.long),
    )
    tp_worker = SimpleNamespace(gpu_id=0, model_runner=SimpleNamespace(model=model))
    return NemotronVoiceChatTalkerModelRunner(tp_worker, output_processor=None)


def test_talker_helpers_run_through_the_real_constructor(monkeypatch):
    runner = _runner(monkeypatch)

    ids, chars, lengths = runner.char_batch([4, PAD_ID])
    assert ids.tolist() == [4, PAD_ID]
    assert chars.tolist() == [[0, 1], [2, 2]]
    assert lengths.tolist() == [2, 1]

    assert runner.pad_codes().tolist() == [[CODEBOOK_SIZE] * 3]

    rows = runner.warmup()
    assert rows.shape == (3, 4)
    assert runner.model.talker.subword_mask.tolist() == [False, True, True]
