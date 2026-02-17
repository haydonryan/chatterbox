# Modified from CosyVoice https://github.com/FunAudioLLM/CosyVoice
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
from typing import Optional

from ..s3tokenizer import SPEECH_VOCAB_SIZE
from .const import S3GEN_SR

# NOTE: Python S3Gen inference internals are disabled.
# Rust implements flow + vocoder now; this module is kept only for embed_ref/tokenizer.


def drop_invalid_tokens(x):
    assert len(x.shape) <= 2 and x.shape[0] == 1, "only batch size of one allowed for now"
    return x[x < SPEECH_VOCAB_SIZE]


class S3Token2Mel(torch.nn.Module):
    """
    S3Gen's CFM decoder maps S3 speech tokens to mel-spectrograms.

    TODO: make these modules configurable?
    """
    def __init__(self, meanflow=False):
        super().__init__()
        # Python S3Gen internals are disabled; Rust handles inference.
        self.tokenizer = None
        self.mel_extractor = None
        self.speaker_encoder = None
        self.meanflow = meanflow
        # Python flow/vocoder modules are intentionally omitted.
        self.flow = None

        self.resamplers = {}

    @property
    def device(self):
        for module in (self.tokenizer, self.speaker_encoder):
            if module is None:
                continue
            params = module.parameters()
            try:
                return next(params).device
            except StopIteration:
                continue
        return torch.device("cpu")

    @property
    def dtype(self):
        for module in (self.tokenizer, self.speaker_encoder):
            if module is None:
                continue
            params = module.parameters()
            try:
                return next(params).dtype
            except StopIteration:
                continue
        return torch.float32

    def embed_ref(
        self,
        ref_wav: torch.Tensor,
        ref_sr: int,
        device="auto",
        ref_fade_out=True,
    ):
        raise RuntimeError("Python S3Gen embed_ref is disabled; use Rust S3Gen pipeline.")

    def forward(
        self,
        speech_tokens: torch.LongTensor,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor],
        ref_sr: Optional[int],
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        n_cfm_timesteps = None,
        finalize: bool = False,
        speech_token_lens=None,
        noised_mels=None,
    ):
        raise RuntimeError("Python S3Gen inference is disabled; use Rust S3Gen pipeline.")


class S3Token2Wav(S3Token2Mel):
    """
    The decoder of S3Gen is a concat of token-to-mel (CFM) and a mel-to-waveform (HiFiGAN) modules.

    TODO: make these modules configurable?
    """

    ignore_state_dict_missing = ("tokenizer._mel_filters", "tokenizer.window")

    def __init__(self, meanflow=False):
        super().__init__(meanflow)
        # Python vocoder is disabled; Rust handles mel->wav.
        self.mel2wav = None

    def forward(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor],
        ref_sr: Optional[int],
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        finalize: bool = False,
        speech_token_lens=None,
        skip_vocoder=False,
        n_cfm_timesteps=None,
        noised_mels=None,

    ):
        raise RuntimeError("Python S3Gen inference is disabled; use Rust S3Gen pipeline.")

    @torch.inference_mode()
    def flow_inference(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: Optional[int] = None,
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        n_cfm_timesteps = None,
        finalize: bool = False,
        speech_token_lens=None,
    ):
        raise RuntimeError("Python S3Gen inference is disabled; use Rust S3Gen pipeline.")

    @torch.inference_mode()
    def hift_inference(self, speech_feat, cache_source: torch.Tensor = None):
        raise RuntimeError("Python S3Gen inference is disabled; use Rust S3Gen pipeline.")

    @torch.inference_mode()
    def inference(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: Optional[int] = None,
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        # left as a kwarg because this can change input/output size ratio
        drop_invalid_tokens=True,
        n_cfm_timesteps=None,
        speech_token_lens=None,
    ):
        raise RuntimeError("Python S3Gen inference is disabled; use Rust S3Gen pipeline.")
