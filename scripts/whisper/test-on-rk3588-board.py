#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
This file test the exported whisper tiny.en rknn model with rknnlite.
"""

try:
    from rknnlite.api import RKNNLite
except:
    print("Please run this file on your board (linux + aarch64 + npu)")
    print("You need to install rknn_toolkit_lite2")
    print(
        " from https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2/packages"
    )
    print(
        "https://github.com/airockchip/rknn-toolkit2/blob/v2.1.0/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.1.0-cp310-cp310-linux_aarch64.whl"
    )
    print("is known to work")
    raise

import base64
import time
from pathlib import Path
from typing import List, Tuple

import kaldi_native_fbank as knf
import numpy as np
import soundfile as sf
import torch


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel

    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def compute_features(samples: np.ndarray, dim: int = 80) -> np.ndarray:
    """
    Returns:
      Return a 1-D float32 tensor of shape (1, 80, 3000) containing the features.
    """
    features = []
    opts = knf.WhisperFeatureOptions()
    opts.dim = dim
    online_whisper_fbank = knf.OnlineWhisperFbank(opts)
    online_whisper_fbank.accept_waveform(16000, samples)
    online_whisper_fbank.input_finished()
    for i in range(online_whisper_fbank.num_frames_ready):
        f = online_whisper_fbank.get_frame(i)
        f = torch.from_numpy(f)
        features.append(f)

    features = torch.stack(features)

    log_spec = torch.clamp(features, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    mel = (log_spec + 4.0) / 4.0
    # mel (T, 80)

    target = 3000
    if mel.shape[0] > target:
        # -50 so that there are some zero tail paddings.
        mel = mel[: target - 50]
        mel = torch.nn.functional.pad(mel, (0, 0, 0, 50), "constant", 0)
    elif mel.shape[0] < target:
        mel = torch.nn.functional.pad(
            mel, (0, 0, 0, target - mel.shape[0]), "constant", 0
        )

    mel = mel.t().unsqueeze(0)

    return mel


def load_tokens(filename):
    tokens = dict()
    with open(filename, "r") as f:
        for line in f:
            t, i = line.split()
            tokens[int(i)] = t
    return tokens


def init_model(filename, target_platform="rk3588"):

    if not Path(filename).is_file():
        exit(f"{filename} does not exist")

    rknn_lite = RKNNLite(verbose=False)
    ret = rknn_lite.load_rknn(path=filename)
    if ret != 0:
        exit(f"Load model {filename} failed!")

    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if ret != 0:
        exit(f"Failed to init rknn runtime for {filename}")
    return rknn_lite


class RKNNModel:
    def __init__(self, encoder: str, decoder: str, target_platform="rk3588"):
        self.encoder = init_model(encoder)
        self.decoder = init_model(decoder)

    def release(self):
        self.encoder.release()
        self.decoder.release()

    def run_encoder(self, x: np.ndarray):
        """
        Args:
          x: (1, 80, 3000), np.float32
        Returns:
          cross_k:
          cross_v:
        """
        out = self.encoder.inference(inputs=[x.numpy()])
        print(out[0].shape, out[1].shape)
        return out[0], out[1]

    def run_decoder(self, tokens: np.ndarray, cross_k: np.ndarray, cross_v):
        """
        Args:
          tokens: (1, 12), np.float32
          cross_k:
          cross_v:
        Returns:
          logit: (1, 12, vocab_size)
        """
        return self.decoder.inference(inputs=[tokens, cross_k, cross_v])[0]


def main():
    model = RKNNModel(
        encoder="./tiny.en-encoder.rknn",
        decoder="./tiny.en-decoder.rknn",
    )
    for i in range(1):
        test(model)


def test(model):
    id2token = load_tokens("./tiny.en-tokens.txt")

    start = time.time()
    samples, sample_rate = load_audio("./0.wav")
    assert sample_rate == 16000, sample_rate

    features = compute_features(samples)
    cross_k, cross_v = model.run_encoder(features)
    sot_sequence = [50257, 50362]
    tokens = sot_sequence + sot_sequence * 5
    idx = len(tokens)

    eot = 50256

    results = []
    for i in range(100):
        logit = model.run_decoder(np.array([tokens], dtype=np.int64), cross_k, cross_v)
        logit = logit[0, -1]
        max_token_id = np.argmax(logit)
        if max_token_id == eot:
            break
        results.append(max_token_id)
        tokens.append(max_token_id)
        if idx > len(sot_sequence):
            idx -= 1
        tokens.pop(idx)
        print(results)
    print()
    print(results)
    s = b""
    for i in results:
        if i in id2token:
            s += base64.b64decode(id2token[i])

    print(s.decode().strip())


if __name__ == "__main__":
    main()
