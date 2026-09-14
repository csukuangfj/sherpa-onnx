#!/usr/bin/env python3
"""
Test a piper tts model using piper_phonemize.

Install piper_phonemize:
  pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html

Download model:
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
  tar xf vits-piper-en_US-amy-low.tar.bz2
"""

import sherpa_onnx
import soundfile as sf

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:
    raise RuntimeError(
        f"{ex}\nPlease run\n"
        "pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html"
    )

tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model="./vits-piper-en_US-amy-low/en_US-amy-low.onnx",
            tokens="./vits-piper-en_US-amy-low/tokens.txt",
        ),
        debug=True,
    ),
)
tts = sherpa_onnx.OfflineTts(tts_config)

text = "Today as always, men fall into two groups: slaves and free men."
sentences = phonemize_espeak(text, "en-us")
phoneme_codepoints = [[ord(c) for c in sentence] for sentence in sentences]

gen_config = sherpa_onnx.GenerationConfig()
gen_config.sid = 0
gen_config.speed = 1.0
gen_config.phoneme_codepoints = phoneme_codepoints

audio = tts.generate("", gen_config)

assert len(audio.samples) > 0, "No audio generated!"
assert audio.sample_rate > 0, "Invalid sample rate!"

sf.write("test-piper-phonemize.wav", audio.samples, samplerate=audio.sample_rate)
print(
    f"OK piper (phonemize): {len(audio.samples)} samples, {audio.sample_rate}Hz, "
    f"{len(audio.samples)/audio.sample_rate:.2f}s"
)
