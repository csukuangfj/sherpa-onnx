#!/usr/bin/env python3
"""
Test OfflineTtsVitsExtImpl with an inflect model using piper_phonemize.

Install piper_phonemize:
  pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html

Download model:
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-inflect-en-nano-v2.tar.bz2
  tar xf vits-inflect-en-nano-v2.tar.bz2
"""

import sherpa_onnx
import soundfile as sf
from piper_phonemize import phonemize_espeak

tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model="./vits-inflect-en-nano-v2/model.onnx",
            tokens="./vits-inflect-en-nano-v2/tokens.txt",
        ),
        debug=True,
    ),
)
tts = sherpa_onnx.OfflineTts(tts_config)

text = "Friends fell out often because life was changing so fast."
sentences = phonemize_espeak(text, "en-us")
phoneme_codepoints = [[ord(c) for c in sentence] for sentence in sentences]

gen_config = sherpa_onnx.GenerationConfig()
gen_config.sid = 0
gen_config.speed = 1.0
gen_config.phoneme_codepoints = phoneme_codepoints

audio = tts.generate("", gen_config)

assert len(audio.samples) > 0, "No audio generated!"
assert audio.sample_rate > 0, "Invalid sample rate!"

sf.write("test-inflect-phonemize.wav", audio.samples, samplerate=audio.sample_rate)
print(
    f"OK inflect (phonemize): {len(audio.samples)} samples, {audio.sample_rate}Hz, "
    f"{len(audio.samples)/audio.sample_rate:.2f}s"
)
