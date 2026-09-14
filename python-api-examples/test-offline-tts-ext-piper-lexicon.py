#!/usr/bin/env python3
"""
Test a piper tts model using lexicon-based phonemization
(no espeak-ng data, no external phonemizer needed).

Download model and lexicon:
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
  tar xf vits-piper-en_US-amy-low.tar.bz2

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt
"""

import sherpa_onnx
import soundfile as sf

tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model="./vits-piper-en_US-amy-low/en_US-amy-low.onnx",
            tokens="./vits-piper-en_US-amy-low/tokens.txt",
            lexicon="./lexicon-en-us.txt",
        ),
        debug=True,
    ),
)
tts = sherpa_onnx.OfflineTts(tts_config)

text = "Today as always, men fall into two groups: slaves and free men."

gen_config = sherpa_onnx.GenerationConfig()
gen_config.sid = 0
gen_config.speed = 1.0

audio = tts.generate(text, gen_config)

assert len(audio.samples) > 0, "No audio generated!"
assert audio.sample_rate > 0, "Invalid sample rate!"

sf.write("test-piper-lexicon.wav", audio.samples, samplerate=audio.sample_rate)
print(
    f"OK piper-lexicon: {len(audio.samples)} samples, {audio.sample_rate}Hz, "
    f"{len(audio.samples)/audio.sample_rate:.2f}s"
)
