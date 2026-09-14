#!/usr/bin/env python3
"""
Test OfflineTtsVitsExtImpl with a coqui model (uses character frontend).

Download model:
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-coqui-de-css10.tar.bz2
  tar xf vits-coqui-de-css10.tar.bz2
"""

import sherpa_onnx
import soundfile as sf

tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model="./vits-coqui-de-css10/model.onnx",
            tokens="./vits-coqui-de-css10/tokens.txt",
        ),
        debug=True,
    ),
)
tts = sherpa_onnx.OfflineTts(tts_config)

text = "Alles hat ein Ende, nur die Wurst hat zwei."

gen_config = sherpa_onnx.GenerationConfig()
gen_config.sid = 0
gen_config.speed = 1.0

audio = tts.generate(text, gen_config)

assert len(audio.samples) > 0, "No audio generated!"
assert audio.sample_rate > 0, "Invalid sample rate!"

sf.write("test-coqui.wav", audio.samples, samplerate=audio.sample_rate)
print(
    f"OK coqui: {len(audio.samples)} samples, {audio.sample_rate}Hz, "
    f"{len(audio.samples)/audio.sample_rate:.2f}s"
)
