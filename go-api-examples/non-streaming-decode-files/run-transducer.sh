#!/usr/bin/env bash

set -ex

export CGO_ENABLED=1

if [ ! -d sherpa-onnx-streaming-zipformer-en-2023-06-26 ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-06-26.tar.bz2
  tar xvf sherpa-onnx-zipformer-en-2023-06-26.tar.bz2
  rm sherpa-onnx-zipformer-en-2023-06-26.tar.bz2
fi

go mod tidy
go build

echo "=== run-transducer.sh: checking binary ==="
ls -lh ./non-streaming-decode-files* 2>/dev/null || echo "no binary"
ls -lh *.dll 2>/dev/null || echo "no dlls"
file ./non-streaming-decode-files.exe 2>/dev/null || file ./non-streaming-decode-files 2>/dev/null || echo "binary not found"

./non-streaming-decode-files \
  --encoder ./sherpa-onnx-zipformer-en-2023-06-26/encoder-epoch-99-avg-1.onnx \
  --decoder ./sherpa-onnx-zipformer-en-2023-06-26/decoder-epoch-99-avg-1.onnx \
  --joiner ./sherpa-onnx-zipformer-en-2023-06-26/joiner-epoch-99-avg-1.onnx \
  --tokens ./sherpa-onnx-zipformer-en-2023-06-26/tokens.txt \
  --model-type transducer \
  --debug 0 \
  ./sherpa-onnx-zipformer-en-2023-06-26/test_wavs/0.wav
