#!/usr/bin/env bash

set -ex

export CGO_ENABLED=1

if [ ! -d sherpa-onnx-paraformer-zh-2023-09-14 ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
  tar xvf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
  rm sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
fi

go mod tidy
go build

echo "=== run-paraformer.sh: checking binary ==="
ls -lh ./non-streaming-decode-files* 2>/dev/null || echo "no binary"
ls -lh *.dll 2>/dev/null || echo "no dlls"
file ./non-streaming-decode-files.exe 2>/dev/null || file ./non-streaming-decode-files 2>/dev/null || echo "binary not found"

./non-streaming-decode-files \
  --paraformer ./sherpa-onnx-paraformer-zh-2023-09-14/model.int8.onnx \
  --tokens ./sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt \
  --model-type paraformer \
  --debug 0 \
  ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/0.wav
