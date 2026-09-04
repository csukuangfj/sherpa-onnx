#!/usr/bin/env bash
set -ex
cd go-api-examples/vad-asr-paraformer
go mod tidy
go build
./run.sh
