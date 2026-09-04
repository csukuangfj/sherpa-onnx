#!/usr/bin/env bash
set -ex
cd go-api-examples/vad-asr-whisper
go mod tidy
go build
./run.sh
