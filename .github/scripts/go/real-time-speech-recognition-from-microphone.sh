#!/usr/bin/env bash
set -ex
cd go-api-examples/real-time-speech-recognition-from-microphone
go mod tidy
go build
./run.sh
