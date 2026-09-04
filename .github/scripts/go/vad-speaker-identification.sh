#!/usr/bin/env bash
set -ex
cd go-api-examples/vad-speaker-identification
go mod tidy
go build
./run.sh
