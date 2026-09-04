#!/usr/bin/env bash
set -ex
cd go-api-examples/vad
go mod tidy
go build
./run.sh
