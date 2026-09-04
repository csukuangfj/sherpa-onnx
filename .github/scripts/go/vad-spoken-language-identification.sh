#!/usr/bin/env bash
set -ex
cd go-api-examples/vad-spoken-language-identification
go mod tidy
go build
./run.sh
