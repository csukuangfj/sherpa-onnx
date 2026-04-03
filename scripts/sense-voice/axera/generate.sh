#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <num-frames>"
  exit 1
fi

num_frames="$1"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"


cd "${script_dir}"
mkdir -p features prompt

for w in *.wav; do
  python3 ./generate_test_data.py \
    --num-frames "${num_frames}" \
    --wav "$w"

  lang="${w%.wav}"
  mv input0.raw "features/${lang}.bin"
  mv input1.raw "prompt/${lang}.bin"
done

tar czvf features.tar.gz features
tar czvf prompt.tar.gz prompt
