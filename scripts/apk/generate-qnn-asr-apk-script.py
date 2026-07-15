#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from pathlib import Path

import jinja2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--total",
        type=int,
        default=1,
        help="Number of runners",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the current runner",
    )
    return parser.parse_args()


@dataclass
class Model:
    model_name: str
    idx: int
    lang: str
    short_name: str = ""
    cmd: str = ""
    release_tag: str = "asr-models-qnn"
    sed_old: str = ""
    sed_new: str = ""


def get_models():
    cmd = """
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """

    models = [
        Model(
            model_name="sherpa-onnx-qnn-streaming-zipformer-transducer-zh-en-2023-03-20-chunk-size-32-android-aarch64",
            idx=9025,
            lang="zh_en",
            short_name="streaming_zipformer_transducer_2023_03_20_chunk_32",
            cmd=cmd,
        ),
    ]

    chunk_sizes = [160, 480, 960, 1920]

    # x-asr streaming
    models.append(Model(
        model_name=f"sherpa-onnx-qnn-x-asr-streaming-zipformer-transducer-zh-en-2026-06-05-chunk-size-{chunk_sizes[0]}ms-android-aarch64",
        idx=9027,
        lang="zh_en",
        short_name=f"x_asr_streaming_zipformer_transducer_2026_06_05_chunk_{chunk_sizes[0]}ms",
        cmd=cmd,
    ))
    for s in chunk_sizes[1:]:
        models.append(Model(
            model_name=f"sherpa-onnx-qnn-x-asr-streaming-zipformer-transducer-zh-en-2026-06-05-chunk-size-{s}ms-android-aarch64",
            idx=9027,
            lang="zh_en",
            short_name=f"x_asr_streaming_zipformer_transducer_2026_06_05_chunk_{s}ms",
            sed_old=f"chunk-size-{chunk_sizes[0]}ms",
            sed_new=f"chunk-size-{s}ms",
            cmd=cmd,
        ))

    # x-asr streaming punct
    models.append(Model(
        model_name=f"sherpa-onnx-qnn-x-asr-streaming-zipformer-transducer-zh-en-punct-2026-06-05-chunk-size-{chunk_sizes[0]}ms-android-aarch64",
        idx=9028,
        lang="zh_en",
        short_name=f"x_asr_streaming_zipformer_transducer_punct_2026_06_05_chunk_{chunk_sizes[0]}ms",
        cmd=cmd,
    ))
    for s in chunk_sizes[1:]:
        models.append(Model(
            model_name=f"sherpa-onnx-qnn-x-asr-streaming-zipformer-transducer-zh-en-punct-2026-06-05-chunk-size-{s}ms-android-aarch64",
            idx=9028,
            lang="zh_en",
            short_name=f"x_asr_streaming_zipformer_transducer_punct_2026_06_05_chunk_{s}ms",
            sed_old=f"chunk-size-{chunk_sizes[0]}ms",
            sed_new=f"chunk-size-{s}ms",
            cmd=cmd,
        ))

    # SM8850 binary x-asr streaming
    models.append(Model(
        model_name=f"sherpa-onnx-qnn-SM8850-binary-x-asr-streaming-zipformer-transducer-zh-en-2026-06-05-chunk-size-{chunk_sizes[0]}ms",
        idx=9029,
        lang="zh_en",
        short_name=f"SM8850_x_asr_streaming_zipformer_transducer_2026_06_05_chunk_{chunk_sizes[0]}ms",
        release_tag="asr-models-qnn-binary",
        cmd=cmd,
    ))
    for s in chunk_sizes[1:]:
        models.append(Model(
            model_name=f"sherpa-onnx-qnn-SM8850-binary-x-asr-streaming-zipformer-transducer-zh-en-2026-06-05-chunk-size-{s}ms",
            idx=9029,
            lang="zh_en",
            short_name=f"SM8850_x_asr_streaming_zipformer_transducer_2026_06_05_chunk_{s}ms",
            release_tag="asr-models-qnn-binary",
            sed_old=f"chunk-size-{chunk_sizes[0]}ms",
            sed_new=f"chunk-size-{s}ms",
            cmd=cmd,
        ))

    # SM8850 binary x-asr streaming punct
    models.append(Model(
        model_name=f"sherpa-onnx-qnn-SM8850-binary-x-asr-streaming-zipformer-transducer-zh-en-punct-2026-06-05-chunk-size-{chunk_sizes[0]}ms",
        idx=9030,
        lang="zh_en",
        short_name=f"SM8850_x_asr_streaming_zipformer_transducer_punct_2026_06_05_chunk_{chunk_sizes[0]}ms",
        release_tag="asr-models-qnn-binary",
        cmd=cmd,
    ))
    for s in chunk_sizes[1:]:
        models.append(Model(
            model_name=f"sherpa-onnx-qnn-SM8850-binary-x-asr-streaming-zipformer-transducer-zh-en-punct-2026-06-05-chunk-size-{s}ms",
            idx=9030,
            lang="zh_en",
            short_name=f"SM8850_x_asr_streaming_zipformer_transducer_punct_2026_06_05_chunk_{s}ms",
            release_tag="asr-models-qnn-binary",
            sed_old=f"chunk-size-{chunk_sizes[0]}ms",
            sed_new=f"chunk-size-{s}ms",
            cmd=cmd,
        ))

    # nemotron streaming models
    nemotron_chunks = [80, 160, 560, 1120]

    # nemotron lib files
    models.append(Model(
        model_name=f"sherpa-onnx-qnn-nemotron-speech-streaming-en-0.6b-{nemotron_chunks[0]}s-android-aarch64",
        idx=9031,
        lang="en",
        short_name=f"nemotron_speech_streaming_en_0.6b_{nemotron_chunks[0]}s",
        release_tag="asr-models-qnn-2",
        cmd=cmd,
    ))
    for s in nemotron_chunks[1:]:
        models.append(Model(
            model_name=f"sherpa-onnx-qnn-nemotron-speech-streaming-en-0.6b-{s}s-android-aarch64",
            idx=9031,
            lang="en",
            short_name=f"nemotron_speech_streaming_en_0.6b_{s}s",
            sed_old=f"{nemotron_chunks[0]}s",
            sed_new=f"{s}s",
            release_tag="asr-models-qnn-2",
            cmd=cmd,
        ))

    # SM8850 binary nemotron
    models.append(Model(
        model_name=f"sherpa-onnx-qnn-SM8850-binary-nemotron-speech-streaming-en-0.6b-{nemotron_chunks[0]}ms",
        idx=9032,
        lang="en",
        short_name=f"SM8850_nemotron_speech_streaming_en_0.6b_{nemotron_chunks[0]}ms",
        release_tag="asr-models-qnn-binary-2",
        cmd=cmd,
    ))
    for s in nemotron_chunks[1:]:
        models.append(Model(
            model_name=f"sherpa-onnx-qnn-SM8850-binary-nemotron-speech-streaming-en-0.6b-{s}ms",
            idx=9032,
            lang="en",
            short_name=f"SM8850_nemotron_speech_streaming_en_0.6b_{s}ms",
            release_tag="asr-models-qnn-binary-2",
            sed_old=f"{nemotron_chunks[0]}ms",
            sed_new=f"{s}ms",
            cmd=cmd,
        ))

    # nemotron-3.5 streaming models
    nemotron35_chunks = [80, 160, 320, 560, 1120]

    # nemotron-3.5 lib files
    models.append(Model(
        model_name=f"sherpa-onnx-qnn-nemotron-3.5-asr-streaming-0.6b-{nemotron35_chunks[0]}s-android-aarch64",
        idx=9033,
        lang="en",
        short_name=f"nemotron_3.5_asr_streaming_0.6b_{nemotron35_chunks[0]}s",
        release_tag="asr-models-qnn-3",
        cmd=cmd,
    ))
    for s in nemotron35_chunks[1:]:
        models.append(Model(
            model_name=f"sherpa-onnx-qnn-nemotron-3.5-asr-streaming-0.6b-{s}s-android-aarch64",
            idx=9033,
            lang="en",
            short_name=f"nemotron_3.5_asr_streaming_0.6b_{s}s",
            sed_old=f"{nemotron35_chunks[0]}s",
            sed_new=f"{s}s",
            release_tag="asr-models-qnn-3",
            cmd=cmd,
        ))

    # SM8850 binary nemotron-3.5
    models.append(Model(
        model_name=f"sherpa-onnx-qnn-SM8850-binary-nemotron-3.5-asr-streaming-0.6b-{nemotron35_chunks[0]}ms",
        idx=9034,
        lang="en",
        short_name=f"SM8850_nemotron_3.5_asr_streaming_0.6b_{nemotron35_chunks[0]}ms",
        release_tag="asr-models-qnn-binary-3",
        cmd=cmd,
    ))
    for s in nemotron35_chunks[1:]:
        models.append(Model(
            model_name=f"sherpa-onnx-qnn-SM8850-binary-nemotron-3.5-asr-streaming-0.6b-{s}ms",
            idx=9034,
            lang="en",
            short_name=f"SM8850_nemotron_3.5_asr_streaming_0.6b_{s}ms",
            release_tag="asr-models-qnn-binary-3",
            sed_old=f"{nemotron35_chunks[0]}ms",
            sed_new=f"{s}ms",
            cmd=cmd,
        ))

    return models


def main():
    args = get_args()
    index = args.index
    total = args.total
    assert 0 <= index < total, (index, total)

    all_model_list = get_models()
    num_models = len(all_model_list)

    num_per_runner = num_models // total
    if num_per_runner <= 0:
        raise ValueError(f"num_models: {num_models}, num_runners: {total}")

    start = index * num_per_runner
    end = start + num_per_runner
    remaining = num_models - args.total * num_per_runner

    print(f"{index}/{total}: {start}-{end}/{num_models}")

    d = {"model_list": all_model_list[start:end]}
    if index < remaining:
        s = args.total * num_per_runner + index
        d["model_list"].append(all_model_list[s])
        print(f"{s}/{num_models}")

    filename_list = [
        "./build-apk-qnn-asr.sh",
    ]
    for filename in filename_list:
        environment = jinja2.Environment()
        if not Path(f"{filename}.in").is_file():
            print(f"skip {filename}")
            continue

        with open(f"{filename}.in") as f:
            s = f.read()
        template = environment.from_string(s)

        s = template.render(**d)
        with open(filename, "w") as f:
            print(s, file=f)


if __name__ == "__main__":
    main()
