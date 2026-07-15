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
    # We will download
    # https://github.com/k2-fsa/sherpa-onnx/releases/download/{release_tag}/{model_name}.tar.bz2
    model_name: str

    # The type of the model, e..g, 0, 1, 2. It is hardcoded in the kotlin code
    idx: int

    # e.g., zh, en, zh_en
    lang: str

    # e.g., whisper, paraformer, zipformer
    short_name: str = ""

    # cmd is used to remove extra file from the model directory
    cmd: str = ""

    rule_fsts: str = ""

    use_hr: bool = False

    release_tag: str = "asr-models-qnn"

    # If non-empty, sed replaces the duration in OfflineRecognizer.kt
    # e.g., sed_old="5s", sed_new="8s" replaces "ja-5s-" with "ja-8s-"
    sed_old: str = ""
    sed_new: str = ""


# See get_2nd_models() in ./generate-asr-2pass-apk-script.py
def get_models():
    models = [
        # sense-voice with different durations
        Model(
            model_name="sherpa-onnx-qnn-5-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8-android-aarch64",
            idx=9000,
            lang="zh_en_ko_ja_yue",
            short_name="5-seconds-sense_voice_2024_07_17_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        *[Model(
            model_name=f"sherpa-onnx-qnn-{s}-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8-android-aarch64",
            idx=9000,
            lang="zh_en_ko_ja_yue",
            short_name=f"{s}-seconds-sense_voice_2024_07_17_int8",
            use_hr=True,
            sed_old="5-seconds-",
            sed_new=f"{s}-seconds-",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ) for s in [8, 10, 13, 15, 18, 20, 23, 25, 28, 30]],
        # zipformer-ctc-zh with different durations
        Model(
            model_name="sherpa-onnx-qnn-5-seconds-zipformer-ctc-zh-2025-07-03-int8-android-aarch64",
            idx=9011,
            lang="zh",
            short_name="5-seconds-zipformer_ctc_2025_07_03_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        *[Model(
            model_name=f"sherpa-onnx-qnn-{s}-seconds-zipformer-ctc-zh-2025-07-03-int8-android-aarch64",
            idx=9011,
            lang="zh",
            short_name=f"{s}-seconds-zipformer_ctc_2025_07_03_int8",
            use_hr=True,
            sed_old="5-seconds-",
            sed_new=f"{s}-seconds-",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ) for s in [8, 10, 13, 15, 18, 20, 23, 25, 28, 30]],
        Model(
            model_name="sherpa-onnx-qnn-SM8850-binary-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8",
            idx=9022,
            lang="zh_en_ko_ja_yue",
            short_name="SM8850_10-seconds-sense_voice_2024_07_17_int8",
            release_tag="asr-models-qnn-binary",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-qnn-5-seconds-paraformer-zh-2023-03-28-int8-android-aarch64",
            idx=9023,
            lang="zh",
            short_name="5-seconds-paraformer_zh_2023_03_28_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-qnn-SM8850-binary-5-seconds-paraformer-zh-2023-03-28-int8",
            idx=9025,
            lang="zh",
            short_name="SM8850_5-seconds-paraformer_zh_2023_03_28_int8",
            release_tag="asr-models-qnn-binary",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        Model(
            model_name="sherpa-onnx-qnn-5-seconds-paraformer-zh-2025-10-07-int8-android-aarch64",
            idx=9024,
            lang="zh",
            short_name="5-seconds-paraformer_zh_2025_10_07_int8",
            use_hr=True,
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        # reazonspeech-zipformer-transducer with different durations
        Model(
            model_name="sherpa-onnx-qnn-reazonspeech-zipformer-transducer-ja-5s-2024-08-01-android-aarch64",
            idx=9026,
            lang="ja",
            short_name="reazonspeech_zipformer_transducer_ja_5s",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        *[Model(
            model_name=f"sherpa-onnx-qnn-reazonspeech-zipformer-transducer-ja-{s}s-2024-08-01-android-aarch64",
            idx=9026,
            lang="ja",
            short_name=f"reazonspeech_zipformer_transducer_ja_{s}s",
            sed_old="ja-5s-",
            sed_new=f"ja-{s}s-",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ) for s in [8, 10, 13, 15, 18, 20, 23, 25, 28, 30]],
        # SM8850 binary reazonspeech-zipformer-transducer with different durations
        Model(
            model_name="sherpa-onnx-qnn-SM8850-binary-reazonspeech-zipformer-transducer-ja-5s-2024-08-01",
            idx=9027,
            lang="ja",
            short_name="SM8850_reazonspeech_zipformer_transducer_ja_5s",
            release_tag="asr-models-qnn-binary",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        *[Model(
            model_name=f"sherpa-onnx-qnn-SM8850-binary-reazonspeech-zipformer-transducer-ja-{s}s-2024-08-01",
            idx=9027,
            lang="ja",
            short_name=f"SM8850_reazonspeech_zipformer_transducer_ja_{s}s",
            release_tag="asr-models-qnn-binary",
            sed_old="ja-5s-",
            sed_new=f"ja-{s}s-",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ) for s in [8, 10, 13, 15, 18, 20, 23, 25, 28, 30]],

        # moonshine QNN models (lib files)
        # All use idx=9030, sed replaces language/model variants
        Model(
            model_name="sherpa-onnx-qnn-moonshine-tiny-en-5s-android-aarch64",
            idx=9030,
            lang="en",
            short_name="moonshine_tiny_en_5s",
            release_tag="asr-models-qnn-3",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        *[Model(
            model_name=f"sherpa-onnx-qnn-moonshine-tiny-en-{s}s-android-aarch64",
            idx=9030,
            lang="en",
            short_name=f"moonshine_tiny_en_{s}s",
            sed_old="tiny-en-5s",
            sed_new=f"tiny-en-{s}s",
            release_tag="asr-models-qnn-3",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ) for s in [8, 10]],
        *[Model(
            model_name=f"sherpa-onnx-qnn-moonshine-tiny-{l}-5s-android-aarch64",
            idx=9030,
            lang=l,
            short_name=f"moonshine_tiny_{l}_5s",
            sed_old="tiny-en-",
            sed_new=f"tiny-{l}-",
            release_tag="asr-models-qnn-3",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ) for l in ["zh", "ja", "ko", "vi", "uk", "ar"]],
        *[Model(
            model_name=f"sherpa-onnx-qnn-moonshine-tiny-{l}-{s}s-android-aarch64",
            idx=9030,
            lang=l,
            short_name=f"moonshine_tiny_{l}_{s}s",
            sed_old="tiny-en-5s",
            sed_new=f"tiny-{l}-{s}s",
            release_tag="asr-models-qnn-3",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ) for l in ["zh", "ja", "ko", "vi", "uk", "ar"] for s in [8, 10]],
        *[Model(
            model_name=f"sherpa-onnx-qnn-moonshine-base-{l}-5s-android-aarch64" if l != "en" else "sherpa-onnx-qnn-moonshine-base-5s-android-aarch64",
            idx=9030,
            lang=l if l != "en" else "en",
            short_name=f"moonshine_base_{l}_5s",
            sed_old="tiny-en-",
            sed_new=f"base-{l}-" if l != "en" else "base-",
            release_tag="asr-models-qnn-3",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ) for l in ["en", "zh", "ja", "ko", "vi", "uk", "ar"]],
        *[Model(
            model_name=f"sherpa-onnx-qnn-moonshine-base-{l}-{s}s-android-aarch64" if l != "en" else f"sherpa-onnx-qnn-moonshine-base-{s}s-android-aarch64",
            idx=9030,
            lang=l if l != "en" else "en",
            short_name=f"moonshine_base_{l}_{s}s",
            sed_old="tiny-en-5s",
            sed_new=f"base-{l}-{s}s" if l != "en" else f"base-{s}s",
            release_tag="asr-models-qnn-3",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ) for l in ["en", "zh", "ja", "ko", "vi", "uk", "ar"] for s in [8, 10]],

        # SM8850 binary moonshine models (for Xiaomi 17 Pro)
        # All use idx=9034, sed replaces language/model variants
        Model(
            model_name="sherpa-onnx-qnn-SM8850-binary-moonshine-tiny-en-5s",
            idx=9034,
            lang="en",
            short_name="SM8850_moonshine_tiny_en_5s",
            release_tag="asr-models-qnn-binary-3",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ),
        *[Model(
            model_name=f"sherpa-onnx-qnn-SM8850-binary-moonshine-tiny-en-{s}s",
            idx=9034,
            lang="en",
            short_name=f"SM8850_moonshine_tiny_en_{s}s",
            release_tag="asr-models-qnn-binary-3",
            sed_old="tiny-en-5s",
            sed_new=f"tiny-en-{s}s",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ) for s in [8, 10]],
        *[Model(
            model_name=f"sherpa-onnx-qnn-SM8850-binary-moonshine-tiny-{l}-5s",
            idx=9034,
            lang=l,
            short_name=f"SM8850_moonshine_tiny_{l}_5s",
            release_tag="asr-models-qnn-binary-3",
            sed_old="tiny-en-",
            sed_new=f"tiny-{l}-",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ) for l in ["zh", "ja", "ko", "vi", "uk", "ar"]],
        *[Model(
            model_name=f"sherpa-onnx-qnn-SM8850-binary-moonshine-tiny-{l}-{s}s",
            idx=9034,
            lang=l,
            short_name=f"SM8850_moonshine_tiny_{l}_{s}s",
            release_tag="asr-models-qnn-binary-3",
            sed_old="tiny-en-5s",
            sed_new=f"tiny-{l}-{s}s",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ) for l in ["zh", "ja", "ko", "vi", "uk", "ar"] for s in [8, 10]],
        *[Model(
            model_name=f"sherpa-onnx-qnn-SM8850-binary-moonshine-base-{l}-5s" if l != "en" else "sherpa-onnx-qnn-SM8850-binary-moonshine-base-5s",
            idx=9034,
            lang=l if l != "en" else "en",
            short_name=f"SM8850_moonshine_base_{l}_5s",
            release_tag="asr-models-qnn-binary-3",
            sed_old="tiny-en-",
            sed_new=f"base-{l}-" if l != "en" else "base-",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ) for l in ["en", "zh", "ja", "ko", "vi", "uk", "ar"]],
        *[Model(
            model_name=f"sherpa-onnx-qnn-SM8850-binary-moonshine-base-{l}-{s}s" if l != "en" else f"sherpa-onnx-qnn-SM8850-binary-moonshine-base-{s}s",
            idx=9034,
            lang=l if l != "en" else "en",
            short_name=f"SM8850_moonshine_base_{l}_{s}s",
            release_tag="asr-models-qnn-binary-3",
            sed_old="tiny-en-5s",
            sed_new=f"base-{l}-{s}s",
            cmd="""
            pushd $model_name

            rm -rfv test_wavs

            ls -lh

            popd
            """,
        ) for l in ["en", "zh", "ja", "ko", "vi", "uk", "ar"] for s in [8, 10]],
    ]
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

    d = dict()
    d["model_list"] = all_model_list[start:end]
    if index < remaining:
        s = args.total * num_per_runner + index
        d["model_list"].append(all_model_list[s])
        print(f"{s}/{num_models}")

    filename_list = [
        "./build-apk-qnn-vad-asr-simulate-streaming.sh",
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
