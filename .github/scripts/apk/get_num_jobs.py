#!/usr/bin/env python3
# Copyright 2026 Xiaomi Corp.        (authors: Fangjun Kuang)

import importlib.util
import json
import os
import sys

MODELS_PER_JOB = 5


def get_apk_script_dir():
    """Get the APK scripts directory relative to this script's location."""
    # This script is at .github/scripts/apk/get_num_jobs.py
    # APK scripts are at scripts/apk/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "..", "..", "..", "scripts", "apk")


def get_num_models(script_name):
    """Get total number of models for the given script."""
    apk_dir = get_apk_script_dir()

    if script_name in ("qnn-asr", "qnn-vad-asr"):
        script_map = {
            "qnn-asr": "generate-qnn-asr-apk-script.py",
            "qnn-vad-asr": "generate-qnn-vad-asr-apk-script.py",
        }
        script_path = os.path.join(apk_dir, script_map[script_name])
        spec = importlib.util.spec_from_file_location("gen_script", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return len(module.get_models())

    elif script_name in ("tts", "tts-engine"):
        script_path = os.path.join(apk_dir, "generate-tts-apk-script.py")
        spec = importlib.util.spec_from_file_location("gen_script", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        all_models = []
        all_models += module.get_vits_models()
        all_models += module.get_piper_models()
        all_models += module.get_mimic3_models()
        all_models += module.get_coqui_models()
        all_models += module.get_matcha_models()
        all_models += module.get_kokoro_models()
        all_models += module.get_kitten_models()
        all_models += module.get_supertonic3_models()
        return len(all_models)
    else:
        raise ValueError(f"Unknown script: {script_name}")


def get_matrix_json(script_name):
    """Generate JSON matrix for GitHub Actions."""
    num_models = get_num_models(script_name)
    num_jobs = max(1, (num_models + MODELS_PER_JOB - 1) // MODELS_PER_JOB)
    indices = list(range(num_jobs))
    return json.dumps({"os": ["ubuntu-latest"], "total": [str(num_jobs)], "index": [str(i) for i in indices]})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, required=True,
                        choices=["qnn-asr", "qnn-vad-asr", "tts", "tts-engine"])
    args = parser.parse_args()

    print(get_matrix_json(args.script))
