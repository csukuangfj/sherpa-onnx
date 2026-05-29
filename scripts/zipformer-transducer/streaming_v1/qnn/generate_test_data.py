#!/usr/bin/env python3
# Copyright      2026  Xiaomi Corp.        (authors: Fangjun Kuang)
import argparse
from pathlib import Path

import numpy as np

from test_onnx import compute_feat, load_audio, load_model, load_tokens


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        required=True,
        help="32, 64, etc",
    )

    parser.add_argument(
        "--wav",
        type=str,
        required=True,
    )
    return parser.parse_args()


def main():
    args = get_args()
    print(vars(args))
    model = load_model(args.chunk_size)
    name = Path(args.wav)

    samples, sample_rate = load_audio(args.wav)

    features = compute_feat(
        samples=samples,
        sample_rate=sample_rate,
    )
    print("features", features.shape)

    states = model.get_encoder_states()

    blank_id = 0

    hyp = [blank_id] * model.context_size
    decoder_out = model.run_decoder(hyp)

    encoder_input_list = []
    decoder_input_list = []
    joiner_input_list = []

    encoder_input_names = [n.name for n in model.encoder.get_inputs()]
    decoder_input_names = [n.name for n in model.decoder.get_inputs()]
    joiner_input_names = [n.name for n in model.joiner.get_inputs()]

    frame_size = model.T
    frame_shift = model.decode_chunk_len
    start = 0
    while start + frame_size < features.shape[0]:
        x = features[start : start + frame_size]
        start += frame_shift
        encoder_input_list.append([x.T[None].copy()] + [s.copy() for s in states])
        for k, n_name in enumerate(encoder_input_names):
            if "cached_key" in n_name or "cached_val" in n_name:
                encoder_input_list[-1][k] = encoder_input_list[-1][k].transpose(
                    0, 2, 3, 1
                )

        x = x[None]
        encoder_out, states = model.run_encoder(x, states)
        num_frames = encoder_out.shape[1]

        for k in range(num_frames):
            cur_encoder_out = encoder_out[0, k : k + 1]

            joiner_input_list.append([cur_encoder_out.copy(), decoder_out.copy()])

            joiner_out = model.run_joiner(cur_encoder_out, decoder_out)
            token_id = joiner_out.argmax()
            if token_id != blank_id:
                hyp.append(token_id)
                decoder_input_list.append(
                    np.array(hyp[-model.context_size :], dtype=np.int32)
                )
                decoder_out = model.run_decoder(hyp[-model.context_size :])
    print(len(encoder_input_list), len(decoder_input_list), len(joiner_input_list))

    stem = name.stem
    with open(f"{stem}-data-encoder.txt", "w") as f:
        for r, input_list in enumerate(encoder_input_list):
            sep = ""
            for i, data in enumerate(input_list):
                n_name = encoder_input_names[i]
                filename = f"{stem}-encoder-in-run-{r}-{i}.raw"
                data.tofile(filename)
                f.write(f"{sep}{n_name}:={filename}")
                sep = " "
            f.write("\n")

    with open(f"{stem}-data-decoder.txt", "w") as f:
        for i, data in enumerate(decoder_input_list):
            n_name = decoder_input_names[0]
            filename = f"{stem}-decoder-in-run-{i}.raw"
            data.tofile(filename)
            f.write(f"{n_name}:={filename}\n")

    with open(f"{stem}-data-joiner.txt", "w") as f:
        for r, input_list in enumerate(joiner_input_list):
            sep = ""
            for i, data in enumerate(input_list):
                n_name = joiner_input_names[i]
                filename = f"{stem}-joiner-in-run-{r}-{i}.raw"
                data.tofile(filename)
                f.write(f"{sep}{n_name}:={filename}")
                sep = " "
            f.write("\n")

    id2token = load_tokens("./2026-05-26/tokens.txt")

    tokens = [id2token[i] for i in hyp[model.context_size :]]
    print(tokens)
    text = "".join(tokens)
    print(text)


if __name__ == "__main__":
    main()
