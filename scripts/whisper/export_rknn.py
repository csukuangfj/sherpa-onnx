#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)

import argparse
import logging
from pathlib import Path
from typing import List

from rknn.api import RKNN

logging.basicConfig(level=logging.WARNING)

g_platforms = [
    #  "rv1103",
    #  "rv1103b",
    #  "rv1106",
    #  "rk2118",
    "rk3562",
    "rk3566",
    "rk3568",
    "rk3576",
    "rk3588",
]


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--target-platform",
        type=str,
        required=True,
        help=f"Supported values are: {','.join(g_platforms)}",
    )

    parser.add_argument(
        "--in-encoder",
        type=str,
        required=True,
        help="Path to the encoder onnx model",
    )

    parser.add_argument(
        "--in-decoder",
        type=str,
        required=True,
        help="Path to the decoder onnx model",
    )

    parser.add_argument(
        "--out-encoder",
        type=str,
        required=True,
        help="Path to the encoder rknn model",
    )

    parser.add_argument(
        "--out-decoder",
        type=str,
        required=True,
        help="Path to the decoder rknn model",
    )

    return parser


def export_rknn(rknn, filename):
    ret = rknn.export_rknn(filename)
    if ret != 0:
        exit("Export rknn model to {filename} failed!")


def init_model(
    filename: str, target_platform: str, custom_string=None, dynamic_input=None
):
    rknn = RKNN(verbose=False)

    rknn.config(
        optimization_level=0,
        target_platform=target_platform,
        custom_string=custom_string,
        dynamic_input=dynamic_input,
    )
    if not Path(filename).is_file():
        exit(f"{filename} does not exist")

    ret = rknn.load_onnx(model=filename)
    if ret != 0:
        exit(f"Load model {filename} failed!")

    ret = rknn.build(do_quantization=False)
    if ret != 0:
        exit("Build model {filename} failed!")

    return rknn


class RKNNModel:
    def __init__(
        self,
        encoder: str,
        decoder: str,
        target_platform: str,
    ):
        self.encoder = init_model(
            encoder,
            target_platform=target_platform,
        )
        self.decoder = init_model(
            decoder,
            target_platform=target_platform,
            #  dynamic_input=[
            #      [
            #          [1, i],
            #          [4, 1, 1500, 384],
            #          [4, 1, 1500, 384],
            #      ]
            #      for i in range(12, 1, -1)
            #  ],
        )

    def export_rknn(self, encoder, decoder):
        export_rknn(self.encoder, encoder)
        export_rknn(self.decoder, decoder)

    def release(self):
        self.encoder.release()
        self.decoder.release()


def main():
    args = get_parser().parse_args()
    print(vars(args))

    model = RKNNModel(
        encoder=args.in_encoder,
        decoder=args.in_decoder,
        target_platform=args.target_platform,
    )

    model.export_rknn(
        encoder=args.out_encoder,
        decoder=args.out_decoder,
    )

    model.release()


if __name__ == "__main__":
    main()
