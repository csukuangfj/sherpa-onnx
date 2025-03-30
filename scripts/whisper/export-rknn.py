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


def get_meta_data(model: str):
    import onnxruntime

    session_opts = onnxruntime.SessionOptions()
    session_opts.inter_op_num_threads = 1
    session_opts.intra_op_num_threads = 1

    m = onnxruntime.InferenceSession(
        model,
        sess_options=session_opts,
        providers=["CPUExecutionProvider"],
    )

    for i in m.get_inputs():
        print(i)

    print("-----")

    for i in m.get_outputs():
        print(i)

    meta = m.get_modelmeta().custom_metadata_map
    del meta["all_language_codes"]
    del meta["all_language_tokens"]
    s = ""
    sep = ""
    for key, value in meta.items():
        s = s + sep + f"{key}={value}"
        sep = ";"
    assert len(s) < 1024
    return s

    """
    {'no_timestamps': '50362', 'sot_lm': '50359', 'transcribe': '50358',
    'non_speech_tokens': '1,2,7,8,9,10,14,25,26,27,28,29,31,58,59,60,61,62,63,90,91,92,93,357,366,438,532,685,705,796,930,1058,1220,1267,1279,1303,1343,1377,1391,1635,1782,1875,2162,2361,2488,3467,4008,4211,4600,4808,5299,5855,6329,7203,9609,9959,10563,10786,11420,11709,11907,13163,13697,13700,14808,15306,16410,16791,17992,19203,19510,20724,22305,22935,27007,30109,30420,33409,34949,40283,40493,40549,47282,49146',
    'no_speech': '50361', 'is_multilingual': '0', 'eot': '50256',
    'sot_index': '0', 'sot': '50257', 'n_mels': '80', 'n_audio_layer': '4',
    'model_type': 'whisper-tiny.en', 'n_text_layer': '4', 'maintainer': 'k2-fsa',
    'sot_sequence': '50257', 'blank_id': '220', 'version': '1',
    'n_audio_state': '384', 'n_audio_head': '6', 'n_vocab': '51864',
    'sot_prev': '50360', 'n_text_ctx': '448', 'n_text_state': '384', 'translate': '50357',
    'n_text_head': '6', 'n_audio_ctx': '1500',
    'all_language_tokens': '50267,50284,50287,50303,50345,50261,50322,50276,50273,50304,50272,50342,50325,50306,50258,50335,50324,50270,50289,50299,50312,50336,50351,50344,50332,50282,50349,50318,50319,50305,50320,50350,50328,50316,50291,50313,50283,50310,50281,50307,50290,50315,50262,50354,50263,50280,50301,50260,50321,50295,50337,50309,50266,50293,50346,50277,50302,50278,50331,50355,50294,50279,50338,50347,50323,50297,50296,50340,50288,50298,50352,50339,50334,50268,50356,50348,50343,50333,50271,50275,50265,50329,50285,50259,50311,50300,50264,50314,50326,50308,50274,50353,50330,50292,50286,50269,50327,50341,50317',
    'all_language_codes': 'tr,da,no,az,my,es,km,fi,it,sl,sv,mt,so,et,en,lo,yo,nl,ur,fa,ne,uz,haw,lb,gu,cs,as,gl,mr,kn,pa,tt,ka,sq,bg,mn,ro,is,ms,mk,hr,kk,ru,ba,ko,el,bn,de,si,ml,fo,eu,pt,la,bo,vi,sr,he,sd,jw,mi,uk,ht,tl,sn,sk,cy,tk,th,te,ln,ps,yi,pl,su,mg,sa,am,ar,hi,ja,be,hu,zh,hy,lv,fr,bs,af,br,id,ha,tg,lt,ta,ca,oc,nn,sw'}
    """


def export_rknn(rknn, filename):
    ret = rknn.export_rknn(filename)
    if ret != 0:
        exit("Export rknn model to {filename} failed!")


def init_model(filename: str, target_platform: str, custom_string=None):
    rknn = RKNN(verbose=False)

    rknn.config(
        optimization_level=0,
        target_platform=target_platform,
        custom_string=custom_string,
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
        meta = get_meta_data(encoder)
        print(meta)

        self.encoder = init_model(
            encoder,
            target_platform=target_platform,
            custom_string=meta,
        )
        self.decoder = init_model(
            decoder,
            target_platform=target_platform,
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
