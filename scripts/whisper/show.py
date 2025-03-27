#!/usr/bin/env python3

import onnxruntime
import onnx


def show(filename):
    session_opts = onnxruntime.SessionOptions()
    session_opts.log_severity_level = 3
    sess = onnxruntime.InferenceSession(filename, session_opts)
    for i in sess.get_inputs():
        print(i)

    print("-----")

    for i in sess.get_outputs():
        print(i)


def main():
    print("=========encoder==========")
    show("./tiny.en-encoder.onnx")

    print("=========decoder==========")
    show("./tiny.en-decoder.onnx")


if __name__ == "__main__":
    main()
