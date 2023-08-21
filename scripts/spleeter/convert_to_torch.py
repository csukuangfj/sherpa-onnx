#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn


def load_graph(frozen_graph_filename):
    # This function is modified from
    # https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc

    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.compat.v1.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        #  tf.import_graph_def(graph_def, name="prefix")
        tf.import_graph_def(graph_def, name="")
    return graph


def generate_waveform():
    np.random.seed(20230821)
    waveform = np.random.rand(60 * 44100).astype(np.float32)

    # (num_samples, num_channels)
    waveform = waveform.reshape(-1, 2)
    return waveform


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 16, kernel_size=5, stride=(2, 2), padding=0)
        self.bn = torch.nn.BatchNorm2d(
            16, track_running_stats=True, eps=1e-3, momentum=0.01
        )

    def forward(self, x):
        x = torch.nn.functional.pad(x, (1, 2, 1, 2), "constant", 0)
        x = self.conv(x)
        x = self.bn(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)

        return x


def get_param(graph, name):
    with tf.compat.v1.Session(graph=graph) as sess:
        constant_ops = [op for op in sess.graph.get_operations() if op.type == "Const"]
        for constant_op in constant_ops:
            if constant_op.name != name:
                continue

            value = sess.run(constant_op.outputs[0])
            return torch.from_numpy(value)


@torch.no_grad()
def main():
    graph = load_graph("./2stems/frozen_model.pb")
    #  for op in graph.get_operations():
    #      print(op.name)
    x = graph.get_tensor_by_name("waveform:0")
    #  y = graph.get_tensor_by_name("Reshape:0")
    y0 = graph.get_tensor_by_name("strided_slice_3:0")
    y1 = graph.get_tensor_by_name("leaky_re_lu/LeakyRelu:0")

    unet = UNet()
    unet.eval()

    # For the conv2d in tensorflow, weight shape is (kernel_h, kernel_w, in_channel, out_channel)
    # default input shape is NHWC

    # For the conv2d in torch, weight shape is (out_channel, in_channel, kernel_h, kernel_w)
    # default input shape is NCHW
    state_dict = unet.state_dict()
    state_dict["conv.weight"] = get_param(graph, "conv2d/kernel").permute(3, 2, 0, 1)
    state_dict["conv.bias"] = get_param(graph, "conv2d/bias")

    state_dict["bn.weight"] = get_param(graph, "batch_normalization/gamma")
    state_dict["bn.bias"] = get_param(graph, "batch_normalization/beta")
    state_dict["bn.running_mean"] = get_param(graph, "batch_normalization/moving_mean")
    state_dict["bn.running_var"] = get_param(
        graph, "batch_normalization/moving_variance"
    )

    print(state_dict["conv.weight"].dtype)
    print(list(state_dict.keys()))
    unet.load_state_dict(state_dict)

    with tf.compat.v1.Session(graph=graph) as sess:
        y0_out, y1_out = sess.run([y0, y1], feed_dict={x: generate_waveform()})
        #  y0_out = sess.run(y0, feed_dict={x: generate_waveform()})
        #  y1_out = sess.run(y1, feed_dict={x: generate_waveform()})
        print(y0_out.shape)
        print(y1_out.shape)

    # for the batchnormalization in tensorflow,
    # default input shape is NHWC

    # for the batchnormalization in torch,
    # default input shape is NCHW

    # NHWC to NCHW
    torch_y1_out = unet(torch.from_numpy(y0_out).permute(0, 3, 1, 2))

    print(torch_y1_out.shape, torch.from_numpy(y1_out).permute(0, 3, 1, 2).shape)
    assert torch.allclose(
        torch_y1_out, torch.from_numpy(y1_out).permute(0, 3, 1, 2), atol=1e-3
    ), ((torch_y1_out - torch.from_numpy(y1_out).permute(0, 3, 1, 2)).abs().max())


if __name__ == "__main__":
    main()
