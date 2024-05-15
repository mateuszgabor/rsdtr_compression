import torch
from torch import nn
from rsdtr import rsdtr
from TR_model import TR_model


def decompose_layer(layer, tol):
    is_bias = torch.is_tensor(layer.bias)
    bias = None
    weights = layer.weight
    factors = rsdtr(weights, tol)

    fourth = factors[0]
    fourth = fourth if fourth.ndim > 2 else fourth.unsqueeze(2)
    first = factors[1]
    first = first if first.ndim > 2 else first.unsqueeze(2)
    second = factors[2]
    second = second if second.ndim > 2 else second.unsqueeze(2)
    third = factors[3]
    third = third if third.ndim > 2 else third.unsqueeze(2)

    c_in_2, H, c_out_2 = second.shape
    c_in_3, W, c_out_3 = third.shape
    second_weights = second.moveaxis(-1, 0).unsqueeze(2).unsqueeze(4)
    third_weights = third.moveaxis(-1, 0).unsqueeze(2).unsqueeze(3)

    conv2 = nn.Conv3d(
        c_in_2,
        c_out_2,
        kernel_size=(1, H, 1),
        stride=(1, layer.stride[0], 1),
        padding=(0, layer.padding[0], 0),
        dilation=layer.dilation[0],
        bias=False,
    )

    conv3 = nn.Conv3d(
        c_in_3,
        c_out_3,
        kernel_size=(1, 1, W),
        stride=(1, 1, layer.stride[1]),
        padding=(0, 0, layer.padding[1]),
        dilation=layer.dilation[0],
        bias=False,
    )

    conv1 = torch.nn.Parameter(first)
    conv2.weight.data = second_weights
    conv3.weight.data = third_weights
    conv4 = torch.nn.Parameter(fourth)

    if is_bias:
        bias = torch.nn.Parameter(layer.bias.data)
        bias = bias.reshape(1, layer.out_channels, 1, 1)

    return TR_model(conv1, conv2, conv3, conv4, bias)
