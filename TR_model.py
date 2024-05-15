import torch
import torch.nn as nn


class TR_model(nn.Module):
    def __init__(self, conv1, conv2, conv3, conv4, bias=None):
        super(TR_model, self).__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.conv4 = conv4
        self.bias = bias

    def forward(self, x):
        x = torch.tensordot(x, self.conv1, dims=([1], [1]))
        x = torch.moveaxis(x, (-1, -2), (1, 2))
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.tensordot(x, self.conv4, dims=([1, 2], [0, 2]))
        x = torch.moveaxis(x, -1, 1)

        if self.bias is not None:
            x += self.bias

        return x
