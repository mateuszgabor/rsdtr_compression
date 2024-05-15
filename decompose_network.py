import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from decompose_layer import decompose_layer
from decompose_resnet import decompose_resnet
from utils import load_model

RESNETS = {
    "cifar10_resnet20",
    "cifar10_resnet32",
    "cifar10_resnet56",
    "cifar10_resnet110",
    "imagenet_resnet18",
    "imagenet_resnet34",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-net", type=str, required=True, help="net type")
    parser.add_argument(
        "-weights",
        type=str,
        required=False,
        help="the weights file of the baseline network",
    )
    parser.add_argument("-p", type=float, default=False, help="RSDTR precision")
    args = parser.parse_args()
    p = Path(__file__)
    if args.net == "imagenet_resnet34":
        net = models.resnet34(True)
    elif args.net == "imagenet_resnet18":
        net = models.resnet18(True)
    else:
        net = load_model(args.weights)

    if args.net in RESNETS:
        decompose_resnet(args, net)
    else:
        for key, layer in net.features._modules.items():
            if isinstance(layer, nn.modules.conv.Conv2d) and key != "0":
                decomposed = decompose_layer(layer, args.p)
                net.features._modules[key] = decomposed

    Path(f"{p.parent}/decomposed_weights/").mkdir(parents=True, exist_ok=True)
    checkpoint = {"model": net, "state_dict": net.state_dict()}
    tmp = str(args.p)
    torch.save(
        checkpoint,
        f"{p.parent}/decomposed_weights/tr_{args.net}_{tmp.replace('.', '_')}.pth",
    )
