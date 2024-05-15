from decompose_layer import decompose_layer


def decompose_resnet(args, net):
    if args.net in {
        "cifar10_resnet20",
        "cifar10_resnet32",
        "cifar10_resnet56",
    }:
        layers = [net.layer1, net.layer2, net.layer3]
    else:
        layers = [net.layer1, net.layer2, net.layer3, net.layer4]

    for layer in layers:
        for block in layer:
            conv1 = block.conv1
            decomposed_conv1 = decompose_layer(conv1, args.p)
            block.conv1 = decomposed_conv1
            conv2 = block.conv2
            decomposed_conv2 = decompose_layer(conv2, args.p)
            block.conv2 = decomposed_conv2
