import torch
import sys


def load_model(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint["model"]
    model.load_state_dict(checkpoint["state_dict"])

    return model


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_network(net_name):
    if net_name == "cifar10_resnet20":
        from models.resnet_cifar import resnet20

        net = resnet20()
    elif net_name == "cifar10_resnet32":
        from models.resnet_cifar import resnet32

        net = resnet32()
    elif net_name == "cifar10_resnet56":
        from models.resnet_cifar import resnet56

        net = resnet56()
    elif net_name == "cifar10_vggnet":
        from models.vggnet import vggnet_cifar10

        net = vggnet_cifar10()
    else:
        print("the network name you have entered is not supported yet")
        sys.exit()

    return net
