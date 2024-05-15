from tqdm import tqdm
from pathlib import Path
import torch.backends.cudnn as cudnn
from utils import load_model
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets, transforms as T

parser = argparse.ArgumentParser()
parser.add_argument("-lr", type=float, default=0.01, required=True)
parser.add_argument("-epochs", type=int, default=90, required=True)
parser.add_argument("-wd", type=float, default=1e-3, required=True)
parser.add_argument("-b", type=int, default=128, required=True)
parser.add_argument("-train", type=str, required=True, help="Train dataset folder")
parser.add_argument("-val", type=str, required=True, help="Test dataset folder")
parser.add_argument("-workers", type=int, default=4)
parser.add_argument("-weights", type=str, required=True)
parser.add_argument("-momentum", type=float, default=0.9, required=False)
args = parser.parse_args()

p = Path(__file__)
weight_path = f"{p.parent}/fine_tuned"
end = args.weights.split("/")[-1]

best_acc1 = 0

WORLD_SIZE = torch.cuda.device_count()
DIST_URL = "tcp://127.0.0.1:5000"
DIST_BACKEND = "nccl"


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


def train(trainloader, model, criterion, optimizer, scheduler, epoch, gpu, scaler):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with tqdm(trainloader, unit="batch") as tepoch:
        for inputs, targets in tepoch:
            optimizer.zero_grad(set_to_none=True)
            tepoch.set_description(f"TRAIN Epoch {epoch}")
            inputs, targets = inputs.cuda(gpu), targets.cuda(gpu)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            tepoch.set_postfix(loss=loss.item(), accuracy=acc1)

    print(
        f"TRAIN, Epoch: {epoch}, Avg. loss: {losses.avg:.4f}, Top-1: {top1.avg:.4f}, Top-5: {top5.avg:.4f}"
    )


def test(testloader, model, criterion, epoch, gpu):
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        with tqdm(testloader, unit="batch") as tepoch:
            for inputs, targets in tepoch:
                inputs, targets = inputs.cuda(gpu), targets.cuda(gpu)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1[0], inputs.size(0))
                top5.update(acc5[0], inputs.size(0))
                tepoch.set_postfix(loss=loss.item(), accuracy=acc1)

    print(
        f"TEST, Epoch: {epoch}, Avg. loss: {losses.avg:.4f}, Top-1: {top1.avg:.4f}, Top-5: {top5.avg:.4f}"
    )

    return top1.avg, top5.avg


def main():
    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1

    if gpu is not None:
        print("Use GPU: {} for training".format(gpu))

    rank = gpu
    dist.init_process_group(
        backend=DIST_BACKEND, init_method=DIST_URL, world_size=WORLD_SIZE, rank=rank
    )
    model = load_model(args.weights)

    if gpu is not None:
        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        args.b = int(args.b / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd
    )

    cudnn.benchmark = True

    traindir = args.train
    valdir = args.val
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize,
            ]
        ),
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.b,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir,
            T.Compose(
                [
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=args.b,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs
    )

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        train(train_loader, model, criterion, optimizer, scheduler, epoch, gpu, scaler)
        acc1, _ = test(val_loader, model, criterion, epoch, gpu)
        if best_acc1 < acc1 and rank == 0:
            checkpoint = {
                "model": model.module,
                "epoch": epoch,
                "state_dict": model.module.state_dict(),
                "best_top1": acc1,
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, f"{weight_path}/{end}")
            best_acc1 = acc1


if __name__ == "__main__":
    main()
