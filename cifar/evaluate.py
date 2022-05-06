from pathlib import Path
import argparse
import json
import os
import random
import signal
import sys
import time
import urllib
from torch import nn, optim
from torchvision import models, datasets, transforms
import torch
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision

parser = argparse.ArgumentParser(description='Evaluate resnet50 features on ImageNet')
parser.add_argument('--data', type=Path, metavar='DIR', default="/mnt/lustre/zhangshaofeng/workspace/dataset/",
                    help='path to dataset')
parser.add_argument('--pretrained', type=Path, metavar='FILE', default="/mnt/lustre/zhangshaofeng/workspace/zerocl32/checkpoint/resnet18-cifar100-256-2048.pth",
                    help='path to pretrained model')
parser.add_argument('--weights', default='freeze', type=str,
                    choices=('finetune', 'freeze'),
                    help='finetune or freeze resnet weights')
parser.add_argument('--train-percent', default=100, type=int,
                    choices=(100, 10, 1),
                    help='size of traing set in percent')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr-backbone', default=0.0, type=float, metavar='LR',
                    help='backbone base learning rate')
parser.add_argument('--lr-classifier', default=0.1, type=float, metavar='LR',
                    help='classifier base learning rate')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='/mnt/lustre/zhangshaofeng/workspace/zerocl32/checkpoint_eval/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')


def main():
    args = parser.parse_args()
    if args.train_percent in {1, 10}:
        args.train_files = urllib.request.urlopen(f'https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/{args.train_percent}percent.txt').readlines()
    args.ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_JOB_ID' in os.environ:
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
    # single-node distributed training
    args.rank = 0
    args.dist_url = f'tcp://localhost:{random.randrange(49152, 65535)}'
    args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):

    if 'cifar100' in str(args.pretrained):
        dataset_name = 'cifar100'
    else:
        dataset_name = 'cifar10'
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    resume = str(args.pretrained).replace("checkpoint", "checkpoint_eval")
    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(resume.replace('pth', 'txt'), 'a', buffering=1)
        print(' '.join(sys.argv))

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    # model = models.resnet.resnet50()
    model = models.resnet.resnet18()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    model.maxpool = nn.Identity()
    if dataset_name == 'cifar100':
        model.fc = nn.Linear(512, 100)
    else:
        model.fc = nn.Linear(512, 10)
    model = model.cuda(gpu)
    state_dict = torch.load(args.pretrained, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []

    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    if args.weights == 'freeze':
        model.requires_grad_(False)
        model.fc.requires_grad_(True)
    classifier_parameters, model_parameters = [], []
    for name, param in model.named_parameters():
        if name in {'fc.weight', 'fc.bias'}:
            classifier_parameters.append(param)
        else:
            model_parameters.append(param)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    criterion = nn.CrossEntropyLoss().cuda(gpu)

    param_groups = [dict(params=classifier_parameters, lr=args.lr_classifier)]
    if args.weights == 'finetune':
        param_groups.append(dict(params=model_parameters, lr=args.lr_backbone))
    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # automatically resume from checkpoint if it exists
    try:
        ckpt = torch.load(resume, map_location='cpu')
        start_epoch = ckpt['epoch']
        best_acc = ckpt['best_acc']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
    except:
        start_epoch = 0
        best_acc = argparse.Namespace(top1=0, top5=0)

    # Data loading code
    traindir = args.data
    valdir = args.data
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))

    if dataset_name == 'cifar100':
        train_dataset = CIFAR100(traindir, True, transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
        val_dataset = CIFAR100(valdir, False, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    else:
        train_dataset = CIFAR10(traindir, True, transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
        val_dataset = CIFAR10(valdir, False, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    if args.train_percent in {1, 10}:
        train_dataset.samples = []
        for fname in args.train_files:
            fname = fname.decode().strip()
            cls = fname.split('_')[0]
            train_dataset.samples.append(
                (traindir / cls / fname, train_dataset.class_to_idx[cls]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    kwargs = dict(batch_size=args.batch_size // args.world_size, num_workers=args.workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        # train
        if args.weights == 'finetune':
            model.train()
        elif args.weights == 'freeze':
            model.eval()
        else:
            assert False
        train_sampler.set_epoch(epoch)
        for step, (images, target) in enumerate(train_loader, start=epoch * len(train_loader)):
            output = model(images.cuda(gpu, non_blocking=True))
            loss = criterion(output, target.cuda(gpu, non_blocking=True))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % args.print_freq == 0:
                torch.distributed.reduce(loss.div_(args.world_size), 0)
                if args.rank == 0:
                    pg = optimizer.param_groups
                    lr_classifier = pg[0]['lr']
                    lr_backbone = pg[1]['lr'] if len(pg) == 2 else 0
                    stats = dict(epoch=epoch, step=step, lr_backbone=lr_backbone,
                                 lr_classifier=lr_classifier, loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)

        # evaluate
        model.eval()
        if args.rank == 0:
            top1 = AverageMeter('Acc@1')
            top5 = AverageMeter('Acc@5')
            with torch.no_grad():
                for images, target in val_loader:
                    output = model(images.cuda(gpu, non_blocking=True))
                    acc1, acc5 = accuracy(output, target.cuda(gpu, non_blocking=True), topk=(1, 5))
                    top1.update(acc1[0].item(), images.size(0))
                    top5.update(acc5[0].item(), images.size(0))
            best_acc.top1 = max(best_acc.top1, top1.avg)
            best_acc.top5 = max(best_acc.top5, top5.avg)
            stats = dict(epoch=epoch, acc1=top1.avg, acc5=top5.avg, best_acc1=best_acc.top1, best_acc5=best_acc.top5)
            print(json.dumps(stats))
            print(json.dumps(stats), file=stats_file)

        # sanity check
        # if args.weights == 'freeze':
        #     reference_state_dict = torch.load(args.pretrained, map_location='cpu')
        #     model_state_dict = model.module.state_dict()
        #     for k in reference_state_dict:
        #         assert torch.equal(model_state_dict[k].cpu(), reference_state_dict[k]), k

        scheduler.step()
        if args.rank == 0:
            state = dict(
                epoch=epoch + 1, best_acc=best_acc, model=model.state_dict(),
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
            torch.save(state, resume)


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass

def adjust_learning_rate(optimizer, epoch):
    """Decay the learning rate based on schedule"""
    lr = 0.1
    if epoch>=60 and epoch<=80:
        lr = 0.01
    elif epoch >= 80:
        lr = 0.001
    optimizer.param_groups[0]['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
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


if __name__ == '__main__':
    main()
