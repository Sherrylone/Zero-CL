from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import numpy as np
import torch.multiprocessing as mp
from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
import torch.distributed as dist
import random
from dataset import ImageNet

parser = argparse.ArgumentParser(description='ZeroCL Training')
parser.add_argument('--data', type=Path, metavar='DIR', default="../SimCLR/data",
                    help='path to dataset')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.3, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='/mnt/lustre/zhangshaofeng/workspace/whitening_224/checkpoint-large/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--mode', default='fea', type=str,
                    metavar='INS', help='instance whitening or feature whitening')
parser.add_argument('--backbone', default='resnet18', type=str, help='backbone network')

def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_JOB_ID' in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
        args.dist_url = f'tcp://{host_name}:58472'
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = 'tcp://localhost:58472'
        args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)
    # main_worker(int(os.environ["SLURM_PROCID"]), args)

def main_worker(gpu, args):
    hidden = args.projector.split('-')[0]
    gpu = int(gpu % args.ngpus_per_node)
    args.rank += gpu
    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / ('stats-'+str(args.mode)+'-hidden-'+str(hidden)+'-batch-'+str(args.batch_size)+'.txt'), 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    torch.cuda.set_device(gpu)

    model = ZeroCL(args)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda(gpu)
    world_size = int(os.environ['SLURM_NTASKS'])
    num_gpus = torch.cuda.device_count() * int(os.getenv('SLURM_NNODES'))
    print("Number of GPUS: %d, GPU id: %d, number of nodes: %d" % (num_gpus, gpu, int(os.getenv('SLURM_NNODES'))))

    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=exclude_bias_and_norm,
                     lars_adaptation_filter=exclude_bias_and_norm)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / ('checkpoint-'+str(args.mode)+'-hidden-'+str(hidden)+'-batch-'+str(args.batch_size)+'.pth')).is_file():
        ckpt = torch.load(args.checkpoint_dir / ('checkpoint-'+str(args.mode)+'-hidden-'+str(hidden)+'-batch-'+str(args.batch_size)+'.pth'),
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0

    dataset = ImageNet('/mnt/lustre/share/images/train/', Transform())
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    assert args.batch_size % world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    print("Property: ", gpu, int(os.getenv('SLURM_NNODES')), args.ngpus_per_node, args.world_size, args.batch_size, per_device_batch_size)
    print("Samples per GPUs", per_device_batch_size)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=16,
        pin_memory=True, sampler=sampler)

    start_time = time.time()
    print("Start training...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, (y1, y2, label) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            loss = model(y1, y2)
            loss.backward()
            optimizer.step()
            if step % args.print_freq == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_biases=optimizer.param_groups[1]['lr'],
                                 loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
        # save checkpoint
        if args.rank == 0:
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / ('checkpoint-'+str(args.mode)+'-hidden-'+str(hidden)+'-batch-'+str(args.batch_size)+'.pth'))
            torch.save(model.module.backbone.state_dict(),
                       args.checkpoint_dir / ('resnet18-'+str(args.mode)+'-hidden-'+str(hidden)+'-batch-'+str(args.batch_size)+'.pth'))


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class ZeroCL(nn.Module):
    def __init__(self, args):
        super(ZeroCL, self).__init__()
        self.args = args
        if args.backbone == 'resnet18':
            self.backbone = torchvision.models.resnet18(zero_init_residual=True)
        elif args.backbone == 'resnet50':
            self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        elif args.backbone == 'resnet101':
            self.backbone = torchvision.models.resnet101(zero_init_residual=True)
        else:
            self.backbone = torchvision.models.resnet152(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [2048] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))
        if self.args.mode == 'ins':
            loss = ins_loss(z1, z2, self.args.world_size)
        elif self.args.mode == 'fea':
            loss = fea_loss(z1, z2, self.args.world_size)
        else:
            loss = barlow_twins_loss(z1, z2, self.args.world_size)
        return loss

def ins_loss(z1, z2, world_size):
    hidden = z1.size(1)
    whiten_net = WhitenTran()
    z1 = standardization(z1)
    z2 = standardization(z2)
    z1 = whiten_net.zca_forward(z1.transpose(0, 1))  # d * N
    z2 = whiten_net.zca_forward(z2.transpose(0, 1))
    c = torch.mm(z1.transpose(0, 1), z2)  # N * N
    c.div_(hidden * world_size)
    torch.distributed.all_reduce(c)
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    loss = on_diag
    return loss

def barlow_twins_loss(z1, z2, world_size, lambd=0.0051):
    batch_size = z1.size(0)
    z1 = standardization(z1, dim=0)
    z2 = standardization(z2, dim=0)
    c = torch.mm(z1.transpose(0, 1), z2)

    # sum the cross-correlation matrix between all gpus
    c.div_(batch_size * world_size)
    torch.distributed.all_reduce(c)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + (lambd * 8192 / 256) * off_diag
    return loss

def fea_loss(z1, z2, world_size):
    batch_size = z1.size(0)
    whiten_net = WhitenTran()
    z1 = standardization(z1, dim=0)
    z2 = standardization(z2, dim=0)
    z1 = whiten_net.zca_forward(z1)  # N * d
    z2 = whiten_net.zca_forward(z2)
    c = torch.mm(z1.transpose(0, 1), z2)
    c.div_(batch_size * world_size)
    torch.distributed.all_reduce(c)
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    loss = on_diag
    return loss

def standardization(data, dim=-1):
    # N * d
    mu = torch.mean(data, dim=dim, keepdim=True)
    sigma = torch.std(data, dim=dim, keepdim=True)
    return (data - mu) / (sigma + 1e-4)

class WhitenTran(nn.Module):
    def __init__(self, eps=0.01, dim=256):
        super(WhitenTran, self).__init__()
        self.eps = eps
        self.dim = dim

    def pca_forward(self, x):
        """normalized tensor"""
        batch_size, feature_dim = x.size()
        f_cov = torch.mm(x.transpose(0, 1), x) / batch_size # d * d
        eye = torch.eye(feature_dim).float().to(f_cov.device)
        f_cov_shrink = (1 - self.eps) * f_cov + self.eps * eye
        inv_sqrt = torch.triangular_solve(eye, torch.linalg.cholesky(f_cov_shrink.float()), upper=False)[0]
        inv_sqrt = inv_sqrt.contiguous().view(feature_dim, feature_dim).detach()
        return torch.mm(inv_sqrt, x.transpose(0, 1)).transpose(0, 1)    # N * d

    def zca_forward(self, x):
        batch_size, feature_dim = x.size()
        f_cov = torch.mm(x.transpose(0, 1), x) / batch_size  # d * d
        U, S, V = torch.svd(f_cov)
        diag = torch.diag(1.0 / torch.sqrt(S + self.eps))
        rotate_mtx = torch.mm(torch.mm(U, diag), U.transpose(0, 1)).detach()  # d * d
        return torch.mm(rotate_mtx, x.transpose(0, 1)).transpose(0, 1)  # N * d


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.02,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


def exclude_bias_and_norm(p):
    return p.ndim == 1


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    # def __call__(self, x):
    #     y1 = self.transform(x)
    #     y2 = self.transform_prime(x)
    #     return y1, y2


if __name__ == '__main__':
    main()
