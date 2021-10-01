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

from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('--data', type=Path, metavar='DIR', required=True,
                    help='path to dataset')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=256, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.3, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.005, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='2048-2048-2048', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--off', default='false', type=str,
                    metavar='OFF', help='using off diagonal loss')
parser.add_argument('--whiten', default='false', type=str,
                    metavar='WHITEN', help='using off diagonal loss')
parser.add_argument('--dataset', default='cifar100', type=str,
                    metavar='dataset')

def main():
    args = parser.parse_args()
    main_worker(0, args)

def main_worker(gpu, args):
    batch_size = int(args.batch_size)
    hidden_dim = int(args.projector.split('-')[0])

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
    print(' '.join(sys.argv))
    print(' '.join(sys.argv), file=stats_file)

    torch.backends.cudnn.benchmark = True

    model = BarlowTwins(args).cuda(gpu)
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    model = torch.nn.parallel.DataParallel(model)
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=exclude_bias_and_norm,
                     lars_adaptation_filter=exclude_bias_and_norm)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / ('checkpoint-'+args.off+'batch-'+str(batch_size)+'-hidden-'+str(hidden_dim)+'.pth')).is_file():
        ckpt = torch.load(args.checkpoint_dir / ('checkpoint-'+args.off+'-batch-'+str(batch_size)+'-hidden-'+str(hidden_dim)+'.pth'),
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0
    if args.dataset == 'ciar100':
        dataset = CIFAR100(root=args.data, train=True, transform=TransformCIFAR(), download=True)
    else:
        dataset = CIFAR10(root=args.data, train=True, transform=TransformCIFAR(), download=True)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=args.workers, shuffle=True,
        pin_memory=True)

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        for step, ((y1, y2), _) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(y1, y2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % args.print_freq == 0:
                stats = dict(epoch=epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_biases=optimizer.param_groups[1]['lr'],
                                 loss=loss.item(),
                                 time=int(time.time() - start_time))
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
        # save checkpoint
        state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
        torch.save(state, args.checkpoint_dir / ('checkpoint-'+args.off+'-batch-'+str(batch_size)+'-hidden-'+str(hidden_dim)+'.pth'))
        torch.save(model.module.backbone.state_dict(),
                       args.checkpoint_dir / ('resnet18-fea_whiten-'+args.off+'-batch-'+str(batch_size)+'-hidden-'+str(hidden_dim)+'.pth'))

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


class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone = torchvision.models.resnet18(zero_init_residual=True)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [512] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        self.whiten_net = WhitenTran()
        self.whiten_bn = nn.BatchNorm1d(sizes[-1], affine=False)

    # feature level with instance mean-0
    def whitened_barlow_twins_loss(self, y1, y2):
        z1 = self.whiten_bn(self.projector(self.backbone(y1)))
        z2 = self.whiten_bn(self.projector(self.backbone(y2)))

        if self.args.whiten=='true':
            z1 = self.whiten_net.zca_forward(z1)    # N * d
            z2 = self.whiten_net.zca_forward(z2)

        c = torch.mm(z1.transpose(0, 1), z2)
        c.div_(self.args.batch_size)
        loss = torch.diagonal(c).add_(-1).pow_(2).sum()
        if self.args.off == 'true':
            off_diag = off_diagonal(c).pow_(2).sum()
            loss += self.args.lambd * off_diag
        return loss

    def forward(self, y1, y2):
        return self.whitened_barlow_twins_loss(y1, y2)

def standardization(data):
    # N * d
    mu = torch.mean(data, dim=-1, keepdim=True)
    sigma = torch.std(data, dim=-1, keepdim=True)
    return (data - mu) / sigma

class WhitenTran(nn.Module):
    def __init__(self, eps=0.01, dim=256):
        super(WhitenTran, self).__init__()
        self.eps = eps
        self.dim = dim

    def cholesky_forward(self, x):
        """normalized tensor"""
        batch_size, feature_dim = x.size()
        f_cov = torch.mm(x.transpose(0, 1), x) / (batch_size - 1) # d * d
        eye = torch.eye(feature_dim).float().to(f_cov.device)
        f_cov_shrink = (1 - self.eps) * f_cov + self.eps * eye
        inv_sqrt = torch.triangular_solve(eye, torch.linalg.cholesky(f_cov_shrink.float()), upper=False)[0]
        inv_sqrt = inv_sqrt.contiguous().view(feature_dim, feature_dim).detach()
        return torch.mm(inv_sqrt, x.transpose(0, 1)).transpose(0, 1)    # N * d

    def zca_forward(self, x):
        batch_size, feature_dim = x.size()
        eps = 0.1
        f_cov = (torch.mm(x.transpose(0, 1), x) / (batch_size - 1)).float()  # d * d
        eye = torch.eye(feature_dim).float().to(f_cov.device)
        # f_cov = torch.mm(x.transpose(0, 1), x).float()  # d * d
        U, S, V = torch.linalg.svd(0.9 * f_cov + 0.1 * eye)
        diag = torch.diag(1.0 / torch.sqrt(S+1e-5))
        rotate_mtx = torch.mm(torch.mm(U, diag), U.transpose(0, 1)).detach() # d * d
        return torch.mm(rotate_mtx, x.transpose(0, 1)).transpose(0, 1)  # N * d

    def pca_for(self, x):
        batch_size, feature_dim = x.size()
        f_cov = (torch.mm(x.transpose(0, 1), x) / (batch_size - 1)).float()  # d * d
        eye = torch.eye(feature_dim).float().to(f_cov.device)
        # f_cov = torch.mm(x.transpose(0, 1), x).float()  # d * d
        U, S, V = torch.linalg.svd(0.99 * f_cov + 0.01 * eye)
        diag = torch.diag(1.0 / torch.sqrt(S + 1e-5))
        rotate_mtx = torch.mm(diag, U.transpose(0, 1)).detach()  # d * d
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


class GaussianBlur:
    def __init__(self, sigma = [0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarization:

    def __call__(self, img: Image) -> Image:
        return ImageOps.solarize(img)

class TransformCIFAR:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.0),
            transforms.RandomApply([Solarization()], p=0.0),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.0),
            transforms.RandomApply([Solarization()], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2

class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(32, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.0),
            transforms.RandomApply([Solarization()], p=0.0),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(32, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.0),
            transforms.RandomApply([Solarization()], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


if __name__ == '__main__':
    main()
