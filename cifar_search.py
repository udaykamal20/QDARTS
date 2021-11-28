#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 19:47:34 2021

@author: root
"""

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torchvision.datasets as dset

from nets.search_model import Network_PC

from utils.preprocessing import cifar_search_transform
from utils.second_order_update import *
from utils.summary import create_summary, create_logger
import numpy as np
from nets.quant_modules import MixActivConv2d
import shutil

torch.backends.cudnn.benchmark = True

# Training settings
parser = argparse.ArgumentParser(description='darts')

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--dist', type=bool, default=False)

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_name', type=str, default='CIFAR10')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')

parser.add_argument('--order', type=str, default='1st', choices=['1st', '2nd'])

parser.add_argument('--w_lr', type=float, default=0.1)
parser.add_argument('--w_min_lr', type=float, default=0.0)
parser.add_argument('--w_wd', type=float, default=3e-4)

parser.add_argument('--a_lr', type=float, default=6e-4)
parser.add_argument('--a_wd', type=float, default=1e-3)
parser.add_argument('--a_start', type=int, default=15)

parser.add_argument('--q_lr', type=float, default=0.01)
parser.add_argument('--q_wd', type=float, default=1e-3)
parser.add_argument('--q_start', type=int, default=-1)

parser.add_argument('--init_ch', type=int, default=16)
parser.add_argument('--num_cells', type=int, default=8)
parser.add_argument('--num_nodes', type=int, default=4)
parser.add_argument('--replica', type=int, default=1)

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--max_epochs', type=int, default=50)

parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--num_workers', type=int, default=4)

parser.add_argument('--complexity-decay', '--cd', default=1e-4, type=float,
                    metavar='W', help='complexity decay (default: 1e-4)')

parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

parser.add_argument('--pc', type=bool, default=True, help='partial channel')

cfg = parser.parse_args()

os.chdir(cfg.root_dir)

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus


def main():
  logger = create_logger(cfg.local_rank, save_dir=cfg.log_dir)
  summary_writer = create_summary(cfg.local_rank, log_dir=cfg.log_dir)
  print = logger.info

  print(cfg)
  num_gpus = torch.cuda.device_count()
  if cfg.dist:
    device = torch.device('cuda:%d' % cfg.local_rank) if cfg.dist else torch.device('cuda')
    torch.cuda.set_device(cfg.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=num_gpus, rank=cfg.local_rank)
  else:
    device = torch.device('cuda')

  print('==> Preparing data..')
  cifar = 100 if 'cifar100' in cfg.log_name else 10
  
  train_transform = cifar_search_transform(is_training=True)
  if cifar==10:
      train_data = dset.CIFAR10(root=cfg.data_dir, train=True, download=True, transform=train_transform)
  else:
      train_data = dset.CIFAR100(root=cfg.data_dir, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(cfg.train_portion * num_train))
    
  train_loader = torch.utils.data.DataLoader(
            train_data,batch_size=cfg.batch_size // num_gpus if cfg.dist else cfg.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            num_workers=cfg.num_workers,
            pin_memory=True)
    
  val_loader = torch.utils.data.DataLoader(
            train_data, batch_size=cfg.batch_size // num_gpus if cfg.dist else cfg.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            num_workers=cfg.num_workers,
            pin_memory=True)


  print('==> Building model..')

  if cfg.pc==True:
      model = Network_PC(C=cfg.init_ch, num_cells=cfg.num_cells,
                      num_nodes=cfg.num_nodes, multiplier=cfg.num_nodes, num_classes=cifar)
  else:
      raise ValueError('partial channel required to speed up training')      

  if not cfg.dist:
    # model = nn.DataParallel(model).to(device)
    model = model.to(device)

  else:
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[cfg.local_rank, ],
                                                output_device=cfg.local_rank)

  # proxy_model is used for 2nd order update
  if cfg.order == '2nd':
    proxy_model = Network_PC(cfg.init_ch, cfg.num_cells, cfg.num_nodes).cuda()

  # count_parameters(model)

  weights = [v for k, v in model.named_parameters() if 'alpha' not in k or 'beta' not in k or 'gamma' not in k]
  alphas = [v for v in model.arch_parameters()]
  betas = [v for k, v in model.named_parameters() if 'gamma' in k]
  

  optimizer_w = optim.SGD(weights, cfg.w_lr, momentum=0.9, weight_decay=cfg.w_wd)
  optimizer_a = optim.Adam(alphas, lr=cfg.a_lr, betas=(0.5, 0.999), weight_decay=cfg.a_wd)
  optimizer_q = optim.SGD(betas, lr=cfg.q_lr, momentum=(0.9), weight_decay=cfg.q_wd)
  
  criterion = nn.CrossEntropyLoss().cuda()
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_w, cfg.max_epochs, eta_min=cfg.w_min_lr)
  scheduler_q = optim.lr_scheduler.CosineAnnealingLR(optimizer_q, cfg.max_epochs, eta_min=cfg.w_min_lr)


  alphas = []
  best_acc1 = -100
  def train(epoch):
    model.train()
    print('\nEpoch: %d lr: %f' % (epoch, scheduler.get_lr()[0]))
    alphas.append([])
    start_time = time.time()

    for batch_idx, ((inputs_w, targets_w), (inputs_a, targets_a)) \
        in enumerate(zip(train_loader, val_loader)):

      inputs_w, targets_w = inputs_w.to(device), targets_w.to(device, non_blocking=True)
      inputs_a, targets_a = inputs_a.to(device), targets_a.to(device, non_blocking=True)

      # 1. update alpha
      if epoch > cfg.a_start:
        optimizer_a.zero_grad()

        if cfg.order == '1st':
          # using 1st order update
          outputs = model(inputs_a)
          val_loss = criterion(outputs, targets_a)
          val_loss.backward()
        else:
          # using 2nd order update
          val_loss = update(model, proxy_model, criterion, optimizer_w,
                            inputs_a, targets_a, inputs_w, targets_w)

        optimizer_a.step()
      else:
        val_loss = torch.tensor([0]).cuda()

      # 2. update weights
      outputs = model(inputs_w)
      cls_loss = criterion(outputs, targets_w)
      
      # 3. complexity penalty
      if cfg.complexity_decay != 0:
        if hasattr(model, 'module'):
            loss_complexity = cfg.complexity_decay * model.module.complexity_loss()
        else:
            loss_complexity = cfg.complexity_decay * model.complexity_loss()
      final_loss = cls_loss +  loss_complexity
      
      optimizer_w.zero_grad()
      optimizer_q.zero_grad()
      
      final_loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
      optimizer_w.step()
      optimizer_q.step()

      if batch_idx % cfg.log_interval == 0:
        step = len(train_loader) * epoch + batch_idx
        duration = time.time() - start_time

        print('[%d/%d - %d/%d] cls_loss: %5f val_loss: %5f complexity_loss: %5f (%d samples/sec)' %
              (epoch, cfg.max_epochs, batch_idx, len(train_loader),
               cls_loss.item(), val_loss.item(), loss_complexity.item(), cfg.batch_size * cfg.log_interval / duration))

        start_time = time.time()
        summary_writer.add_scalar('cls_loss', cls_loss.item(), step)
        summary_writer.add_scalar('val_loss', val_loss.item(), step)
        summary_writer.add_scalar('complexity_loss', loss_complexity.item(), step)
        summary_writer.add_scalar('learning rate', optimizer_w.param_groups[0]['lr'], step)

        alphas[-1].append(model.alpha_normal.detach().cpu().numpy())
        alphas[-1].append(model.alpha_reduce.detach().cpu().numpy())
    return

  def eval(epoch):
    model.eval()

    correct = 0
    total_loss = 0
    with torch.no_grad():
      for step, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)

        outputs = model(inputs)
        total_loss += criterion(outputs, targets).item()
        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(targets.data).cpu().sum().item()

      acc = 100. * correct / len(val_loader.dataset)
      total_loss = total_loss / len(val_loader)
      print('Val_loss==> %.5f Precision@1 ==> %.2f%% \n' % (total_loss, acc))
      summary_writer.add_scalar('Precision@1', acc, global_step=epoch)
      summary_writer.add_scalar('val_loss_per_epoch', total_loss, global_step=epoch)
    return acc

  for epoch in range(cfg.max_epochs):
    # train_sampler.set_epoch(epoch)
    # val_sampler.set_epoch(epoch)
    train(epoch)
    acc = eval(epoch)
    scheduler.step(epoch)
    scheduler_q.step(epoch)
    # print(model.module.genotype())
    is_best = acc > best_acc1
    best_acc1 = max(acc, best_acc1)
    if is_best:
        best_epoch = epoch

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer_w': optimizer_w.state_dict(),
        'optimizer_a': optimizer_a.state_dict(),
        'optimizer_q': optimizer_q.state_dict(),
        'best_epoch': best_epoch
    }, is_best, epoch)
    
    if cfg.local_rank == 0:
      torch.save(alphas, os.path.join(cfg.ckpt_dir, 'alphas.t7'))
      torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, 'search_checkpoint.t7'))
      torch.save({'genotype': model.genotype()}, os.path.join(cfg.ckpt_dir, 'genotype.t7'))

  summary_writer.close()

def save_checkpoint(state, is_best, epoch, filename=f'darts_checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'darts_model_best.pth.tar')
        
if __name__ == '__main__':
  if cfg.local_rank == 0:
    main()
