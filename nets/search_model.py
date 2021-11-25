#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 22:39:45 2021

@author: root
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.operations import FactorizedReduce, ReLUConvBN, MixedLayer, MixedLayer_PC

from genotypes import PRIMITIVES, Genotype


class Cell(nn.Module):

  def __init__(self, num_nodes, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, conv_func):
    """
    :param num_nodes: 4, number of layers inside a cell
    :param multiplier: 4
    :param C_prev_prev: 48
    :param C_prev: 48
    :param C: 16
    :param reduction: indicates whether to reduce the output maps width
    :param reduction_prev: when previous cell reduced width, s1_d = s0_d//2
    in order to keep same shape between s1 and s0, we adopt prep0 layer to
    reduce the s0 width by half.
    """
    super(Cell, self).__init__()

    # indicating current cell is reduction or not
    self.reduction = reduction
    self.reduction_prev = reduction_prev

    # preprocess0 deal with output from prev_prev cell
    if reduction_prev:
      # if prev cell has reduced channel/double width,
      # it will reduce width by half
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, conv_func, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, kernel_size=1,
                                    stride=1, padding=0, conv_func=conv_func, affine=False)
    # preprocess1 deal with output from prev cell
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, conv_func=conv_func, affine=False)

    # steps inside a cell
    self.num_nodes = num_nodes  # 4
    self.multiplier = multiplier  # 4

    self.layers = nn.ModuleList()

    for i in range(self.num_nodes):
      # for each i inside cell, it connects with all previous output
      # plus previous two cells' output
      for j in range(2 + i):
        # for reduction cell, it will reduce the heading 2 inputs only
        stride = 2 if reduction and j < 2 else 1
        layer = MixedLayer(C, stride, op_names=PRIMITIVES, conv_func=conv_func)
        self.layers.append(layer)

  def forward(self, s0, s1, weights):
    """
    :param s0:
    :param s1:
    :param weights: [14, 8]
    :return:
    """
    # print('s0:', s0.shape,end='=>')
    s0 = self.preprocess0(s0)  # [40, 48, 32, 32], [40, 16, 32, 32]
    # print(s0.shape, self.reduction_prev)
    # print('s1:', s1.shape,end='=>')
    s1 = self.preprocess1(s1)  # [40, 48, 32, 32], [40, 16, 32, 32]
    # print(s1.shape)

    states = [s0, s1]
    offset = 0
    # for each node, receive input from all previous intermediate nodes and s0, s1
    for i in range(self.num_nodes):  # 4
      # [40, 16, 32, 32]
      s = sum(self.layers[offset + j](h, weights[offset + j])
              for j, h in enumerate(states)) / len(states)
      offset += len(states)
      # append one state since s is the elem-wise addition of all output
      states.append(s)
      # print('node:',i, s.shape, self.reduction)

    # concat along dim=channel
    return torch.cat(states[-self.multiplier:], dim=1)  # 6 of [40, 16, 32, 32]

class Cell_PC(nn.Module):

  def __init__(self, num_nodes, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, conv_func):
    """
    :param num_nodes: 4, number of layers inside a cell
    :param multiplier: 4
    :param C_prev_prev: 48
    :param C_prev: 48
    :param C: 16
    :param reduction: indicates whether to reduce the output maps width
    :param reduction_prev: when previous cell reduced width, s1_d = s0_d//2
    in order to keep same shape between s1 and s0, we adopt prep0 layer to
    reduce the s0 width by half.
    """
    super(Cell_PC, self).__init__()

    # indicating current cell is reduction or not
    self.reduction = reduction
    self.reduction_prev = reduction_prev

    # preprocess0 deal with output from prev_prev cell
    if reduction_prev:
      # if prev cell has reduced channel/double width,
      # it will reduce width by half
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, conv_func, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, kernel_size=1,
                                    stride=1, padding=0, conv_func=conv_func, affine=False)
    # preprocess1 deal with output from prev cell
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, conv_func=conv_func, affine=False)

    # steps inside a cell
    self.num_nodes = num_nodes  # 4
    self.multiplier = multiplier  # 4

    self.layers = nn.ModuleList()

    for i in range(self.num_nodes):
      # for each i inside cell, it connects with all previous output
      # plus previous two cells' output
      for j in range(2 + i):
        # for reduction cell, it will reduce the heading 2 inputs only
        stride = 2 if reduction and j < 2 else 1
        layer = MixedLayer_PC(C, stride, op_names=PRIMITIVES, conv_func=conv_func)
        self.layers.append(layer)

  def forward(self, s0, s1, weights, weights2):
    """
    :param s0:
    :param s1:
    :param weights: [14, 8]
    :return:
    """
    # print('s0:', s0.shape,end='=>')
    s0 = self.preprocess0(s0)  # [40, 48, 32, 32], [40, 16, 32, 32]
    # print(s0.shape, self.reduction_prev)
    # print('s1:', s1.shape,end='=>')
    s1 = self.preprocess1(s1)  # [40, 48, 32, 32], [40, 16, 32, 32]
    # print(s1.shape)

    states = [s0, s1]
    offset = 0
    # for each node, receive input from all previous intermediate nodes and s0, s1
    for i in range(self.num_nodes):  # 4
      # [40, 16, 32, 32]
      # import pdb; pdb.set_trace()
      s = sum(weights2[offset+j]*self.layers[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      # append one state since s is the elem-wise addition of all output
      states.append(s)
      # print('node:',i, s.shape, self.reduction)

    # concat along dim=channel
    return torch.cat(states[-self.multiplier:], dim=1)  # 6 of [40, 16, 32, 32]

class Network(nn.Module):
  """
  stack number:layer of cells and then flatten to fed a linear layer
  """

  def __init__(self, C, num_cells, conv_func, 
               num_nodes=4, multiplier=4, stem_multiplier=3, num_classes=10, img_channel=3):
    """
    :param C: 16
    :param num_cells: number of cells of current network
    :param num_nodes: nodes num inside cell
    :param multiplier: output channel of cell = multiplier * ch
    :param stem_multiplier: output channel of stem net = stem_multiplier * ch
    :param num_classes: 10
    """
    super(Network, self).__init__()

    self.C = C
    self.num_classes = num_classes
    self.num_cells = num_cells
    self.num_nodes = num_nodes
    self.multiplier = multiplier
    self.conv_func = conv_func

    # stem_multiplier is for stem network,
    # and multiplier is for general cell
    C_curr = stem_multiplier * C  # 3*16
    # stem network, convert 3 channel to c_curr
    self.stem = nn.Sequential(  # 3 => 48
      nn.Conv2d(img_channel, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr))

    # c_curr means a factor of the output channels of current cell
    # output channels = multiplier * c_curr
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C  # 48, 48, 16
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(num_cells):

      # for layer in the middle [1/3, 2/3], reduce via stride=2
      if i in [num_cells // 3, 2 * num_cells // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False

      # [cp, h, h] => [multiplier*c_curr, h/h//2, h/h//2]
      # the output channels = multiplier * c_curr
      cell = Cell(num_nodes, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.conv_func, pc=False)
      # update reduction_prev
      reduction_prev = reduction

      self.cells += [cell]

      C_prev_prev, C_prev = C_prev, multiplier * C_curr

    # adaptive pooling output size to 1x1
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    # since cp records last cell's output channels
    # it indicates the input channel number
    self.classifier = nn.Linear(C_prev, num_classes)

    # k is the total number of edges inside single cell, 14
    k = sum(1 for i in range(self.num_nodes) for j in range(2 + i))
    num_ops = len(PRIMITIVES)  # 8

    self.alpha_normal = nn.Parameter(torch.randn(k, num_ops))
    self.alpha_reduce = nn.Parameter(torch.randn(k, num_ops))
    with torch.no_grad():
      # initialize to smaller value
      self.alpha_normal.mul_(1e-3)
      self.alpha_reduce.mul_(1e-3)
    self._arch_parameters = [self.alpha_normal, self.alpha_reduce]

  def forward(self, x):
    """
    in: torch.Size([3, 3, 32, 32])
    stem: torch.Size([3, 48, 32, 32])
    cell: 0 torch.Size([3, 64, 32, 32]) False
    cell: 1 torch.Size([3, 64, 32, 32]) False
    cell: 2 torch.Size([3, 128, 16, 16]) True
    cell: 3 torch.Size([3, 128, 16, 16]) False
    cell: 4 torch.Size([3, 128, 16, 16]) False
    cell: 5 torch.Size([3, 256, 8, 8]) True
    cell: 6 torch.Size([3, 256, 8, 8]) False
    cell: 7 torch.Size([3, 256, 8, 8]) False
    pool:   torch.Size([16, 256, 1, 1])
    linear: [b, 10]
    :param x:
    :return:
    """
    # print('in:', x.shape)
    # s0 & s1 means the last cells' output
    s0 = s1 = self.stem(x)  # [b, 3, 32, 32] => [b, 48, 32, 32]
    # print('stem:', s0.shape)

    for i, cell in enumerate(self.cells):
      # architecture weights are shared across all reduction cell or normal cell
      # according to current cell's type, it choose which architecture parameters
      # to use
      if cell.reduction:  # if current cell is reduction cell
        weights = F.softmax(self.alpha_reduce, dim=-1)
      else:
        weights = F.softmax(self.alpha_normal, dim=-1)  # [14, 8]
      # execute cell() firstly and then assign s0=s1, s1=result
      s0, s1 = s1, cell(s0, s1, weights)  # [40, 64, 32, 32]
      # print('cell:',i, s1.shape, cell.reduction, cell.reduction_prev)
      # print('\n')

    # s1 is the last cell's output
    out = self.global_pooling(s1)
    # print('pool', out.shape)
    logits = self.classifier(out.view(out.size(0), -1))

    return logits

  def genotype(self):
    def _parse(weights):
      """
      :param weights: [14, 8]
      :return:
      """
      gene = []
      n = 2
      start = 0
      for i in range(self.num_nodes):  # for each node
        end = start + n
        W = weights[start:end].copy()  # shape=[2, 8], [3, 8], [4, 8], [5, 8]
        # i+2 is the number of connection for node i
        # sort by descending order, get strongest 2 edges
        # note here we assume the 0th op is none op, if it's not the case this will be wrong!
        edges = np.argsort(-np.max(W[:, 1:], axis=1))[:2]
        ops = np.argmax(W[edges, 1:], axis=1) + 1
        gene += [(PRIMITIVES[op], edge) for op, edge in zip(ops, edges)]
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alpha_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alpha_reduce, dim=-1).data.cpu().numpy())

    concat = range(2 + self.num_nodes - self.multiplier, self.num_nodes + 2)
    genotype = Genotype(normal=gene_normal, normal_concat=concat,
                        reduce=gene_reduce, reduce_concat=concat)

    return genotype

  def complexity_loss(self):
        size_product = []
        loss = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                loss += m.complexity_loss()
                size_product += [m.size_product]
        normalizer = size_product[0].item()
        loss /= normalizer
        return loss

class Network_PC(nn.Module):
  """
  stack number:layer of cells and then flatten to fed a linear layer
  """

  def __init__(self, C, num_cells, conv_func, 
               num_nodes=4, multiplier=4, stem_multiplier=3, num_classes=10, img_channel=3):
    """
    :param C: 16
    :param num_cells: number of cells of current network
    :param num_nodes: nodes num inside cell
    :param multiplier: output channel of cell = multiplier * ch
    :param stem_multiplier: output channel of stem net = stem_multiplier * ch
    :param num_classes: 10
    """
    super(Network_PC, self).__init__()

    self.C = C
    self.num_classes = num_classes
    self.num_cells = num_cells
    self.num_nodes = num_nodes
    self.multiplier = multiplier
    self.conv_func = conv_func

    # stem_multiplier is for stem network,
    # and multiplier is for general cell
    C_curr = stem_multiplier * C  # 3*16
    # stem network, convert 3 channel to c_curr
    self.stem = nn.Sequential(  # 3 => 48
      nn.Conv2d(img_channel, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr))

    # c_curr means a factor of the output channels of current cell
    # output channels = multiplier * c_curr
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C  # 48, 48, 16
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(num_cells):

      # for layer in the middle [1/3, 2/3], reduce via stride=2
      if i in [num_cells // 3, 2 * num_cells // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False

      # [cp, h, h] => [multiplier*c_curr, h/h//2, h/h//2]
      # the output channels = multiplier * c_curr
      # import pdb; pdb.set_trace()
      cell = Cell_PC(num_nodes, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.conv_func)
      # update reduction_prev
      reduction_prev = reduction

      self.cells += [cell]

      C_prev_prev, C_prev = C_prev, multiplier * C_curr

    # adaptive pooling output size to 1x1
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    # since cp records last cell's output channels
    # it indicates the input channel number
    self.classifier = nn.Linear(C_prev, num_classes)

    # k is the total number of edges inside single cell, 14
    k = sum(1 for i in range(self.num_nodes) for j in range(2 + i))
    num_ops = len(PRIMITIVES)  # 8

    self.alpha_normal = nn.Parameter(torch.randn(k, num_ops))
    self.alpha_reduce = nn.Parameter(torch.randn(k, num_ops))
    self.beta_normal = nn.Parameter(torch.randn(k))
    self.beta_reduce = nn.Parameter(torch.randn(k))
    
    with torch.no_grad():
      # initialize to smaller value
      self.alpha_normal.mul_(1e-3)
      self.alpha_reduce.mul_(1e-3)
      self.beta_normal.mul_(1e-3)
      self.beta_reduce.mul_(1e-3)
      
    self._arch_parameters = [self.alpha_normal, self.alpha_reduce, self.beta_normal, self.beta_reduce]

  def arch_parameters(self):
    return self._arch_parameters

  def forward(self, x):
    """
    in: torch.Size([3, 3, 32, 32])
    stem: torch.Size([3, 48, 32, 32])
    cell: 0 torch.Size([3, 64, 32, 32]) False
    cell: 1 torch.Size([3, 64, 32, 32]) False
    cell: 2 torch.Size([3, 128, 16, 16]) True
    cell: 3 torch.Size([3, 128, 16, 16]) False
    cell: 4 torch.Size([3, 128, 16, 16]) False
    cell: 5 torch.Size([3, 256, 8, 8]) True
    cell: 6 torch.Size([3, 256, 8, 8]) False
    cell: 7 torch.Size([3, 256, 8, 8]) False
    pool:   torch.Size([16, 256, 1, 1])
    linear: [b, 10]
    :param x:
    :return:
    """
    # print('in:', x.shape)
    # s0 & s1 means the last cells' output
    s0 = s1 = self.stem(x)  # [b, 3, 32, 32] => [b, 48, 32, 32]
    # print('stem:', s0.shape)
          
    for i, cell in enumerate(self.cells):
      # architecture weights are shared across all reduction cell or normal cell
      # according to current cell's type, it choose which architecture parameters
      # to use
      if cell.reduction:  # if current cell is reduction cell
        weights = F.softmax(self.alpha_reduce, dim=-1)
        n = 3
        start = 2
        weights2 = F.softmax(self.beta_reduce[0:2], dim=-1)
        for i in range(self.num_nodes-1):
          end = start + n
          tw2 = F.softmax(self.beta_reduce[start:end], dim=-1)
          start = end
          n += 1
          weights2 = torch.cat([weights2,tw2],dim=0)
      else:
        weights = F.softmax(self.alpha_normal, dim=-1)  # [14, 8]
        n = 3
        start = 2
        weights2 = F.softmax(self.beta_normal[0:2], dim=-1)
        for i in range(self.num_nodes-1):
          end = start + n
          tw2 = F.softmax(self.beta_normal[start:end], dim=-1)
          start = end
          n += 1
          weights2 = torch.cat([weights2,tw2],dim=0)
      # execute cell() firstly and then assign s0=s1, s1=result
      # import pdb; pdb.set_trace()
      s0, s1 = s1, cell(s0, s1, weights, weights2)  # [40, 64, 32, 32]
      # print('cell:',i, s1.shape, cell.reduction, cell.reduction_prev)
      # print('\n')

    # s1 is the last cell's output
    out = self.global_pooling(s1)
    # print('pool', out.shape)
    logits = self.classifier(out.view(out.size(0), -1))

    return logits

  def genotype(self):

    def _parse(weights,weights2):
      gene = []
      n = 2
      start = 0
      for i in range(self.num_nodes):
        end = start + n
        W = weights[start:end].copy()
        W2 = weights2[start:end].copy()
        for j in range(n):
          W[j,:]=W[j,:]*W2[j]
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        ops = np.argmax(W[edges, 1:], axis=1) + 1
        gene += [(PRIMITIVES[op], edge) for op, edge in zip(ops, edges)]
        start = end
        n += 1
      return gene
  
    n = 3
    start = 2
    weightsr2 = F.softmax(self.beta_reduce[0:2], dim=-1)
    weightsn2 = F.softmax(self.beta_normal[0:2], dim=-1)
    for i in range(self._steps-1):
      end = start + n
      tw2 = F.softmax(self.beta_reduce[start:end], dim=-1)
      tn2 = F.softmax(self.beta_normal[start:end], dim=-1)
      start = end
      n += 1
      weightsr2 = torch.cat([weightsr2,tw2],dim=0)
      weightsn2 = torch.cat([weightsn2,tn2],dim=0)
    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(),weightsn2.data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(),weightsr2.data.cpu().numpy())

    concat = range(2+self.num_nodes-self._multiplier, self.num_nodes+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

  def complexity_loss(self):
        size_product = []
        loss = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                loss += m.complexity_loss()
                size_product += [m.size_product]
        normalizer = size_product[0].item()
        loss /= normalizer
        return loss


# if __name__ == '__main__':
#   import numpy as np
#   from utils.utils import create_logger
#
#
#   def hook(self, input, output):
#     # print(output.data.cpu().numpy().shape)
#     pass
#
#
#   logger = create_logger(0)
#   net = Network(16, 8, 4)
#   print(net.genotype())
#   logger.info(net.genotype())
#   print(net.genotype())
#
#   for m in net.modules():
#     if isinstance(m, nn.Conv2d):
#       m.register_forward_hook(hook)
#
#   y = net(torch.randn(1, 3, 32, 32))
#   print(y.size())
#
#   sep_size = 0
#   for k, v in net.named_parameters():
#     print('%s: %f MB' % (k, v.numel() / 1024 / 1024))
#     if '4.op' in k or '5.op' in k:
#       sep_size += v.numel() / 1024 / 1024
#   print("Sep conv size = %f MB" % sep_size)
#   print("Total param size = %f MB" % (sum(v.numel() for v in net.parameters()) / 1024 / 1024))