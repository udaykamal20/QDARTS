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

from nets.operations import FactorizedReduce, ReLUConvBN, MixedLayer_PC

from genotypes import PRIMITIVES, Genotype
from torch.nn.parameter import Parameter

gaussian_steps = {1: 1.596, 2: 0.996, 3: 0.586, 4: 0.336}
hwgq_steps = {1: 0.799, 2: 0.538, 3: 0.3217, 4: 0.185}


class _gauss_quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, step, bit):
        lvls = 2 ** bit / 2
        gamma = x.std().item()
        step *= gamma
        y = (torch.round(x/step+0.5)-0.5) * step
        thr = (lvls-0.5)*step
        y = y.clamp(min=-thr, max=thr)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class _gauss_quantize_resclaed_step(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, step, bit):
        lvls = 2 ** bit / 2
        y = (torch.round(x/step+0.5)-0.5) * step
        thr = (lvls-0.5)*step
        y = y.clamp(min=-thr, max=thr)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class _hwgq(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, step):
        y = torch.round(x / step) * step
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class HWGQ(nn.Module):
    def __init__(self, bit=2):
        super(HWGQ, self).__init__()
        self.bit = bit
        if bit < 32:
            self.step = hwgq_steps[bit]
        else:
            self.step = None

    def forward(self, x):
        if self.bit >= 32:
            return x.clamp(min=0.0)
        lvls = float(2 ** self.bit - 1)
        clip_thr = self.step * lvls
        y = x.clamp(min=0.0, max=clip_thr)
        return _hwgq.apply(y, self.step)



class MixQuantActiv(nn.Module):

    def __init__(self, bits):
        super(MixQuantActiv, self).__init__()
        self.bits = bits
        self.mix_activ = nn.ModuleList()
        for bit in self.bits:
            self.mix_activ.append(HWGQ(bit=bit))

    def forward(self, input, sw):
        outs = []
        for i, branch in enumerate(self.mix_activ):
            outs.append(branch(input) * sw[i])
        activ = sum(outs)
        return activ
    
class SharedMixQuantConv2d(nn.Module):

    def __init__(self, inplane, outplane, bits, **kwargs):
        super(SharedMixQuantConv2d, self).__init__()
        assert not kwargs['bias']
        self.bits = bits
        self.conv = nn.Conv2d(inplane, outplane, **kwargs)
        self.steps = []
        for bit in self.bits:
            assert 0 < bit < 32
            self.steps.append(gaussian_steps[bit])

    def forward(self, input, sw):
        mix_quant_weight = []
        conv = self.conv
        weight = conv.weight
        # save repeated std computation for shared weights
        weight_std = weight.std().item()
        for i, bit in enumerate(self.bits):
            step = self.steps[i] * weight_std
            quant_weight = _gauss_quantize_resclaed_step.apply(weight, step, bit)
            scaled_quant_weight = quant_weight * sw[i]
            mix_quant_weight.append(scaled_quant_weight)
        mix_quant_weight = sum(mix_quant_weight)
        out = F.conv2d(
            input, mix_quant_weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        return out
    
class MixActivConv2d(nn.Module):

    def __init__(self, inplane, outplane, wbits=[2,4], abits=[2,4], share_weight=True, **kwargs):
        super(MixActivConv2d, self).__init__()
        if wbits is None:
            self.wbits = [1, 2]
        else:
            self.wbits = wbits
        if abits is None:
            self.abits = [1, 2]
        else:
            self.abits = abits
        # build mix-precision branches
        self.mix_activ = MixQuantActiv(self.abits)
        self.share_weight = share_weight
        if share_weight:
            self.mix_weight = SharedMixQuantConv2d(inplane, outplane, self.wbits, **kwargs)
        else:
            raise ValueError('Cant find shared weight')
        # complexities
        stride = kwargs['stride'] if 'stride' in kwargs else 1
        if isinstance(kwargs['kernel_size'], tuple):
            kernel_size = kwargs['kernel_size'][0] * kwargs['kernel_size'][1]
        else:
            kernel_size = kwargs['kernel_size'] * kwargs['kernel_size']
        self.param_size = inplane * outplane * kernel_size * 1e-6
        self.filter_size = self.param_size / float(stride ** 2.0)
        self.register_buffer('size_product', torch.tensor(0, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))
        # self.register_buffer('gamma_act', torch.tensor([0,0], dtype=torch.float))
        # self.register_buffer('gamma_wt', torch.tensor([0,0], dtype=torch.float))
        self.gamma_act = torch.tensor([0,0])
        self.gamma_wt = torch.tensor([0,0])
        


    def forward(self, input, gamma_act, gamma_wt):
        ## buffers the quant weights after each forward pass 
        self.gamma_act = gamma_act
        self.gamma_wt = gamma_wt
        in_shape = input.shape
        tmp = torch.tensor(in_shape[1] * in_shape[2] * in_shape[3] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(tmp)
        tmp = torch.tensor(self.filter_size * in_shape[-1] * in_shape[-2], dtype=torch.float)
        self.size_product.copy_(tmp)
        out = self.mix_activ(input, gamma_act)
        out = self.mix_weight(out, gamma_wt)
        return out

    def complexity_loss(self):
        sw = self.gamma_act
        mix_abit = 0
        abits = self.mix_activ.bits
        for i in range(len(abits)):
            mix_abit += sw[i] * abits[i]
        sw = self.gamma_wt
        mix_wbit = 0
        wbits = self.mix_weight.bits
        for i in range(len(wbits)):
            mix_wbit += sw[i] * wbits[i]
        complexity = self.size_product.item() * mix_abit * mix_wbit
        return complexity

    def fetch_best_arch(self, layer_idx):
        size_product = float(self.size_product.cpu().numpy())
        memory_size = float(self.memory_size.cpu().numpy())
        prob_activ = F.softmax(self.mix_activ.gamma_activ, dim=0)
        prob_activ = prob_activ.detach().cpu().numpy()
        best_activ = prob_activ.argmax()
        mix_abit = 0
        abits = self.mix_activ.bits
        for i in range(len(abits)):
            mix_abit += prob_activ[i] * abits[i]
        prob_weight = F.softmax(self.mix_weight.gamma_weight, dim=0)
        prob_weight = prob_weight.detach().cpu().numpy()
        best_weight = prob_weight.argmax()
        mix_wbit = 0
        wbits = self.mix_weight.bits
        for i in range(len(wbits)):
            mix_wbit += prob_weight[i] * wbits[i]
        if self.share_weight:
            weight_shape = list(self.mix_weight.conv.weight.shape)
        else:
            weight_shape = list(self.mix_weight.conv_list[0].weight.shape)
        print('idx {} with shape {}, activ gamma: {}, comp: {:.3f}M * {:.3f} * {:.3f}, '
              'memory: {:.3f}K * {:.3f}'.format(layer_idx, weight_shape, prob_activ, size_product,
                                                mix_abit, mix_wbit, memory_size, mix_abit))
        print('idx {} with shape {}, weight gamma: {}, comp: {:.3f}M * {:.3f} * {:.3f}, '
              'param: {:.3f}M * {:.3f}'.format(layer_idx, weight_shape, prob_weight, size_product,
                                               mix_abit, mix_wbit, self.param_size, mix_wbit))
        best_arch = {'best_activ': [best_activ], 'best_weight': [best_weight]}
        bitops = size_product * abits[best_activ] * wbits[best_weight]
        bita = memory_size * abits[best_activ]
        bitw = self.param_size * wbits[best_weight]
        mixbitops = size_product * mix_abit * mix_wbit
        mixbita = memory_size * mix_abit
        mixbitw = self.param_size * mix_wbit
        return best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw



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

  def forward(self, s0, s1, weights, weights2, preact0, prewt0, preact1, prewt1, gamma_act, gamma_wt):
    """
    :param s0:
    :param s1:
    :param weights: [14, 8]
    :return:
    """
    # print('s0:', s0.shape,end='=>')
    s0 = self.preprocess0(s0, preact0, prewt0)  # [40, 48, 32, 32], [40, 16, 32, 32]
    # print(s0.shape, self.reduction_prev)
    # print('s1:', s1.shape,end='=>')
    s1 = self.preprocess1(s1, preact1, prewt1)  # [40, 48, 32, 32], [40, 16, 32, 32]
    # print(s1.shape)

    states = [s0, s1]
    offset = 0
    # for each node, receive input from all previous intermediate nodes and s0, s1
    for i in range(self.num_nodes):  # 4
      # [40, 16, 32, 32]
      # import pdb; pdb.set_trace()
      s = sum(weights2[offset+j]*self.layers[offset+j](h, weights[offset+j], gamma_act[offset+j], gamma_wt[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      # append one state since s is the elem-wise addition of all output
      states.append(s)
      # print('node:',i, s.shape, self.reduction)

    # concat along dim=channel
    return torch.cat(states[-self.multiplier:], dim=1)  # 6 of [40, 16, 32, 32]



class Network_PC(nn.Module):
  """
  stack number:layer of cells and then flatten to fed a linear layer
  """

  def __init__(self, C, num_cells, 
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
    self.conv_func = MixActivConv2d
    self.quant_search_num = 2

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
    
    self.gamma_activ_celln_preprocess0 = nn.Parameter(torch.randn(self.quant_search_num))
    self.gamma_weight_celln_preprocess0 = nn.Parameter(torch.randn(self.quant_search_num))

    self.gamma_activ_celln_preprocess1 = nn.Parameter(torch.randn(self.quant_search_num))
    self.gamma_weight_celln_preprocess1 = nn.Parameter(torch.randn(self.quant_search_num))

    self.gamma_activ_celln = nn.Parameter(torch.randn(k, num_ops-2, self.quant_search_num)) ## 5 conv operation among 8 search primitives
    self.gamma_weight_celln = nn.Parameter(torch.randn(k, num_ops-2, self.quant_search_num)) ## 5 conv operation among 8 search primitives

    self.gamma_activ_cellr_preprocess0 = nn.Parameter(torch.randn(self.quant_search_num))
    self.gamma_weight_cellr_preprocess0 = nn.Parameter(torch.randn(self.quant_search_num))

    self.gamma_activ_cellr_preprocess1 = nn.Parameter(torch.randn(self.quant_search_num))
    self.gamma_weight_cellr_preprocess1 = nn.Parameter(torch.randn(self.quant_search_num))

    self.gamma_activ_cellr = nn.Parameter(torch.randn(k, num_ops-2, self.quant_search_num)) ## 5 conv operation among 8 search primitives
    self.gamma_weight_cellr = nn.Parameter(torch.randn(k, num_ops-2, self.quant_search_num)) ## 5 conv operation among 8 search primitives
    
    with torch.no_grad():
      # initialize to smaller value
      self.alpha_normal.mul_(1e-3)
      self.alpha_reduce.mul_(1e-3)
      self.beta_normal.mul_(1e-3)
      self.beta_reduce.mul_(1e-3)

      ##for normal cell      
      self.gamma_activ_celln_preprocess0.mul_(1e-3)
      self.gamma_weight_celln_preprocess0.mul_(1e-3)
      self.gamma_activ_celln_preprocess1.mul_(1e-3)
      self.gamma_weight_celln_preprocess1.mul_(1e-3)
      self.gamma_activ_celln.mul_(1e-3)
      self.gamma_weight_celln.mul_(1e-3)
      
      ##for reduction cell      
      self.gamma_activ_cellr_preprocess0.mul_(1e-3)
      self.gamma_weight_cellr_preprocess0.mul_(1e-3)
      self.gamma_activ_cellr_preprocess1.mul_(1e-3)
      self.gamma_weight_cellr_preprocess1.mul_(1e-3)
      self.gamma_activ_cellr.mul_(1e-3)
      self.gamma_weight_cellr.mul_(1e-3)
      
    self._arch_parameters = [self.alpha_normal, self.alpha_reduce, self.beta_normal, self.beta_reduce]
    self._quant_parameters = [self.gamma_activ_celln_preprocess0, self.gamma_weight_celln_preprocess0, self.gamma_activ_celln_preprocess1, self.gamma_weight_celln_preprocess1, self.gamma_activ_celln, self.gamma_weight_celln,
                              self.gamma_activ_cellr_preprocess0, self.gamma_weight_cellr_preprocess0, self.gamma_activ_cellr_preprocess1, self.gamma_weight_cellr_preprocess1, self.gamma_activ_cellr, self.gamma_weight_cellr]

  def arch_parameters(self):
    return self._arch_parameters

  def quant_parameters(self):
    return self._quant_parameters

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
        pre_act0 = F.softmax(self.gamma_activ_cellr_preprocess0, dim=-1)
        pre_wt0 = F.softmax(self.gamma_weight_cellr_preprocess0, dim=-1)
        pre_act1 = F.softmax(self.gamma_activ_cellr_preprocess1, dim=-1)
        pre_wt1 = F.softmax(self.gamma_weight_cellr_preprocess1, dim=-1)
        act = F.softmax(self.gamma_activ_cellr, dim=-1)
        wt = F.softmax(self.gamma_weight_cellr, dim=-1)        
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
        pre_act0 = F.softmax(self.gamma_activ_celln_preprocess0, dim=-1)
        pre_wt0 = F.softmax(self.gamma_weight_celln_preprocess0, dim=-1)
        pre_act1 = F.softmax(self.gamma_activ_celln_preprocess1, dim=-1)
        pre_wt1 = F.softmax(self.gamma_weight_celln_preprocess1, dim=-1)
        act = F.softmax(self.gamma_activ_celln, dim=-1)
        wt = F.softmax(self.gamma_weight_celln, dim=-1)
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
      s0, s1 = s1, cell(s0, s1, weights, weights2, pre_act0, pre_wt0, pre_act1, pre_wt1, act, wt)  # [40, 64, 32, 32]
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