#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 22:51:21 2021

@author: root
"""

import torch
import torch.nn as nn
from genotypes import *
from nets.quant_modules import MixActivConv2d
import pdb

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape (n, 32, 224, 224)==>(n, 4, 8, 224, 224)
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)
    
    # (n, 4, 8, 224, 224) ==> (n, 8, 4, 224, 224)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class MixedLayer_PC(nn.Module):

  def __init__(self, C, stride, op_names, conv_func):
    super(MixedLayer_PC, self).__init__()
    self.op_names = op_names
    self.layers = nn.ModuleList()
    self.mp = nn.MaxPool2d(2,2)
    self.k = 4

    OPS = {'none': lambda C, stride, affine: Zero(stride),
           'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride,
                                                                  padding=1, count_include_pad=False),
           'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
           'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, conv_func, affine=affine),
           'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine, conv_func=conv_func),
           'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine, conv_func=conv_func),
           'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine, conv_func=conv_func),
           'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine, conv_func=conv_func),
           'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine,conv_func=conv_func),
    
           'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
             nn.ReLU(inplace=False),
             nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
             nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
             nn.BatchNorm2d(C, affine=affine))}
    
    for primitive in op_names:
      layer = OPS[primitive](C //self.k, stride, False)
      if 'pool' in primitive:
        layer = nn.Sequential(layer, nn.BatchNorm2d(C //self.k, affine=False))
      self.layers.append(layer)


  def forward(self, x, weights):
    #channel proportion k=4  
    dim_2 = x.shape[1]
    xtemp = x[ : , :  dim_2//self.k, :, :]
    xtemp2 = x[ : ,  dim_2//self.k:, :, :]
    temp1 = sum(w * op(xtemp) for w, op in zip(weights, self.layers))
    #reduction cell needs pooling before concat
    if temp1.shape[2] == x.shape[2]:
      ans = torch.cat([temp1,xtemp2],dim=1)
    else:
      ans = torch.cat([temp1,self.mp(xtemp2)], dim=1)
    ans = channel_shuffle(ans,self.k)
    #ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
    #except channe shuffle, channel shift also works
    return ans



class MixedLayer(nn.Module):
  def __init__(self, c, stride, op_names, conv_func):
    super(MixedLayer, self).__init__()
    self.op_names = op_names
    self.layers = nn.ModuleList()
    """
    PRIMITIVES = [
                'none',
                'max_pool_3x3',
                'avg_pool_3x3',
                'skip_connect',
                'sep_conv_3x3',
                'sep_conv_5x5',
                'dil_conv_3x3',
                'dil_conv_5x5'
            ]
    """
# OPS is a set of layers with same input/output channel.

    OPS = {'none': lambda C, stride, affine: Zero(stride),
           'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride,
                                                                  padding=1, count_include_pad=False),
           'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
           'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, conv_func, affine=affine),
           'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine, conv_func=conv_func),
           'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine, conv_func=conv_func),
           'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine, conv_func=conv_func),
           'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine, conv_func=conv_func),
           'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine,conv_func=conv_func),
    
           'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
             nn.ReLU(inplace=False),
             nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
             nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
             nn.BatchNorm2d(C, affine=affine))}
    
    for primitive in op_names:
      layer = OPS[primitive](c, stride, False)
      if 'pool' in primitive:
        layer = nn.Sequential(layer, nn.BatchNorm2d(c, affine=False))

      self.layers.append(layer)

  def forward(self, x, weights):
    return sum([w * layer(x) for w, layer in zip(weights, self.layers)])





class ReLUConvBN(nn.Module):
  """
  Stack of relu-conv-bn
  """

  def __init__(self, C_in, C_out, kernel_size, stride, padding, conv_func, affine=True):
    """
    :param C_in:
    :param C_out:
    :param kernel_size:
    :param stride:
    :param padding:
    :param affine:
    """
    super(ReLUConvBN, self).__init__()

    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Sequential(conv_func(inplane=C_in, outplane=C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)),
      nn.BatchNorm2d(C_out, affine=affine))

  def forward(self, x):
    return self.op(x)


class DilConv(nn.Module):
  """
  relu-dilated conv-bn
  """

  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, conv_func, affine=True):
    """
    :param C_in:
    :param C_out:
    :param kernel_size:
    :param stride:
    :param padding: 2/4
    :param dilation: 2
    :param affine:
    """
    super(DilConv, self).__init__()

    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Sequential(conv_func(inplane=C_in, outplane=C_in, kernel_size=kernel_size, stride=stride, padding=padding, 
                     dilation=dilation, groups=C_in, bias=False)),
      nn.Sequential(MixActivConv2d(inplane=C_in, outplane=C_out, kernel_size=1, padding=0, bias=False)),
     
      # nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
      #           dilation=dilation, groups=C_in, bias=False),
      # nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine))

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
  """
  implemented separate convolution via pytorch groups parameters
  """

  def __init__(self, C_in, C_out, kernel_size, stride, padding, conv_func, affine=True):
    """
    :param C_in:
    :param C_out:
    :param kernel_size:
    :param stride:
    :param padding: 1/2
    :param affine:
    """
    super(SepConv, self).__init__()

    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Sequential(conv_func(inplane=C_in, outplane=C_in, kernel_size=kernel_size, stride=stride, padding=padding, 
                     groups=C_in, bias=False)),
      nn.Sequential(conv_func(inplane=C_in, outplane=C_in, kernel_size=1, padding=0, bias=False)),
      # nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
      #           groups=C_in, bias=False),
      # nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Sequential(conv_func(inplane=C_in, outplane=C_in, kernel_size=kernel_size, stride=1, padding=padding, 
                     groups=C_in, bias=False)),
      nn.Sequential(conv_func(inplane=C_in, outplane=C_out, kernel_size=1, padding=0, bias=False)),
      
      # nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding,
      #           groups=C_in, bias=False),
      # nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      
      nn.BatchNorm2d(C_out, affine=affine))

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):
  """
  zero by stride
  """

  def __init__(self, stride):
    super(Zero, self).__init__()

    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:, :, ::self.stride, ::self.stride].mul(0.)

def conv3x3(conv_func, in_plane, out_plane, kernel_size, stride=1, padding=0, bias=False, **kwargs):
    "3x3 convolution with padding"
    return conv_func(in_plane, out_plane, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=bias, **kwargs)

class FactorizedReduce(nn.Module):
  """
  reduce feature maps height/width by half while keeping channel same
  """

  def __init__(self, C_in, C_out, conv_func, affine=True):
    """
    :param C_in:
    :param C_out:
    :param affine:
    """
    super(FactorizedReduce, self).__init__()

    assert C_out % 2 == 0

    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = conv_func(inplane=C_in, outplane=C_out // 2, kernel_size=1, stride=2, padding=0, bias=False)
    self.conv_2 = conv_func(C_in, C_out // 2, kernel_size=1, stride=2, padding=0, bias=False)
    # self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    # self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)

    # x: torch.Size([32, 32, 32, 32])
    # conv1: [b, c_out//2, d//2, d//2]
    # conv2: []
    # out: torch.Size([32, 32, 16, 16])
    # pdb.set_trace()
    
    out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
    out = self.bn(out)
    return out