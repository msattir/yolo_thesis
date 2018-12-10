from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def parse_cfg(cfgfile):
     file = open(cfgfile, 'r')
     lines = file.read().split('\n')
     lines = [x for x in lines if len(x) > 0]
     lines = [x for x in lines if x[0] != '#']
     lines = [x.rstrip().lstrip() for x in lines]

     block = {}
     blocks = []
     
     for line in lines:
          if line[0] == '[':
               if len(blocks) != 0:
                    blocks.append(block)
                    block = {}
               block["type"] = line[1:-1].rstrip()
          else:
               key,value = line.split("=")
               block[key.rstrip()] = value.lstrip()
     blocks.append(block)

     return blocks


def create_modules(blocks):
     net_info = blocks[0]
     module_list = nn.ModuleList()
     prev_filters = 3 $#RGB Image
     output_filters = []
  
     for index, x in enumerate(blocks[1:]):
           module = nn.Sequential()

           if(x["type"] == "convolutional"):
                 activation = x["activation"]
                 try:
                       batch_norm = int(x["batch_normalize"])
                       bias = False
                 except:
                       batch_norm = 0
                       bias = True

                 filters = int(x["filters"])
                 padding = int(x["pad"])
                 kernel_size = int(x["size"])
                 stride = int(x["stride"])

                 if padding:
                       pad = (kernel_size -1)//2
                 else:
                       pad = 0
    
                 #add THE layer
                 conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
                 module.add_module("conv_{0}".format(index), conv)


b=parse_cfg("cfg/yolov3.cfg")
a=4
