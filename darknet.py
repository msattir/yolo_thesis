from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils import *


def get_test_input():
     img = cv2.imread("dog-cycle-car.png")
     img = cv2.resize(img, (416,416))
     img_ = img[:,:,::-1].transpose((2,0,1)) #BGR->RGB
     img_ = img_[np.newaxis,:,:,:]/255.0
     img_ = torch.from_numpy(img_).float()
     img_ = Variable(img_)
     return img_


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
               if len(block) != 0:
                    blocks.append(block)
                    block = {}
               block["type"] = line[1:-1].rstrip()
          else:
               key,value = line.split("=")
               block[key.rstrip()] = value.lstrip()
     blocks.append(block)

     return blocks

class EmptyLayer(nn.Module):
     def __init__(self):
           super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
     def __init__(self, anchors):
           super(DetectionLayer, self).__init__()
           self.anchors = anchors

     def forward(self, x, inp_dim, num_classes, confidence):
           x = x.data
           global CUDA
           prediction = x
           prediction = predict_transform(prediction, inp_dim, self.anchors, num_clases, confidence, CUDA)
           return prediciton


def create_modules(blocks):
     net_info = blocks[0]
     module_list = nn.ModuleList()
     prev_filters = 3 #RGB Image
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
                 
                 #Add Batch Norm
                 if batch_norm:
                       bn = nn.BatchNorm2d(filters)
                       module.add_module("batch_norm_{0}".format(index), bn)

                 #Activvation
                 if activation == "leaky":
                       activn = nn.LeakyReLU(0.1, inplace = True)
                       module.add_module("leaky_{0}".format(index), activn)
           
           #Upsample Layer
           elif x["type"] == "upsample":
                 stride = int(x["stride"])
                 upsample = nn.Upsample(scale_factor=2, mode="bilinear")
                 module.add_module("upsample_{}".format(index),upsample)

           #Routing Layer
           elif (x["type"] == "route"):
                 x["layers"] = x["layers"].split(',')
                 
                 start = int(x["layers"][0])
                 try:
                       end = int(x["layers"][1])
                 except:
                       end = 0
                 
                 if start > 0:
                       start = start - index
                 if end > 0:
                       end = end - index
               
                 route = EmptyLayer()
                 module.add_module("route_{0}".format(index), route)
                 
                 if end < 0:
                       filters = output_filters[index + start] + output_filters[index + end]
                 else:
                       filters = output_filters[index + start]

           #Shortcut Layer
           elif x["type"] == "shortcut":
                 shortcut = EmptyLayer()
                 module.add_module("shortcut_{0}".format(index), shortcut)
           
           #Yolo Layer
           elif x["type"] == "yolo":
                 mask = x["mask"].split(",")
                 mask = [int(x) for x in mask]
                 
                 anchors = x["anchors"].split(",")
                 anchors = [int(a) for a in anchors]
                 anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
                 anchors = [anchors[i] for i in mask]

                 detection = DetectionLayer(anchors)
                 module.add_module("Detection_{}".format(index), detection)

           module_list.append(module)
           prev_filters = filters
           output_filters.append(filters)
     
     return (net_info, module_list)       
      
          

class Darknet(nn.Module):
     def __init__(self, cfgfile):
           super(Darknet, self).__init__()
           self.blocks = parse_cfg(cfgfile)
           self.net_info, self.module_list = create_modules(self.blocks)

     def forward(self, x, CUDA):
           modules = self.blocks[1:]
           outputs = {}

           write = 0
           if CUDA:
                 x = x.cuda()
           for i, module in enumerate(modules):
                 module_type = (module["type"])

                 if module_type == "convolutional" or module_type == "upsample":
                       #print (i, self.module_list[i], x)
                       x = self.module_list[i](x)
       
                 elif module_type == "route":
                       layers = module["layers"]
                       layers = [int(a) for a in layers]
        
                       if (layers[0]) > 0:
                             layers[0] = layers[0] -i
                  
                       if len(layers) == 1:
                             x = outputs[i + (layers[0])]
                    
                       else:
                             if (layers[1]) > 0:
                                   layers[1] = layers[1] -i
 
                             map1 = outputs[i + layers[0]]
                             map2 = outputs[i + layers[1]]
                      
                             x = torch.cat((map1, map2), 1)
       
                 elif module_type == "shortcut":
                       from_ = int(module["from"])
                       x = outputs[i-1] + outputs[i+from_]

                 elif module_type == "yolo":
                       #Get anchors and input dimensions
                       anchors = self.module_list[i][0].anchors
                       inp_dim = int(self.net_info["height"])

                       num_classes = int(module["classes"])

                       #Transform
                       #x = x.data
                       x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                       if not write:
                             detections = x
                             write = 1
                       else:
                             detections = torch.cat((detections, x),1)
 
                 outputs[i] = x
             
           return detections



     def load_weights(self, weightsfile):
          fp = open(weightsfile, "rb")
          header = np.fromfile(fp, dtype = np.int32, count=5)
          self.header = torch.from_numpy(header)
          self.seen = self.header[3]
     
          weights = np.fromfile(fp, dtype = np.float32)
          ptr = 0
          for i in range(len(self.module_list)):
                module_type = self.blocks[i+1]["type"]
     
                if module_type == "convolutional":
                      model = self.module_list[i]
                      try:
                            batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                      except:
                            batch_normalize = 0
                    
                      conv = model[0]
                      
                      if (batch_normalize):
                            bn = model[1]
     
                            #Get the number of weights of Batch Norm Layer
                            num_bn_biases = bn.bias.numel()
     
                            #Load the weights
                            bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                            ptr += num_bn_biases
     
                            bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                            ptr  += num_bn_biases
     
                            bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                            ptr  += num_bn_biases
     
                            bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                            ptr  += num_bn_biases
     
                            #Cast the loaded weights into dims of model weights. 
                            bn_biases = bn_biases.view_as(bn.bias.data)
                            bn_weights = bn_weights.view_as(bn.weight.data)
                            bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                            bn_running_var = bn_running_var.view_as(bn.running_var)
     
                            #Copy the data to model
                            bn.bias.data.copy_(bn_biases)
                            bn.weight.data.copy_(bn_weights)
                            bn.running_mean.copy_(bn_running_mean)
                            bn.running_var.copy_(bn_running_var)
      
                      else:
                            #Number of biases
                            num_biases = conv.bias.numel()
     
                            #Load the weights
                            conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                            ptr = ptr + num_biases
     
                            #reshape the loaded weights according to the dims of the model weights
                            conv_biases = conv_biases.view_as(conv.bias.data)
     
                            #Finally copy the data
                            conv.bias.data.copy_(conv_biases)
     
                      #Let us load the weights for the Convolutional layers
                      num_weights = conv.weight.numel()
                      
                      #Do the same as above for weights
                      conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                      ptr = ptr + num_weights
                      
                      conv_weights = conv_weights.view_as(conv.weight.data)
                      conv.weight.data.copy_(conv_weights)
           
                                

#blocks=parse_cfg("cfg/yolov3.cfg")
#print(create_modules(blocks))

model = Darknet("cfg/yolov3.cfg")
inp = get_test_input()
#model.load_weights("yolov3.weights")
model = model.cuda()

pred = model(inp, torch.cuda.is_available())#torch.cuda.is_available())
#torch.save(pred, 'tensor.pt')
#buffer = io.BytesIO()
#torch.save(x, buffer)

pred1 = torch.load('tensor.pt')
#with open('tensor.pt') as f:
#     buffer = io.BytesIO(f.red())

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(1000):
     y_pred1 = model(inp, torch.cuda.is_available())
     #y_pred1.requires_grad=True
     loss = loss_fn(y_pred1.cuda(), pred1.cuda())
     print(t, loss.item())

     optimizer.zero_grad()
     loss.backward()
     optimizer.step()


w_pred = write_results(y_pred, 0.5, 80, 0.4)
w_pred1 = write_results(pred1, 0.5, 80, 0.4)
print (w_pred)
print (w_pred1)
