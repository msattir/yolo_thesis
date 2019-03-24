from __future__ import division
import torch
import time
import torch.nn as nn
import torch.autograd as Variable
import numpy as np
import cv2
from utils import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random
from preprocess import gt_pred

def load_classes(fileName):
     fp = open(fileName, "r")
     names = fp.read().split("\n")[:-1]
     return names



def arg_parse():
     
     parser = argparse.ArgumentParser(description='Yolo V3 Detection Module')
     parser.add_argument("--images", dest = 'images', help = "Images dir to perform Detection on", default = "imgs/", type = str)
     parser.add_argument("--det", dest = 'det', help = "Images dir to store output Detection in", default = "det/", type = str)
     parser.add_argument("--bs", dest = 'bs', help = "Batch Size", default = 1)
     parser.add_argument("--confidence", dest = "confidence", help = "Min Confidence", default = 0.6)
     parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshold percent", default = 0.4)
     parser.add_argument("--cfg", dest = 'cfgfile', help = "Configure file", default = "cfg/yolov3.cfg", type = str)
     parser.add_argument("--weights", dest = 'weightsfile', help = "weightsfile", default = "yolov3.weights", type = str)
     parser.add_argument("--reso", dest = 'reso', help = "Input resolution of the network", default = "416", type = str)
     parser.add_argument("--train", dest = 'train_dir', help = "Images Dir where images to train on are present", default = None, type = str)
     parser.add_argument("--iter", dest = 'num_iter', help = "Number of iterations in training", default = 10000)
     parser.add_argument("--load_ckpt", dest = 'ckpt_load_dir', help = "Path of saved model for checkpoint training", default = None, type = str)
     parser.add_argument("--save_ckpt", dest = 'ckpt_save_dir', help = "Path to save checkpoints of model", default = None, type = str)
     
     return parser.parse_args()

args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
ckpt_load_dir = args.ckpt_load_dir
start = 0
CUDA = torch.cuda.is_available()
train_dir = args.train_dir
ckpt_save_dir = args.ckpt_save_dir

num_classes = 2
classes = load_classes("data/bosch.names")
training = False

if ckpt_load_dir is None:
     checkpoint = 0
     print ("Loading Network")
     model = Darknet(args.cfgfile)
else:
     print ("Restoring Ckeckpoint")
     checkpoint = 1
     model = Darknet(args.cfgfile)
     model = nn.DataParallel(model)
     ckpt = torch.load(ckpt_load_dir)
     model.load_state_dict(ckpt['model_state_dict'])


if train_dir is None:
     if checkpoint == 0:
           model.load_weights(args.weightsfile)
           print ("Weights Loaded")

else:
     training = True
     images = train_dir #Update images dir to training images
     num_iter = int(args.num_iter)
     print ("Training Enabled")

#model.net_info["height"] = args.reso
inp_dim = 416#int(model.net_info["height"])
#assert inp_dim % 32 == 0
#assert inp_dim > 32


#Enable CUDA if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if CUDA:
     if torch.cuda.device_count() > 1:
           print ("Using ", torch.cuda.device_count(), " GPUs to train")
           if checkpoint != 1:
                 model = nn.DataParallel(model)
           model.to(device)
     else:
           model.cuda()

read_dir = time.time()

try:
     imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
     imlist = []
     imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
     print ("Input Dir not found {}".format(images))
     exit()

imlist.sort()

if training:
     #labels = (images.rsplit('train', 1)[0] + "labels" + images.rsplit('train', 1)[1]).replace('.jpg', '.txt')
     labels = images.replace('images', 'labels').replace('.jpg', '.txt')
      
     try:
          labellist = [osp.join(osp.realpath('.'), labels, lab) for lab in os.listdir(labels)]
     except NotADirectoryError:
          labellist = []
          labellist.append(osp.join(osp.realpath('.'), labels))
     except FileNotFoundError:
          print ("Labels Dir not found {}".format(labels))
          exit()

     labellist.sort()
     
     try:
           label_tensor = torch.load("bdd_9929.pt", map_location=lambda storage, loc: storage.cuda(0))

     except FileNotFoundError:
           print ("Generating Labels")
           label_tensor = gt_pred(imlist, labellist, CUDA,  num_classes)
           label_test = []
           for mi, bl in zip(imlist, labellist):
                 label_test.append(mi.rsplit('/', 1)[1][:-3][1:] == bl.rsplit('/', 1)[1][:-3][1:])
           assert all(label_test) #Check to see if all labels correspond to images
     lab_batches = label_tensor.unsqueeze(0)
     y_mask=torch.zeros(label_tensor.shape)
     y_mask[(label_tensor[:,:,4] == 1).nonzero()[:,0], (label_tensor[:,:,4] == 1).nonzero()[:,1], :] = 1.0
     y_mask[:,:,4] = 1.0
     y_mask_batches = y_mask.unsqueeze(0)
     
     

     


if not os.path.exists(args.det):
     os.makedirs(args.det)

load_batch = time.time()
loaded_ims = [cv2.imread(x) for x in imlist]

im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))

im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

if CUDA:
     im_dim_list = im_dim_list.cuda()

leftover = 0
if (len(im_dim_list) % batch_size):
     leftover = 1

if batch_size != 1:
     num_batches = len(imlist) // batch_size + leftover
     im_batches = [torch.cat((im_batches[i*batch_size : min((i+1)*batch_size, len(im_batches))])) for i in range (num_batches)]
     if training:
           lab_batches = [(label_tensor[i*batch_size : min((i+1)*batch_size, len(label_tensor)),:,:]) for i in range (num_batches)]
           y_mask_batches = [(y_mask[i*batch_size : min((i+1)*batch_size, len(y_mask)),:,:]) for i in range (num_batches)]
#else: #batch size is 1
     
     
     

##################
### Detection Loop
##################
if not training:
     model.eval()
     write = 0
     start_det_loop = time.time()
     
     for i, batch in enumerate(im_batches):
           #Load Image
           start = time.time()
           if CUDA:
                 batch = batch.cuda()
           with torch.no_grad():
                 prediction = model(Variable(batch), CUDA)

           y_temp = prediction[:,:,0].clone()
           prediction[:,:,0] = prediction[:,:,1]
           prediction[:,:,1] = y_temp

           print (prediction.size(0))
           prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thresh)
           print (prediction)
           end = time.time()
           
           if type(prediction) == int:
                 for im_num, image in enumerate(imlist[i*batch_size: min((i+1)*batch_size, len(imlist))]):
                       im_id = i*batch_size + im_num
                       print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
                       print("{0:20s} {1:s}".format("Objects Detected:", ""))
                       print("----------------------------------------------------------")
                 continue
      
           prediction[:,0] += i*batch_size
           if not write:
                 output = prediction
                 write = 1
      
           else:
                 output = torch.cat((output, prediction))
      
           for im_num, image in enumerate(imlist[i*batch_size: min ((i+1)*batch_size, len(imlist))]):
                 im_id = i*batch_size + im_num
                 objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
                 print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
                 print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
                 print("----------------------------------------------------------")
      
           if CUDA:
                 torch.cuda.synchronize()
      
      
     
     try:
           output
     except NameError:
           print ("No detections made")
           exit()

##########################
## Trainig Loop #########
#########################

else:
     write = 0 #Concatinate predictions
     start_det_loop = time.time()
     loss_fn = torch.nn.MSELoss(reduction='sum')
     learning_rate =1e-3

     if checkpoint == 0:
           optimizer = torch.optim.Adam(model.parameters())
           scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.25) 
           epoc = int(args.num_iter)
           start = 0

     else:
           optimizer = torch.optim.Adam(model.parameters())
           scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.2) 
           scheduler.load_state_dict(ckpt['scheduler_state_dict'])
           optimizer.load_state_dict(ckpt['optimizer_state_dict'])
           start = ckpt['epoch']
           epoc = int(args.num_iter) + start
           loss = ckpt['loss']
           
     rnd_int = int(torch.randint(0,10000,(1,)).item())
     write_str = ''

     for e in range(start, epoc): 
            
           for t in range(1):
                 for b, (batch, label, y_mask) in enumerate(zip(im_batches, lab_batches, y_mask_batches)):
                       if CUDA:
                             batch = batch.cuda()
                             gt_pred1 = label.cuda()
                             y_mask = y_mask.cuda()
                             #model = torch.nn.DataParallel(model).cuda()
                       f=open('logs/training_'+str(rnd_int)+'.log', 'a+')
                       y_pred1 = model(batch, CUDA)
                       y_temp = y_pred1[:,:,0].clone()
                       y_pred1[:,:,0] = y_pred1[:,:,1]
                       y_pred1[:,:,1] = y_temp

                       y_pred1 = torch.mul(y_pred1, y_mask)
                        
                       gt_obj = gt_pred1[:,:,:] > 0.0000001
                       gt_noobj = gt_pred1[:,:,:] < 0

                       loss_mse = torch.nn.MSELoss()
                       loss_xywh_obj = loss_mse(y_pred1[:,:,0:4], gt_pred1[:,:,0:4])

                       #loss_wh_obj = loss_mse(torch.pow(torch.abs(y_pred1_obj[:,:,2:4]), 0.5), torch.pow(torch.abs(gt_pred1[:,:,2:4]), 0.5))
                       y_pred1_obj_box = y_pred1.new_empty([y_pred1.shape[0], y_pred1.shape[1], 4])
                       y_pred1_obj_box[:,:,0] = (y_pred1[:,:,0] - y_pred1[:,:,2]/2) 
                       y_pred1_obj_box[:,:,1] = (y_pred1[:,:,1] - y_pred1[:,:,3]/2) 
                       y_pred1_obj_box[:,:,2] = (y_pred1[:,:,0] + y_pred1[:,:,2]/2) 
                       y_pred1_obj_box[:,:,3] = (y_pred1[:,:,1] + y_pred1[:,:,3]/2) 
                       
                       gt_pred1_box = gt_pred1.new_empty([gt_pred1.shape[0], gt_pred1.shape[1], 4])
                       gt_pred1_box[:,:,0] = (gt_pred1[:,:,0] - gt_pred1[:,:,2]/2) 
                       gt_pred1_box[:,:,1] = (gt_pred1[:,:,1] - gt_pred1[:,:,3]/2) 
                       gt_pred1_box[:,:,2] = (gt_pred1[:,:,0] + gt_pred1[:,:,2]/2) 
                       gt_pred1_box[:,:,3] = (gt_pred1[:,:,1] + gt_pred1[:,:,3]/2)

                       iou = torch.empty([gt_pred1.shape[0], gt_pred1.shape[1]])

                       for ix in range(gt_pred1.shape[0]):
                             iou[ix, :] = box_iou(y_pred1_obj_box[ix, :, :], gt_pred1_box[ix, :, :])


                       if CUDA:
                             iou = iou.cuda()

                       iou = torch.mul(iou, (gt_pred1[:,:,4]>0.00001).float())

                       loss_conf = loss_mse(y_pred1[:,:,4], iou)
                       #loss_aspect = loss_mse(y_pred1[:,:,5], gt_pred1[:,:,5])
                       loss_ce = torch.nn.CrossEntropyLoss()
                       loss_class = loss_mse(y_pred1[:,:,6:], gt_pred1[:,:,6:])
      
                       loss = 5.0*loss_xywh_obj + 1.0*loss_class + 1.0*loss_conf #+ 1.0*loss_aspect

                              
                       #y_pred1[gt_zeros[:,0],gt_zeros[:,1],:] *= 0
                       #all_zeros = torch.zeros(y_pred1.shape).cuda()
                       #gt = torch.load('I1_2009_12_14_drive_0071_000051.pt')
                       #gt[:,(gt[:,:,4] < 0.5).nonzero(),5:] *= 0 
                       #gt_pred1[:,(gt_pred1[:,:,4] < 0.5).nonzero(),5:] += 1e-5 
                       #y_pred1[:,(gt_pred1[:,:,4] < 0.5).nonzero(),5:] *= 0
                       #loss = loss_fn(y_pred1, gt_pred1)
 
                      # a = list(model.parameters())[0].clone()
                       
               
                       optimizer.zero_grad()
                       loss.backward()
                       optimizer.step()

                       pos_loc = gt_pred1[0,:,4].nonzero()
                       pos_item=''
                       for p in pos_loc:
                             pos_item += str(y_pred1[0,p,4].item())+','

                      # b = list(model.parameters())[0].clone()
                      # with open("train.txt", "a") as myfile:
                      #       txt = str(e) + " " + str(loss.item()) + "\n"
                      #       myfile.write(txt)
                       print (e,'-', b, loss.item(), loss_class.item(), y_pred1[0,10094,4].item(), y_pred1[0,pos_loc[0],4].item(), scheduler.get_lr()) #loss_obj.item(), loss_noobj.item(), loss_xy_obj.item(), loss_wh_obj.item(), loss_class.item())#y_pred1[:,:,:].nonzero().sum().data[0], diff.sum().data[0], torch.equal(a.data, b.data))
                       write_str+=str(e)+','+str(b)+','+str(loss.item())+','+str(loss_class.item())+','+str(y_pred1[0,10094,4].item())+','+(pos_item)+ str(scheduler.get_lr())+'\n' 
          # print ("Epoc {}".format(e))
           scheduler.step()
           if ckpt_save_dir is not None:
                 if e % 1000 == 0:
                       torch.save({'epoch': e, 'model_state_dict':model.state_dict(), 'optimizer_state_dict':optimizer.state_dict(), 'scheduler_state_dict':scheduler.state_dict(), 'loss':loss}, '{}/batch_model_{}.pb'.format(ckpt_save_dir, e))
           if e % 1000 == 0:      
                 f.write(write_str)
                 write_str=''
                 f.close()

     model=model.eval()
     for i, batch in enumerate(im_batches):
           #Load Image
           start = time.time()
           if CUDA:
                 batch = batch.cuda()
           with torch.no_grad():
                 prediction = model(Variable(batch), CUDA)


           y_temp = prediction[:,:,0].clone()
           prediction[:,:,0] = prediction[:,:,1]
           prediction[:,:,1] = y_temp

           prediction = write_results(prediction.data, confidence, num_classes, nms_conf = nms_thresh)
           print (prediction)
           end = time.time()

                           
           if type(prediction) == int:
                 for im_num, image in enumerate(imlist[i*batch_size: min((i+1)*batch_size, len(imlist))]):
                       im_id = i*batch_size + im_num
                       print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
                       print("{0:20s} {1:s}".format("Objects Detected:", ""))
                       print("----------------------------------------------------------")
                 continue
      
           prediction[:,0] += i*batch_size
           if not write:
                 output = prediction
                 write = 1
      
           else:
                 output = torch.cat((output, prediction))
      
           for im_num, image in enumerate(imlist[i*batch_size: min ((i+1)*batch_size, len(imlist))]):
                 im_id = i*batch_size + im_num
                 objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
                 print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
                 print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
                 print("----------------------------------------------------------")
      
           if CUDA:
                 torch.cuda.synchronize()
      
      
      
     try:
           output
     except NameError:
           print ("No detections made")
           exit()

im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)

output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2

output[:,1:5] /= scaling_factor

for i in range(output.shape[0]):
     output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
     output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])

output_recast = time.time()
class_load = time.time()
colors = pkl.load(open("pallete", "rb"))


def write_fn(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


draw = time.time()

list(map(lambda x: write_fn(x, loaded_ims), output))
det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det, x.split("/")[-1]))

list(map(cv2.imwrite, det_names, loaded_ims))
end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")


torch.cuda.empty_cache()
