from __future__ import division
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import numpy as np


def letterbox_image(img, inp_dim):
     #Resize box keweping aspect ratio const
     img_w, img_h = img.shape[1], img.shape[0]
     w, h = inp_dim
     new_w = int(img_w * min(w/img_w, h/img_h))
     new_h = int(img_h * min(w/img_w, h/img_h))
     resized_image = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
     canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
     canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
     return canvas


def letterbox_image2(img, inp_dim):
     #Resize image maintaining aspect ration but padding grey cells and return resized_image size

     img_w, img_h = img.shape[1], img.shape[0]
     w, h = inp_dim
     new_w = int(img_w*min(w/img_w, h/img_h))
     new_h = int(img_h*min(w/img_w, h/img_h))
     resized_image = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
     return resized_image.shape

def letterbox_image3(img, inp_dim):
  #Resize image maintaining aspect ration but padding grey cells and return resized_image size

  img_w, img_h = img.shape[1], img.shape[0]
  w, h = inp_dim
  new_w = int(img_w*min(w/img_w, h/img_h))
  new_h = int(img_h*min(w/img_w, h/img_h))
  resized_image = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
 # resized = (resized_image[:,:,0]+resized_image[:,:,1]+resized_image[:,:,2])//3
  canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
  canvas[(h-new_h)//2:(h-new_h)//2+new_h, (w-new_w)//2:(w-new_w)//2+new_w, :] = resized_image
  return canvas, resized_image.shape


def prep_image(img, inp_dim):
     img = letterbox_image(img, (inp_dim, inp_dim))
     img = img[:,:,::-1].transpose((2,0,1)).copy()
     img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
     return img

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
     #Convert a output feature map to 2D tensor

     batch_size = prediction.size(0)
     stride = inp_dim // prediction.size(2)
     grid_size = inp_dim // stride
     num_anchors = len(anchors)
     bbox_attrs = 6 + num_classes

     prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
     prediction = prediction.transpose(1,2).contiguous()
     prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

     anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

     #Sigmoid center_x,y and objectness score
     prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
     prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
     prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

     #Center offsets
     grid = np.arange(grid_size)
     a, b = np.meshgrid(grid, grid)

     x_offset = torch.FloatTensor(a).view(-1,1)
     y_offset = torch.FloatTensor(b).view(-1,1)

     if CUDA:
           x_offset = x_offset.cuda()
           y_offset = y_offset.cuda()

     x_y_offset = torch.cat((x_offset,y_offset),1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
     prediction[:,:,:2] += x_y_offset

     #Log space transform the height and wisth
     anchors = torch.FloatTensor(anchors)

     if CUDA:
           anchors = anchors.cuda()

     anchors = anchors.repeat(grid_size*grid_size,1).unsqueeze(0)
     prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

     prediction[:,:,6: 6+num_classes] = torch.sigmoid((prediction[:,:,6:6+num_classes]))
     prediction[:,:,:4] *= stride

     return prediction



def gt_predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
     #Convert a output feature map to 2D tensor

     batch_size = prediction.size(0)
     stride = inp_dim // prediction.size(2)
     grid_size = inp_dim // stride
     num_anchors = len(anchors)
     bbox_attrs = 6 + num_classes

     prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
     prediction = prediction.transpose(1,2).contiguous()
     prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

     #anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

     #Sigmoid center_x,y and objectness score
     #prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
     #prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
     #prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

     #Center offsets
     #grid = np.arange(grid_size)
     #a, b = np.meshgrid(grid, grid)

     #x_offset = torch.FloatTensor(a).view(-1,1)
     #y_offset = torch.FloatTensor(b).view(-1,1)

     #if CUDA:
     #      x_offset = x_offset.cuda()
     #      y_offset = y_offset.cuda()

     #x_y_offset = torch.cat((x_offset,y_offset),1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
     #prediction[:,:,:2] += x_y_offset
 
     prediction[:,:,0] *= prediction[:,:,4] #Zero out non prediction boxes
     prediction[:,:,1] *= prediction[:,:,4] #Zero out non prediction boxes
     prediction[:,:,2] *= prediction[:,:,4] #Zero out non prediction boxes
     prediction[:,:,3] *= prediction[:,:,4] #Zero out non prediction boxes

     #Log space transform the height and wisth
     #anchors = torch.FloatTensor(anchors)

     #if CUDA:
     #      anchors = anchors.cuda()

     #anchors = anchors.repeat(grid_size*grid_size,1).unsqueeze(0)
     #prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

     #prediction[:,:,5: 5+num_classes] = torch.sigmoid((prediction[:,:,5:5+num_classes]))
     #prediction[:,:,:2] *= stride

     return prediction



def unique(tensor):
     tensor_np = tensor.cpu().numpy()
     unique_np = np.unique(tensor_np)
     unique_tensor = torch.from_numpy(unique_np)
     
     tensor_res = tensor.new(unique_tensor.shape)
     tensor_res.copy_(unique_tensor)
     return tensor_res

def bbox_iou(box1, box2):
     #Returns IoU of two boxes

     b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
     b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

     #Coord of intersect rect
     inter_rect_x1 = torch.max(b1_x1, b2_x1)
     inter_rect_y1 = torch.max(b1_y1, b2_y1)
     inter_rect_x2 = torch.min(b1_x2, b2_x2)
     inter_rect_y2 = torch.min(b1_y2, b2_y2)

     #Intersection area
     inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 +1, min=0)
     
     #Union Area
     b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
     b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

     iou = inter_area / (b1_area + b2_area - inter_area)

     return iou

def box_iou(box1, box2):
     if len(box1.nonzero()) == 0 or len(box2.nonzero()) == 0:
           return 0.0

     b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
     b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

     #Coord of intersect rect
     inter_rect_x1 = torch.max(b1_x1, b2_x1)
     inter_rect_y1 = torch.max(b1_y1, b2_y1)
     inter_rect_x2 = torch.min(b1_x2, b2_x2)
     inter_rect_y2 = torch.min(b1_y2, b2_y2)
     
     #Intersection area
     inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 +1, min=0)
     
     #Union Area
     b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
     b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

     iou = inter_area / (b1_area + b2_area - inter_area)

     return iou
     


def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
     #Convert 10647 bounding boxes to 1 per detection
     conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
     prediction = prediction*conf_mask #Makes entire prediction zero (below threshold)

     box_corner = prediction.new(prediction.shape) #Below converts x,y, width, height to x,y top left, x,y bottom right
     box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
     box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
     box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)
     box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
     prediction[:,:,:4] = box_corner[:,:,:4]

 
     batch_size = prediction.size(0)
     write = False

     for ind in range(batch_size):
           image_pred = prediction[ind]

           max_conf, max_conf_score = torch.max(image_pred[:,6:6+num_classes], 1) #max_conf_score has class label, max_conf has prob
           max_conf = max_conf.float().unsqueeze(1)
           max_conf_score = max_conf_score.float().unsqueeze(1)
           seq = (image_pred[:,:6], max_conf, max_conf_score)
           image_pred = torch.cat(seq, 1)
 
           non_zero_ind = (torch.nonzero(image_pred[:,4]))
           try:
                 image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,8)
           except:
                 continue

           if image_pred_.shape[0] == 0:
                 continue

           #Getting classes detected in an image
           img_classes = unique(image_pred_[:,-1])

           for cls in img_classes:
                 #Performing NMS
                 cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
                 class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
                 image_pred_class = image_pred_[class_mask_ind].view(-1,8)

                 #Sorting detections such that objectness score is at top
                 conf_sort_index = torch.sort(image_pred_class[:,4], descending = True)[1] 
                 image_pred_class = image_pred_class[conf_sort_index]
                 idx = image_pred_class.size(0) #Number of detections


                 for i in range(idx):
                       #IoUs of boxes that come after the one we looked at
                       try:
                             ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                       except ValueError:
                             break
                       except IndexError:
                             break

                       iou_mask = (ious < nms_conf).float().unsqueeze(1)
                       image_pred_class[i+1:] *= iou_mask

                       #Remove non-zero entries
                       non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                       image_pred_class = image_pred_class[non_zero_ind].view(-1,8)

                 batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
                 seq = batch_ind, image_pred_class

                 if not write:
                       output = torch.cat(seq,1)
                       write = True
                 else:
                       out = torch.cat(seq, 1)
                       output = torch.cat((output, out))

     try:
           return output
     except: 
           return 0      
                             
 
