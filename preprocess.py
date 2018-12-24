from __future__ import division
import cv2
import numpy as np
from utils import letterbox_image2
import torch

def y_pred(filter_size, masks, det, max_size, num_classes=1):
  anchors = [(10,13),  (16,30),  (33,23),  (30,61),  (62,45),  (59,119),  (116,90),  (156,198),  (373,326)]
  anc_masks = []
  anc_masks = [anchors[i] for i in masks]
  anc_masks_aspect = [anc_masks[i][0]/anc_masks[i][1] for i in range(3)]
  anc_masks_aspect = np.asarray(anc_masks_aspect)
  x_dir, y_dir = filter_size
  jump_cell = np.zeros(2)
  jump_cell[0] = max_size[0]/filter_size[0]
  jump_cell[1] = max_size[1]/filter_size[1]
  predictions = np.zeros([3, filter_size[0]*filter_size[1], 5+num_classes])
#  print (predictions.shape)
  p = 0
  for i in range(1,filter_size[0]):
     for j in range(1,filter_size[1]):
        a1 = det[:,0]-jump_cell[0]*i >= 0
        a2 = det[:,0]-jump_cell[0]*i < jump_cell[0]
        b1 = det[:,1]-jump_cell[1]*j >= 0
        b2 = det[:,1]-jump_cell[1]*j < jump_cell[1]
        if any((a1*a2)*(b1*b2)):
           loc_x = np.where(a1*a2)[0]
           loc_y = np.where(b1*b2)[0]
           loc = list(set(loc_x) & set(loc_y))
 #          print (p,i,j,'---', (det[loc[0],0]-(jump_cell[0]*i))/jump_cell[0], (det[loc[0],1]-(jump_cell[1]*j))/jump_cell[1],det[loc[0],2], det[loc[0],3], int(det[loc[0],4]==1), int(det[loc[0],4]==2), det[loc[0],:]) #Doesn't handle 2 boxes in same cell
           #check IOU of bbox with anc_masks and populate those 
           
           cx = (det[loc[0],0]-(jump_cell[0]*i))/jump_cell[0]
           cy = (det[loc[0],1]-(jump_cell[1]*j))/jump_cell[1]
           class_prob = np.eye(num_classes)[det[loc[0],4]-1]
           im_aspect = det[loc[0],2]/det[loc[0],3]
           m = np.argmin(abs(anc_masks_aspect-im_aspect))
           predictions[m, p,:] = np.concatenate((np.array([1, cx, cy, det[loc[0],2], det[loc[0],3]]), class_prob))
           #print (predictions[p,:])
        p=p+1
  
  t_pred = torch.FloatTensor(predictions)

  return (torch.cat((t_pred[0,:,:], t_pred[1,:,:], t_pred[2,:,:]),0))

def gt_pred(img, filt, masks):
  img = cv2.imread('datasets/I1_2009_09_08_drive_0004_000951.png')
  det = np.genfromtxt('datasets/my_csv.csv', delimiter=',')
  
  canvas_shape = []
  img4, canvas_shape  = letterbox_image2(img,[416, 416])
  
  img2 = img4.astype(np.uint8)
  
  x_fact = canvas_shape[0]/img.shape[0]
  y_fact = canvas_shape[1]/img.shape[1]
  
  det[:,0] = det[:,0]*y_fact
  det[:,2] = det[:,2]*y_fact
  det[:,1] = det[:,1]*x_fact+int((416-canvas_shape[0])/2)
  det[:,3] = det[:,3]*x_fact
  
  det2 = det.copy()
  
  det[:,2] = det[:,0]+det[:,2]
  det[:,3] = det[:,1]+det[:,3]
  
  
  det = det.astype(int)
  for i in range(0,det.shape[0]):
    cv2.rectangle(img2, (det[i,0],det[i,1]), (det[i,2],det[i,3]), (0, 255, 0), 2)
  
  
  det2[:,0] = det2[:,0]+det2[:,2]/2
  det2[:,1] = det2[:,1]+det2[:,3]/2
  
  det2 = det2.astype(int)
  for i in range(0,det.shape[0]):
    cv2.circle(img2, (det2[i,0],det2[i,1]), 1, (255, 0, 0), 2)
  
  pred=y_pred(filt, masks, det2, [416, 416], 2)
  return (pred)
#cv2.imshow('image', img2)
#cv2.waitKey(0)
#cv2.imshow('image',img)


