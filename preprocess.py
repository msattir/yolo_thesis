from __future__ import division
import cv2
import numpy as np
from utils import letterbox_image2, letterbox_image3
import torch
from utils import gt_predict_transform

def y_pred(filter_size, masks, det, max_size, CUDA, num_classes=1):
     anchors = [(10,13),  (16,30),  (33,23),  (30,61),  (62,45),  (59,119),  (116,90),  (156,198),  (373,326)]
     anc_masks = []
     anc_masks = [anchors[i] for i in masks]
     anc_masks_aspect = [anc_masks[i][0]/anc_masks[i][1] for i in range(3)]
     anc_masks_aspect = np.asarray(anc_masks_aspect)
     x_dir, y_dir = filter_size
     jump_cell = np.zeros(2)
     jump_cell[0] = max_size[0]/filter_size[0]
     jump_cell[1] = max_size[1]/filter_size[1]
     predictions = np.zeros([3*(5+51+num_classes), filter_size[0], filter_size[1]])
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
 #                      print (p,i,j,'---', (det[loc[0],0]-(jump_cell[0]*i))/jump_cell[0], (det[loc[0],1]-(jump_cell[1]*j))/jump_cell[1],det[loc[0],2], det[loc[0],3], int(det[loc[0],4]==1), int(det[loc[0],4]==2), det[loc[0],:]) #Doesn't handle 2 boxes in same cell
                       #check IOU of bbox with anc_masks and populate those 
                       
                       cx = (det[loc[0],0]-(jump_cell[0]*i))/jump_cell[0]
                       cy = (det[loc[0],1]-(jump_cell[1]*j))/jump_cell[1]
                       class_prob = np.array([1])#np.eye(num_classes)[det[loc[0],4]-1]
                       im_aspect = det[loc[0],2]/det[loc[0],3]
                       m = np.argmin(abs(anc_masks_aspect-im_aspect))
                       start = m*(5+51+num_classes)
                       end = start+(5+51+num_classes)
                       predictions[start:end,i,j] = np.concatenate((np.concatenate((np.array([det[loc[0],0], det[loc[0],1], det[loc[0],2], det[loc[0],3], 1.0]), det[loc[0],4:])),class_prob))
                       #print (predictions[p,:])
                 p=p+1
     
     t_pred = torch.FloatTensor(predictions)

     x = t_pred.unsqueeze(0)
     #if CUDA:
     #      x = x.cuda()
     ret = gt_predict_transform(x, 416, anc_masks, num_classes, CUDA)
     return(ret)



def gt_pred(imlist, labellist, CUDA, num_classes):
     #Inputs:  imlist - List of Images
     #         labellist - List of Labels
     #         filt - Yolo grid sizes (defaults = 13, 26, 52)
     #         masks - Yolo dataset specific anchors (defaults = coco defaults)
     #         num_classes - No of classes in dataset
     #Returns: a tensor of shape [N, 10647, 5+num classes] of GT labels (10647 is for 416x416 with 13, 26, and 52 grid sizes and 3 anchors
     #         N - Number of images in imlist and labellist or batch size
     write = 0 
     for index, (im, lb) in enumerate(zip(imlist, labellist)):

           img = cv2.imread(im)
           #lab = (im.rsplit('train', 1)[0] + "labels" + im.rsplit('train', 1)[1]).replace('.jpg', '.txt') 
           lab = im.replace('images', 'labels').replace('.jpg', '.txt')
           o_det = np.genfromtxt(lab, delimiter=',')

           if o_det.ndim == 1:
                 o_det = o_det.reshape(1,-1)
 
           det = o_det[:,:-1]

           if det.ndim == 1:
                 det = det.reshape(1, -1)          
 
           canvas_shape = []
           img4, canvas_shape  = letterbox_image3(img,[416, 416])
           
           img2 = img4.astype(np.uint8)
           
           x_fact = canvas_shape[0]/img.shape[0]
           y_fact = canvas_shape[1]/img.shape[1]
           

           #Convert 2nd and 3rd to width and height
           #if any(det[:,2] < det[:,0]) or any(det[:,3] < det[:,1]):
           #      print ("wrong det", det)
           #      exit(0)
 
           #det[:,2] = det[:,2] - det[:,0] 
           #det[:,3] = det[:,3] - det[:,1]

           det[:,0] = det[:,0]*y_fact
           det[:,2] = det[:,2]*y_fact
           det[:,1] = det[:,1]*x_fact+int((416-canvas_shape[0])/2)
           det[:,3] = det[:,3]*x_fact
           
           det2 = det.copy()
           det2[:,0] = det2[:,0]+det2[:,2]/2
           det2[:,1] = det2[:,1]+det2[:,3]/2
           
           det2[:,2] = det[:,2]
           det2[:,3] = det[:,3]
           
           det2 = det2.astype(int)

           dele=[]
           for ix, d in enumerate(det2):
                 if any(d[:] <= 0):
                       dele.append(ix)

           det2 = np.delete(det2, dele, axis=0) 
        
          #for i in range(0,det.shape[0]):
          #       cv2.rectangle(img2, (det[i,0],det[i,1]), (det[i,2],det[i,3]), (0, 255, 0), 2)

          # cv2.imshow('image', img2)
          # cv2.waitKey(0)

          # for i in range(0,det2.shape[0]):
          #       cv2.circle(img2, (det2[i,0], det2[i,1]), 1, (255, 0, 0), 2)

          # for i in range(0,det2.shape[0]):
          #       cv2.circle(img2, (int(det2[i,0]-det2[i,2]/2),int(det2[i,1]-det2[i,3]/2)), 1, (0, 255, 0), 2)

          # for i in range(0,det2.shape[0]):
          #       cv2.circle(img2, (int(det2[i,0]+det2[i,2]/2),int(det2[i,1]+det2[i,3]/2)), 1, (0, 0, 255), 2)

          # cv2.imshow('image', img2)
          # cv2.waitKey(0)

        
           filts = [[13,13], [26,26], [52,52]]
           all_masks = [[6,7,8], [3,4,5], [0,1,2]]
           w_pred = 0
           for f, m in zip (filts, all_masks):
                 pred_1=y_pred(f, m, det2, [416, 416], CUDA, num_classes)
                 if w_pred == 0:
                       pred = pred_1
                       w_pred = 1
                 else:
                       pred = torch.cat((pred, pred_1),1)
           if write == 0:
                 output = pred
                 #output = output.unsqueeze(0)
                 write = 1
           else:
                 output = torch.cat((output, pred))
           
     return (output)


