#Convert person_keypoint.json to independent label files

import json
import sys

if len(sys.argv) < 3:
    print ('Usage: python coco_prepare.py <path_to_annotations_file.json>')
    exit(0)

read_txt = str(sys.argv[2])

with open(read_txt) as f:
    data=json.load(f)

for i in range(4):#range(len(data['annotations'])):
    lab_txt = str(data['annotations'][i]['image_id']).rjust(12,'0')+'.txt'
    f = open(lab_txt, 'w')

    write_str = str(data['annotations'][i]['bbox'])[1:-1]+','
    write_str += str(data['annotations'][i]['keypoints'])[1:-1]+'\n'
 
    f.write(write_str)
    f.close()

    seg_txt = 'seg/'+str(data['annotations'][i]['image_id']).rjust(12,'0')+'.txt'
    f_seg = open(seg_txt, 'w')

    write_str = str(data['annotations'][i]['bbox'])[1:-1])+','
    write_str += str(data['annotations'][i]['keypoints'])[1:-1]+','
    write_str += str(data['annotations'][i]['segmentation'])+'\n'
 
    f_seg.write(write_str)
    f_seg.close()
