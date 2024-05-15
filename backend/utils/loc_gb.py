import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import tqdm

data_path = '/media/miao/ptsdisk/dataset/localization/ILSVRC/'
with open(data_path + 'val_list.txt', 'r') as f:
    lines = f.readlines()
img_list = [line[:-1].split()[0] for line in lines]
with open('/media/miao/ptsdisk/dataset/localization/ILSVRC/val_box.txt') as f:
    lines1 = f.readlines()
label_list = [line[:-1].split()[1] for line in lines1]

annotations = scio.loadmat(data_path + 'val_box.mat')['rec'][0]
bboxes = []
for ia in range(len(annotations)):
    bbox = []
    for bb in range(len(annotations[ia][0][0])):
        xyxy = annotations[ia][0][0][bb][1][0]
        bbox.append(xyxy.tolist())
    bboxes.append(bbox)
print('Number of processing images: ', len(bboxes)) 

def large_rect(rect):
    # find largest recteangles
    large_area = 0
    target = 0
    for i in range(len(rect)):
        area = rect[i][2]*rect[i][3]
        if large_area < area:
            large_area = area
            target = i

    x = rect[target][0]
    y = rect[target][1]
    w = rect[target][2]
    h = rect[target][3]

    return x, y, w, h

data_path1 = '/media/miao/ptsdisk/Project/hcrm/vgg_imagenet/exp2/hcrm/relu3_3/'
save_path = '/media/miao/ptsdisk/Project/hcrm/vgg_imagenet/exp2/hcrm/results/relu3_3/'

for i in range(0, 10):
    im_name = img_list[i]
    la = label_list[i]
    im = cv2.imread(data_path + 'val/' + im_name[:-5] + '.JPEG', 1)
    cam = cv2.imread('{}{}_{}.png'.format(data_path1, im_name[:-5], la), 0) / 255.0
    height, width = cam.shape[:2]
    # gradcut  
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[cam > 0.4] = 1
    mask[cam <= 0.4] = 2
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    I1, bg, fg = cv2.grabCut(im, mask, None, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_MASK)
    cam1 = (np.where((I1 == 2) | (I1 == 0), 0, 1) * 255).astype(np.uint8)

    # _, thred_gray_heatmap = cv2.threshold(cam1,int(0.3*255), 255, cv2.THRESH_TOZERO)
    # _, contours, _ = cv2.findContours(thred_gray_heatmap, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # # calculate bbox coordinates
    # rect = []
    # for c in contours:
    #     x, y, w, h = cv2.boundingRect(c)
    #     rect.append([x,y,w,h])

    # if rect == []:
    #     estimated_box = [0,0,1,1] #dummy
    # else:
    #     x,y,w,h = large_rect(rect)
    #     estimated_box = [x,y,x+w,y+h]
    # 	cv2.rectangle(hot_cam, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 	for j in range(len(bboxes[i])):
    # 		x1,y1,x2,y2 = bboxes[i][j]
    # 		cv2.rectangle(hot_cam, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(save_path + im_name[:-5] + '.png', cam1)

