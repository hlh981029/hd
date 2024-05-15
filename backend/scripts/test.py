import sys
sys.path.append('/home/miao/Projects/hcrm/')

import os
import cv2
import torch
import shutil
import argparse
import numpy as np
import torchvision
from models import *
import torch.nn as nn
from tqdm import tqdm
import scipy.io as scio
from os.path import exists
from torch.optim import SGD
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.Restore import restore
from torch.utils.data import DataLoader
from utils.imutils import large_rect, bb_IOU
from torchvision import models, transforms
from utils.LoadData import test_data_loader, test_msf_data_loader, test_data_loader_caffe



ROOT_DIR = '/'.join(os.getcwd().split('/')[:-1])

def get_arguments():
    parser = argparse.ArgumentParser(description='ACoL')
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR)
    parser.add_argument("--save_dir", type=str, default='')
    parser.add_argument("--save_dir1", type=str, default='')
    parser.add_argument("--img_dir", type=str, default='')
    parser.add_argument("--train_list", type=str, default='')
    parser.add_argument("--test_list", type=str, default='')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--scales", type=list, default=[0.5, 0.75, 1, 1.25, 1.5, 2])
    parser.add_argument("--arch", type=str,default='vgg_v0')
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tencrop", type=str, default='False')
    parser.add_argument("--onehot", type=str, default='True')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--disp_interval", type=int, default=40)
    parser.add_argument("--snapshot_dir", type=str, default='')
    parser.add_argument("--resume", type=str, default='True')
    parser.add_argument("--restore_from", type=str, default='')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--current_epoch", type=int, default=0)

    return parser.parse_args()

#def get_model(args):
#    #model = eval(args.arch).model(num_classes=args.num_classes, args=args, threshold=args.threshold)
#    model = eval(args.arch).vgg16(pretrained=False, num_classes=args.num_classes)
#    #model = eval(args.arch).resnet50(pretrained=False, num_classes=args.num_classes)
#    model = model.cuda()
#    
#    pretrained_dict = torch.load(args.restore_from)['state_dict']
#    model_dict = model.state_dict()
#    
#    print(model_dict.keys())
#    print(pretrained_dict.keys())
#    
#    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
#    print("Weights cannot be loaded:")
#    print([k for k in model_dict.keys() if k not in pretrained_dict.keys()])
#
#    model_dict.update(pretrained_dict)
#    model.load_state_dict(model_dict)
#
#    return  model

def get_model(args):
    print(args.arch)
    #model = eval(args.arch).model(num_classes=args.num_classes, args=args, threshold=args.threshold)
    model = eval(args.arch).vgg16(pretrained=False, num_classes=args.num_classes)
    model = model.cuda()
  
    pretrained_dict = torch.load(args.restore_from)['state_dict']
    #pretrained_dict = torch.load(args.restore_from)
    model_dict = model.state_dict()
  
    print(model_dict.keys())
    print(pretrained_dict.keys())
  
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict.keys()}
    print("Weights cannot be loaded:")
    print([k for k in model_dict.keys() if k not in pretrained_dict.keys()])

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return  model

def get_torch_model(args):
    model = eval(args.arch).vgg16(pretrained=False, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.restore_from))
    model = model.cuda()
    
    return  model


##################### CAM ####################
#def val_cam(args, model=None, current_epoch=0):
#    if model is None:
#        model = get_model(args)
#    model.eval()
#    train_loader, val_loader = test_data_loader(args)
#    target_layers = [37]
#    if not exists(args.save_dir):
#        os.mkdir(args.save_dir)
#
#    for idx, dat in tqdm(enumerate(val_loader)):
#        img_name, img, label_in = dat
#        img = img.cuda(non_blocking=True)
#        ####forward         
#        feature_maps, logits = model.forward_cam(img, target_layers)
#        ####extract feature
#        target = feature_maps[0].cpu().data.numpy()[0]
#        logits = torch.sigmoid(logits)
#        
#        cv_im = cv2.imread(img_name[0])
#        cv_im_gray = cv2.cvtColor(cv_im, cv2.COLOR_BGR2GRAY)
#        height, width = cv_im.shape[:2]
#        
#        im_name = args.save_dir + img_name[0].split('/')[-1][:-4]
#        labels = label_in.long().numpy()[0]
#
#        for i in range(int(args.num_classes)):
#            if labels[i] == 1:                
#                cam = target[i] 
#                cam = np.maximum(cam, 0)
#                cam /= (cam.max() + 1e-8)  # Normalize between 0-1
#                cam = np.uint8(cam * 255)  
#                cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)
#
#                # Scale between 0-255 to visualize
#                out_name1 = im_name + '_{}.png'.format(i)
#                cam = cv_im_gray * 0.2 + cam * 0.8
#                cv2.imwrite(out_name1, att)
#                #plt.imsave(out_name2, cam, cmap=plt.cm.viridis)


def val_gradcam(args, model=None, current_epoch=0):   
    if model is None:
        model = get_model(args)
    model.eval()
    train_loader, val_loader = test_data_loader(args)
    target_layers = [18]
    if not exists(args.save_dir):
        os.mkdir(args.save_dir)

    for idx, dat in tqdm(enumerate(val_loader)):
        img_name, img, label_in = dat
        img = img.cuda(non_blocking=True)
        ####forward         
        feature_maps, logits = model.forward_cam(img, target_layers)
        ####extract feature
        target = feature_maps[0].cpu().data.numpy()[0]
        logits1 = torch.sigmoid(logits)
        
        cv_im = cv2.imread(img_name[0])
        cv_im_gray = cv2.cvtColor(cv_im, cv2.COLOR_BGR2GRAY)
        height, width = cv_im.shape[:2]
        
        im_name = args.save_dir + img_name[0].split('/')[-1][:-4]
        labels = label_in.long().numpy()[0]

        for i in range(int(args.num_classes)):
            if labels[i] == 1:
                # Target for backprop
                one_hot_output = torch.FloatTensor(1, logits.size()[-1]).zero_()
                one_hot_output[0][i] = 1
                one_hot_output = one_hot_output.cuda(non_blocking=True)
                # Zero grads
                model.zero_grad()
                # Backward pass with specified target
                logits.backward(gradient=one_hot_output, retain_graph=True)
               
                guided_gradients = model.gradients[0].cpu().numpy()[0]
                model.gradients = []
               
                weights = np.mean(guided_gradients, axis=(1, 2))  
                cam = np.zeros(target.shape[1:], dtype=np.float32) 
                for c, w in enumerate(weights):
                    cam += w * target[c, :, :]
                
                cam = np.maximum(cam, 0)
                cam /= (cam.max() + 1e-8)  # Normalize between 0-1
                cam = np.uint8(cam * 255)  
                cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)

                # Scale between 0-255 to visualize
                out_name = im_name + '_{}.png'.format(i)
                cam = cv_im_gray * 0.2 + cam * 0.8
                #cv2.imwrite(out_name, att)
                plt.imsave(out_name, cam, cmap=plt.cm.viridis)
                
def val_cub_gradcam(args, model=None, current_epoch=0):
    if model is None:
        model = get_model(args)
    model.eval()
    train_loader, val_loader = test_data_loader(args)
    global_counter = 0
    cls_counter = 0
    cls5_counter = 0
    loc_counter = 0
    loc5_counter = 0
    gt_known = 0
    all_recalls = 0.0 
    all_ious = 0.0
    
    with open('/media/miao/ptsdisk/dataset/localization/CUB_200_2011/test_bounding_boxes.txt') as f:
        lines = f.readlines()
    bboxes = [line[:-1].split() for line in lines] 
    print('Number of processing images: ', len(bboxes)) 
    
    target_layers = [29]
    if not exists(args.save_dir):
        os.makedirs(args.save_dir)

    for idx, dat in tqdm(enumerate(val_loader)):
        img_name, img, label_in = dat
        img = img.cuda()
        label = label_in.long().numpy()[0]
        
        bbox = bboxes[global_counter]
        gt_box = [float(coor) for coor in bbox[:-2]]
        gt_box[2] += gt_box[0]
        gt_box[3] += gt_box[1]
        height, width = int(bbox[-2]), int(bbox[-1]) 
        
        ####forward         
        feature_maps, logits = model.forward_cam(img, target_layers)
        ####extract feature
        target = feature_maps[0].cpu().data.numpy()[0]
        logits1 = torch.sigmoid(logits)
        
        #######top1 cls
        prob, ind = logits1.max(1)
        ind = ind.cpu().data.numpy()
        if ind == label:
            cls_counter += 1  
        #######top5 cls
        probs, inds = torch.sort(-logits1, 1)
        inds = inds.cpu().data.numpy()      
        if label in inds[0, :5]:
            cls5_counter += 1
        global_counter += 1    
        
        cv_im = cv2.imread(img_name[0])
        cv_im_gray = cv2.cvtColor(cv_im, cv2.COLOR_BGR2GRAY)        
        im_name = args.save_dir + img_name[0].split('/')[-1][:-4]

        # Target for backprop
        one_hot_output = torch.FloatTensor(1, logits.size()[-1]).zero_()
        one_hot_output[0][label] = 1
        one_hot_output = one_hot_output.cuda(non_blocking=True)
        # Zero grads
        model.zero_grad()
        # Backward pass with specified target
        logits.backward(gradient=one_hot_output, retain_graph=True)

        guided_gradients = model.gradients[0].cpu().numpy()[0]
        model.gradients = []

        weights = np.mean(guided_gradients, axis=(1, 2))  
        pre_cam = np.reshape(weights, (-1, 1, 1)) * target
        cam = np.sum(pre_cam, 0)
        #for c, w in enumerate(weights):
        #    cam += w * target[c, :, :]

        cam = np.maximum(cam, 0)
        cam /= (cam.max() + 1e-8)  # Normalize between 0-1
        cam = np.uint8(cam * 255)  
        att = cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)
                      
        _, binary_att = cv2.threshold(att, 0.15*255, 255, cv2.THRESH_BINARY) 
        _, contours, _ = cv2.findContours(binary_att, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rect = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            rect.append([x, y, w, h])
        if rect == []:
            estimated_box = [0, 0, 1, 1]
        else:
            x, y, w, h = large_rect(rect)
            estimated_box = [x,y,x+w,y+h]

        #print(estimated_box, gt_box)
        iou, recall = bb_IOU(gt_box, estimated_box)
        all_recalls += recall
        all_ious += iou
        #print('iou', iou, estimated_box, gt_box)
        if iou >= 0.5:
            gt_known += 1
            if ind == label:
                loc_counter += 1
            if label in inds[:5]:
                loc5_counter += 1
    
    gt_acc = gt_known / float(global_counter)
    cls_acc = cls_counter / float(global_counter)
    cls5_acc = cls5_counter / float(global_counter)
    loc_acc = loc_counter / float(global_counter)
    loc5_acc = loc5_counter / float(global_counter)
    ave_recall = all_recalls / float(global_counter)
    ave_iou = all_ious / float(global_counter)

    print('Top 1 classification accuracy: {}\t'
          'Top 5 classification accuracy: {}\t'
          'GT-known accuracy: {}\t'
          'Top 1 localization accuracy: {}\t'
          'Top 5 localization accuracy: {}\t'
          'The mean recall of top-1 class: {}\t'
         'The mean iou of top-1 class: {}\t'.format(cls_acc, cls5_acc, gt_acc, loc_acc, loc5_acc, ave_recall, ave_iou))

# grad-cam with single crop
def val_sc_gradcam(args, model=None, current_epoch=0):   
    #print('in val_oc_gradcam: relu, 0.15, 29')
    if model is None:
        #model = get_model(args)
        model = get_torch_model(args)
    model.eval()
    val_loader = test_data_loader_caffe(args)
    global_counter = 0
    cls_counter = 0
    cls5_counter = 0
    loc_counter = 0
    loc5_counter = 0
    gt_known = 0
    all_recalls = 0.0 
    all_ious = 0.0
        
    annotations = scio.loadmat('/media/miao/ptsdisk/dataset/localization/ILSVRC/val_box.mat')['rec'][0]
    bboxes = []
    for ia in range(len(annotations)):
        bbox = []
        for bb in range(len(annotations[ia][0][0])):
            xyxy = annotations[ia][0][0][bb][1][0]
            bbox.append(xyxy.tolist())
        bboxes.append(bbox)
    print('Number of processing images: ', len(bboxes)) 
    
    with open('/media/miao/ptsdisk/dataset/localization/ILSVRC/val_box.txt') as f:
        lines = f.readlines()
    labels = [line[:-1].split()[1] for line in lines]
    
    # # for relu5_3
    #target_layers = [29]

    # # for relu4_3
    #target_layers = [22]

    # for relu3_3
    #target_layers = [15]

    # for relu2_2
    #target_layers = [8]

    # for relu1_2
    target_layers = [3]

    if not exists(args.save_dir):
        os.makedirs(args.save_dir)

    for idx, dat in tqdm(enumerate(val_loader)):
        img_name, img, label_in = dat
        img = img.cuda() 
        bbox = bboxes[global_counter]
        la = int(labels[idx])
        
        ####forward         
        feature_maps, logits = model.forward_cam(img, target_layers)

        ####extract feature
        target = feature_maps[0]  #target 1*512*14*14
        logits1 = torch.softmax(logits, 1)

        #######top1 cls
        prob, ind = logits1.max(1)
        ind = ind.cpu().data.numpy()
        if ind == la:
            cls_counter += 1  
        #######top5 cls
        probs, inds = torch.sort(-logits1, 1)
        inds = inds.cpu().data.numpy()
        
        if la in inds[0, :5]:
            cls5_counter += 1
        global_counter += 1
        
        # read original image
        cv_im = cv2.imread(img_name[0])
        cv_im_gray = cv2.cvtColor(cv_im, cv2.COLOR_BGR2GRAY)
        height, width = cv_im.shape[:2]
        
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, logits.size()[-1]).zero_()
        one_hot_output[0][la] = 1
        one_hot_output = one_hot_output.cuda(non_blocking=True)
        # Zero grads
        model.zero_grad()
        # Backward pass with specified target
        logits.backward(gradient=one_hot_output, retain_graph=True)
        # grad-cam
        guided_gradients = model.gradients[0]
        model.gradients = []
        weights = torch.mean(guided_gradients, dim=(2, 3))
        weights = weights.cuda()
        pre_cam = (target * (weights.view(1, -1, 1, 1)))
        #pre_cam = torch.nn.functional.relu(pre_cam)
        pre_cam = pre_cam.sum(1).unsqueeze(0)
        cam = torch.nn.functional.interpolate(pre_cam, size=(224,224), \
                                mode='bilinear', align_corners=True).squeeze(0) #1*224*224
        cam = cam.detach().cpu().numpy()[0]
        
        # center crop fill 
        # cam = np.zeros((256, 256), dtype=np.float32)
        # cam[16:240, 16:240] = pre_cam
        
        cam = np.maximum(cam, 0)
        cam /= (cam.max() + 1e-8)  # Normalize between 0-1
        cam = np.uint8(cam * 255)  
        cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)

        # Scale between 0-255 to visualize
        im_name = args.save_dir + img_name[0].split('/')[-1][:-5]
        out_name = im_name + '_{}_1.png'.format(la)
        #cam1 = cv_im_gray * 0.5 + cam * 0.5
        #cv2.imwrite(out_name, cam1)
        #plt.imsave(out_name, cam1, cmap=plt.cm.viridis)
        
        cmap = plt.cm.jet_r(cam)[..., :3] * 255.0
        cam1 = (cmap.astype(np.float) + cv_im.astype(np.float)) / 2
        cv2.imwrite(out_name, cam1)

        _, binary_att = cv2.threshold(cam, 0.15*255, 255, cv2.THRESH_BINARY) 
        _, contours, _ = cv2.findContours(binary_att, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rect = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            rect.append([x, y, w, h])
        if rect == []:
            estimated_box = [0, 0, 1, 1]
        else:
            x, y, w, h = large_rect(rect)
            estimated_box = [x,y,x+w,y+h]

        #print(estimated_box, gt_box)
        ious = []
        recalls = []
        for ii in range(len(bbox)):
            iou, recall = bb_IOU(bbox[ii], estimated_box)
            ious.append(iou)
            recalls.append(recall)
        iou = max(ious)
        recall = recalls[ious.index(iou)]
        
        all_recalls += recall
        all_ious += iou
        #print('iou', iou, estimated_box, gt_box)
        if iou >= 0.5:
            gt_known += 1
            if ind == la:
                loc_counter += 1
            if la in inds[0, :5]:
                loc5_counter += 1

    gt_acc = gt_known / float(global_counter)
    cls_acc = cls_counter / float(global_counter)
    cls5_acc = cls5_counter / float(global_counter)
    loc_acc = loc_counter / float(global_counter)
    loc5_acc = loc5_counter / float(global_counter)
    ave_recall = all_recalls / float(global_counter)
    ave_iou = all_ious / float(global_counter)

    print('Top 1 classification accuracy: {}\t'
          'Top 5 classification accuracy: {}\t'
          'GT-known accuracy: {}\t'
          'top 1 localization accuracy: {}\t'
          'top 5 localization accuracy: {}\t'
          'The mean recall of top-1 class: {}\t'
         'The mean iou of top-1 class: {}\t'.format(cls_acc, cls5_acc, gt_acc, loc_acc, loc5_acc, ave_recall, ave_iou))

## grad-cam with ten crop
def val_oc_gradcam(args, model=None, current_epoch=0):   
    #print('in val_oc_gradcam: relu, 0.15, 29')
    args.tencrop = True
    if model is None:
        #model = get_model(args)
        model = get_torch_model(args)
    model.eval()
    val_loader = test_data_loader_caffe(args)
    global_counter = 0
    cls_counter = 0
    cls5_counter = 0
    loc_counter = 0
    loc5_counter = 0
    gt_known = 0
    all_recalls = 0.0 
    all_ious = 0.0
        
    annotations = scio.loadmat('/media/miao/ptsdisk/dataset/localization/ILSVRC/val_box.mat')['rec'][0]
    bboxes = []
    for ia in range(len(annotations)):
        bbox = []
        for bb in range(len(annotations[ia][0][0])):
            xyxy = annotations[ia][0][0][bb][1][0]
            bbox.append(xyxy.tolist())
        bboxes.append(bbox)
    print('Number of processing images: ', len(bboxes)) 
    
    with open('/media/miao/ptsdisk/dataset/localization/ILSVRC/val_box.txt') as f:
        lines = f.readlines()
    labels = [line[:-1].split()[1] for line in lines]
    
    #target-layer
    target_layers = [15]
    if not exists(args.save_dir):
        os.makedirs(args.save_dir)

    for idx, dat in tqdm(enumerate(val_loader)):
        img_name, img, label_in = dat
        img = img.cuda() 
        img = img.squeeze(0)
        bbox = bboxes[global_counter]
        la = int(labels[idx])
        
        ####forward         
        feature_maps, logits = model.forward_cam(img, target_layers)

        ####extract feature
        target = feature_maps[0]  #target 10*512*14*14
        logits = logits.sum(dim=0).unsqueeze(0)
        logits1 = torch.softmax(logits, dim=1)
        
        #######top1 cls
        prob, ind = logits1.max(1)
        ind = ind.cpu().data.numpy()
        if ind == la:
            cls_counter += 1  
        #######top5 cls
        probs, inds = torch.sort(-logits1, 1)
        inds = inds.cpu().data.numpy()
        
        if la in inds[0, :5]:
            cls5_counter += 1
        global_counter += 1
        
        # read original image
        cv_im = cv2.imread(img_name[0])
        cv_im_gray = cv2.cvtColor(cv_im, cv2.COLOR_BGR2GRAY)
        height, width = cv_im.shape[:2]
        
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, logits.size()[-1]).zero_()
        one_hot_output[0][la] = 1
        one_hot_output = one_hot_output.cuda(non_blocking=True)
        # Zero grads
        model.zero_grad()
        # Backward pass with specified target
        logits.backward(gradient=one_hot_output, retain_graph=True)
        # grad-cam
        guided_gradients = model.gradients[0]
        model.gradients = []
        weights = torch.mean(guided_gradients, dim=(2, 3))
        weights = weights.cuda()
        cam_ten_crop = (target * (weights.view(10, -1, 1, 1)))
        #cam_ten_crop = torch.nn.functional.relu(cam_ten_crop)
        cam_ten_crop = cam_ten_crop.sum(1)

        #upsample cam_ten_crop to 224*224
        cam_ten_crop_upsample = torch.unsqueeze(cam_ten_crop, 1) #10*1*14*14
        cam_ten_crop_upsample = torch.nn.functional.interpolate(cam_ten_crop_upsample, size=(224,224), \
                                mode='bilinear', align_corners=True).squeeze(0) #10*224*224
        #0 - topleft
        #1 - topright
        #2 - bottomleft
        #3 - bottomright
        #4 - center
        #5 - topleft - flip
        #6 - topright - flip
        #7 - bottomleft - flip
        #8 - bottomright - flip
        #9 - center - flip
        # cam_ten_crop_upsample_pad: a list of upsampled 10 * [224 * 224], padding_value = 0
        cam_ten_crop_upsample_pad = []
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[0], pad=(0, 32, 0, 32))]
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[1], pad=(32, 0, 0, 32))]
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[2], pad=(0, 32, 32, 0))]
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[3], pad=(32, 0, 32, 0))]
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[4], pad=(16, 16, 16, 16))]
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[5], pad=(0, 32, 0, 32)).flip([2])] #1-vertical, 2-hori
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[6], pad=(32, 0, 0, 32)).flip([2])]
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[7], pad=(0, 32, 32, 0)).flip([2])]
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[8], pad=(32, 0, 32, 0)).flip([2])]
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[9], pad=(16, 16, 16, 16)).flip([2])]

        cam = torch.tensor(np.zeros((256, 256), dtype=np.float32)).cuda()

        for i in range(len(cam_ten_crop_upsample_pad)):
            cam += cam_ten_crop_upsample_pad[i][0]
        
        cam = cam.detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam /= (cam.max() + 1e-8)  # Normalize between 0-1
        cam = np.uint8(cam * 255)  
        cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)

        # # Scale between 0-255 to visualize
        # im_name = args.save_dir + img_name[0].split('/')[-1][:-5]
        # out_name = im_name + '_{}.png'.format(la)
        # #cam1 = cv_im_gray * 0.2 + cam * 0.8
        # cv2.imwrite(out_name, cam)
        # #plt.imsave(out_name, cam1, cmap=plt.cm.viridis)

        _, binary_att = cv2.threshold(cam, 0.15*255, 255, cv2.THRESH_BINARY) 
        _, contours, _ = cv2.findContours(binary_att, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rect = []
        for c in contours:
           x, y, w, h = cv2.boundingRect(c)
           rect.append([x, y, w, h])
        if rect == []:
           estimated_box = [0, 0, 1, 1]
        else:
           x, y, w, h = large_rect(rect)
           estimated_box = [x,y,x+w,y+h]

        #print(estimated_box, gt_box)
        ious = []
        recalls = []
        for ii in range(len(bbox)):
           iou, recall = bb_IOU(bbox[ii], estimated_box)
           ious.append(iou)
           recalls.append(recall)
        iou = max(ious)
        recall = recalls[ious.index(iou)]

        all_recalls += recall
        all_ious += iou
        #print('iou', iou, estimated_box, gt_box)
        if iou >= 0.5:
           gt_known += 1
           if ind == la:
               loc_counter += 1
           if la in inds[0, :5]:
               loc5_counter += 1

    gt_acc = gt_known / float(global_counter)
    cls_acc = cls_counter / float(global_counter)
    cls5_acc = cls5_counter / float(global_counter)
    loc_acc = loc_counter / float(global_counter)
    loc5_acc = loc5_counter / float(global_counter)
    ave_recall = all_recalls / float(global_counter)
    ave_iou = all_ious / float(global_counter)

    print('Top 1 classification accuracy: {}\t'
         'Top 5 classification accuracy: {}\t'
         'GT-known accuracy: {}\t'
         'top 1 localization accuracy: {}\t'
         'top 5 localization accuracy: {}\t'
         'The mean recall of top-1 class: {}\t'
        'The mean iou of top-1 class: {}\t'.format(cls_acc, cls5_acc, gt_acc, loc_acc, loc5_acc, ave_recall, ave_iou))

               
##################### hcrm ####################
def val_hcrm(args, model=None, current_epoch=0):   
    if model is None:
        #model = get_model(args)
        model = get_torch_model(args)
    model.eval()
    train_loader, val_loader = test_data_loader(args)
    target_layers = [29]
    if not exists(args.save_dir):
        os.mkdir(args.save_dir)

    for idx, dat in tqdm(enumerate(val_loader)):
        img_name, img, label_in = dat
        img = img.cuda(non_blocking=True)
        ####forward         
        feature_maps, logits = model.forward_cam(img, target_layers)
        ####extract feature
        target = feature_maps[0].cpu().data.numpy()[0]
        logits1 = torch.sigmoid(logits)
        
        cv_im = cv2.imread(img_name[0])
        cv_im_gray = cv2.cvtColor(cv_im, cv2.COLOR_BGR2GRAY)
        height, width = cv_im.shape[:2]
        
        im_name = args.save_dir + img_name[0].split('/')[-1][:-4]
        labels = label_in.long().numpy()[0]

        for i in range(int(args.num_classes)):
            if labels[i] == 1:
                # Target for backprop
                one_hot_output = torch.FloatTensor(1, logits.size()[-1]).zero_()
                one_hot_output[0][i] = 1
                one_hot_output = one_hot_output.cuda(non_blocking=True)
                # Zero grads
                model.zero_grad()
                # Backward pass with specified target
                logits.backward(gradient=one_hot_output, retain_graph=True)
               
                guided_gradients = model.gradients[0].cpu().numpy()[0]
                model.gradients = []
               
                pre_cam = target * guided_gradients
                pre_cam[pre_cam < 0] = 0
                cam = np.sum(pre_cam, axis=0)

                cam = np.maximum(cam, 0)
                cam /= (cam.max() + 1e-8)  # Normalize between 0-1
                cam = np.uint8(cam * 255)  
                cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)

                # Scale between 0-255 to visualize
                out_name = im_name + '_{}_pool4.png'.format(i)
                cam = cv_im_gray * 0.2 + cam * 0.8
                #cv2.imwrite(out_name, cam)
                plt.imsave(out_name, cam, cmap=plt.cm.viridis)

#########hcrm with single crop
def val_sc_hcrm(args, model=None, current_epoch=0):   
    args.tencrop = False
    if model is None:
        #model = get_model(args)
        model = get_torch_model(args)
    model.eval()
    val_loader = test_data_loader_caffe(args)
    #train_loader, val_loader = test_data_loader(args)
    global_counter = 0
    cls_counter = 0
    cls5_counter = 0
    loc_counter = 0
    loc5_counter = 0
    gt_known = 0
    all_recalls = 0.0 
    all_ious = 0.0
        
    annotations = scio.loadmat('/media/miao/ptsdisk/dataset/localization/ILSVRC/val_box.mat')['rec'][0]
    bboxes = []
    for ia in range(len(annotations)):
        bbox = []
        for bb in range(len(annotations[ia][0][0])):
            xyxy = annotations[ia][0][0][bb][1][0]
            bbox.append(xyxy.tolist())
        bboxes.append(bbox)
    print('Number of processing images: ', len(bboxes)) 
    
    with open('/media/miao/ptsdisk/dataset/localization/ILSVRC/val_box.txt') as f:
        lines = f.readlines()
    labels = [line[:-1].split()[1] for line in lines]
    
    #target_layers = [29, 30]
    #length = 14 
    
    #target_layers = [22, 23]
    #length = 28
    
    target_layers = [15, 16]
    length = 56
    
    # target_layers = [3, 4]
    # length = 224

    if not exists(args.save_dir):
        os.makedirs(args.save_dir)

    for idx, dat in tqdm(enumerate(val_loader)):
        img_name, img, label_in = dat
        img = img.cuda()
        bbox = bboxes[global_counter]
        la = int(labels[idx])
        #la = label_in.long().numpy()[0]
        ####forward         
        feature_maps, logits = model.forward_cam(img, target_layers)
        ####extract feature
        target = feature_maps[0] 
        logits = logits.sum(dim=0).unsqueeze(0)
        logits1 = torch.softmax(logits, dim=1)
        
        #######top1 cls
        prob, ind = logits1.max(1)
        ind = ind.cpu().data.numpy()
        if ind == la:
            cls_counter += 1  
        #######top5 cls
        probs, inds = torch.sort(-logits1, 1)
        inds = inds.cpu().data.numpy()
        if la in inds[0, :5]:
            cls5_counter += 1
        global_counter += 1

        cv_im = cv2.imread(img_name[0])
        cv_im_gray = cv2.cvtColor(cv_im, cv2.COLOR_BGR2GRAY)
        height, width = cv_im.shape[:2]
        
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, logits.size()[-1]).zero_()
        one_hot_output[0][la] = 1
        one_hot_output = one_hot_output.cuda(non_blocking=True)
        # Zero grads
        model.zero_grad()
        # Backward pass with specified target
        logits.backward(gradient=one_hot_output, retain_graph=True)
      
        ############
        guided_gradients = model.gradients[0]
        model.gradients = []
        guided_gradients_upsample = torch.nn.functional.interpolate(guided_gradients, 
                                    size=(length, length), mode='bilinear', align_corners=True)

        ############
        pre_cam = target * guided_gradients_upsample
        pre_cam = torch.nn.functional.relu(pre_cam) #relu
        pre_cam = torch.sum(pre_cam, dim=1)

        pre_cam = torch.unsqueeze(pre_cam, 1) 
        cam = torch.nn.functional.interpolate(pre_cam, size=(224,224), \
                                mode='bilinear', align_corners=True).squeeze()
        
        cam = cam.detach().cpu().numpy()
        
        cam = np.maximum(cam, 0)
        cam[cam < cam.max()*0.02] = 0
        cam /= (cam.max() + 1e-8)  # Normalize between 0-1
        cam = np.uint8(cam * 255)  
 
        cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)

        # Scale between 0-255 to visualize
        im_name = args.save_dir + img_name[0].split('/')[-1][:-5]
        out_name = im_name + '_{}.png'.format(la)
        #cam1 = cv_im_gray * 0.2 + cam * 0.8
        cv2.imwrite(out_name, cam)
        
        im_name1 = args.save_dir1 + img_name[0].split('/')[-1][:-5]
        out_name1 = im_name1 + '_{}.png'.format(la)

        #cmap = plt.cm.jet_r(cam)[..., :3] * 255.0
        #cam1 = (cmap.astype(np.float) + cv_im.astype(np.float)) / 2
        #cv2.imwrite(out_name1, cmap)
        plt.imsave(out_name1, cam, cmap=plt.cm.viridis)
        
    #    _, binary_att = cv2.threshold(cam, 0.15*255, 255, cv2.THRESH_BINARY) 
    #    _, contours, _ = cv2.findContours(binary_att, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #    rect = []
    #    for c in contours:
    #        x, y, w, h = cv2.boundingRect(c)
    #        rect.append([x, y, w, h])
    #    if rect == []:
    #        estimated_box = [0, 0, 1, 1]
    #    else:
    #        x, y, w, h = large_rect(rect)
    #        estimated_box = [x,y,x+w,y+h]

    #    #print(estimated_box, gt_box)
    #    ious = []
    #    recalls = []
    #    for ii in range(len(bbox)):
    #        iou, recall = bb_IOU(bbox[ii], estimated_box)
    #        ious.append(iou)
    #        recalls.append(recall)
    #    iou = max(ious)
    #    recall = recalls[ious.index(iou)]
    #    
    #    all_recalls += recall
    #    all_ious += iou
    #    #print('iou', iou, estimated_box, gt_box)
    #    if iou >= 0.5:
    #        gt_known += 1
    #        if ind == la:
    #            loc_counter += 1
    #        if la in inds[0, :5]:
    #            loc5_counter += 1
    #   
    #gt_acc = gt_known / float(global_counter)
    #cls_acc = cls_counter / float(global_counter)
    #cls5_acc = cls5_counter / float(global_counter)
    #loc_acc = loc_counter / float(global_counter)
    #loc5_acc = loc5_counter / float(global_counter)
    #ave_recall = all_recalls / float(global_counter)
    #ave_iou = all_ious / float(global_counter)
    #
    #print('Top 1 classification accuracy: {}\t'
    #      'Top 5 classification accuracy: {}\t'
    #      'GT-known accuracy: {}\t'
    #      'top 1 localization accuracy: {}\t'
    #      'top 5 localization accuracy: {}\t'
    #      'The mean recall of top-1 class: {}\t'
    #     'The mean iou of top-1 class: {}\t'.format(cls_acc, cls5_acc, gt_acc, loc_acc, loc5_acc, ave_recall, ave_iou))

#####hcrm with tencrop
def val_oc_hcrm(args, model=None, current_epoch=0):   
    #print('in_val_oc_hcrm, threshold = 0.15, layer=29, bilinear-gradient with zero-area')
    if model is None:
        #model = get_model(args)
        model = get_torch_model(args)
    model.eval()
    val_loader = test_data_loader_caffe(args)
    # pytorch dataloader
    #train_loader, val_loader = test_data_loader(args)
    global_counter = 0
    cls_counter = 0
    cls5_counter = 0
    loc_counter = 0
    loc5_counter = 0
    gt_known = 0
    all_recalls = 0.0 
    all_ious = 0.0
        
    annotations = scio.loadmat('/media/miao/ptsdisk/dataset/localization/ILSVRC/val_box.mat')['rec'][0]
    bboxes = []
    for ia in range(len(annotations)):
        bbox = []
        for bb in range(len(annotations[ia][0][0])):
            xyxy = annotations[ia][0][0][bb][1][0]
            bbox.append(xyxy.tolist())
        bboxes.append(bbox)
    print('Number of processing images: ', len(bboxes)) 
    
    with open('/media/miao/ptsdisk/dataset/localization/ILSVRC/val_box.txt') as f:
        lines = f.readlines()
    labels = [line[:-1].split()[1] for line in lines]
    ## for relu5_3
    #target_layers = [29, 30]
    #length = 14 

    # # for relu4_3
    #target_layers = [22, 23]
    #length = 28 

    # for relu3_3
    target_layers = [15, 16]
    length = 56 

    # for relu2_2
    #target_layers = [8, 9]
    #length = 112

    # for relu1_2
    target_layers = [3, 4]
    length = 224

    if not exists(args.save_dir):
        os.makedirs(args.save_dir)

    for idx, dat in tqdm(enumerate(val_loader)):
        img_name, img, label_in = dat
        img = img.cuda()
        img = img.squeeze(0)
        bbox = bboxes[global_counter]
        la = int(labels[idx])
        #la = label_in.long().numpy()[0]
        ####forward         
        feature_maps, logits = model.forward_cam(img, target_layers)
        ####extract feature
        target = feature_maps[0] 
        logits = logits.sum(dim=0).unsqueeze(0)
        logits1 = torch.softmax(logits, dim=1)
        
        #######top1 cls
        prob, ind = logits1.max(1)
        ind = ind.cpu().data.numpy()
        if ind == la:
            cls_counter += 1  
        #######top5 cls
        probs, inds = torch.sort(-logits1, 1)
        inds = inds.cpu().data.numpy()
        if la in inds[0, :5]:
            cls5_counter += 1
        global_counter += 1

        cv_im = cv2.imread(img_name[0])
        cv_im_gray = cv2.cvtColor(cv_im, cv2.COLOR_BGR2GRAY)
        height, width = cv_im.shape[:2]
        
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, logits.size()[-1]).zero_()
        one_hot_output[0][la] = 1
        one_hot_output = one_hot_output.cuda(non_blocking=True)
        # Zero grads
        model.zero_grad()
        # Backward pass with specified target
        logits.backward(gradient=one_hot_output, retain_graph=True)
        # Obtain gradients for 30 layer
        guided_gradients = model.gradients[0]
        model.gradients = []
        # Upsmaple the gradients
        guided_gradients_upsample = torch.nn.functional.interpolate(guided_gradients, 
                                           size=(length,length), mode='bilinear', align_corners=True)
        
        # another operation
        #guided_gradients_upsample = torch.nn.functional.interpolate(guided_gradients, 
        #                                   size=(length,length), mode='nearest')
        #target_gradients = model.gradients[1]
        #model.gradients = []
        #guided_gradients_upsample[target_gradients!=0] = target_gradients[target_gradients!=0]
        
        #compute cam
        pre_cam = target * guided_gradients_upsample
        #pre_cam = torch.nn.functional.relu(pre_cam) #relu
        # means = torch.mean(torch.mean(pre_cam, 2), 2)
        # means = means.view(means.shape[0], -1,  1, 1)
        # pre_cam[pre_cam < means] = 0
        pre_cam = torch.sum(pre_cam, dim=1)
        #pre_cam = torch.nn.functional.relu(pre_cam) #relu

        cam_ten_crop_upsample = torch.unsqueeze(pre_cam, 1) 
        cam_ten_crop_upsample = torch.nn.functional.interpolate(cam_ten_crop_upsample, size=(224,224), \
                                mode='bilinear', align_corners=True).squeeze(0) #10*224*224
        
        # #guided_gradients_upsample = torch.abs(guided_gradients_upsample)
        # guided_gradients_upsample[guided_gradients_upsample < 0] = 0
        # gb, ind = torch.max(guided_gradients_upsample, 1)
        # gb = gb.squeeze()[0]
        # gb = gb.detach().cpu().numpy()
        # gb /= (gb.max() + 1e-8)  # Normalize between 0-1
        # gb = np.uint8(gb * 255)  
 
        # gb = cv2.resize(gb, (width, height), interpolation=cv2.INTER_CUBIC)

        #0 - topleft         #1 - topright         #2 - bottomleft        #3 - bottomright        #4 - center
        #5 - topleft - flip  #6 - topright - flip  #7 - bottomleft - flip #8 - bottomright - flip #9 - center - flip
        cam_ten_crop_upsample_pad = []
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[0], pad=(0, 32, 0, 32))]
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[1], pad=(32, 0, 0, 32))]
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[2], pad=(0, 32, 32, 0))]
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[3], pad=(32, 0, 32, 0))]
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[4], pad=(16, 16, 16, 16))]
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[5], pad=(0, 32, 0, 32)).flip([2])] #1-vertical, 2-hori
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[6], pad=(32, 0, 0, 32)).flip([2])]
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[7], pad=(0, 32, 32, 0)).flip([2])]
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[8], pad=(32, 0, 32, 0)).flip([2])]
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[9], pad=(16, 16, 16, 16)).flip([2])]

        cam = torch.tensor(np.zeros((256, 256), dtype=np.float32)).cuda()
        for i in range(len(cam_ten_crop_upsample_pad)):
            cam += cam_ten_crop_upsample_pad[i][0]

        cam = cam.detach().cpu().numpy()

        cam = np.maximum(cam, 0)
        cam /= (cam.max() + 1e-8)  # Normalize between 0-1
        cam = np.uint8(cam * 255)  
 
        cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)

        # grabcut
        #if target_layers[0] == 3:
        #    mask = np.zeros((height, width), dtype=np.uint8)
        #    mask[cam > 0.2*255] = 1
        #    mask[cam <= 0.2*255] = 2
        #    bgdModel = np.zeros((1,65),np.float64)
        #    fgdModel = np.zeros((1,65),np.float64)
        #    I1, bg, fg = cv2.grabCut(cv_im, mask, None, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
        #    cam1 = (np.where((I1 == 2) | (I1 == 0), 0, 1) * 255).astype(np.uint8)
        #    cam = cam1
        
        ### Scale between 0-255 to visualize
        im_name = args.save_dir + img_name[0].split('/')[-1][:-5]
        out_name = im_name + '_{}.png'.format(la)
        #cam1 = cv_im_gray * 0.3 + cam * 0.7
        cv2.imwrite(out_name, cam)
        #plt.imsave(out_name, cam, cmap=plt.cm.jet)
        # im_name1 = args.save_dir1 + img_name[0].split('/')[-1][:-5]
        # out_name1 = im_name1 + '_{}.png'.format(la)

        #cmap = plt.cm.jet_r(cam)[..., :3] * 255.0
        #cam1 = (cmap.astype(np.float) + cv_im.astype(np.float)) / 2
        #cv2.imwrite(out_name1, cmap)
        #plt.imsave(out_name1, cam, cmap=plt.cm.viridis)

        _, binary_att = cv2.threshold(cam, 0.15*255, 255, cv2.THRESH_BINARY) 
        _, contours, _ = cv2.findContours(binary_att, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rect = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            rect.append([x, y, w, h])
        if rect == []:
            estimated_box = [0, 0, 1, 1]
        else:
            x, y, w, h = large_rect(rect)
            estimated_box = [x,y,x+w,y+h]

        #print(estimated_box, gt_box)
        ious = []
        recalls = []
        for ii in range(len(bbox)):
            iou, recall = bb_IOU(bbox[ii], estimated_box)
            ious.append(iou)
            recalls.append(recall)
        iou = max(ious)
        recall = recalls[ious.index(iou)]
        
        all_recalls += recall
        all_ious += iou
        #print('iou', iou, estimated_box, gt_box)
        if iou >= 0.5:
            gt_known += 1
            if ind == la:
                loc_counter += 1
            if la in inds[0, :5]:
                loc5_counter += 1
    
    gt_acc = gt_known / float(global_counter)
    cls_acc = cls_counter / float(global_counter)
    cls5_acc = cls5_counter / float(global_counter)
    loc_acc = loc_counter / float(global_counter)
    loc5_acc = loc5_counter / float(global_counter)
    ave_recall = all_recalls / float(global_counter)
    ave_iou = all_ious / float(global_counter)
    
    print('Top 1 classification accuracy: {}\t'
          'Top 5 classification accuracy: {}\t'
          'GT-known accuracy: {}\t'
          'top 1 localization accuracy: {}\t'
          'top 5 localization accuracy: {}\t'
          'The mean recall of top-1 class: {}\t'
         'The mean iou of top-1 class: {}\t'.format(cls_acc, cls5_acc, gt_acc, loc_acc, loc5_acc, ave_recall, ave_iou))

def val_grad(args, model=None, current_epoch=0):   
    args.tencrop = False
    if model is None:
        #model = get_model(args)
        model = get_torch_model(args)
    model.eval()
    val_loader = test_data_loader_caffe(args)
    #train_loader, val_loader = test_data_loader(args)
    global_counter = 0
    cls_counter = 0
    cls5_counter = 0
    loc_counter = 0
    loc5_counter = 0
    gt_known = 0
    all_recalls = 0.0 
    all_ious = 0.0
        
    annotations = scio.loadmat('/media/miao/ptsdisk/dataset/localization/ILSVRC/val_box.mat')['rec'][0]
    bboxes = []
    for ia in range(len(annotations)):
        bbox = []
        for bb in range(len(annotations[ia][0][0])):
            xyxy = annotations[ia][0][0][bb][1][0]
            bbox.append(xyxy.tolist())
        bboxes.append(bbox)
    print('Number of processing images: ', len(bboxes)) 
    
    with open('/media/miao/ptsdisk/dataset/localization/ILSVRC/val_box.txt') as f:
        lines = f.readlines()
    labels = [line[:-1].split()[1] for line in lines]
    
    #target_layers = [29, 30]
    #length = 14 
    
    #target_layers = [22, 23]
    #length = 28

    if not exists(args.save_dir):
        os.makedirs(args.save_dir)

    for idx, dat in tqdm(enumerate(val_loader)):
        img_name, img, label_in = dat
        img = img.cuda()
        bbox = bboxes[global_counter]
        la = int(labels[idx])
        #la = label_in.long().numpy()[0]
        ####forward         
        feature_maps, logits = model.forward_cam(img, target_layers)
        ####extract feature
        target = feature_maps[0] 
        logits = logits.sum(dim=0).unsqueeze(0)
        logits1 = torch.softmax(logits, dim=1)
        
        #######top1 cls
        prob, ind = logits1.max(1)
        ind = ind.cpu().data.numpy()
        if ind == la:
            cls_counter += 1  
        #######top5 cls
        probs, inds = torch.sort(-logits1, 1)
        inds = inds.cpu().data.numpy()
        if la in inds[0, :5]:
            cls5_counter += 1
        global_counter += 1

        cv_im = cv2.imread(img_name[0])
        cv_im_gray = cv2.cvtColor(cv_im, cv2.COLOR_BGR2GRAY)
        height, width = cv_im.shape[:2]
        
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, logits.size()[-1]).zero_()
        one_hot_output[0][la] = 1
        one_hot_output = one_hot_output.cuda(non_blocking=True)
        # Zero grads
        model.zero_grad()
        # Backward pass with specified target
        logits.backward(gradient=one_hot_output, retain_graph=True)
      
        ############
        guided_gradients = model.gradients[0]
        model.gradients = []
        guided_gradients_upsample = torch.nn.functional.interpolate(guided_gradients, 
                                    size=(length, length), mode='bilinear', align_corners=True)

        ############
        pre_cam = target * guided_gradients_upsample
        pre_cam = torch.nn.functional.relu(pre_cam) #relu
        pre_cam = torch.sum(pre_cam, dim=1)

        pre_cam = torch.unsqueeze(pre_cam, 1) 
        cam = torch.nn.functional.interpolate(pre_cam, size=(224,224), \
                                mode='bilinear', align_corners=True).squeeze()
        
        cam = cam.detach().cpu().numpy()
        
        cam = np.maximum(cam, 0)
        cam /= (cam.max() + 1e-8)  # Normalize between 0-1
        cam = np.uint8(cam * 255)  
 
        cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)

        # Scale between 0-255 to visualize
        im_name = args.save_dir + img_name[0].split('/')[-1][:-5]
        out_name = im_name + '_{}_3.png'.format(la)
        #cam1 = cv_im_gray * 0.2 + cam * 0.8
        cv2.imwrite(out_name, cam)
        #plt.imsave(out_name, cam, cmap=plt.cm.jet)
        
    #    _, binary_att = cv2.threshold(cam, 0.15*255, 255, cv2.THRESH_BINARY) 
    #    _, contours, _ = cv2.findContours(binary_att, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #    rect = []
    #    for c in contours:
    #        x, y, w, h = cv2.boundingRect(c)
    #        rect.append([x, y, w, h])
    #    if rect == []:
    #        estimated_box = [0, 0, 1, 1]
    #    else:
    #        x, y, w, h = large_rect(rect)
    #        estimated_box = [x,y,x+w,y+h]

    #    #print(estimated_box, gt_box)
    #    ious = []
    #    recalls = []
    #    for ii in range(len(bbox)):
    #        iou, recall = bb_IOU(bbox[ii], estimated_box)
    #        ious.append(iou)
    #        recalls.append(recall)
    #    iou = max(ious)
    #    recall = recalls[ious.index(iou)]
    #    
    #    all_recalls += recall
    #    all_ious += iou
    #    #print('iou', iou, estimated_box, gt_box)
    #    if iou >= 0.5:
    #        gt_known += 1
    #        if ind == la:
    #            loc_counter += 1
    #        if la in inds[0, :5]:
    #            loc5_counter += 1
    #   
    #gt_acc = gt_known / float(global_counter)
    #cls_acc = cls_counter / float(global_counter)
    #cls5_acc = cls5_counter / float(global_counter)
    #loc_acc = loc_counter / float(global_counter)
    #loc5_acc = loc5_counter / float(global_counter)
    #ave_recall = all_recalls / float(global_counter)
    #ave_iou = all_ious / float(global_counter)
    #
    #print('Top 1 classification accuracy: {}\t'
    #      'Top 5 classification accuracy: {}\t'
    #      'GT-known accuracy: {}\t'
    #      'top 1 localization accuracy: {}\t'
    #      'top 5 localization accuracy: {}\t'
    #      'The mean recall of top-1 class: {}\t'
    #     'The mean iou of top-1 class: {}\t'.format(cls_acc, cls5_acc, gt_acc, loc_acc, loc5_acc, ave_recall, ave_iou))

#####hcrm with tencrop, without find bounding boxes
def val_oc1_hcrm(args, model=None, current_epoch=0):   
    #print('in_val_oc_hcrm, threshold = 0.15, layer=29, bilinear-gradient with zero-area')
    if model is None:
        #model = get_model(args)
        model = get_torch_model(args)
    model.eval()
    val_loader = test_data_loader_caffe(args)
    global_counter = 0
    cls_counter = 0
    cls5_counter = 0
            
    with open('/media/miao/ptsdisk/dataset/localization/ILSVRC/val_box.txt') as f:
        lines = f.readlines()
    labels = [line[:-1].split()[1] for line in lines]
    # for relu5_3
    #target_layers = [29, 30]
    #length = 14 
    
    # for relu3_3
    target_layers = [15, 16]
    length = 56 

    # # for relu2_2
    # target_layers = [8, 9]
    # length = 112 

    if not exists(args.save_dir):
        os.makedirs(args.save_dir)

    for idx, dat in tqdm(enumerate(val_loader)):
        img_name, img, label_in = dat
        img = img.cuda()
        img = img.squeeze(0)
        bbox = bboxes[global_counter]
        la = int(labels[idx])
        #la = label_in.long().numpy()[0]
        ####forward         
        feature_maps, logits = model.forward_cam(img, target_layers)
        ####extract feature
        target = feature_maps[0] 
        logits = logits.sum(dim=0).unsqueeze(0)
        logits1 = torch.softmax(logits, dim=1)
        
        #######top1 cls
        prob, ind = logits1.max(1)
        ind = ind.cpu().data.numpy()
        if ind == la:
            cls_counter += 1  
        #######top5 cls
        probs, inds = torch.sort(-logits1, 1)
        inds = inds.cpu().data.numpy()
        if la in inds[0, :5]:
            cls5_counter += 1
        global_counter += 1

        cv_im = cv2.imread(img_name[0])
        cv_im_gray = cv2.cvtColor(cv_im, cv2.COLOR_BGR2GRAY)
        height, width = cv_im.shape[:2]
        
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, logits.size()[-1]).zero_()
        one_hot_output[0][la] = 1
        one_hot_output = one_hot_output.cuda(non_blocking=True)
        # Zero grads
        model.zero_grad()
        # Backward pass with specified target
        logits.backward(gradient=one_hot_output, retain_graph=True)
        # Obtain gradients for 30 layer
        guided_gradients = model.gradients[0]
        model.gradients = []
        # Upsmaple the gradients
        guided_gradients_upsample = torch.nn.functional.interpolate(guided_gradients, 
                                           size=(length,length), mode='bilinear', align_corners=True)       
        
        #compute cam
        pre_cam = target * guided_gradients_upsample
        pre_cam = torch.nn.functional.relu(pre_cam) #relu
        pre_cam = torch.sum(pre_cam, dim=1)

        cam_ten_crop_upsample = torch.unsqueeze(pre_cam, 1) 
        cam_ten_crop_upsample = torch.nn.functional.interpolate(cam_ten_crop_upsample, size=(224,224), \
                                mode='bilinear', align_corners=True).squeeze(0) #10*224*224
        
        #0 - topleft         #1 - topright         #2 - bottomleft        #3 - bottomright        #4 - center
        #5 - topleft - flip  #6 - topright - flip  #7 - bottomleft - flip #8 - bottomright - flip #9 - center - flip
        cam_ten_crop_upsample_pad = []
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[0], pad=(0, 32, 0, 32))]
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[1], pad=(32, 0, 0, 32))]
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[2], pad=(0, 32, 32, 0))]
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[3], pad=(32, 0, 32, 0))]
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[4], pad=(16, 16, 16, 16))]
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[5], pad=(0, 32, 0, 32)).flip([2])] #1-vertical, 2-hori
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[6], pad=(32, 0, 0, 32)).flip([2])]
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[7], pad=(0, 32, 32, 0)).flip([2])]
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[8], pad=(32, 0, 32, 0)).flip([2])]
        cam_ten_crop_upsample_pad += [torch.nn.functional.pad(cam_ten_crop_upsample[9], pad=(16, 16, 16, 16)).flip([2])]

        cam = torch.tensor(np.zeros((256, 256), dtype=np.float32)).cuda()
        for i in range(len(cam_ten_crop_upsample_pad)):
            cam += cam_ten_crop_upsample_pad[i][0]

        cam = cam.detach().cpu().numpy()

        cam = np.maximum(cam, 0)
        cam /= (cam.max() + 1e-8)  # Normalize between 0-1
        cam = np.uint8(cam * 255)  
 
        cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)

        # Scale between 0-255 to visualize
        im_name = args.save_dir + img_name[0].split('/')[-1][:-5]
        out_name = im_name + '_{}.png'.format(la)
        #cam1 = cv_im_gray * 0.2 + cam * 0.8
        cv2.imwrite(out_name, cam)
        # plt.imsave(out_name, cam1, cmap=plt.cm.viridis)

    cls_acc = cls_counter / float(global_counter)
    cls5_acc = cls5_counter / float(global_counter)

    
    print('Top 1 classification accuracy: {}\t'
          'Top 5 classification accuracy: {}\t'.format(cls_acc, cls5_acc))


def voc_hcrm(args, model=None, current_epoch=0):   
    if model is None:
        model = get_model(args)
        #model = get_torch_model(args)
    model.eval()
    train_loader, val_loader = test_data_loader(args)
    
    # for conv5_3
    target_layers = [29, 30]
    length = 28
    #target_layers = [8, 9]
    #length = 224
    #target_layers = ['layer4']

    if not exists(args.save_dir):
        os.mkdir(args.save_dir)
   
    for idx, dat in tqdm(enumerate(val_loader)):
        img_name, img, label_in = dat
        img = img.cuda(non_blocking=True)
        ####forward         
        feature_maps, logits = model.forward_cam(img, target_layers)
        ####extract feature
        target = feature_maps[0] 
        #target = feature_maps[0].cpu().data.numpy()[0]
        logits1 = torch.sigmoid(logits)
        
        cv_im = cv2.imread(img_name[0])
        cv_im_gray = cv2.cvtColor(cv_im, cv2.COLOR_BGR2GRAY)
        height, width = cv_im.shape[:2]
        
        im_name = args.save_dir + img_name[0].split('/')[-1][:-4]
        labels = label_in.long().numpy()[0]
        
        for i in range(int(args.num_classes)):
            if labels[i] == 1:
                # Target for backprop
                one_hot_output = torch.FloatTensor(1, logits.size()[-1]).zero_()
                one_hot_output[0][i] = 1
                one_hot_output = one_hot_output.cuda(non_blocking=True)
                # Zero grads
                model.zero_grad()
                # Backward pass with specified target
                logits.backward(gradient=one_hot_output, retain_graph=True)
                # Obtain gradients for 30 layer
                guided_gradients = model.gradients[0]
                model.gradients = []
                # Upsmaple the gradients
                guided_gradients_upsample = torch.nn.functional.interpolate(guided_gradients, 
                                                   size=(length,length), mode='bilinear', align_corners=True)    

                pre_cam = target * guided_gradients_upsample
                pre_cam = torch.nn.functional.relu(pre_cam) #relu
                pre_cam = torch.sum(pre_cam, dim=1)

                cam = pre_cam.detach().cpu().numpy()[0]
                cam /= (cam.max() + 1e-8)  # Normalize between 0-1
                cam = np.uint8(cam * 255)  
                cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)


                # Scale between 0-255 to visualize
                out_name = im_name + '_{}_5.png'.format(i)
                #cam1 = cv_im_gray * 0.2 + cam * 0.8
                #cv2.imwrite(out_name, cam)
                plt.imsave(out_name, cam, cmap=plt.cm.viridis)
                
def voc_gradcam(args, model=None, current_epoch=0):   
    if model is None:
        model = get_model(args)
        #model = get_torch_model(args)
    model.eval()
    train_loader, val_loader = test_data_loader(args)
    
    # for conv5_3
    target_layers = ['layer4']
    
    if not exists(args.save_dir):
        os.mkdir(args.save_dir)
   
    for idx, dat in tqdm(enumerate(val_loader)):
        img_name, img, label_in = dat
        img = img.cuda(non_blocking=True)
        ####forward         
        feature_maps, logits = model.forward_cam(img, target_layers)
        ####extract feature
        target = feature_maps[0]
        #target = feature_maps[0].cpu().data.numpy()[0]
        logits1 = torch.sigmoid(logits)
        
        cv_im = cv2.imread(img_name[0])
        cv_im_gray = cv2.cvtColor(cv_im, cv2.COLOR_BGR2GRAY)
        height, width = cv_im.shape[:2]
        
        im_name = args.save_dir + img_name[0].split('/')[-1][:-4]
        labels = label_in.long().numpy()[0]
        
        for i in range(int(args.num_classes)):
            if labels[i] == 1:
                # Target for backprop
                one_hot_output = torch.FloatTensor(1, logits.size()[-1]).zero_()
                one_hot_output[0][i] = 1
                one_hot_output = one_hot_output.cuda(non_blocking=True)
                # Zero grads
                model.zero_grad()
                # Backward pass with specified target
                logits.backward(gradient=one_hot_output, retain_graph=True)
                # grad-cam
                guided_gradients = model.gradients[0]
                model.gradients = []
                weights = torch.mean(guided_gradients, dim=(2, 3))
                weights = weights.cuda()
                weights = torch.nn.functional.relu(-weights)
                #weights = torch.abs(weights)
                pre_cam = (target * (weights.view(1, -1, 1, 1)))
                #pre_cam = torch.nn.functional.relu(pre_cam)
                pre_cam = pre_cam.sum(1).unsqueeze(0)

                cam = torch.nn.functional.interpolate(pre_cam, size=(224,224), \
                                        mode='bilinear', align_corners=True).squeeze(0) #1*224*224
                cam = cam.detach().cpu().numpy()[0]
                
                # center crop fill 
                # cam = np.zeros((256, 256), dtype=np.float32)
                # cam[16:240, 16:240] = pre_cam
                
                cam = np.maximum(cam, 0)
                cam /= (cam.max() + 1e-10)  # Normalize between 0-1
                cam = np.uint8(cam * 255)  
                cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)

                # Scale between 0-255 to visualize
                out_name = im_name + '_{}_5.png'.format(i)
                #cam1 = cv_im_gray * 0.2 + cam * 0.8
                cv2.imwrite(out_name, cam)
                #plt.imsave(out_name, cam1, cmap=plt.cm.viridis)

if __name__ == '__main__':
    args = get_arguments()
    #val_gradcam(args)
    #val_cub_gradcam(args)
    #val_oc_hcrm(args)
    val_oc_hcrm(args)
    #val_oc_gradcam(args)
    #val_sc_gradcam(args)
    #voc_hcrm(args)
    #voc_gradcam(args)
