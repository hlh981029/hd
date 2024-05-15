import sys

sys.path.append('D:/lab/project/hcrm/')

import os
import cv2
import torch
import shutil
import argparse
import torchvision
import numpy as np
from models import *
import torch.nn as nn
from tqdm import tqdm, trange
from PIL import Image
import scipy.io as scio
from os.path import exists
from torch.optim import SGD
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.Restore import restore
from torch.utils.data import DataLoader
from torchvision import models, transforms
from utils.imutils import large_rect, bb_IOU
from utils.LoadData import test_data_loader, test_msf_data_loader, test_data_loader_caffe, test_voc_data_loader

ROOT_DIR = '/'.join(os.getcwd().split('/')[:-1])
data_root = 'D:/lab/data/VOCdevkit/VOC2012/'

def get_arguments():
    parser = argparse.ArgumentParser(description='ACoL')
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR)
    # parser.add_argument("--save_dir", type=str, default='')
    parser.add_argument("--save_dir", type=str, default=data_root+'hcrm/vis_2007_000170_noenhanced_0/')
    # parser.add_argument("--img_dir", type=str, default='')
    parser.add_argument("--img_dir", type=str, default=data_root+'JPEGImages/')
    # parser.add_argument("--train_list", type=str, default='')
    parser.add_argument("--train_list", type=str, default=data_root+'ImageSets/Segmentation/train_cls.txt')
    # parser.add_argument("--test_list", type=str, default='')
    parser.add_argument("--test_list", type=str, default=data_root+'ImageSets/Segmentation/train_cls.txt')
    parser.add_argument("--batch_size", type=int, default=1)
    # parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--crop_size", type=int, default=224)
    # parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--dataset", type=str, default='pascal_voc')
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--scales", type=list, default=[0.5, 0.75, 1, 1.25, 1.5, 2])
    # parser.add_argument("--arch", type=str, default='vgg_v0')
    parser.add_argument("--arch", type=str, default='vgg_voc')
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tencrop", type=str, default='False')
    parser.add_argument("--onehot", type=str, default='True')
    # parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--disp_interval", type=int, default=40)
    parser.add_argument("--snapshot_dir", type=str, default='')
    parser.add_argument("--resume", type=str, default='True')
    # parser.add_argument("--restore_from", type=str, default='')
    parser.add_argument("--restore_from", type=str, default='D:/lab/project/hcrm/pascal_voc_epoch_14.pth')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--current_epoch", type=int, default=0)

    return parser.parse_args()


def get_model(args, bp_type='BP'):
    # model = eval(args.arch).model(num_classes=args.num_classes, args=args, threshold=args.threshold)
    model = eval(args.arch).vgg16(pretrained=False, num_classes=args.num_classes, bp_type=bp_type)
    # model = eval(args.arch).resnet50(pretrained=False, num_classes=args.num_classes)
    model = model.cuda()

    pretrained_dict = torch.load(args.restore_from)['state_dict']
    model_dict = model.state_dict()

    print(model_dict.keys())
    print(pretrained_dict.keys())

    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict.keys()}
    print("Weights cannot be loaded:")
    print([k for k in model_dict.keys() if k not in pretrained_dict.keys()])

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


def get_torch_model(args, bp_type='BP'):
    model = eval(args.arch).vgg16(pretrained=False, num_classes=args.num_classes, bp_type=bp_type)
    model.load_state_dict(torch.load(args.restore_from))
    model = model.cuda()

    return model


def get_torch_resnet_model(args):
    model = eval(args.arch).resnet50(pretrained=False, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.restore_from))
    model = model.cuda()

    return model


## Principal Component Analysis (PCA)
def pca(data, n):
    newData = data - np.mean(data, axis=0)
    covMat = np.cov(newData, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValIndice = np.argsort(eigVals)
    n_eigValIndice = eigValIndice[-1:-(n + 1):-1]
    n_eigVect = eigVects[:, n_eigValIndice]
    lowDDataMat = newData * n_eigVect

    return lowDDataMat


def voc_find_kernel_pattern(args, model=None, current_epoch=0):
    if model is None:
        model = get_model(args)
        gb_model = get_model(args, bp_type='GBP')
    model.eval()
    gb_model.eval()
    val_loader, transform = test_voc_data_loader(args)

    # if not exists(args.save_dir):
    #     os.makedirs(args.save_dir)
    # if not exists(args.save_dir+'conv1/pattern/'):
    #     os.makedirs(args.save_dir+'conv1/pattern/')
    # if not exists(args.save_dir+'conv2/pattern/'):
    #     os.makedirs(args.save_dir+'conv2/pattern/')
    # if not exists(args.save_dir+'conv3/pattern/'):
    #     os.makedirs(args.save_dir+'conv3/pattern/')
    # if not exists(args.save_dir+'conv4/pattern/'):
    #     os.makedirs(args.save_dir+'conv4/pattern/')
    if not exists(args.save_dir + 'conv5/pattern/'):
        os.makedirs(args.save_dir + 'conv5/pattern/')

    target_layers = [3, 8, 15, 22, 29]
    channel_patterns = []
    for cc in range(1472):
        channel_patterns.append([])
        for _ in range(9):
            channel_patterns[cc].append('')
    channel_activations = torch.zeros(1472, 9)

    for idx, dat in tqdm(enumerate(val_loader)):
        if idx == 20:
            break

        img_name, img, label_in = dat
        img = img.cuda()
        input_var = torch.autograd.Variable(img, requires_grad=True)
        labels = label_in.long().numpy()[0]

        ####forward          
        feature_maps, logits = model.forward_cam(input_var, target_layers)
        model.gradients = []
        chn_counter = 0
        for ff in range(len(feature_maps)):
            with torch.no_grad():
                target = feature_maps[ff]
                mas, _ = torch.max(torch.max(target, 3)[0], 2)
                mas = mas.cpu()
                N, C, H, W = target.shape
                for j in range(C):
                    per_activation = channel_activations[chn_counter + j]
                    for gg in range(9):
                        if per_activation[gg] <= mas[0, j]:
                            for hh in range(8, gg, -1):
                                channel_activations[chn_counter + j][hh] = channel_activations[chn_counter + j][hh - 1]
                                channel_patterns[chn_counter + j][hh] = channel_patterns[chn_counter + j][hh - 1]
                            channel_activations[chn_counter + j][gg] = mas[0, j]
                            channel_patterns[chn_counter + j][gg] = img_name[0]
                            break
                chn_counter += C

    for idx in trange(1472):
        if idx < 960:
            continue
        for j in range(9):
            img_name = channel_patterns[idx][j]
            img = Image.open(img_name).convert('RGB')
            img = transform(img).unsqueeze(0)
            img = img.cuda()
            input_var = torch.autograd.Variable(img, requires_grad=True)
            input_var1 = torch.autograd.Variable(img, requires_grad=True)

            cv_im = cv2.imread(img_name)
            cv_im_gray = cv2.cvtColor(cv_im, cv2.COLOR_BGR2GRAY)
            height, width = cv_im.shape[:2]

            ####forward          
            feature_maps, logits = model.forward_cam(input_var, target_layers)
            gb_feature_maps, logits = gb_model.forward_cam(input_var1, target_layers)
            gb_model.gradients = []

            if idx < 64:
                target = feature_maps[0]
                gb_target = gb_feature_maps[0]

                unit_label = torch.zeros_like(target)
                ma = torch.max(target[:, idx])
                unit_label[:, idx][target[:, idx] == ma] = 1

                gb_model.zero_grad()
                input_var1.grad = None

                gb_target.backward(gradient=unit_label, retain_graph=True)
                grad = input_var1.grad.detach().cpu().numpy()[0]

                grad = grad.transpose(1, 2, 0)
                grad = cv2.resize(grad, (width, height), interpolation=cv2.INTER_CUBIC)
                grad -= grad.min()
                grad /= (grad.max() + 1e-20)

                accu_grad = np.sum(grad, 2)
                ma = np.max(accu_grad)
                pos = np.where(accu_grad == ma)
                y, x = pos[0][0], pos[1][0]

                y1 = max(y - 10, 0)
                y2 = min(y + 10, height)
                if y2 - y1 != 20:
                    if y1 == 0:
                        y2 = 20
                    if y2 == height:
                        y1 = y2 - 20

                x1 = max(x - 10, 0)
                x2 = min(x + 10, width)
                if x2 - x1 != 20:
                    if x1 == 0:
                        x2 = 20
                    if x2 == width:
                        x1 = x2 - 20

                crop_grad = grad[y1:y2, x1:x2, :]

                ## Scale between 0-255 to visualize
                # im_name = '{}/conv1/pattern/{}_{}.png'.format(args.save_dir, idx, j, img_name.split('/')[-1][-15:-4])
                im_name = '{}/conv1/pattern/{}_{}.png'.format(args.save_dir, idx, j)
                plt.imsave(im_name, crop_grad)

            elif idx < 192:
                idd = idx - 64
                target = feature_maps[1]
                gb_target = gb_feature_maps[1]

                unit_label = torch.zeros_like(target)
                ma = torch.max(target[:, idd])
                unit_label[:, idd][target[:, idd] == ma] = 1

                gb_model.zero_grad()
                input_var1.grad = None

                gb_target.backward(gradient=unit_label, retain_graph=True)
                grad = input_var1.grad.detach().cpu().numpy()[0]

                grad = grad.transpose(1, 2, 0)
                grad = cv2.resize(grad, (width, height), interpolation=cv2.INTER_CUBIC)
                grad -= grad.min()
                grad /= (grad.max() + 1e-20)

                accu_grad = np.sum(grad, 2)
                ma = np.max(accu_grad)
                pos = np.where(accu_grad == ma)
                y, x = pos[0][0], pos[1][0]

                y1 = max(y - 20, 0)
                y2 = min(y + 20, height)
                if y2 - y1 != 40:
                    if y1 == 0:
                        y2 = 40
                    if y2 == height:
                        y1 = y2 - 40

                x1 = max(x - 20, 0)
                x2 = min(x + 20, width)
                if x2 - x1 != 40:
                    if x1 == 0:
                        x2 = 40
                    if x2 == width:
                        x1 = x2 - 40

                crop_grad = grad[y1:y2, x1:x2, :]

                ## Scale between 0-255 to visualize
                im_name = '{}/conv2/pattern/{}_{}.png'.format(args.save_dir, idd, j)
                plt.imsave(im_name, crop_grad)

            elif idx < 448:
                idd = idx - 192
                target = feature_maps[2]
                gb_target = gb_feature_maps[2]

                unit_label = torch.zeros_like(target)
                ma = torch.max(target[:, idd])
                unit_label[:, idd][target[:, idd] == ma] = 1

                gb_model.zero_grad()
                input_var1.grad = None

                gb_target.backward(gradient=unit_label, retain_graph=True)
                grad = input_var1.grad.detach().cpu().numpy()[0]

                grad = grad.transpose(1, 2, 0)
                grad = cv2.resize(grad, (width, height), interpolation=cv2.INTER_CUBIC)
                grad -= grad.min()
                grad /= (grad.max() + 1e-20)

                accu_grad = np.sum(grad, 2)
                ma = np.max(accu_grad)
                pos = np.where(accu_grad == ma)
                y, x = pos[0][0], pos[1][0]

                y1 = max(y - 40, 0)
                y2 = min(y + 40, height)
                if y2 - y1 != 80:
                    if y1 == 0:
                        y2 = 80
                    if y2 == height:
                        y1 = y2 - 80

                x1 = max(x - 40, 0)
                x2 = min(x + 40, width)
                if x2 - x1 != 80:
                    if x1 == 0:
                        x2 = 80
                    if x2 == width:
                        x1 = x2 - 80

                crop_grad = grad[y1:y2, x1:x2, :]

                ## Scale between 0-255 to visualize
                im_name = '{}/conv3/pattern/{}_{}.png'.format(args.save_dir, idd, j)
                plt.imsave(im_name, crop_grad)

            elif idx < 960:
                idd = idx - 448
                target = feature_maps[3]
                gb_target = gb_feature_maps[3]

                unit_label = torch.zeros_like(target)
                ma = torch.max(target[:, idd])
                unit_label[:, idd][target[:, idd] == ma] = 1

                gb_model.zero_grad()
                input_var1.grad = None

                gb_target.backward(gradient=unit_label, retain_graph=True)
                grad = input_var1.grad.detach().cpu().numpy()[0]

                grad = grad.transpose(1, 2, 0)
                grad = cv2.resize(grad, (width, height), interpolation=cv2.INTER_CUBIC)
                grad -= grad.min()
                grad /= (grad.max() + 1e-20)

                accu_grad = np.sum(grad, 2)
                ma = np.max(accu_grad)
                pos = np.where(accu_grad == ma)
                y, x = pos[0][0], pos[1][0]

                y1 = max(y - 80, 0)
                y2 = min(y + 80, height)
                if y2 - y1 != 160:
                    if y1 == 0:
                        y2 = 160
                    if y2 == height:
                        y1 = y2 - 160

                x1 = max(x - 80, 0)
                x2 = min(x + 80, width)
                if x2 - x1 != 160:
                    if x1 == 0:
                        x2 = 160
                    if x2 == width:
                        x1 = x2 - 160

                crop_grad = grad[y1:y2, x1:x2, :]

                ## Scale between 0-255 to visualize
                im_name = '{}/conv4/pattern/{}_{}.png'.format(args.save_dir, idd, j)
                plt.imsave(im_name, crop_grad)

            else:
                idd = idx - 960
                target = feature_maps[4]
                gb_target = gb_feature_maps[4]

                unit_label = torch.zeros_like(target)
                ma = torch.max(target[:, idd])
                unit_label[:, idd][target[:, idd] == ma] = 1

                gb_model.zero_grad()
                input_var1.grad = None

                gb_target.backward(gradient=unit_label, retain_graph=True)
                grad = input_var1.grad.detach().cpu().numpy()[0]

                grad = grad.transpose(1, 2, 0)
                grad = cv2.resize(grad, (width, height), interpolation=cv2.INTER_CUBIC)
                grad -= grad.min()
                grad /= (grad.max() + 1e-20)

                accu_grad = np.sum(grad, 2)
                ma = np.max(accu_grad)
                pos = np.where(accu_grad == ma)
                y, x = pos[0][0], pos[1][0]

                length = min(height, width)
                y1 = max(y - length // 2, 0)
                y2 = min(y + length // 2 + length % 2, height)
                if y2 - y1 != length:
                    if y1 == 0:
                        y2 = length
                    if y2 == height:
                        y1 = y2 - length

                x1 = max(x - length // 2, 0)
                x2 = min(x + length // 2 + length % 2, width)
                if x2 - x1 != length:
                    if x1 == 0:
                        x2 = length
                    if x2 == width:
                        x1 = x2 - length
                crop_grad = grad[y1:y2, x1:x2, :]

                crop_grad -= crop_grad.min()
                crop_grad /= crop_grad.max()

                xx = crop_grad.copy()
                crop_grad[xx > 0.65] = crop_grad[xx > 0.65] * 1.3
                crop_grad[np.logical_and(xx <= 0.65, xx >= 0.5)] = crop_grad[
                                                                       np.logical_and(xx <= 0.65, xx >= 0.5)] * 1.1
                crop_grad[crop_grad > 1.0] = 1.
                # crop_grad *= 255.0
                # cv2.imwrite(filename, np.uint8(crop_grad))

                # scale = np.max(crop_grad)
                # maps = crop_grad / scale * 0.5
                # maps += 0.5
                # maps = cm.bwr_r(maps)[..., :3]
                # maps = np.uint8(maps * 255.0)

                ## Scale between 0-255 to visualize
                im_name = '{}conv5/test/{}_{}.png'.format(args.save_dir, idd, j)
                plt.imsave(im_name, crop_grad)


def voc_hcrm_unit_kernel1(args, model=None, current_epoch=0):
    if model is None:
        model = get_model(args)
        gb_model = get_model(args, bp_type='GBP')
    model.eval()
    gb_model.eval()
    val_loader, _ = test_voc_data_loader(args)

    threshold = 0.5
    target_layers = [3, 8, 15, 22, 29, 30]
    length = 14

    if not exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not exists(args.save_dir + 'conv5/gray/'):
        os.makedirs(args.save_dir + 'conv5/gray/')
    if not exists(args.save_dir + 'conv5/color/'):
        os.makedirs(args.save_dir + 'conv5/color/')
    if not exists(args.save_dir + 'conv1/kernel/'):
        os.makedirs(args.save_dir + 'conv1/kernel/')
    if not exists(args.save_dir + 'conv2/kernel/'):
        os.makedirs(args.save_dir + 'conv2/kernel/')
    if not exists(args.save_dir + 'conv3/kernel/'):
        os.makedirs(args.save_dir + 'conv3/kernel/')
    if not exists(args.save_dir + 'conv4/kernel/'):
        os.makedirs(args.save_dir + 'conv4/kernel/')
    if not exists(args.save_dir + 'conv5/kernel/'):
        os.makedirs(args.save_dir + 'conv5/kernel/')

    f = open(args.save_dir + 'conv5/conv5_chn.txt', 'w')
    f1 = open(args.save_dir + 'conv4/conv4_chn.txt', 'w')
    f2 = open(args.save_dir + 'conv3/conv3_chn.txt', 'w')
    f3 = open(args.save_dir + 'conv2/conv2_chn.txt', 'w')
    f4 = open(args.save_dir + 'conv1/conv1_chn.txt', 'w')

    for idx, dat in tqdm(enumerate(val_loader)):
        if idx == 5:
            img_name, img, label_in = dat
            img = img.cuda()
            input_var = torch.autograd.Variable(img, requires_grad=True)
            input_var1 = torch.autograd.Variable(img, requires_grad=True)
            labels = label_in.long().numpy()[0]

            ####forward         
            feature_maps, logits = model.forward_cam(input_var, target_layers)
            target = feature_maps[4]
            conv4_target = feature_maps[3]
            conv3_target = feature_maps[2]
            conv2_target = feature_maps[1]
            conv1_target = feature_maps[0]

            gb_feature_maps, gb_logits = gb_model.forward_cam(input_var1, target_layers)
            gb_target = gb_feature_maps[4]
            gb_conv4_target = gb_feature_maps[3]
            gb_conv3_target = gb_feature_maps[2]
            gb_conv2_target = gb_feature_maps[1]
            gb_conv1_target = gb_feature_maps[0]

            cv_im = cv2.imread(img_name[0])
            cv_im_gray = cv2.cvtColor(cv_im, cv2.COLOR_BGR2GRAY)
            height, width = cv_im.shape[:2]

            for la in range(int(args.num_classes)):
                if la == 14:
                    # if labels[la] == 1:
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
                    guided_gradients_upsample = torch.nn.functional.interpolate(guided_gradients, \
                                                                                size=(length, length), mode='bilinear',
                                                                                align_corners=True)

                    model.gradients = []

                    # compute cam
                    pre_cam = target * guided_gradients_upsample
                    cam = torch.sum(pre_cam, dim=1).squeeze()
                    cam[cam < 0] = 0
                    cam /= (cam.max() + 1e-8)

                    # cam_np = cam.detach().cpu().numpy()
                    # cam_np = np.uint8(cam_np * 255)
                    # cam_np = cv2.resize(cam_np, (width, height), interpolation=cv2.INTER_CUBIC)

                    # im_gray_name = args.save_dir + 'gray/' + img_name[0].split('/')[-1][:-4]
                    # im_color_name = args.save_dir + 'color/' + img_name[0].split('/')[-1][:-4]

                    # out_gray_name = im_gray_name + '_{}.png'.format(la)
                    # out_color_name = im_color_name + '_{}.png'.format(la)
                    # cv2.imwrite(out_gray_name, cam_np)

                    # cmap = plt.cm.jet_r(cam_np)[..., :3] * 255.0
                    # cam1 = (cmap.astype(np.float) + cv_im.astype(np.float)) / 2
                    # cv2.imwrite(out_color_name, cam1)

                    pre_cam[pre_cam < 0] = 0
                    pre_cam[0, :, cam < 0.5] = 0

                    mas, _ = torch.max(torch.max(pre_cam, 3)[0], 2)
                    pre_cam_max = torch.zeros_like(target)
                    for cc in range(pre_cam_max.shape[1]):
                        pre_cam_max[0, cc][pre_cam[0, cc] == mas[0, cc]] = mas[0, cc]

                    # compute cam
                    chn_cam = torch.sum(pre_cam_max, dim=(2, 3)).squeeze()
                    _, inds = torch.sort(-chn_cam)

                    for j in range(9):
                        ind = inds[j]
                        gb_model.gradients = []
                        unit_label = torch.zeros_like(pre_cam_max)
                        ma = torch.max(pre_cam_max[:, ind])
                        unit_label[:, ind][pre_cam_max[:, ind] == ma] = 1

                        gb_model.zero_grad()
                        input_var1.grad = None
                        gb_target.backward(gradient=unit_label, retain_graph=True)

                        grad = input_var1.grad.detach().cpu().numpy()[0]
                        grad = grad.transpose(1, 2, 0)
                        grad = cv2.resize(grad, (width, height), interpolation=cv2.INTER_CUBIC)
                        grad -= grad.min()
                        grad /= (grad.max() + 1e-20)

                        # xx = grad.copy()
                        # grad[xx > 0.65] = grad[xx > 0.65] * 1.3
                        # grad[np.logical_and(xx <= 0.65, xx >= 0.5)] = grad[np.logical_and(xx <= 0.65, xx >= 0.5)] * 1.1
                        # grad[grad > 1.0] = 1.

                        ### Scale between 0-255 to visualize
                        ind = ind.cpu().numpy()
                        im_name = '{}/conv5/kernel/{}'.format(args.save_dir, img_name[0].split('/')[-1][:-4])
                        out_name = im_name + '_{}_{}.png'.format(la, ind)
                        plt.imsave(out_name, grad)
                        f.write(str(ind) + ' ')

                        conv4_grad = gb_model.gradients[1]
                        print('conv4: ', conv4_grad.shape)
                        conv4_cam = conv4_grad * conv4_target
                        conv4_cam_max = torch.zeros_like(conv4_cam)
                        for ccc in range(conv4_cam_max.shape[1]):
                            ma1 = torch.max(conv4_cam[0, ccc])
                            conv4_cam_max[0, ccc][conv4_cam[0, ccc] == ma1] = ma1

                        # compute cam
                        chn_cam1 = torch.sum(conv4_cam_max, dim=(2, 3)).squeeze()
                        _, inds1 = torch.sort(-chn_cam1)

                        for k in range(9):
                            ind1 = inds1[k]
                            unit_label = torch.zeros_like(conv4_cam_max)
                            unit_label[:, ind1][conv4_cam_max[:, ind1] > 0] = 1

                            gb_model.gradients = []
                            gb_model.zero_grad()
                            input_var1.grad = None
                            gb_conv4_target.backward(gradient=unit_label, retain_graph=True)

                            grad = input_var1.grad.detach().cpu().numpy()[0]
                            grad = grad.transpose(1, 2, 0)
                            grad = cv2.resize(grad, (width, height), interpolation=cv2.INTER_CUBIC)
                            grad -= grad.min()
                            grad /= (grad.max() + 1e-20)

                            # xx = grad.copy()
                            # grad[xx > 0.65] = grad[xx > 0.65] * 1.3
                            # grad[np.logical_and(xx <= 0.65, xx >= 0.5)] = grad[np.logical_and(xx <= 0.65, xx >= 0.5)] * 1.1
                            # grad[grad > 1.0] = 1.

                            accu_grad = np.sum(grad, 2)
                            ma = np.max(accu_grad)
                            pos = np.where(accu_grad == ma)
                            y, x = pos[0][0], pos[1][0]

                            y1 = max(y - 80, 0)
                            y2 = min(y + 80, height)
                            if y2 - y1 != 160:
                                if y1 == 0:
                                    y2 = 160
                                if y2 == height:
                                    y1 = y2 - 160

                            x1 = max(x - 80, 0)
                            x2 = min(x + 80, width)
                            if x2 - x1 != 160:
                                if x1 == 0:
                                    x2 = 160
                                if x2 == width:
                                    x1 = x2 - 160

                            crop_grad = grad[y1:y2, x1:x2, :]

                            ### Scale between 0-255 to visualize
                            ind1 = ind1.cpu().numpy()
                            im_name = '{}/conv4/kernel/{}'.format(args.save_dir, img_name[0].split('/')[-1][:-4])
                            out_name = im_name + '_{}_{}_{}.png'.format(la, ind, ind1)
                            plt.imsave(out_name, crop_grad)
                            f1.write(str(ind1) + ' ')

                            conv3_grad = gb_model.gradients[1]
                            print('conv3: ', conv3_grad.shape)
                            conv3_cam = conv3_grad * conv3_target
                            conv3_cam_max = torch.zeros_like(conv3_cam)
                            for ccc in range(conv3_cam_max.shape[1]):
                                ma2 = torch.max(conv3_cam[0, ccc])
                                conv3_cam_max[0, ccc][conv3_cam[0, ccc] == ma2] = ma2

                            # compute cam
                            chn_cam2 = torch.sum(conv3_cam_max, dim=(2, 3)).squeeze()
                            _, inds2 = torch.sort(-chn_cam2)

                            for l in range(9):
                                ind2 = inds2[l]
                                unit_label = torch.zeros_like(conv3_cam_max)
                                unit_label[:, ind2][conv3_cam_max[:, ind2] > 0] = 1

                                gb_model.gradients = []
                                gb_model.zero_grad()
                                input_var1.grad = None
                                gb_conv3_target.backward(gradient=unit_label, retain_graph=True)

                                grad = input_var1.grad.detach().cpu().numpy()[0]
                                grad = grad.transpose(1, 2, 0)
                                grad = cv2.resize(grad, (width, height), interpolation=cv2.INTER_CUBIC)
                                grad -= grad.min()
                                grad /= (grad.max() + 1e-20)

                                # xx = grad.copy()
                                # grad[xx > 0.65] = grad[xx > 0.65] * 1.3
                                # grad[np.logical_and(xx <= 0.65, xx >= 0.5)] = grad[np.logical_and(xx <= 0.65, xx >= 0.5)] * 1.1
                                # grad[grad > 1.0] = 1.

                                accu_grad = np.sum(grad, 2)
                                ma = np.max(accu_grad)
                                pos = np.where(accu_grad == ma)
                                y, x = pos[0][0], pos[1][0]

                                y1 = max(y - 40, 0)
                                y2 = min(y + 40, height)
                                if y2 - y1 != 80:
                                    if y1 == 0:
                                        y2 = 80
                                    if y2 == height:
                                        y1 = y2 - 80

                                x1 = max(x - 40, 0)
                                x2 = min(x + 40, width)
                                if x2 - x1 != 80:
                                    if x1 == 0:
                                        x2 = 80
                                    if x2 == width:
                                        x1 = x2 - 80
                                crop_grad = grad[y1:y2, x1:x2, :]

                                ### Scale between 0-255 to visualize
                                ind2 = ind2.cpu().numpy()
                                im_name = '{}/conv3/kernel/{}'.format(args.save_dir, img_name[0].split('/')[-1][:-4])
                                out_name = im_name + '_{}_{}_{}_{}.png'.format(la, ind, ind1, ind2)
                                plt.imsave(out_name, crop_grad)
                                f2.write(str(ind2) + ' ')

                                conv2_grad = gb_model.gradients[1]
                                print('conv2: ', conv2_grad.shape)
                                conv2_cam = conv2_grad * conv2_target
                                conv2_cam_max = torch.zeros_like(conv2_cam)
                                for ccc in range(conv2_cam_max.shape[1]):
                                    ma3 = torch.max(conv2_cam[0, ccc])
                                    conv2_cam_max[0, ccc][conv2_cam[0, ccc] == ma3] = ma3

                                # compute cam
                                chn_cam3 = torch.sum(conv2_cam_max, dim=(2, 3)).squeeze()
                                _, inds3 = torch.sort(-chn_cam3)

                                for m in range(9):
                                    ind3 = inds3[m]
                                    unit_label = torch.zeros_like(conv2_cam_max)
                                    unit_label[:, ind3][conv2_cam_max[:, ind3] > 0] = 1

                                    gb_model.gradients = []
                                    gb_model.zero_grad()
                                    input_var1.grad = None
                                    gb_conv2_target.backward(gradient=unit_label, retain_graph=True)

                                    grad = input_var1.grad.detach().cpu().numpy()[0]
                                    grad = grad.transpose(1, 2, 0)
                                    grad = cv2.resize(grad, (width, height), interpolation=cv2.INTER_CUBIC)
                                    grad -= grad.min()
                                    grad /= (grad.max() + 1e-20)

                                    # xx = grad.copy()
                                    # grad[xx > 0.65] = grad[xx > 0.65] * 1.3
                                    # grad[np.logical_and(xx <= 0.65, xx >= 0.5)] = grad[np.logical_and(xx <= 0.65, xx >= 0.5)] * 1.1
                                    # grad[grad > 1.0] = 1.

                                    accu_grad = np.sum(grad, 2)
                                    ma = np.max(accu_grad)
                                    pos = np.where(accu_grad == ma)
                                    y, x = pos[0][0], pos[1][0]

                                    y1 = max(y - 20, 0)
                                    y2 = min(y + 20, height)
                                    if y2 - y1 != 40:
                                        if y1 == 0:
                                            y2 = 40
                                        if y2 == height:
                                            y1 = y2 - 40

                                    x1 = max(x - 20, 0)
                                    x2 = min(x + 20, width)
                                    if x2 - x1 != 40:
                                        if x1 == 0:
                                            x2 = 40
                                        if x2 == width:
                                            x1 = x2 - 40

                                    crop_grad = grad[y1:y2, x1:x2, :]

                                    #### Scale between 0-255 to visualize
                                    ind3 = ind3.cpu().numpy()
                                    im_name = '{}/conv2/kernel/{}'.format(args.save_dir,
                                                                          img_name[0].split('/')[-1][:-4])
                                    out_name = im_name + '_{}_{}_{}_{}_{}.png'.format(la, ind, ind1, ind2, ind3)
                                    plt.imsave(out_name, crop_grad)
                                    f3.write(str(ind3) + ' ')

                                    conv1_grad = gb_model.gradients[1]
                                    print('conv1: ', conv1_grad.shape)
                                    conv1_cam = conv1_grad * conv1_target
                                    conv1_cam_max = torch.zeros_like(conv1_cam)
                                    for ccc in range(conv1_cam_max.shape[1]):
                                        ma4 = torch.max(conv1_cam[0, ccc])
                                        conv1_cam_max[0, ccc][conv1_cam[0, ccc] == ma4] = ma4

                                    # compute cam
                                    chn_cam4 = torch.sum(conv1_cam_max, dim=(2, 3)).squeeze()
                                    _, inds4 = torch.sort(-chn_cam4)

                                    for n in range(9):
                                        ind4 = inds4[n]
                                        unit_label = torch.zeros_like(conv1_cam_max)
                                        unit_label[:, ind4][conv1_cam_max[:, ind4] > 0] = 1

                                        gb_model.gradients = []
                                        gb_model.zero_grad()
                                        input_var1.grad = None
                                        gb_conv1_target.backward(gradient=unit_label, retain_graph=True)

                                        grad = input_var1.grad.detach().cpu().numpy()[0]
                                        grad = grad.transpose(1, 2, 0)
                                        grad = cv2.resize(grad, (width, height), interpolation=cv2.INTER_CUBIC)
                                        grad -= grad.min()
                                        grad /= (grad.max() + 1e-20)

                                        # xx = grad.copy()
                                        # grad[xx > 0.65] = grad[xx > 0.65] * 1.3
                                        # grad[np.logical_and(xx <= 0.65, xx >= 0.5)] = grad[np.logical_and(xx <= 0.65, xx >= 0.5)] * 1.1
                                        # grad[grad > 1.0] = 1.

                                        accu_grad = np.sum(grad, 2)
                                        ma = np.max(accu_grad)
                                        pos = np.where(accu_grad == ma)
                                        y, x = pos[0][0], pos[1][0]

                                        y1 = max(y - 10, 0)
                                        y2 = min(y + 10, height)
                                        if y2 - y1 != 20:
                                            if y1 == 0:
                                                y2 = 20
                                            if y2 == height:
                                                y1 = y2 - 20

                                        x1 = max(x - 10, 0)
                                        x2 = min(x + 10, width)
                                        if x2 - x1 != 20:
                                            if x1 == 0:
                                                x2 = 20
                                            if x2 == width:
                                                x1 = x2 - 20

                                        crop_grad = grad[y1:y2, x1:x2, :]

                                        #### Scale between 0-255 to visualize
                                        ind4 = ind4.cpu().numpy()
                                        im_name = '{}/conv1/kernel/{}'.format(args.save_dir,
                                                                              img_name[0].split('/')[-1][:-4])
                                        out_name = im_name + '_{}_{}_{}_{}_{}_{}.png'.format(la, ind, ind1, ind2, ind3,
                                                                                             ind4)
                                        plt.imsave(out_name, crop_grad)

                                        f4.write(str(ind4) + ' ')

                    f4.write('\n')
                    f3.write('\n')
                    f2.write('\n')
                    f1.write('\n')
                    f.write('\n')


if __name__ == '__main__':
    args = get_arguments()
    print(args)
    # voc_hcrm_unit(args)
    # voc_hcrm_unit1(args)
    # voc_find_kernel_pattern(args)
    voc_hcrm_unit_kernel1(args)
