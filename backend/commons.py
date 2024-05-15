import torch
from models import vgg_voc, vgg
from utils import transforms
import pickle
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from functools import wraps
import matplotlib as mpl
import cv2


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
    model.eval()

    return model

def get_torch_model(args, bp_type='BP'):
    model = eval(args.arch).vgg16(pretrained=False, num_classes=args.num_classes, bp_type=bp_type)
    model.load_state_dict(torch.load(args.restore_from))
    model = model.cuda()

    return  model

def get_transform(args):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    crop_size = args.crop_size
    return transforms.Compose([transforms.Resize(crop_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_vals, std_vals)])


def init_position():
    index = [[] for i in range(10)]
    for i in range(9):
        index[i].extend([[] for i in range(10)])
        for ii in range(9):
            index[i][ii].extend([[] for i in range(10)])
            for iii in range(9):
                index[i][ii][iii].extend([[] for i in range(10)])
                for iiii in range(9):
                    index[i][ii][iii][iiii].extend([[-1] * 2 for i in range(9)])
                index[i][ii][iii][9] = [[-1] * 4 for i in range(9)]
            index[i][ii][9] = [[-1] * 4 for i in range(9)]
        index[i][9] = [[-1] * 4 for i in range(9)]
    index[9] = [[-1] * 4 for i in range(9)]
    return index


class Timer:
    def __init__(self):
        self.start = time.time()
        self.count = 0
        self.duration = 0
    def tick(self, message=None):
        torch.cuda.synchronize()
        self.duration = time.time() - self.start
        if message:
            print(message, str(self.duration))
        else:
            print(str(self.duration))
        self.start = time.time()
    def set_start(self):
        torch.cuda.synchronize()
        self.start = time.time()
    def reset_count(self):
        self.count = 0
    def add(self):
        torch.cuda.synchronize()
        self.duration += time.time() - self.start
        self.count += 1


def timeit(func):
    @wraps(func)
    def timer(*args, **kwargs):
        start = time.time()
        for i in range(len(args)):
            print(f'[{func.__name__}][{func.__code__.co_varnames[i]}] Value: {args[i]}')
        res = func(*args, **kwargs)
        print(f'[{func.__name__}] Time used: {(time.time()-start)*1000:.3f}ms')
        return res
    return timer


def pickle_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def pickle_dump(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def max_crop(img, size):
    height, width = img.shape[:2]
    half_size = size // 2
    sum_img = np.sum(img, 2)
    ma = np.max(sum_img)
    pos = np.where(sum_img == ma)
    y, x = pos[0][0], pos[1][0]

    y1 = max(y - half_size, 0)
    y2 = min(y + half_size, height)
    if y2 - y1 != size:
        if y1 == 0 and y2 != height:
            y2 = size
        if y2 == height and y1 != 0:
            y1 = y2 - size

    x1 = max(x - half_size, 0)
    x2 = min(x + half_size, width)
    if x2 - x1 != size:
        if x1 == 0 and x2 != width:
            x2 = size
        if x2 == width and x1 != 0:
            x1 = x2 - size
    x1 = max(x1, 0)
    x2 = min(x2, width)
    y1 = max(y1, 0)
    y2 = min(y2, height)
    return img[y1:y2, x1:x2, :], (y, x)
    # return img, (y, x)


colormaps = ['#000000', '#7F0000', '#007F00', '#7F7F00', '#00007F', '#7F007F', '#007F7F', '#7F7F7F', '#3F0000',
             '#BF0000', '#3F7F00',
             '#BF7F00', '#3F007F', '#BF007F', '#3F7F7F', '#BF7F7F', '#003F00', '#7F3F00', '#00BF00', '#7FBF00',
             '#003F7F']


def colormap(index):
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', [colormaps[0], colormaps[index + 1], '#FFFFFF'], 256)

def save_cam(cam, cv_im, out_name):
    height, width = cv_im.shape[:2]
    # height, width = (224, 224)
    # cv_im_resize = cv2.resize(cv_im, (width, height), interpolation=cv2.INTER_CUBIC)
    cam_np = cam.clone().detach().cpu().numpy()
    cam_np = np.uint8(cam_np * 255)  
    cam_np = cv2.resize(cam_np, (width, height), interpolation=cv2.INTER_CUBIC)
    
    out_gray_name = out_name + '_graycam.jpg'
    out_color_name = out_name + '_cam.jpg'    
    cv2.imwrite(out_gray_name, cam_np)
    cmap = plt.cm.jet_r(cam_np)[..., :3] * 255.0
    cam1 = (cmap.astype(np.float) + cv_im.astype(np.float)) / 2
    # cam1 = (cmap.astype(np.float) + cv_im_resize.astype(np.float)) / 2
    cv2.imwrite(out_color_name, cam1)

def save_crop_cam(cam, cv_im, out_name, crop_center, size):
    height, width = cv_im.shape[:2]
    # height, width = (224, 224)
    # cv_im_resize = cv2.resize(cv_im, (width, height), interpolation=cv2.INTER_CUBIC)
    cam_np = cam.clone().detach().cpu().numpy()
    cam_np = np.uint8(cam_np * 255)  
    cam_np = cv2.resize(cam_np, (width, height), interpolation=cv2.INTER_CUBIC)

    half_size = size // 2
    y, x =crop_center

    y1 = max(y - half_size, 0)
    y2 = min(y + half_size, height)
    # print('y1, y2', y1, y2)
    if y2 - y1 != size:
        if y1 == 0:
            y2 = size
        if y2 == height:
            y1 = y2 - size

    x1 = max(x - half_size, 0)
    x2 = min(x + half_size, width)
    # print('x1, x2', x1, x2)
    if x2 - x1 != size:
        if x1 == 0:
            x2 = size
        if x2 == width:
            x1 = x2 - size

    x1 = max(x1, 0)
    x2 = min(x2, width)
    y1 = max(y1, 0)
    y2 = min(y2, height)

    cam_np = cam_np[y1:y2, x1:x2]
    # cv_im_resize = cv_im_resize[y1:y2, x1:x2, :]
    cv_im = cv_im[y1:y2, x1:x2, :]

    # print(x1, x2, y1, y2)
    out_gray_name = out_name + '_graycam.jpg'
    out_color_name = out_name + '_cam.jpg'    
    cv2.imwrite(out_gray_name, cam_np)
    cmap = plt.cm.jet_r(cam_np)[..., :3] * 255.0
    cam1 = (cmap.astype(np.float) + cv_im.astype(np.float)) / 2
    # cam1 = (cmap.astype(np.float) + cv_im_resize.astype(np.float)) / 2
    cv2.imwrite(out_color_name, cam1)

def activation_sort(activation):
    mas, _ = torch.max(torch.max(activation, 3)[0], 2)
    mas = mas.squeeze()

    avg = torch.sum(torch.sum(activation, 3), 2)
    avg = avg.squeeze()
    avg_sort, inds = torch.sort(-avg)
    avg_sort = -avg_sort
    avg_all = torch.sum(avg_sort)
    avg_sort /= (avg_all + 1e-10)
    return mas, inds, avg_sort

def activation_max_sort(activation):
    mas, _ = torch.max(torch.max(activation, 3)[0], 2)
    mas = mas.squeeze()
    mas_sort, inds = torch.sort(-mas)
    mas_sort = -mas_sort
    mas_all = torch.sum(mas_sort)
    mas_sort /= (mas_all + 1e-10)
    return mas, inds, mas_sort

def dupl_removal(conv_cam, inds, avg_sort, att_threshold=0.4, iou_threshold=0.4):
    inds_np = inds.clone().cpu().numpy()
    avg_sort_np = avg_sort.clone().detach().cpu().numpy()
    conv_cam_np = conv_cam.clone().detach().cpu().numpy()[0]
    conv_cam_np /= (conv_cam_np.max((1, 2)).reshape(-1, 1, 1) + 1e-10)
    contrib = np.sum(conv_cam_np, (1, 2))[inds_np]
    inds_np = inds_np[contrib > 0.0]
    avg_sort_np = avg_sort_np[contrib > 0.0]
    #origin_len = len(inds_np)

    keep_inds = []
    keep_inds_overlap = []
    keep_inds_iou = []
    keep_avg_sort = []
    print(len(avg_sort_np))
    while len(inds_np) > 0:
        cam_ii = conv_cam_np[inds_np[0]] > att_threshold
        cam_jj = conv_cam_np[inds_np[1:]] > att_threshold
        intect = np.sum(cam_ii * cam_jj > 0, (1, 2))
        union = np.sum((cam_ii + cam_jj) > 0, (1, 2))
        iou = intect / (union + 1e-20)

        avg_sort_np[0] += avg_sort_np[1:][iou >= iou_threshold].sum()
        keep_inds.append(inds_np[0])
        keep_avg_sort.append(avg_sort_np[0])
        
        overlap_ind = inds_np[1:][iou >= iou_threshold]
        keep_inds_overlap.append(overlap_ind)
        keep_inds_iou.append(iou[iou >= iou_threshold])
        inds_np = inds_np[1:][iou < iou_threshold]
        avg_sort_np = avg_sort_np[1:][iou < iou_threshold]

    return torch.tensor(keep_inds).type_as(inds), torch.tensor(keep_avg_sort).type_as(avg_sort)

