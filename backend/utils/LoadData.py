# from torchvision import transforms
from .transforms import transforms
from torch.utils.data import DataLoader
from .mydataset import dataset as my_dataset, dataset_caffe, ocdataset, datasetMSF, dataset_with_mask, VOCDataset, VOCDatasetMSF, VOCDatasetWBOX
import torchvision
import torch
import numpy as np
from .imutils import * 

def train_data_loader(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
       
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([#transforms.Resize(input_size),  
                                     RandomResizeLong(256, 512),
                                     #transforms.RandomCrop(crop_size), #321
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     RandomCrop(args.crop_size)
                                     ])

    if args.tencrop == 'True':
        func_transforms = [transforms.Resize(input_size),
                           transforms.TenCrop(crop_size),
                           transforms.Lambda(
                               lambda crops: torch.stack(
                                   [transforms.Normalize(mean_vals, std_vals)(transforms.ToTensor()(crop)) for crop in crops])),
                           ]
    else:
        func_transforms = []
        # print input_size, crop_size
        if input_size == 0 or crop_size == 0:
            pass
        else:
            func_transforms.append(transforms.Resize(input_size))
            func_transforms.append(transforms.CenterCrop(crop_size))

        func_transforms.append(transforms.ToTensor())
        func_transforms.append(transforms.Normalize(mean_vals, std_vals))

    tsfm_test = transforms.Compose(func_transforms)

    img_train = VOCDataset(args.train_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_train, test=True)
    img_test = VOCDataset(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, test=True)
    #img_train = ocdataset(args.train_list, root_dir=args.img_dir, transform=tsfm_train, with_path=True)
    #img_test = ocdataset(args.test_list, root_dir=args.img_dir, transform=tsfm_test, with_path=True)

    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader

def test_data_loader(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]

    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms.Resize(input_size),  #356
                                     transforms.RandomCrop(crop_size), #321
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)
                                     ])

    if args.tencrop == 'True':
        func_transforms = [transforms.Resize(input_size),
                           transforms.TenCrop(crop_size),
                           transforms.Lambda(
                               lambda crops: torch.stack(
                                   [transforms.Normalize(mean_vals, std_vals)(transforms.ToTensor()(crop)) for crop in crops])),
                           ]
    else:
        func_transforms = []
        # print input_size, crop_size
        if input_size == 0 or crop_size == 0:
            pass
        else:
            #func_transforms.append(transforms.Resize(input_size))
            func_transforms.append(transforms.Resize(crop_size))
            #func_transforms.append(transforms.CenterCrop(crop_size))

        func_transforms.append(transforms.ToTensor())
        func_transforms.append(transforms.Normalize(mean_vals, std_vals))

    tsfm_test = transforms.Compose(func_transforms)
    
    #####VOC dataset
    #img_train = VOCDataset(args.train_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_train, test=True)
    #img_test = VOCDataset(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, test=True)
   
    #####cub and imagenet
    img_train = ocdataset(args.train_list, root_dir=args.img_dir, transform=tsfm_train, with_path=True)
    img_test = ocdataset(args.test_list, root_dir=args.img_dir, transform=tsfm_test, with_path=True)  
    
    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader

def test_msf_data_loader(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]


    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms.Resize(input_size),  #356
                                     transforms.RandomCrop(crop_size), #321
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)
                                     ])

    if args.tencrop == 'True':
        func_transforms = [transforms.Resize(input_size),
                           transforms.TenCrop(crop_size),
                           transforms.Lambda(
                               lambda crops: torch.stack(
                                   [transforms.Normalize(mean_vals, std_vals)(transforms.ToTensor()(crop)) for crop in crops])),
                           ]
    else:
        func_transforms = []
        func_transforms.append(transforms.ToTensor())
        func_transforms.append(transforms.Normalize(mean_vals, std_vals))

    tsfm_test = transforms.Compose(func_transforms)

    img_train = VOCDatasetMSF(args.train_list, root_dir=args.img_dir, num_classes=args.num_classes, scales=args.scales, transform=tsfm_train, test=True)
    img_test = VOCDatasetMSF(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, scales=args.scales, transform=tsfm_test, test=True)
    #img_train = datasetMSF(args.train_list, root_dir=args.img_dir, scales=args.scales, transform=tsfm_train, with_path=True)
    #img_test = datasetMSF(args.test_list, root_dir=args.img_dir, scales=args.scales, transform=tsfm_test, with_path=True)

    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader

def test_data_loader_caffe(args, test_path=False, segmentation=False):
    # from torchvision import transforms
    # mean_vals = [104.00699/255.0, 116.66877/255.0, 122.6789/255.0]
    # std_vals = [1.0, 1.0, 1.0]

    # input_size = int(args.input_size)
    # crop_size = int(args.crop_size)
    # if args.tencrop == True:
    #     func_transforms = [transforms.Resize(input_size),
    #                        transforms.TenCrop(crop_size, vertical_flip=False),
    #                        transforms.Lambda(
    #                            lambda crops: torch.stack(
    #                                [transforms.Normalize(mean_vals, std_vals)(transforms.ToTensor()(crop)) for crop in crops])),
    #                        ]
    # else:
    #     func_transforms = []
    #     func_transforms.append(transforms.Resize(input_size))
    #     func_transforms.append(transforms.CenterCrop(crop_size))
    #     func_transforms.append(transforms.ToTensor())
    #     func_transforms.append(transforms.Normalize(mean_vals, std_vals))

    # tsfm_test = transforms.Compose(func_transforms)
    
    #####caffe model on imagenet
    img_test = dataset_caffe(args.test_list, root_dir=args.img_dir, transform=None, with_path=True)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader


def test_msf_data_loader(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]


    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms.Resize(input_size),  #356
                                     transforms.RandomCrop(crop_size), #321
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)
                                     ])

    if args.tencrop == 'True':
        func_transforms = [transforms.Resize(input_size),
                           transforms.TenCrop(crop_size),
                           transforms.Lambda(
                               lambda crops: torch.stack(
                                   [transforms.Normalize(mean_vals, std_vals)(transforms.ToTensor()(crop)) for crop in crops])),
                           ]
    else:
        func_transforms = []
        func_transforms.append(transforms.ToTensor())
        func_transforms.append(transforms.Normalize(mean_vals, std_vals))

    tsfm_test = transforms.Compose(func_transforms)

    #img_train = VOCDatasetMSF(args.train_list, root_dir=args.img_dir, num_classes=args.num_classes, scales=args.scales, transform=tsfm_train, test=True)
    #img_test = VOCDatasetMSF(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, scales=args.scales, transform=tsfm_test, test=True)
    img_train = datasetMSF(args.train_list, root_dir=args.img_dir, scales=args.scales, transform=tsfm_train, with_path=True)
    img_test = datasetMSF(args.test_list, root_dir=args.img_dir, scales=args.scales, transform=tsfm_test, with_path=True)

    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader

def test_wbox_data_loader(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]


    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms.Resize(input_size),  #356
                                     transforms.RandomCrop(crop_size), #321
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)
                                     ])

    if args.tencrop == 'True':
        func_transforms = [transforms.Resize(input_size),
                           transforms.TenCrop(crop_size),
                           transforms.Lambda(
                               lambda crops: torch.stack(
                                   [transforms.Normalize(mean_vals, std_vals)(transforms.ToTensor()(crop)) for crop in crops])),
                           ]
    else:
        func_transforms = []
        if input_size == 0 or crop_size == 0:
            pass
        else:
            func_transforms.append(transforms.Resize(input_size))
            #func_transforms.append(transforms.CenterCrop(crop_size))

        func_transforms.append(transforms.ToTensor())
        func_transforms.append(transforms.Normalize(mean_vals, std_vals))

    tsfm_test = transforms.Compose(func_transforms)

    img_train = VOCDatasetWBOX(args.train_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_train, test=True)

    img_test = VOCDatasetWBOX(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, test=True)

    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader

def test_voc_data_loader(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]

    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    
    func_transforms = []
    func_transforms.append(transforms.Resize(crop_size))
    func_transforms.append(transforms.ToTensor())
    func_transforms.append(transforms.Normalize(mean_vals, std_vals))

    tsfm_test = transforms.Compose(func_transforms)
    img_test = VOCDataset(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, test=True)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader, tsfm_test
