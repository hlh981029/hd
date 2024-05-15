from commons import colormap, timeit, get_model, get_torch_model, get_transform, init_position, max_crop, \
                    Timer, pickle_dump, pickle_load, save_cam, activation_sort, activation_max_sort, save_crop_cam
from PIL import Image, ImageEnhance
import configs
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

test_filename = 'demo1.jpg'
if configs.dataset == 'pascal_voc':
    model = get_model(configs)
    gb_model = get_model(configs, bp_type='GBP')
elif configs.dataset == 'imagenet':
    model = get_torch_model(configs)
    gb_model = get_torch_model(configs, bp_type='GBP')
model.eval()
gb_model.eval()


transforms = get_transform(configs)
target_layers = [3, 4, 8, 9, 15, 16, 22, 23, 29, 30]
target_layers1 = [3, 8, 15, 22, 29]
length = 14

def gen_sample():
    sample_list = os.listdir(configs.sample_dir)
    for i in sample_list:
        filename = i
        img_dir = configs.upload_dir + filename + '/'
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        img = Image.open(configs.sample_dir + filename)
        img.save(img_dir + filename)

@timeit
def get_labels(filename):
    img_dir = configs.upload_dir + filename + '/'
    img_name = img_dir + filename
    img = Image.open(img_name).convert('RGB')
    img = transforms(img).unsqueeze(0)
    # img_np = img.clone().detach().cpu().numpy().squeeze()
    # img_np -= img_np.min()
    # img_np /= img_np.max()
    # img_np *= 255
    # img_np = img_np.astype(np.uint8)
    # img_np = np.transpose(img_np, (1, 2, 0))
    # print(img_np.shape, img_np.dtype)
    # img_np = Image.fromarray(img_np)
    # img_np = img_np.convert('RGB')
    # img_np.save(img_dir + 'img_np.png')
    img = img.cuda()
    if configs.dataset == 'pascal_voc':
        logits = torch.sigmoid(model.forward(img))[0]
    elif configs.dataset == 'imagenet':
        logits = torch.softmax(model.forward(img), dim=1)[0]
    # print(logits)
    return logits.tolist()


@timeit
def get_kernel_images_5(filename, label):
    top_n = configs.top_n
    atten_thres_all = configs.atten_thres_all
    img_dir = configs.upload_dir + filename + '/'
    if not os.path.exists(img_dir + f'{label}_position.data'):
        position = init_position()
    else:
        position = pickle_load(img_dir + f'{label}_position.data')
    if sum([i[0] for i in position[9]]) != -9:
        print('found')
        return ([i[0] for i in position[9]], [i[3] for i in position[9]])
    prefix = f'{label}_5_'
    result, result_mas = [], []
    img_name = img_dir + filename
    img = Image.open(img_name)
    height, width = img.height, img.width
    img = transforms(img).unsqueeze(0)
    img = img.cuda()

    input_var = torch.autograd.Variable(img, requires_grad=True)
    input_var1 = torch.autograd.Variable(img, requires_grad=True)

    feature_maps, logits = model.forward_cam(input_var, target_layers)
    target = feature_maps[8]
    gb_feature_maps, gb_logits = gb_model.forward_cam(input_var1, target_layers1)
    gb_target = gb_feature_maps[4]

    one_hot_output = torch.FloatTensor(1, configs.num_classes).zero_()
    one_hot_output[0][label] = 1
    one_hot_output = one_hot_output.cuda(non_blocking=True)
    
    model.gradients = []
    model.zero_grad()
    logits.backward(gradient=one_hot_output, retain_graph=True)
    conv5_grad = model.gradients[0]
    conv5_grad_up = F.interpolate(conv5_grad, size=(target.size()[2], target.size()[3]), \
                              mode='bilinear', align_corners=True)
    # compute grad-cam
    pre_cam = target * conv5_grad_up
    weights = torch.mean(conv5_grad_up, dim=(2, 3))
    cam = torch.sum(weights.view(1, -1, 1, 1) * target, dim=1).squeeze()
    cam[cam < 0] = 0
    cam /= (cam.max() + 1e-8)
    out_name = img_dir + f'{label}'
    cv_im = cv2.imread(img_name)
    save_cam(cam, cv_im, out_name)    

    pre_cam[pre_cam < 0] = 0
    pre_cam_cp = pre_cam.clone()
    pre_cam[0, :, cam < atten_thres_all] = 0
    mas, inds, avg_sort = activation_sort(pre_cam)
    #mas, inds, avg_sort = activation_max_sort(pre_cam)
    # print(avg_sort)  
    
    for j in range(top_n):
        ind = inds[j]
        res = avg_sort[j]
        # guided back to visulize each kernel  
        unit_label = torch.zeros_like(target)
        ma = mas[ind]
        unit_label[:, ind][pre_cam[:, ind] == ma] = 1
        assert unit_label.nonzero().shape[0] == 1
        indices = unit_label[0, ind].nonzero().tolist()
        position[9][j][:] = int(ind), *indices[0], float(res)

        gb_model.gradients = []   
        gb_model.zero_grad()
        input_var1.grad = None
        gb_target.backward(gradient=unit_label, retain_graph=True)
        grad = input_var1.grad.detach().cpu().numpy()[0]
        grad = grad.transpose(1, 2, 0)
        grad = cv2.resize(grad, (width, height), interpolation=cv2.INTER_CUBIC)
        grad -= grad.min()
        grad /= (grad.max() + 1e-20)
        crop_grad, crop_center = max_crop(grad, 280)
        ind = ind.cpu().numpy()
        
        out_name = img_dir + prefix + str(ind) + '.jpg'
        origin_image = Image.fromarray(np.asarray((crop_grad * 255).astype(np.uint8)))
        converter = ImageEnhance.Color(origin_image)
        converted_image = converter.enhance(5)
        converted_image.save(img_dir + prefix + str(ind) + f'_e.jpg')
        plt.imsave(out_name, crop_grad)
        # save cam for each filter
        conv5_cam_loc = pre_cam[0, ind].clone() 
        conv5_cam_loc /= (conv5_cam_loc.max() + 1e-8)
        # save_cam(conv5_cam_loc, cv_im, out_name[:-4])
        save_crop_cam(conv5_cam_loc, cv_im, out_name[:-4], crop_center, 280)
        
        result.append(int(ind))
        result_mas.append(float(res))
        
    pickle_dump(position, img_dir + f'{label}_position.data')
    return result, result_mas

@timeit
def get_kernel_images_4(filename, label, index):
    top_n = configs.top_n
    img_dir = configs.upload_dir + filename + '/'
    position = pickle_load(img_dir + f'{label}_position.data')
    channel_position = position[index]
    if sum([i[0] for i in channel_position[9]]) != -9:
        print('found')
        return ([i[0] for i in channel_position[9]], [i[3] for i in channel_position[9]])
    conv5_channel = position[9][index][0]
    prefix = f'{label}_4_{conv5_channel}_'
    
    result, result_mas = [], []
    img_name = img_dir + filename
    img = Image.open(img_name)
    height, width = img.height, img.width
    img = transforms(img).unsqueeze(0)
    img = img.cuda()
    cv_im = cv2.imread(img_name)
    
    input_var = torch.autograd.Variable(img, requires_grad=True)
    input_var1 = torch.autograd.Variable(img, requires_grad=True)

    feature_maps, logits = model.forward_cam(input_var, target_layers)
    target_5 = feature_maps[8]
    target_4 = feature_maps[6]
    
    gb_feature_maps, gb_logits = gb_model.forward_cam(input_var1, target_layers1)
    gb_target_5 = gb_feature_maps[4]
    gb_target_4 = gb_feature_maps[3]

    unit_label = torch.zeros_like(target_5)
    unit_label[0][tuple(position[9][index][:3])] = 1
    model.gradients = []
    model.zero_grad()
    input_var.grad = None
    target_5.backward(gradient=unit_label, retain_graph=True)
    conv4_grad = model.gradients[1]
    conv4_grad_up = F.interpolate(conv4_grad, size=(target_4.size()[2], target_4.size()[3]), \
                              mode='bilinear', align_corners=True)
    conv4_cam = conv4_grad_up * target_4
    conv4_cam[conv4_cam < 0] = 0
    mas, inds, avg_sort = activation_sort(conv4_cam)
    #mas, inds, avg_sort = activation_max_sort(conv4_cam)
    # print(avg_sort)  
    
    for j in range(top_n):
        ind = inds[j]
        res = avg_sort[j]
        unit_label = torch.zeros_like(conv4_cam)
        ma = mas[ind]
        unit_label[:, ind][conv4_cam[:, ind] == ma] = 1
        assert unit_label.nonzero().shape[0] == 1
        indices = unit_label[0, ind].nonzero().tolist()
        channel_position[9][j][:] = int(ind), *indices[0], float(res)
        
        gb_model.gradients = []
        gb_model.zero_grad()
        input_var1.grad = None
        gb_target_4.backward(gradient=unit_label, retain_graph=True)
        grad = input_var1.grad.detach().cpu().numpy()[0]
        grad = grad.transpose(1, 2, 0)
        grad = cv2.resize(grad, (width, height), interpolation=cv2.INTER_CUBIC)
        grad -= grad.min()
        grad /= (grad.max() + 1e-20)
        crop_grad, crop_center = max_crop(grad, 160)
        
        ind = ind.cpu().numpy()
        out_name = img_dir + prefix + str(ind) + '.jpg'
        origin_image = Image.fromarray(np.asarray((crop_grad * 255).astype(np.uint8)))
        converter = ImageEnhance.Color(origin_image)
        converted_image = converter.enhance(5)
        converted_image.save(img_dir + prefix + str(ind) + f'_e.jpg')
        plt.imsave(out_name, crop_grad)
        # save cam for each filter
        conv4_cam_loc = conv4_cam[0, ind].clone() 
        conv4_cam_loc /= (conv4_cam_loc.max() + 1e-8)
        # save_cam(conv4_cam_loc, cv_im, out_name[:-4])  
        save_crop_cam(conv4_cam_loc, cv_im, out_name[:-4], crop_center, 160)  
      
        result.append(int(ind))
        result_mas.append(float(res))
    pickle_dump(position, img_dir + f'{label}_position.data')
    return result, result_mas


@timeit
def get_kernel_images_3(filename, label, index_5, index_4):
    top_n = configs.top_n
    img_dir = configs.upload_dir + filename + '/'
    position = pickle_load(img_dir + f'{label}_position.data')
    channel_position = position[index_5][index_4]
    if sum([i[0] for i in channel_position[9]]) != -9:
        print('found')
        return ([i[0] for i in channel_position[9]], [i[3] for i in channel_position[9]])
    conv5_channel = position[9][index_5][0]
    conv4_channel = position[index_5][9][index_4][0]
    prefix = f'{label}_3_{conv5_channel}_{conv4_channel}_'

    result, result_mas = [], []
    img_name = img_dir + filename
    img = Image.open(img_name)
    height, width = img.height, img.width
    img = transforms(img).unsqueeze(0)
    img = img.cuda()
    cv_im = cv2.imread(img_name)
    
    input_var = torch.autograd.Variable(img, requires_grad=True)
    input_var1 = torch.autograd.Variable(img, requires_grad=True)

    feature_maps, logits = model.forward_cam(input_var, target_layers)
    target_4 = feature_maps[6]
    target_3 = feature_maps[4]
    
    gb_feature_maps, gb_logits = gb_model.forward_cam(input_var1, target_layers1)
    gb_target_3 = gb_feature_maps[2]
    
    unit_label = torch.zeros_like(target_4)
    unit_label[0][tuple(position[index_5][9][index_4][:3])] = 1
    
    model.gradients = []
    model.zero_grad()
    input_var.grad = None
    target_4.backward(gradient=unit_label, retain_graph=True)
    conv3_grad = model.gradients[1]
    
    conv3_grad_up = F.interpolate(conv3_grad, size=(target_3.size()[2], target_3.size()[3]), \
                              mode='bilinear', align_corners=True)
    conv3_cam = conv3_grad_up * target_3
    conv3_cam[conv3_cam < 0] = 0
    mas, inds, avg_sort = activation_sort(conv3_cam)
    #mas, inds, avg_sort = activation_max_sort(conv3_cam)
    # print(avg_sort)  
    
    for j in range(top_n):
        ind = inds[j]
        res = avg_sort[j]
        unit_label = torch.zeros_like(conv3_cam)
        ma = mas[ind]
        unit_label[:, ind][conv3_cam[:, ind] == ma] = 1
        assert unit_label.nonzero().shape[0] == 1
        indices = unit_label[0, ind].nonzero().tolist()
        channel_position[9][j][:] = int(ind), *indices[0], float(res)
        
        gb_model.gradients = []
        gb_model.zero_grad()
        input_var1.grad = None
        gb_target_3.backward(gradient=unit_label, retain_graph=True)
        
        grad = input_var1.grad.detach().cpu().numpy()[0]
        grad = grad.transpose(1, 2, 0)
        grad = cv2.resize(grad, (width, height), interpolation=cv2.INTER_CUBIC)
        grad -= grad.min()
        grad /= (grad.max() + 1e-20)
        crop_grad, crop_center = max_crop(grad, 80)
        ind = ind.cpu().numpy()
        
        out_name = img_dir + prefix + str(ind) + '.jpg'
        origin_image = Image.fromarray(np.asarray((crop_grad * 255).astype(np.uint8)))
        converter = ImageEnhance.Color(origin_image)
        converted_image = converter.enhance(5)
        converted_image.save(img_dir + prefix + str(ind) + f'_e.jpg')
        plt.imsave(out_name, crop_grad)
        # save cam for each filter
        conv3_cam_loc = conv3_cam[0, ind].clone() 
        conv3_cam_loc /= (conv3_cam_loc.max() + 1e-8)
        # save_cam(conv3_cam_loc, cv_im, out_name[:-4])  
        save_crop_cam(conv3_cam_loc, cv_im, out_name[:-4], crop_center, 80)  
        
        result.append(int(ind))
        result_mas.append(float(res))
    pickle_dump(position, img_dir + f'{label}_position.data')
    return result, result_mas


@timeit
def get_kernel_images_2(filename, label, index_5, index_4, index_3):
    top_n = configs.top_n
    img_dir = configs.upload_dir + filename + '/'
    position = pickle_load(img_dir + f'{label}_position.data')
    channel_position = position[index_5][index_4][index_3]
    if sum([i[0] for i in channel_position[9]]) != -9:
        print('found')
        return ([i[0] for i in channel_position[9]], [i[3] for i in channel_position[9]])
    conv5_channel = position[9][index_5][0]
    conv4_channel = position[index_5][9][index_4][0]
    conv3_channel = position[index_5][index_4][9][index_3][0]
    prefix = f'{label}_2_{conv5_channel}_{conv4_channel}_{conv3_channel}_'

    result, result_mas = [], []
    img_name = img_dir + filename
    img = Image.open(img_name)
    height, width = img.height, img.width
    img = transforms(img).unsqueeze(0)
    img = img.cuda()
    cv_im = cv2.imread(img_name)
    
    input_var = torch.autograd.Variable(img, requires_grad=True)
    input_var1 = torch.autograd.Variable(img, requires_grad=True)

    feature_maps, logits = model.forward_cam(input_var, target_layers)
    target_3 = feature_maps[4]
    target_2 = feature_maps[2]

    gb_feature_maps, gb_logits = gb_model.forward_cam(input_var1, target_layers1)
    gb_target_3 = gb_feature_maps[2]
    gb_target_2 = gb_feature_maps[1]

    unit_label = torch.zeros_like(target_3)
    unit_label[0][tuple(position[index_5][index_4][9][index_3][:3])] = 1
    
    model.gradients = []
    model.zero_grad()
    input_var.grad = None
    target_3.backward(gradient=unit_label, retain_graph=True)
    conv2_grad = model.gradients[1]
    conv2_grad_up = F.interpolate(conv2_grad, size=(target_2.size()[2], target_2.size()[3]), \
                              mode='bilinear', align_corners=True)
    conv2_cam = conv2_grad_up * target_2
    conv2_cam[conv2_cam < 0] = 0
    mas, inds, avg_sort = activation_sort(conv2_cam)
    #mas, inds, avg_sort = activation_max_sort(conv2_cam)
    # print(avg_sort) 
    
    for j in range(top_n):
        ind = inds[j]
        res = avg_sort[j]
        unit_label = torch.zeros_like(conv2_cam)
        ma = mas[ind]
        unit_label[:, ind][conv2_cam[:, ind] == ma] = 1
        assert unit_label.nonzero().shape[0] == 1
        indices = unit_label[0, ind].nonzero().tolist()
        channel_position[9][j][:] = int(ind), *indices[0], float(res)

        gb_model.gradients = []
        gb_model.zero_grad()
        input_var1.grad = None
        gb_target_2.backward(gradient=unit_label, retain_graph=True)
        
        grad = input_var1.grad.detach().cpu().numpy()[0]
        grad = grad.transpose(1, 2, 0)
        grad = cv2.resize(grad, (width, height), interpolation=cv2.INTER_CUBIC)
        grad -= grad.min()
        grad /= (grad.max() + 1e-20)
        crop_grad, crop_center = max_crop(grad, 40)
        ind = ind.cpu().numpy()
        
        out_name = img_dir + prefix + str(ind) + '.jpg'
        origin_image = Image.fromarray(np.asarray((crop_grad * 255).astype(np.uint8)))
        converter = ImageEnhance.Color(origin_image)
        converted_image = converter.enhance(5)
        converted_image.save(img_dir + prefix + str(ind) + f'_e.jpg')
        plt.imsave(out_name, crop_grad)
        # save cam for each filter
        conv2_cam_loc = conv2_cam[0, ind].clone() 
        conv2_cam_loc /= (conv2_cam_loc.max() + 1e-8)
        # save_cam(conv2_cam_loc, cv_im, out_name[:-4])  
        save_crop_cam(conv2_cam_loc, cv_im, out_name[:-4], crop_center, 40)          

        result.append(int(ind))
        result_mas.append(float(res))
    pickle_dump(position, img_dir + f'{label}_position.data')
    return result, result_mas


@timeit
def get_kernel_images_1(filename, label, index_5, index_4, index_3, index_2):
    top_n = configs.top_n
    img_dir = configs.upload_dir + filename + '/'
    position = pickle_load(img_dir + f'{label}_position.data')
    channel_position = position[index_5][index_4][index_3][index_2]
    print(channel_position)
    if sum([i[0] for i in channel_position]) != -9:
        print('found')
        return ([i[0] for i in channel_position], [i[1] for i in channel_position]) 
    conv5_channel = position[9][index_5][0]
    conv4_channel = position[index_5][9][index_4][0]
    conv3_channel = position[index_5][index_4][9][index_3][0]
    conv2_channel = position[index_5][index_4][index_3][9][index_2][0]
    prefix = f'{label}_1_{conv5_channel}_{conv4_channel}_{conv3_channel}_{conv2_channel}_'

    result, result_mas = [], []
    img_name = img_dir + filename
    img = Image.open(img_name)
    height, width = img.height, img.width
    img = transforms(img).unsqueeze(0)
    img = img.cuda()
    cv_im = cv2.imread(img_name)
    
    input_var = torch.autograd.Variable(img, requires_grad=True)
    input_var1 = torch.autograd.Variable(img, requires_grad=True)

    feature_maps, logits = model.forward_cam(input_var, target_layers)
    target_2 = feature_maps[2]
    target_1 = feature_maps[0]

    gb_feature_maps, gb_logits = gb_model.forward_cam(input_var1, target_layers1)
    gb_target_2 = gb_feature_maps[1]
    gb_target_1 = gb_feature_maps[0]

    unit_label = torch.zeros_like(target_2)
    unit_label[0][tuple(position[index_5][index_4][index_3][9][index_2][:3])] = 1
    
    model.gradients = []
    model.zero_grad()
    input_var.grad = None
    target_2.backward(gradient=unit_label, retain_graph=True)
    conv1_grad = model.gradients[1]
    conv1_grad_up = F.interpolate(conv1_grad, size=(target_1.size()[2], target_1.size()[3]), \
                              mode='bilinear', align_corners=True)
    conv1_cam = conv1_grad_up * target_1
    conv1_cam[conv1_cam < 0] = 0
    mas, inds, avg_sort = activation_sort(conv1_cam) 
    #mas, inds, avg_sort = activation_max_sort(conv1_cam)
    # print(avg_sort)     
    
    for j in range(top_n):
        ind = inds[j]
        res = avg_sort[j]
        unit_label = torch.zeros_like(conv1_cam)
        ma = mas[ind]
        unit_label[:, ind][conv1_cam[:, ind] == ma] = 1
        assert unit_label.nonzero().shape[0] == 1
        indices = unit_label[0, ind].nonzero().tolist()
        channel_position[j] = int(ind), float(res)
        
        gb_model.gradients = []
        gb_model.zero_grad()
        input_var1.grad = None
        gb_target_1.backward(gradient=unit_label, retain_graph=True)
        grad = input_var1.grad.detach().cpu().numpy()[0]
        grad = grad.transpose(1, 2, 0)
        grad = cv2.resize(grad, (width, height), interpolation=cv2.INTER_CUBIC)
        grad -= grad.min()
        grad /= (grad.max() + 1e-20)
        crop_grad, crop_center = max_crop(grad, 20)
        ind = ind.cpu().numpy()
        
        out_name = img_dir + prefix + str(ind) + '.jpg'
        origin_image = Image.fromarray(np.asarray((crop_grad * 255).astype(np.uint8)))
        converter = ImageEnhance.Color(origin_image)
        converted_image = converter.enhance(5)
        converted_image.save(img_dir + prefix + str(ind) + f'_e.jpg')
        plt.imsave(out_name, crop_grad)
        # save cam for each filter
        conv1_cam_loc = conv1_cam[0, ind].clone() 
        conv1_cam_loc /= (conv1_cam_loc.max() + 1e-8)
        # save_cam(conv1_cam_loc, cv_im, out_name[:-4])  
        save_crop_cam(conv1_cam_loc, cv_im, out_name[:-4], crop_center, 20)  
        
        result.append(int(ind))
        result_mas.append(float(res))

    pickle_dump(position, img_dir + f'{label}_position.data')
    return result, result_mas

