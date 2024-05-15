from commons import colormap, timeit, get_model, get_transform, init_position, max_crop, Timer, pickle_dump, pickle_load
from PIL import Image, ImageEnhance
import configs
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

test_filename = 'demo1.jpg'
model = get_model(configs)
gb_model = get_model(configs, bp_type='GBP')

transforms = get_transform(configs)
target_layers = [3, 8, 15, 22, 29, 30]
target_layers1 = [3, 4, 8, 9, 15, 16, 22, 23, 29, 30]

@timeit
def get_labels(filename):
    img_dir = configs.upload_dir + filename + '/'
    img_name = img_dir + filename
    img = Image.open(img_name)
    img = transforms(img).unsqueeze(0)
    img = img.cuda()
    logits = torch.sigmoid(model.forward(img))[0]
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
        print('Successfully loaded....')
    if sum([i[0] for i in position[9]]) != -9:
        print('found')
        return ([i[0] for i in position[9]], [i[3] for i in position[9]])
    prefix = f'{label}_5_'
    result = []
    result_mas = []
    img_name = img_dir + filename
    img = Image.open(img_name)
    height = img.height
    width = img.width
    img = transforms(img).unsqueeze(0)
    img = img.cuda()

    input_var = torch.autograd.Variable(img, requires_grad=True)
    input_var1 = torch.autograd.Variable(img, requires_grad=True)

    feature_maps, logits = model.forward_cam(input_var, target_layers)
    target = feature_maps[4]
    gb_feature_maps, gb_logits = gb_model.forward_cam(input_var1, target_layers1)
    gb_target = gb_feature_maps[8]

    one_hot_output = torch.FloatTensor(1, configs.num_classes).zero_()
    one_hot_output[0][label] = 1
    one_hot_output = one_hot_output.cuda(non_blocking=True)
    
    model.gradients = []
    # Zero grads
    model.zero_grad()
    # Backward pass with specified target
    logits.backward(gradient=one_hot_output, retain_graph=True)
    # Obtain gradients for 30 layer
    guided_gradients = model.gradients[0]
    guided_gradients_upsample = torch.nn.functional.interpolate(guided_gradients,
                                                                size=(target.shape[-2], target.shape[-1]), mode='bilinear',
                                                                align_corners=True)
    # compute cam
    pre_cam = target * guided_gradients_upsample
    cam = torch.sum(pre_cam, dim=1).squeeze()
    cam[cam < 0] = 0
    cam /= (cam.max() + 1e-8)
    
    cam_np = cam.detach().cpu().numpy()
    cam_np = np.uint8(cam_np * 255)
    cam_np = cv2.resize(cam_np, (width, height), interpolation=cv2.INTER_CUBIC)
    cam_np_min = np.min(cam_np)
    cam_np_max = np.max(cam_np)
    cam_np = (cam_np - cam_np_min) / (cam_np_max - cam_np_min + 1e-8)
    cv_im = cv2.imread(img_name)
    out_color_name = img_dir + f'{label}_cam.png'
    cmap = plt.cm.jet_r(cam_np)[..., :3] * 255.0
    cam1 = (cmap.astype(np.float) + cv_im.astype(np.float)) / 2
    cv2.imwrite(out_color_name, cam1)
    
    #cv_im_gray = cv2.cvtColor(cv_im, cv2.COLOR_BGR2GRAY)
    #cam_np = cv_im_gray * 0.2 + cam_np * 0.8 * 255
    #plt.imsave(out_color_name, cam_np, cmap=colormap(label))

    pre_cam[pre_cam < 0] = 0
    pre_cam_cp = pre_cam.clone()
    pre_cam[0, :, cam < atten_thres_all] = 0
    
    mas, _ = torch.max(torch.max(pre_cam, 3)[0], 2)
    mas = mas.squeeze()
    mas_sort, inds = torch.sort(-mas)
    
    mas_sort = -mas_sort
    mas_all = torch.sum(mas_sort)
    mas_sort /= (mas_all + 1e-10)
    print(mas_sort)
    ########new added##########
#     avg = torch.sum(torch.sum(pre_cam, 3), 2)
#     avg = avg.squeeze()
#     avg_sort, inds = torch.sort(-avg)
#     avg_sort = -avg_sort
#     avg_all = torch.sum(avg_sort)
#     avg_sort /= (avg_all + 1e-10)
#     print(avg_sort)    
    
    for j in range(top_n):
        ind = inds[j]
        res = mas_sort[j]
        ###save kernel-based attention to guide where to decompose###
        #conv5_cam_loc = pre_cam_cp[0, ind].clone()
        conv5_cam_loc = pre_cam[0, ind].clone() 
        conv5_cam_loc /= (conv5_cam_loc.max() + 1e-8)
        
        conv5_cam_loc_np = conv5_cam_loc.detach().cpu().numpy()
        im_atten_name = img_dir + f'{label}_{ind}'
        np.save(im_atten_name, conv5_cam_loc_np)

        conv5_cam_loc_np = np.uint8(conv5_cam_loc_np * 255)  
        conv5_cam_loc_np = cv2.resize(conv5_cam_loc_np, (width, height), interpolation=cv2.INTER_CUBIC)

        im_gray_name1 = img_dir + f'{label}_{ind}_graycam.png'
        im_color_name1 = img_dir + f'{label}_{ind}_cam.png'
        # cv2.imwrite(im_gray_name1, conv5_cam_loc_np)

        cmap = plt.cm.jet_r(conv5_cam_loc_np)[..., :3] * 255.0
        conv5_cam_loc1 = (cmap.astype(np.float) + cv_im.astype(np.float)) / 2
        cv2.imwrite(im_color_name1, conv5_cam_loc1)
        ##############################################################

        unit_label = torch.zeros_like(target)
        ma = mas[ind]
        unit_label[:, ind][pre_cam[:, ind] == ma] = 1
        print(unit_label.nonzero().shape)
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
        crop_grad = max_crop(grad, 280)
        ### Scale between 0-255 to visualize
        ind = ind.cpu().numpy()
        out_name = img_dir + prefix + str(ind) + '.png'
        origin_image = Image.fromarray(np.asarray((grad * 255).astype(np.uint8)))
        converter = ImageEnhance.Color(origin_image)
        converted_image = converter.enhance(5)
        converted_image.save(img_dir + prefix + str(ind) + f'_e.png')

        plt.imsave(out_name, crop_grad)
        result.append(int(ind))
        result_mas.append(float(res))
    pickle_dump(position, img_dir + f'{label}_position.data')
    return result, result_mas

@timeit
def get_kernel_images_4(filename, label, index):
    top_n = configs.top_n
    atten_thres_s5 = configs.atten_thres_s5
    img_dir = configs.upload_dir + filename + '/'
    position = pickle_load(img_dir + f'{label}_position.data')
    channel_position = position[index]
    if sum([i[0] for i in channel_position[9]]) != -9:
        print('found')
        return ([i[0] for i in channel_position[9]], [i[3] for i in channel_position[9]])
    result = []
    result_mas = []
    conv5_channel = position[9][index][0]
    prefix = f'{label}_4_{conv5_channel}_'

    img_name = img_dir + filename
    img = Image.open(img_name)
    height = img.height
    width = img.width
    cv_im = cv2.imread(img_name)
    #cv_im_gray = cv2.cvtColor(cv_im, cv2.COLOR_BGR2GRAY)

    img = transforms(img).unsqueeze(0)
    img = img.cuda()

    input_var = torch.autograd.Variable(img, requires_grad=True)

    gb_feature_maps, gb_logits = gb_model.forward_cam(input_var, target_layers1)
    gb_target_5 = gb_feature_maps[8]
    gb_target_4 = gb_feature_maps[6]

    unit_label = torch.zeros_like(gb_target_5)
    unit_label[0][tuple(position[9][index][:3])] = 1
    gb_model.gradients = []
    gb_model.zero_grad()
    input_var.grad = None
    gb_target_5.backward(gradient=unit_label, retain_graph=True)
    conv4_grad = gb_model.gradients[1]
    guided_conv4_gradients_upsample = F.interpolate(conv4_grad, \
                        size=(gb_target_4.shape[-2], gb_target_4.shape[-1]), mode='bilinear', align_corners=True)
    conv4_cam = guided_conv4_gradients_upsample * gb_target_4
    conv4_cam[conv4_cam < 0] = 0
    conv4_cam_cp = conv4_cam.clone()

    ###load kernel-based attention to guide where to decompose###
    im_atten_name = img_dir + f'{label}_{conv5_channel}.npy'
    conv5_kernel_atten = torch.tensor(np.load(im_atten_name))
    conv5_kernel_atten_upsample = F.interpolate(conv5_kernel_atten[None, None, :, :], \
                        size=(gb_target_4.shape[-2], gb_target_4.shape[-1]), mode='bilinear', align_corners=True) 
    #conv4_cam[:, :, conv5_kernel_atten_upsample.squeeze() < atten_thres_s5] = 0
    ##############################################################
    mas, _ = torch.max(torch.max(conv4_cam, 3)[0], 2)
    mas = mas.squeeze()
    mas_sort, inds = torch.sort(-mas)
    
    mas_sort = -mas_sort
    mas_all = torch.sum(mas_sort)
    mas_sort /= (mas_all + 1e-10)
    print(mas_sort)
    #assert mas_sort[0] == mas[inds[0]]
    
    for j in range(top_n):
        ind = inds[j]
        res = mas_sort[j]
        ######save kernel-based attention for conv3_3###########
        #conv4_cam_loc = conv4_cam_cp[0, ind] 
        conv4_cam_loc = conv4_cam[0, ind].clone() 
        conv4_cam_loc /= (conv4_cam_loc.max() + 1e-8)
        conv4_cam_loc_np = conv4_cam_loc.detach().cpu().numpy()
        save_atten_name = img_dir + f'{label}_{conv5_channel}_{ind}'
        np.save(save_atten_name, conv4_cam_loc_np)

        conv4_cam_loc_np = np.uint8(conv4_cam_loc_np * 255)  
        conv4_cam_loc_np = cv2.resize(conv4_cam_loc_np, (width, height), interpolation=cv2.INTER_CUBIC)

        im_gray_name1 = img_dir + f'{label}_{conv5_channel}_{ind}_graycam.png'
        im_color_name1 = img_dir + f'{label}_{conv5_channel}_{ind}_cam.png'
        # cv2.imwrite(im_gray_name1, conv4_cam_loc_np)

        cmap = plt.cm.jet_r(conv4_cam_loc_np)[..., :3] * 255.0
        conv4_cam_loc1 = (cmap.astype(np.float) + cv_im.astype(np.float)) / 2
        cv2.imwrite(im_color_name1, conv4_cam_loc1)
        #############################################################
        
        unit_label = torch.zeros_like(conv4_cam)
        ma = mas[ind]
        unit_label[:, ind][conv4_cam[:, ind] == ma] = 1
        assert unit_label.nonzero().shape[0] == 1
        indices = unit_label[0, ind].nonzero().tolist()
        channel_position[9][j][:] = int(ind), *indices[0], float(res)
        
        gb_model.gradients = []
        gb_model.zero_grad()
        input_var.grad = None
        gb_target_4.backward(gradient=unit_label, retain_graph=True)
        grad = input_var.grad.detach().cpu().numpy()[0]
        grad = grad.transpose(1, 2, 0)
        grad = cv2.resize(grad, (width, height), interpolation=cv2.INTER_CUBIC)
        grad -= grad.min()
        grad /= (grad.max() + 1e-20)
        crop_grad = max_crop(grad, 160)
        
        ind = ind.cpu().numpy()
        out_name = img_dir + prefix + str(ind) + '.png'
        origin_image = Image.fromarray(np.asarray((crop_grad * 255).astype(np.uint8)))
        converter = ImageEnhance.Color(origin_image)
        converted_image = converter.enhance(5)
        converted_image.save(img_dir + prefix + str(ind) + f'_e.png')
        plt.imsave(out_name, crop_grad)
        result.append(int(ind))
        result_mas.append(float(res))
    pickle_dump(position, img_dir + f'{label}_position.data')
    return result, result_mas


@timeit
def get_kernel_images_3(filename, label, index_5, index_4):
    top_n = configs.top_n
    atten_thres_s4 = configs.atten_thres_s4
    img_dir = configs.upload_dir + filename + '/'
    position = pickle_load(img_dir + f'{label}_position.data')
    channel_position = position[index_5][index_4]
    if sum([i[0] for i in channel_position[9]]) != -9:
        print('found')
        return ([i[0] for i in channel_position[9]], [i[3] for i in channel_position[9]])
    result = []
    result_mas = []
    conv5_channel = position[9][index_5][0]
    conv4_channel = position[index_5][9][index_4][0]
    prefix = f'{label}_3_{conv5_channel}_{conv4_channel}_'

    img_name = img_dir + filename
    img = Image.open(img_name)
    height = img.height
    width = img.width
    img = transforms(img).unsqueeze(0)
    img = img.cuda()
    cv_im = cv2.imread(img_name)

    input_var = torch.autograd.Variable(img, requires_grad=True)

    gb_feature_maps, gb_logits = gb_model.forward_cam(input_var, target_layers1)
    gb_target_4 = gb_feature_maps[6]
    gb_target_3 = gb_feature_maps[4]
    
    unit_label = torch.zeros_like(gb_target_4)
    unit_label[0][tuple(position[index_5][9][index_4][:3])] = 1
    
    gb_model.gradients = []
    gb_model.zero_grad()
    input_var.grad = None
    gb_target_4.backward(gradient=unit_label, retain_graph=True)
    conv3_grad = gb_model.gradients[1]
    guided_conv3_gradients_upsample = F.interpolate(conv3_grad, \
                        size=(gb_target_3.shape[-2], gb_target_3.shape[-1]), mode='bilinear', align_corners=True)
    conv3_cam = guided_conv3_gradients_upsample * gb_target_3
    conv3_cam[conv3_cam < 0] = 0
    conv3_cam_cp = conv3_cam.clone()

    ###load kernel-based attention to guide where to decompose###
    im_atten_name = img_dir + f'{label}_{conv5_channel}_{conv4_channel}.npy'
    conv4_kernel_atten = torch.tensor(np.load(im_atten_name))
    conv4_kernel_atten_upsample = F.interpolate(conv4_kernel_atten[None, None, :, :], \
                        size=(gb_target_3.shape[-2], gb_target_3.shape[-1]), mode='bilinear', align_corners=True) 
    #conv3_cam[:, :, conv4_kernel_atten_upsample.squeeze() < atten_thres_s4] = 0
    ##############################################################

    mas, _ = torch.max(torch.max(conv3_cam, 3)[0], 2)
    mas = mas.squeeze()
    mas_sort, inds = torch.sort(-mas)

    mas_sort = -mas_sort
    mas_all = torch.sum(mas_sort)
    mas_sort /= (mas_all + 1e-10)
    print(mas_sort)
    #assert mas_sort[0] == mas[inds[0]]
    
    for j in range(top_n):
        ind = inds[j]
        res = mas_sort[j]
        ######save kernel-based attention for conv3_3###########
        #conv3_cam_loc = conv3_cam_cp[0, ind] 
        conv3_cam_loc = conv3_cam[0, ind].clone() 
        conv3_cam_loc /= (conv3_cam_loc.max() + 1e-8)
        conv3_cam_loc_np = conv3_cam_loc.detach().cpu().numpy()
        save_atten_name = img_dir + f'{label}_{conv5_channel}_{conv4_channel}_{ind}'
        np.save(save_atten_name, conv3_cam_loc_np)

        conv3_cam_loc_np = np.uint8(conv3_cam_loc_np * 255)  
        conv3_cam_loc_np = cv2.resize(conv3_cam_loc_np, (width, height), interpolation=cv2.INTER_CUBIC)

        im_gray_name1 = img_dir + f'{label}_{conv5_channel}_{conv4_channel}_{ind}_graycam.png'
        im_color_name1 = img_dir + f'{label}_{conv5_channel}_{conv4_channel}_{ind}_cam.png'
        # cv2.imwrite(im_gray_name1, conv4_cam_loc_np)

        cmap = plt.cm.jet_r(conv3_cam_loc_np)[..., :3] * 255.0
        conv3_cam_loc1 = (cmap.astype(np.float) + cv_im.astype(np.float)) / 2
        cv2.imwrite(im_color_name1, conv3_cam_loc1)
        #############################################################
        
        unit_label = torch.zeros_like(conv3_cam)
        ma = mas[ind]
        unit_label[:, ind][conv3_cam[:, ind] == ma] = 1
        assert unit_label.nonzero().shape[0] == 1
        indices = unit_label[0, ind].nonzero().tolist()
        channel_position[9][j][:] = int(ind), *indices[0], float(res)
        
        gb_model.gradients = []
        gb_model.zero_grad()
        input_var.grad = None
        gb_target_3.backward(gradient=unit_label, retain_graph=True)
        
        grad = input_var.grad.detach().cpu().numpy()[0]
        grad = grad.transpose(1, 2, 0)
        grad = cv2.resize(grad, (width, height), interpolation=cv2.INTER_CUBIC)
        grad -= grad.min()
        grad /= (grad.max() + 1e-20)
        crop_grad = max_crop(grad, 80)
        ind = ind.cpu().numpy()
        out_name = img_dir + prefix + str(ind) + '.png'
        origin_image = Image.fromarray(np.asarray((crop_grad * 255).astype(np.uint8)))
        converter = ImageEnhance.Color(origin_image)
        converted_image = converter.enhance(5)
        converted_image.save(img_dir + prefix + str(ind) + f'_e.png')
        plt.imsave(out_name, crop_grad)
        result.append(int(ind))
        result_mas.append(float(res))
    pickle_dump(position, img_dir + f'{label}_position.data')
    return result, result_mas


@timeit
def get_kernel_images_2(filename, label, index_5, index_4, index_3):
    top_n = configs.top_n
    atten_thres_s3 = configs.atten_thres_s3
    img_dir = configs.upload_dir + filename + '/'
    position = pickle_load(img_dir + f'{label}_position.data')
    channel_position = position[index_5][index_4][index_3]
    if sum([i[0] for i in channel_position[9]]) != -9:
        print('found')
        return ([i[0] for i in channel_position[9]], [i[3] for i in channel_position[9]])
    result = []
    result_mas = []
    conv5_channel = position[9][index_5][0]
    conv4_channel = position[index_5][9][index_4][0]
    conv3_channel = position[index_5][index_4][9][index_3][0]
    prefix = f'{label}_2_{conv5_channel}_{conv4_channel}_{conv3_channel}_'

    img_name = img_dir + filename
    img = Image.open(img_name)
    height = img.height
    width = img.width
    img = transforms(img).unsqueeze(0)
    img = img.cuda()
    cv_im = cv2.imread(img_name)

    input_var = torch.autograd.Variable(img, requires_grad=True)

    gb_feature_maps, gb_logits = gb_model.forward_cam(input_var, target_layers1)
    gb_target_3 = gb_feature_maps[4]
    gb_target_2 = gb_feature_maps[2]

    unit_label = torch.zeros_like(gb_target_3)
    unit_label[0][tuple(position[index_5][index_4][9][index_3][:3])] = 1
    
    gb_model.gradients = []
    gb_model.zero_grad()
    input_var.grad = None
    gb_target_3.backward(gradient=unit_label, retain_graph=True)
    conv2_grad = gb_model.gradients[1]
    guided_conv2_gradients_upsample = F.interpolate(conv2_grad, \
                        size=(gb_target_2.shape[-2], gb_target_2.shape[-1]), mode='bilinear', align_corners=True)
    conv2_cam = guided_conv2_gradients_upsample * gb_target_2
    conv2_cam[conv2_cam < 0] = 0
    conv2_cam_cp = conv2_cam.clone()

    ###load kernel-based attention to guide where to decompose###
    im_atten_name = img_dir + f'{label}_{conv5_channel}_{conv4_channel}_{conv3_channel}.npy'
    conv3_kernel_atten = torch.tensor(np.load(im_atten_name))
    conv3_kernel_atten_upsample = F.interpolate(conv3_kernel_atten[None, None, :, :], \
                        size=(gb_target_2.shape[-2], gb_target_2.shape[-1]), mode='bilinear', align_corners=True) 
    #conv2_cam[:, :, conv3_kernel_atten_upsample.squeeze() < atten_thres_s3] = 0
    ##############################################################    
    
    mas, _ = torch.max(torch.max(conv2_cam, 3)[0], 2)
    mas = mas.squeeze()
    mas_sort, inds = torch.sort(-mas)
    
    mas_sort = -mas_sort
    mas_all = torch.sum(mas_sort)
    mas_sort /= (mas_all + 1e-10)
    print(mas_sort)
    #assert mas_sort[0] == mas[inds[0]]
    
    for j in range(top_n):
        ind = inds[j]
        res = mas_sort[j]
        ######save kernel-based attention for conv3_3###########
        #conv2_cam_loc = conv2_cam_cp[0, ind] 
        conv2_cam_loc = conv2_cam[0, ind].clone()  
        conv2_cam_loc /= (conv2_cam_loc.max() + 1e-8)
        conv2_cam_loc_np = conv2_cam_loc.detach().cpu().numpy()
        save_atten_name = img_dir + f'{label}_{conv5_channel}_{conv4_channel}_{conv3_channel}_{ind}'
        np.save(save_atten_name, conv2_cam_loc_np)

        conv2_cam_loc_np = np.uint8(conv2_cam_loc_np * 255)  
        conv2_cam_loc_np = cv2.resize(conv2_cam_loc_np, (width, height), interpolation=cv2.INTER_CUBIC)

        im_gray_name1 = img_dir + f'{label}_{conv5_channel}_{conv4_channel}_{conv3_channel}_{ind}_graycam.png'
        im_color_name1 = img_dir + f'{label}_{conv5_channel}_{conv4_channel}_{conv3_channel}_{ind}_cam.png'
        # cv2.imwrite(im_gray_name1, conv4_cam_loc_np)

        cmap = plt.cm.jet_r(conv2_cam_loc_np)[..., :3] * 255.0
        conv2_cam_loc1 = (cmap.astype(np.float) + cv_im.astype(np.float)) / 2
        cv2.imwrite(im_color_name1, conv2_cam_loc1)
        #############################################################

        unit_label = torch.zeros_like(conv2_cam)
        ma = mas[ind]
        unit_label[:, ind][conv2_cam[:, ind] == ma] = 1
        assert unit_label.nonzero().shape[0] == 1
        indices = unit_label[0, ind].nonzero().tolist()
        channel_position[9][j][:] = int(ind), *indices[0], float(res)

        gb_model.gradients = []
        gb_model.zero_grad()
        input_var.grad = None
        gb_target_2.backward(gradient=unit_label, retain_graph=True)
        
        grad = input_var.grad.detach().cpu().numpy()[0]
        grad = grad.transpose(1, 2, 0)
        grad = cv2.resize(grad, (width, height), interpolation=cv2.INTER_CUBIC)
        grad -= grad.min()
        grad /= (grad.max() + 1e-20)
        crop_grad = max_crop(grad, 40)
        ind = ind.cpu().numpy()
        out_name = img_dir + prefix + str(ind) + '.png'
        origin_image = Image.fromarray(np.asarray((crop_grad * 255).astype(np.uint8)))
        converter = ImageEnhance.Color(origin_image)
        converted_image = converter.enhance(5)
        converted_image.save(img_dir + prefix + str(ind) + f'_e.png')
        plt.imsave(out_name, crop_grad)
        result.append(int(ind))
        result_mas.append(float(res))
    pickle_dump(position, img_dir + f'{label}_position.data')
    return result, result_mas


@timeit
def get_kernel_images_1(filename, label, index_5, index_4, index_3, index_2):
    top_n = configs.top_n
    atten_thres_s2 = configs.atten_thres_s2
    img_dir = configs.upload_dir + filename + '/'
    position = pickle_load(img_dir + f'{label}_position.data')
    channel_position = position[index_5][index_4][index_3][index_2]
    print(channel_position)
    if sum([i[0] for i in channel_position]) != -9:
        print('found')
        return ([i[0] for i in channel_position], [i[1] for i in channel_position]) 
    result = []
    result_mas = []
    conv5_channel = position[9][index_5][0]
    conv4_channel = position[index_5][9][index_4][0]
    conv3_channel = position[index_5][index_4][9][index_3][0]
    conv2_channel = position[index_5][index_4][index_3][9][index_2][0]
    prefix = f'{label}_1_{conv5_channel}_{conv4_channel}_{conv3_channel}_{conv2_channel}_'

    img_name = img_dir + filename
    img = Image.open(img_name)
    height = img.height
    width = img.width
    img = transforms(img).unsqueeze(0)
    img = img.cuda()

    input_var = torch.autograd.Variable(img, requires_grad=True)

    gb_feature_maps, gb_logits = gb_model.forward_cam(input_var, target_layers1)
    gb_target_2 = gb_feature_maps[2]
    gb_target_1 = gb_feature_maps[0]

    unit_label = torch.zeros_like(gb_target_2)
    unit_label[0][tuple(position[index_5][index_4][index_3][9][index_2][:3])] = 1
    
    gb_model.gradients = []
    gb_model.zero_grad()
    input_var.grad = None
    gb_target_2.backward(gradient=unit_label, retain_graph=True)
    conv1_grad = gb_model.gradients[1]
    guided_conv1_gradients_upsample = F.interpolate(conv1_grad, \
                        size=(gb_target_1.shape[-2], gb_target_1.shape[-1]), mode='bilinear', align_corners=True)
    conv1_cam = guided_conv1_gradients_upsample * gb_target_1
    conv1_cam[conv1_cam < 0] = 0
    conv1_cam_cp = conv1_cam.clone()
    ###load kernel-based attention to guide where to decompose###
    im_atten_name = img_dir + f'{label}_{conv5_channel}_{conv4_channel}_{conv3_channel}_{conv2_channel}.npy'
    conv2_kernel_atten = torch.tensor(np.load(im_atten_name))
    conv2_kernel_atten_upsample = F.interpolate(conv2_kernel_atten[None, None, :, :], \
                        size=(gb_target_1.shape[-2], gb_target_1.shape[-1]), mode='bilinear', align_corners=True) 
    conv1_cam[:, :, conv2_kernel_atten_upsample.squeeze() < atten_thres_s2] = 0
    ##############################################################    

    mas, _ = torch.max(torch.max(conv1_cam, 3)[0], 2)
    mas = mas.squeeze()
    mas_sort, inds = torch.sort(-mas)
    
    mas_sort = -mas_sort
    mas_all = torch.sum(mas_sort)
    mas_sort /= (mas_all + 1e-10)
    print(mas_sort)
    #assert mas_sort[0] == mas[inds[0]]
    
    for j in range(top_n):
        ind = inds[j]
        res = mas_sort[j]
        unit_label = torch.zeros_like(conv1_cam)
        ma = mas[ind]
        unit_label[:, ind][conv1_cam[:, ind] == ma] = 1
        assert unit_label.nonzero().shape[0] == 1
        indices = unit_label[0, ind].nonzero().tolist()
        channel_position[j] = int(ind), float(res)
        
        gb_model.gradients = []
        gb_model.zero_grad()
        input_var.grad = None
        gb_target_1.backward(gradient=unit_label, retain_graph=True)
        grad = input_var.grad.detach().cpu().numpy()[0]
        grad = grad.transpose(1, 2, 0)
        grad = cv2.resize(grad, (width, height), interpolation=cv2.INTER_CUBIC)
        grad -= grad.min()
        grad /= (grad.max() + 1e-20)
        crop_grad = max_crop(grad, 20)
        ind = ind.cpu().numpy()
        out_name = img_dir + prefix + str(ind) + '.png'
        origin_image = Image.fromarray(np.asarray((crop_grad * 255).astype(np.uint8)))
        converter = ImageEnhance.Color(origin_image)
        converted_image = converter.enhance(5)
        converted_image.save(img_dir + prefix + str(ind) + f'_e.png')
        plt.imsave(out_name, crop_grad)
        result.append(int(ind))
        result_mas.append(float(res))
    pickle_dump(position, img_dir + f'{label}_position.data')
    return result, result_mas
