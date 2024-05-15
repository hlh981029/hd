from commons import colormap, timeit, get_model, get_transform, init_position, max_crop, Timer, pickle_dump, pickle_load
from PIL import Image, ImageEnhance
import configs
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

test_filename = 'demo1.jpg'
model = get_model(configs)
gb_model = get_model(configs, bp_type='GBP')

transforms = get_transform(configs)
target_layers = [3, 8, 15, 22, 29, 30]
length = 14

@timeit
def get_labels(filename):
    img_dir = configs.upload_dir + filename + '/'
    img_name = img_dir + filename
    img = Image.open(img_name)
    img = transforms(img).unsqueeze(0)
    img = img.cuda()
    logits = torch.nn.functional.softmax(model.forward(img), dim=1)[0]
    return logits.tolist()


@timeit
def get_kernel_images_5(filename, label):
    top_n = configs.top_n
    img_dir = configs.upload_dir + filename + '/'
    if not os.path.exists(img_dir + f'{label}_position.data'):
        position = init_position()
    else:
        position = pickle_load(img_dir + f'{label}_position.data')
    if sum([i[0] for i in position[9]]) != -9:
        print('found')
        return [i[0] for i in position[9]]
    prefix = f'{label}_5_'
    result = []
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
    gb_feature_maps, gb_logits = gb_model.forward_cam(input_var1, target_layers)
    gb_target = gb_feature_maps[4]

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
                                                                size=(length, length), mode='bilinear',
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
    cv_im_gray = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    cam_np = cv_im_gray * 0.2 + cam_np * 0.8 * 255
    out_color_name = img_dir + f'{label}_cam.png'
    # cmap = plt.cm.jet_r(cam_np)[..., :3] * 255.0
    # cam1 = (cmap.astype(np.float) + cv_im.astype(np.float)) / 2
    # cv2.imwrite(out_color_name, cam1)
    plt.imsave(out_color_name, cam_np, cmap=colormap(label))

    pre_cam[pre_cam < 0] = 0
    pre_cam[0, :, cam < 0.5] = 0
    mas, _ = torch.max(torch.max(pre_cam, 3)[0], 2)
    mas = mas.squeeze()
    _, inds = torch.sort(-mas)
    for j in range(top_n):
        ind = inds[j]
        unit_label = torch.zeros_like(target)
        ma = mas[ind]
        unit_label[:, ind][pre_cam[:, ind] == ma] = 1
        assert unit_label.nonzero().shape[0] == 1
        indices = unit_label[0, ind].nonzero().tolist()
        position[9][j][:] = int(ind), *indices[0]

        gb_model.gradients = []   
        gb_model.zero_grad()
        input_var1.grad = None
        gb_target.backward(gradient=unit_label, retain_graph=True)
        grad = input_var1.grad.detach().cpu().numpy()[0]
        grad = grad.transpose(1, 2, 0)
        grad = cv2.resize(grad, (width, height), interpolation=cv2.INTER_CUBIC)
        grad -= grad.min()
        grad /= (grad.max() + 1e-20)
        ### Scale between 0-255 to visualize
        ind = ind.cpu().numpy()
        out_name = img_dir + prefix + str(ind) + '.png'
        origin_image = Image.fromarray(np.asarray((grad * 255).astype(np.uint8)))
        converter = ImageEnhance.Color(origin_image)
        converted_image = converter.enhance(5)
        converted_image.save(img_dir + prefix + str(ind) + f'_e.png')

        plt.imsave(out_name, grad)
        result.append(int(ind))
    pickle_dump(position, img_dir + f'{label}_position.data')
    return result

@timeit
def get_kernel_images_4(filename, label, index):
    top_n = configs.top_n
    img_dir = configs.upload_dir + filename + '/'
    position = pickle_load(img_dir + f'{label}_position.data')
    channel_position = position[index]
    if sum([i[0] for i in channel_position[9]]) != -9:
        print('found')
        return [i[0] for i in channel_position[9]]
    result = []
    conv5_channel = position[9][index][0]
    prefix = f'{label}_4_{conv5_channel}_'

    img_name = img_dir + filename
    img = Image.open(img_name)
    height = img.height
    width = img.width
    img = transforms(img).unsqueeze(0)
    img = img.cuda()

    input_var = torch.autograd.Variable(img, requires_grad=True)

    gb_feature_maps, gb_logits = gb_model.forward_cam(input_var, target_layers)
    gb_target_5 = gb_feature_maps[4]
    gb_target_4 = gb_feature_maps[3]

    unit_label = torch.zeros_like(gb_target_5)
    unit_label[0][tuple(position[9][index])] = 1
    gb_model.gradients = []
    gb_model.zero_grad()
    input_var.grad = None
    gb_target_5.backward(gradient=unit_label, retain_graph=True)
    conv4_grad = gb_model.gradients[1]
    conv4_cam = conv4_grad * gb_target_4
    mas, _ = torch.max(torch.max(conv4_cam, 3)[0], 2)
    mas = mas.squeeze()
    _, inds = torch.sort(-mas)

    for j in range(top_n):
        ind = inds[j]
        unit_label = torch.zeros_like(conv4_cam)
        ma = mas[ind]
        unit_label[:, ind][conv4_cam[:, ind] == ma] = 1
        assert unit_label.nonzero().shape[0] == 1
        indices = unit_label[0, ind].nonzero().tolist()
        channel_position[9][j][:] = int(ind), *indices[0]
        
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
    pickle_dump(position, img_dir + f'{label}_position.data')
    return result


@timeit
def get_kernel_images_3(filename, label, index_5, index_4):
    top_n = configs.top_n
    img_dir = configs.upload_dir + filename + '/'
    position = pickle_load(img_dir + f'{label}_position.data')
    channel_position = position[index_5][index_4]
    if sum([i[0] for i in channel_position[9]]) != -9:
        print('found')
        return [i[0] for i in channel_position[9]]
    result = []
    conv5_channel = position[9][index_5][0]
    conv4_channel = position[index_5][9][index_4][0]
    prefix = f'{label}_3_{conv5_channel}_{conv4_channel}_'

    img_name = img_dir + filename
    img = Image.open(img_name)
    height = img.height
    width = img.width
    img = transforms(img).unsqueeze(0)
    img = img.cuda()

    input_var = torch.autograd.Variable(img, requires_grad=True)
   
    gb_feature_maps, gb_logits = gb_model.forward_cam(input_var, target_layers)
    gb_target_4 = gb_feature_maps[3]
    gb_target_3 = gb_feature_maps[2]
    
    unit_label = torch.zeros_like(gb_target_4)
    unit_label[0][tuple(position[index_5][9][index_4])] = 1
    
    gb_model.gradients = []
    gb_model.zero_grad()
    input_var.grad = None
    gb_target_4.backward(gradient=unit_label, retain_graph=True)
    conv3_grad = gb_model.gradients[1]
    conv3_cam = conv3_grad * gb_target_3
    mas, _ = torch.max(torch.max(conv3_cam, 3)[0], 2)
    mas = mas.squeeze()
    _, inds = torch.sort(-mas)
    
    for j in range(top_n):
        ind = inds[j]
        unit_label = torch.zeros_like(conv3_cam)
        ma = mas[ind]
        unit_label[:, ind][conv3_cam[:, ind] == ma] = 1
        assert unit_label.nonzero().shape[0] == 1
        indices = unit_label[0, ind].nonzero().tolist()
        channel_position[9][j][:] = int(ind), *indices[0]
        
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
    pickle_dump(position, img_dir + f'{label}_position.data')
    return result


@timeit
def get_kernel_images_2(filename, label, index_5, index_4, index_3):
    top_n = configs.top_n
    img_dir = configs.upload_dir + filename + '/'
    position = pickle_load(img_dir + f'{label}_position.data')
    channel_position = position[index_5][index_4][index_3]
    if sum([i[0] for i in channel_position[9]]) != -9:
        print('found')
        return [i[0] for i in channel_position[9]]
    result = []
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

    input_var = torch.autograd.Variable(img, requires_grad=True)

    gb_feature_maps, gb_logits = gb_model.forward_cam(input_var, target_layers)
    gb_target_3 = gb_feature_maps[2]
    gb_target_2 = gb_feature_maps[1]

    unit_label = torch.zeros_like(gb_target_3)
    unit_label[0][tuple(position[index_5][index_4][9][index_3])] = 1
    
    gb_model.gradients = []
    gb_model.zero_grad()
    input_var.grad = None
    gb_target_3.backward(gradient=unit_label, retain_graph=True)
    conv2_grad = gb_model.gradients[1]
    conv2_cam = conv2_grad * gb_target_2
    
    mas, _ = torch.max(torch.max(conv2_cam, 3)[0], 2)
    mas = mas.squeeze()
    _, inds = torch.sort(-mas)
    for j in range(top_n):
        ind = inds[j]
        unit_label = torch.zeros_like(conv2_cam)
        ma = mas[ind]
        unit_label[:, ind][conv2_cam[:, ind] == ma] = 1
        assert unit_label.nonzero().shape[0] == 1
        indices = unit_label[0, ind].nonzero().tolist()
        channel_position[9][j][:] = int(ind), *indices[0]

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
    pickle_dump(position, img_dir + f'{label}_position.data')
    return result


@timeit
def get_kernel_images_1(filename, label, index_5, index_4, index_3, index_2):
    top_n = configs.top_n
    img_dir = configs.upload_dir + filename + '/'
    position = pickle_load(img_dir + f'{label}_position.data')
    channel_position = position[index_5][index_4][index_3][index_2]
    print(channel_position)
    if sum(channel_position) != -9:
        print('found')
        return channel_position
    result = []
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

    gb_feature_maps, gb_logits = gb_model.forward_cam(input_var, target_layers)
    gb_target_2 = gb_feature_maps[1]
    gb_target_1 = gb_feature_maps[0]

    unit_label = torch.zeros_like(gb_target_2)
    unit_label[0][tuple(position[index_5][index_4][index_3][9][index_2])] = 1
    
    gb_model.gradients = []
    gb_model.zero_grad()
    input_var.grad = None
    gb_target_2.backward(gradient=unit_label, retain_graph=True)
    conv1_grad = gb_model.gradients[1]
    conv1_cam = conv1_grad * gb_target_1
    mas, _ = torch.max(torch.max(conv1_cam, 3)[0], 2)
    mas = mas.squeeze()
    _, inds = torch.sort(-mas)
    for j in range(top_n):
        ind = inds[j]
        unit_label = torch.zeros_like(conv1_cam)
        ma = mas[ind]
        unit_label[:, ind][conv1_cam[:, ind] == ma] = 1
        assert unit_label.nonzero().shape[0] == 1
        indices = unit_label[0, ind].nonzero().tolist()
        channel_position[j] = int(ind)
        
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
    pickle_dump(position, img_dir + f'{label}_position.data')
    return result