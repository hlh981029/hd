import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
from matplotlib import cm
import numpy as np
from PIL import Image

from shutil import copy
import configs

class_label = configs.class_label
line_color = cm.tab10.colors[0]
figsize = (10,15)
src_box_height = 0.1
src_box_bottom = 0.87
vis_box_height = 0.07
vis_box_width = 0.105
left_box_width = 0.10
layer_box_height = 0.16
layer_box_bottom = [src_box_bottom - 0.14 - i * layer_box_height for i in range(5)]
filter_box_width = 0.225
filter_box_padding = 0.0075
arrow_path_top_1 = 0.125
arrow_path_bottom_1 = 0.105
arrow_path_top_2 = 0.105
arrow_path_bottom_2 = 0.07
filter_text_box_height = 0.025
center_x = (1 + left_box_width) / 2
arrow_args = {
    'coordsA': 'data',
    'coordsB': 'data',
    'arrowstyle': '-|>',
    'shrinkA': 0,
    'shrinkB': 5,
    'mutation_scale': 20
}
line_args = {
    'color': line_color,
    'lw': 3
}


def gen_demo(filename, result, label, select, inds, contris):
    fig = plt.figure(figsize=(10,15), dpi=200)
    ax = fig.add_axes((0,0,1,1))
    ax.set_axis_off()
    # for i in range(11):
    #     ax.axvline(0.1*i, ls='-', lw=0.5, color='#777777')
    #     ax.axhline(0.1*i, ls='-', lw=0.5, color='#777777')

    src_img = Image.open(configs.upload_dir + filename + '/' + filename)
    width, height = src_img.size
    ax_width = src_box_height / height * width * 1.5
    ax_src = fig.add_axes((center_x-ax_width/2,src_box_bottom,ax_width,src_box_height))
    ax_src.set_axis_off()
    ax_src.imshow(src_img)


    ax.text((left_box_width + center_x-ax_width/2)/2, 0.9, f'Class: {class_label[label]}\nConfidence: {result[label]:.2%}', size='large', va="center", ha="center", multialignment="left", linespacing=1.5)
    ax.text(left_box_width / 2, 0.9, 'Input\nImage', size='x-large', va="center", ha="center", multialignment="center")
    for i in range(5):
        ax.text(left_box_width / 2, layer_box_bottom[i] + vis_box_height / 2, f'Stage {5-i}', size='x-large', va="center", ha="center", multialignment="center")
        for j in range(4):
            ax.text(left_box_width + filter_box_padding + filter_box_width * j + vis_box_width, layer_box_bottom[i] - filter_text_box_height, f'Filter {inds[i][j]}: {contris[i][j]:.2%}', 
                size='large', va="bottom", ha="center", multialignment="center")


    for i in range(4):
        ax.add_artist(patches.Rectangle(
                (left_box_width + filter_box_padding + filter_box_width * select[i], layer_box_bottom[i]), 
                vis_box_width*2, vis_box_height, color=line_color, lw=8))

    for i in range(5):
        for j in range(4):
            img_prefix = ''
            for k in range(i):
                img_prefix += f'_{inds[k][select[k]]}'
            ax_temp = fig.add_axes((left_box_width + filter_box_padding + filter_box_width * j, layer_box_bottom[i], vis_box_width, vis_box_height))
            ax_temp.set_axis_off()
            temp_img = Image.open(f'{configs.upload_dir + filename}/{label}_{5-i}{img_prefix}_{inds[i][j]}_cam.jpg')
            ax_temp.imshow(temp_img, aspect='auto')
            ax_temp = fig.add_axes((left_box_width + filter_box_padding + filter_box_width * j + vis_box_width, layer_box_bottom[i], vis_box_width, vis_box_height))
            ax_temp.set_axis_off()
            temp_img = Image.open(f'{configs.upload_dir + filename}/{label}_{5-i}{img_prefix}_{inds[i][j]}.jpg')
            ax_temp.imshow(temp_img, aspect='auto')

    for i in range(5):
        if i == 0:
            ax.add_artist(lines.Line2D(
                [center_x, center_x], 
                [layer_box_bottom[i] + arrow_path_top_1, layer_box_bottom[i] + arrow_path_bottom_1],
                **line_args))
        else:
            ax.add_artist(lines.Line2D(
                [left_box_width + filter_box_padding + filter_box_width * select[i-1] + vis_box_width, left_box_width + filter_box_padding + filter_box_width * select[i-1] + vis_box_width], 
                [layer_box_bottom[i] + arrow_path_top_1, layer_box_bottom[i] + arrow_path_bottom_1],
                **line_args))
        ax.add_artist(lines.Line2D(
            [left_box_width + filter_box_padding + vis_box_width, left_box_width + filter_box_padding + filter_box_width * 3 + vis_box_width], 
            [layer_box_bottom[i] + arrow_path_bottom_1, layer_box_bottom[i] + arrow_path_bottom_1],
            **line_args))
        for j in range(4):
            ax.add_artist(patches.ConnectionPatch(
                (left_box_width + filter_box_padding + filter_box_width * j + vis_box_width, layer_box_bottom[i] + arrow_path_top_2), 
                (left_box_width + filter_box_padding + filter_box_width * j + vis_box_width, layer_box_bottom[i] + arrow_path_bottom_2),
                **arrow_args, **line_args))


    fig.savefig(f'{configs.upload_dir + filename}/{filename}.pdf')
    return 0