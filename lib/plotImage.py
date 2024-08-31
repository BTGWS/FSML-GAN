# coding:utf-8
# @TIME         : 2022/4/13 10:32 
# @Author       : BTG
# @Project      : NBGAN
# @File Name    : plotImage.py
#
#                          _ooOoo_
#                         o8888888o
#                         88" . "88                              
#                         (| ^_^ |)
#                         O\  =  /O
#                      ____/`---'\____
#                    .'  \\|     |//  `.
#                   /  \\|||  :  |||//  \
#                  /  _||||| -:- |||||-  \
#                  |   | \\\  -  /// |   |
#                  | \_|  ''\---/''  |   |
#                  \  .-\__  `-`  ___/-. /
#                ___`. .'  /--.--\  `. . ___
#              ."" '<  `.___\_<|>_/___.'  >'"".
#            | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#            \  \ `-.   \_ __\ /__ _/   .-` /  /
#      ========`-.____`-.___\_____/___.-`____.-'========
#                           `=---='
#      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                 佛祖保佑             永无BUG
import os

import numpy as np
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from numpy import sqrt, power
import torch

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def plot_images(images, title_name: str, batch_size, info_dict=None, showImages=True):
    plt.axis("off")
    plt.title(title_name)
    plt.imshow(np.transpose(vutils.make_grid(images.cpu()[:len(images)]), (1, 2, 0)))
    if info_dict is not None:
        idx = info_dict['taskIdx']
        save_folderPath = info_dict['save_folderPath']
        dataset = info_dict['Dataset']
        path = save_folderPath + '/' + dataset + '/{}'
        temp = title_name.split('_')[0]
        if os.path.exists(path.format(temp)) is False:
            os.makedirs(path.format(temp))
        path += '/{}.png'

        plt.savefig(path.format(temp, idx))
    if showImages is True:
        plt.show()
    plt.clf()


def adjust_dynamic_range(data, drange_in=(-1, 1), drange_out=(0, 1)):
    """
    adjust the dynamic colour range of the given input data
    :param data: input image data
    :param drange_in: original range of input
    :param drange_out: required range of output
    :return: img => colour range adjusted images
    """
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return torch.clamp(data, min=0, max=1)


def saveImages(img, dataset, className, num):
    # path = './generateImages/' + dataset + '/{}'
    path = './Images/' + dataset
    if os.path.exists(path) is False:
        os.makedirs(path)
    path += '/{}_{}.png'
    # plt.savefig(path.format(temp, idx))
    vutils.save_image(img, path.format(className, num), nrow=int(sqrt(img.shape[0])), normalize=True, scale_each=True,
                      padding=0)


def saveImages_mutil(img, dataset, className, num):
    # path = './generateImages/' + dataset + '/{}'

    # path = './Images/{}/{}'.format(dataset, className)
    path = '/home/hadoop/Datasets/{}/{}'.format(dataset, className)
    if os.path.exists(path) is False:
        print(path + ' is not found.')
        os.makedirs(path)
    path += '/fake_{}_{}.png'
    # plt.savefig(path.format(temp, idx))
    vutils.save_image(img, path.format(className, num), nrow=int(sqrt(img.shape[0])), normalize=True, scale_each=True,
                      padding=0)
