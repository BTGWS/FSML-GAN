# coding:utf-8
# @TIME         : 2022/5/26 10:23 
# @Author       : BTG
# @Project      : NBGAN
# @File Name    : trainGD.py
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
# 判别器对原始图像进行判别
import random
import time

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from GAN_net.Discriminator import DiscriminatorPlus
from GAN_net.Encoder import Encoder
from GAN_net.Generator import GeneratorPlus
from lib.getDatasets import ReadImageMetaEnv
from lib.loss_function import compute_gradient_penalty, hinge_loss_d, compute_FID, ContrastiveLoss, crop_image_by_part
from lib.loss_function import loss_fn as c_loss_fn
from lib.loss_function import loss_fn_test

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)


percept = lpips.LPIPS(net='vgg').to('cuda')


def train_D_org(netD, x_real, x_fake, enc_x, real_labels, fake_labels,
                isBCE=True, isHL=False, isGP=True, isCloss=True, device='cuda'):
    gradient_penalty = 0
    c_loss = 0
    # 判别器对原始图像进行判别
    real_out, real_feat = netD(x_real)

    # 将两个fake_images送入判别器判别
    fake1_out, fake1_feat_d = netD(x_fake[0].detach())
    fake2_out, fake2_feat_d = netD(x_fake[1].detach())

    if isCloss is True:
        c_loss1 = c_loss_fn(enc_x[1], fake1_feat_d[0], flag=False)
        c_loss2 = c_loss_fn(enc_x[2], fake2_feat_d[0], flag=False)
        c_loss3 = c_loss_fn(enc_x[0], real_feat[0], flag=False)
        c_loss = (c_loss1 + c_loss2 + c_loss3) / 3
        # c_loss = c_loss_fn(enc_x, fuck[0], flag=False)

    if isGP is True:
        # 对判别器梯度更新
        gradient_penalty1 = compute_gradient_penalty(netD, x_real, x_fake[0])
        gradient_penalty2 = compute_gradient_penalty(netD, x_real, x_fake[1])
        gradient_penalty = (gradient_penalty1 + gradient_penalty2) / 2

    if isBCE is True and isHL is False:
        BCE_stable = nn.BCEWithLogitsLoss().to(device)
        errD1 = (BCE_stable(real_out - torch.mean(fake1_out), real_labels) + BCE_stable(
            fake1_out - torch.mean(real_out), fake_labels)) / 2
        errD2 = (BCE_stable(real_out - torch.mean(fake2_out), real_labels) + BCE_stable(
            fake2_out - torch.mean(real_out), fake_labels)) / 2

    if isHL is True and isBCE is False:
        errD1 = hinge_loss_d(real_out, fake1_out)
        errD2 = hinge_loss_d(real_out, fake2_out)
    errD = errD1 + errD2 + gradient_penalty + c_loss
    errD.backward()
    return errD


def train_D_pro(netD, x_real, x_fake, enc_x, real_labels=None, fake_labels=None,
                isCloss=True, isBCE=True, isHL=False, isGP=True, isConLoss=False,
                device='cuda'):
    gradient_penalty = 0
    c_loss = 0
    loss_contra = 0
    BCE_stable = nn.BCEWithLogitsLoss().to(device)
    # D_criterionL1 = nn.L1Loss().to(device)
    # ContrastiveLoss_func = ContrastiveLoss(batch_size=x_real.shape[0]).to(device)

    part = random.randint(0, 3)
    pred_real, [real_feat, rec_big, rec_part] = netD(x_real, 'real', part=part)

    real_part = crop_image_by_part(x_real, part)
    errD = percept(rec_big, F.interpolate(x_real, rec_big.shape[2])).sum() + \
           percept(rec_part, F.interpolate(real_part, rec_part.shape[2])).sum()
    # dl1_loss1 = D_criterionL1(F.interpolate(x_real, rec_part.shape[2]), torch.sigmoid(rec_big))
    # dl1_loss2 = D_criterionL1(F.interpolate(real_part, rec_part.shape[2]), torch.sigmoid(rec_part))
    # errD += dl1_loss1 + dl1_loss2
    # errD = percept(rec_big, F.interpolate(x_real, rec_big.shape[2])).sum()

    # pred_real, [real_feat] = netD(x_real)
    # errD = (compute_FID(x_real, x_fake[0], Device=device) + compute_FID(x_real, x_fake[1], Device=device)) / 2
    # errD = percept(x_real, x_fake[0]).sum() + \
    #        percept(x_real, x_fake[1]).sum()

    # 将两个fake_images送入判别器判别
    pred_fake1, fake1_feat_d = netD(x_fake[0].detach(), 'fake')
    pred_fake2, fake2_feat_d = netD(x_fake[1].detach(), 'fake')

    rec_big_out, rec_big_feat = netD(rec_big)
    if isCloss is True:
        c_loss1 = c_loss_fn(enc_x[1], fake1_feat_d[0], flag=False)
        c_loss2 = c_loss_fn(enc_x[2], fake2_feat_d[0], flag=False)
        c_loss3 = c_loss_fn(enc_x[0], real_feat, flag=False)
        # c_loss = 0.2 * c_loss1 + 0.2 * c_loss2 + 0.6 * c_loss3
        c_loss = (c_loss1 + c_loss2 + c_loss3) / 3
        # c_loss = c_loss_fn(enc_x, fuck[0], flag=False)

    if isConLoss is True:
        # F.interpolate(x_fake[0].detach(), rec_big.shape[2])
        loss_contra = loss_fn_test(real=rec_big,
                                   fake=[x_fake[0].detach(),
                                         x_fake[1].detach()],
                                   real_feat=rec_big_feat[0],
                                   fake_feat=[fake1_feat_d[0], fake2_feat_d[0]])

        # tf_aug = transforms.Compose([
        #     transforms.RandomCrop(x_real.shape[2] / 2),
        #     # transforms.RandomInvert(0.5),  # 颜色倒置器(无效)
        #     # transforms.RandomRotation((0, 360)),  # 旋转图片，能有利于在原训练集基础上能好的拟合模型
        #     transforms.RandomHorizontalFlip(0.5),  # 随机水平翻转
        #     # transforms.RandomVerticalFlip(0.5),  # 随机垂直翻转
        #     # transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5)]),
        #     #                        p=1),  # 随机调整HSV
        # ])

    if isBCE is True and isHL is False:
        errD1 = (BCE_stable(pred_real - torch.mean(pred_fake1), real_labels) + BCE_stable(
            pred_fake1 - torch.mean(pred_real), fake_labels)) / 2
        errD2 = (BCE_stable(pred_real - torch.mean(pred_fake2), real_labels) + BCE_stable(
            pred_fake2 - torch.mean(pred_real), fake_labels)) / 2

    if isHL is True and isBCE is False:
        errD1 = hinge_loss_d(pred_real, pred_fake1)
        errD2 = hinge_loss_d(pred_real, pred_fake2)

    if isGP is True:
        # 对判别器梯度更新
        # gradient_penalty1 = compute_gradient_penalty(netD, x_real, x_fake[0])
        # gradient_penalty2 = compute_gradient_penalty(netD, x_real, x_fake[1])
        # gradient_penalty = (gradient_penalty1 + gradient_penalty2) / 2
        fake_factor = Tensor(np.random.random((x_real.size(0), 1, 1, 1))).to(device)
        x_fake_mix = (fake_factor * x_fake[0] + ((1 - fake_factor) * x_fake[1])).requires_grad_(True)
        gradient_penalty = compute_gradient_penalty(netD, x_real, x_fake_mix)

    errD = errD + errD1 + errD2 + gradient_penalty + c_loss + loss_contra

    errD.backward()
    # return errD, [real_feat, rec_big, rec_part]
    return errD, []


def train_D_new(netD, x_real, x_fake, enc_x, real_labels=None, fake_labels=None,
                isCloss=True, isBCE=True, isHL=False, isGP=True, isConLoss=False,
                device='cuda'):
    gradient_penalty = 0
    c_loss = 0
    loss_contra = 0
    BCE_stable = nn.BCEWithLogitsLoss().to(device)
    # ContrastiveLoss_func = ContrastiveLoss(batch_size=x_real.shape[0]).to(device)

    part = random.randint(0, 3)
    pred_real, [real_feat, rec_big, rec_part] = netD(x_real, 'real', part=part)
    # rec_enc = netD.decoder_big(enc_x[3])
    errD = percept(rec_big, F.interpolate(x_real, rec_big.shape[2])).sum() + \
           percept(rec_part, F.interpolate(crop_image_by_part(x_real, part), rec_part.shape[2])).sum()

    # pred_real, [real_feat] = netD(x_real)
    # errD = (compute_FID(x_real, x_fake[0], Device=device) + compute_FID(x_real, x_fake[1], Device=device)) / 2
    # errD = percept(x_real, x_fake[0]).sum() + \
    #        percept(x_real, x_fake[1]).sum()

    # 将两个fake_images送入判别器判别
    pred_fake1, fake1_feat_d = netD(x_fake[0].detach(), 'fake')
    pred_fake2, fake2_feat_d = netD(x_fake[1].detach(), 'fake')

    if isCloss is True:
        c_loss1 = c_loss_fn(enc_x[1], fake1_feat_d[0], flag=False)
        c_loss2 = c_loss_fn(enc_x[2], fake2_feat_d[0], flag=False)
        c_loss3 = c_loss_fn(enc_x[0], real_feat, flag=False)
        # c_loss = 0.2 * c_loss1 + 0.2 * c_loss2 + 0.6 * c_loss3
        c_loss = (c_loss1 + c_loss2 + c_loss3) / 3
        # c_loss = c_loss_fn(enc_x, fuck[0], flag=False)

    if isConLoss is True:
        loss_contra1 = ContrastiveLoss_func(real_feat, fake1_feat_d[0])
        loss_contra2 = ContrastiveLoss_func(real_feat, fake2_feat_d[0])
        loss_contra = (loss_contra1 + loss_contra2) / 2
        # tf_aug = transforms.Compose([
        #     transforms.RandomCrop(x_real.shape[2] / 2),
        #     # transforms.RandomInvert(0.5),  # 颜色倒置器(无效)
        #     # transforms.RandomRotation((0, 360)),  # 旋转图片，能有利于在原训练集基础上能好的拟合模型
        #     transforms.RandomHorizontalFlip(0.5),  # 随机水平翻转
        #     # transforms.RandomVerticalFlip(0.5),  # 随机垂直翻转
        #     # transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5)]),
        #     #                        p=1),  # 随机调整HSV
        # ])

    if isBCE is True and isHL is False:
        errD1 = (BCE_stable(pred_real - torch.mean(pred_fake1), real_labels) + BCE_stable(
            pred_fake1 - torch.mean(pred_real), fake_labels)) / 2
        errD2 = (BCE_stable(pred_real - torch.mean(pred_fake2), real_labels) + BCE_stable(
            pred_fake2 - torch.mean(pred_real), fake_labels)) / 2

    if isHL is True and isBCE is False:
        errD1 = hinge_loss_d(pred_real, pred_fake1)
        errD2 = hinge_loss_d(pred_real, pred_fake2)

    if isGP is True:
        # 对判别器梯度更新
        gradient_penalty1 = compute_gradient_penalty(netD, x_real, x_fake[0])
        gradient_penalty2 = compute_gradient_penalty(netD, x_real, x_fake[1])
        gradient_penalty = (gradient_penalty1 + gradient_penalty2) / 2
    # print('gradient_penalty: ', gradient_penalty)
    # print('c_loss: ', c_loss)
    # print('loss_contra: ', loss_contra)
    errD = errD + errD1 + errD2 + gradient_penalty + c_loss + loss_contra

    errD.backward()
    # return errD, [real_feat, rec_big, rec_part]
    return errD, []


if __name__ == '__main__':
    img_size = 64
    batch_size = 4
    channel = 3
    z_shape = 128
    device = 'cuda'
    z1 = torch.randn((batch_size, z_shape), dtype=torch.float, device=device)
    z2 = torch.randn((batch_size, z_shape), dtype=torch.float, device=device)
    d_pro = DiscriminatorPlus(image_size=img_size, n_colors=channel).to(device)
    g_pro = GeneratorPlus(z_size=2 * z_shape, image_size=img_size, n_colors=channel).to(device)
    e = Encoder(img_size, channel, outDim=z_shape).to(device)

    a = ReadImageMetaEnv(image_size=img_size, set_name='train', DatasetsName='vggFace')
    batch, classId = a.load_images(batch_size)
    batch = batch.to(device)
    print(batch.shape, type(batch))
    print(classId)
    # plot_images(batch, title_name='real image', batch_size=batch_size)
    enc_x = e(batch)
    x_fake1 = g_pro(enc_x, z1)
    x_fake2 = g_pro(enc_x, z2)

    # plot_images(x_fake1, title_name='fuck1 image', batch_size=batch_size)
    # plot_images(x_fake2, title_name='fuck2 image', batch_size=batch_size)

    # errD, pre, fuck = train_D(d_pro, batch, [x_fake1, x_fake2])
    # print('errD: ', errD, type(errD))
    s = time.time()
    a1 = compute_FID(batch, x_fake1, Device=device, flag=1)
    e = time.time()
    print('time1: ', e - s)
    s = time.time()
    a2 = compute_FID(batch, x_fake1, Device=device, flag=2)
    e = time.time()
    print('time2: ', e - s)
    print('a1: {} \na2: {}'.format(a1, a2))
    # print('a1: {} \na2: {}'.format(a1, 0))
