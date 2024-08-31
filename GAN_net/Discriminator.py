# coding:utf-8
# @TIME         : 2022/3/26 21:07 
# @Author       : BTG
# @Project      : NBGAN
# @File Name    : Discriminator.py
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
import random

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm  # 频谱归一化
import torch.nn.functional as F
from torchvision import transforms
from lib.loss_function import crop_image_by_part
from GAN_net.BaseNet import DownBlock, DownBlockComp, SEBlock, conv2d, batchNorm2d, GLU
from GAN_net.Decoder import SimpleDecoder
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, input_size, n_colors, spectral_D=True, Tanh_GD=False, no_batch_norm_D=False, Proj=False,
                 ProjDim=128):
        """

        :param input_size: 输入的size
        :param n_colors:
        :param spectral_D: (bool) default:　True, 用于判断生成器是否添加频谱归一化， 默认添加
        :param Tanh_GD: (bool)  default:    False,  判断是否使用tanh激活函数
        :param no_batch_norm_D: (bool)  default: False, 用于判断生成器是否添加BN层，默认添加
        """
        super(Discriminator, self).__init__()
        self.n_features = int(input_size / 8)
        self.Proj = Proj

        if self.Proj is False:
            self.dense = nn.Linear(512 * self.n_features * self.n_features, 1)
        else:
            self.proj_head = nn.Linear(512 * self.n_features * self.n_features, ProjDim)
            self.dense = nn.Linear(ProjDim, 1)

        if spectral_D:
            model = [spectral_norm(nn.Conv2d(n_colors, 64, kernel_size=3, stride=1, padding=1, bias=True)),
                     nn.LeakyReLU(0.1, inplace=True),
                     spectral_norm(nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True)),
                     nn.LeakyReLU(0.1, inplace=True),

                     spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)),
                     nn.LeakyReLU(0.1, inplace=True),
                     spectral_norm(nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)),
                     nn.LeakyReLU(0.1, inplace=True),

                     spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)),
                     nn.LeakyReLU(0.1, inplace=True),
                     spectral_norm(nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=True)),
                     nn.LeakyReLU(0.1, inplace=True),

                     spectral_norm(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)),
                     nn.LeakyReLU(0.1, inplace=True)]
        else:
            model = [nn.Conv2d(n_colors, 64, kernel_size=3, stride=1, padding=1, bias=True)]
            if not no_batch_norm_D:
                model += [nn.BatchNorm2d(64)]
            if Tanh_GD:
                model += [nn.Tanh()]
            else:
                model += [nn.LeakyReLU(0.1, inplace=True)]

            model += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)]
            if not no_batch_norm_D:
                model += [nn.BatchNorm2d(64)]
            if Tanh_GD:
                model += [nn.Tanh()]
            else:
                model += [nn.LeakyReLU(0.1, inplace=True)]

            model += [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)]
            if not no_batch_norm_D:
                model += [nn.BatchNorm2d(128)]
            if Tanh_GD:
                model += [nn.Tanh()]
            else:
                model += [nn.LeakyReLU(0.1, inplace=True)]

            model += [nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)]
            if not no_batch_norm_D:
                model += [nn.BatchNorm2d(128)]
            if Tanh_GD:
                model += [nn.Tanh()]
            else:
                model += [nn.LeakyReLU(0.1, inplace=True)]

            model += [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)]
            if not no_batch_norm_D:
                model += [nn.BatchNorm2d(256)]
            if Tanh_GD:
                model += [nn.Tanh()]
            else:
                model += [nn.LeakyReLU(0.1, inplace=True)]

            model += [nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=True)]
            if not no_batch_norm_D:
                model += [nn.BatchNorm2d(256)]
            if Tanh_GD:
                model += [nn.Tanh()]
            else:
                model += [nn.LeakyReLU(0.1, inplace=True)]
            model += [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)]
            if Tanh_GD:
                model += [nn.Tanh()]
            else:
                model += [nn.LeakyReLU(0.1, inplace=True)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        p = self.model(x)  # [bs, 512, 4, 4]
        # print('p: ', p.shape)
        proj = p.view(-1, 512 * self.n_features * self.n_features)
        if self.Proj is True:
            proj = self.proj_head(proj)
        output = self.dense(proj).view(-1)
        # output = torch.softmax(output, dim=0)

        return output, proj


class DiscriminatorPro(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512):
        super(DiscriminatorPro, self).__init__()
        self.ndf = ndf
        self.im_size = im_size
        # self.resize = transforms.Resize((256, 256))

        nfc_multi = {4: 16, 8: 16, 16: 8, 32: 4, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ndf)
        # print(nfc)
        # {4: 1024, 8: 1024, 16: 512, 32: 256, 64: 128,
        # 128: 64, 256: 32, 512: 16, 1024: 8}

        # 定义down_from_big(image), 将输入图像降到16 256
        if im_size == 1024:
            """
               in_dim = nc = 3
               out_dim = nfc[1024] = 8
               kernel = 4
               stride = 2
               padding = 1

            """
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[1024], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(nfc[1024], nfc[512], 4, 2, 1, bias=False),
                batchNorm2d(nfc[512]),
                nn.LeakyReLU(0.2, inplace=True))
        elif im_size == 512:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True))
        elif im_size <= 256:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True))

        self.down_2 = DownBlockComp(16, 32)  # 128
        self.down_4 = DownBlockComp(32, 64)  # 64
        self.down_8 = DownBlockComp(64, 128)  # 32
        self.down_16 = DownBlockComp(128, 256)  # 16
        self.down_32 = DownBlockComp(256, 512)  # 8

        self.rf_big = nn.Sequential(
            # 512,8 -> 1024, 8
            conv2d(nfc[16], nfc[8], 1, 1, 0, bias=False),
            batchNorm2d(nfc[8]), nn.LeakyReLU(0.2, inplace=True),
            conv2d(nfc[8], 1, 4, 1, 0, bias=False))

        self.se_big_8 = SEBlock(16, 128)  # dim: 16 -> 128
        self.se_2_16 = SEBlock(32, 256)  # dim: 32 -> 256
        self.se_4_32 = SEBlock(64, 512)  # dim: 64 -> 512

        self.down_from_small = nn.Sequential(
            conv2d(nc, nfc[256], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            DownBlock(nfc[256], nfc[128]),
            DownBlock(nfc[128], nfc[64]),
            DownBlock(nfc[64], nfc[32]), )

        self.rf_small = conv2d(nfc[32], 1, 4, 1, 0, bias=False)

        self.decoder_big = SimpleDecoder(nfc[16], nc)
        self.decoder_part = SimpleDecoder(nfc[32], nc)
        self.decoder_small = SimpleDecoder(nfc[32], nc)

    def forward(self, imgs, label='fake', part=None):

        if type(imgs) is not list:
            imgs = [F.interpolate(imgs, size=self.im_size), F.interpolate(imgs, size=128)]

        if imgs[0].shape[2] < 256:
            imgs = [F.interpolate(imgs[0], size=256), F.interpolate(imgs[1], size=128)]
            # imgs[0] = self.resize(imgs[0])

        feat_big = self.down_from_big(imgs[0])  # 16 256
        # print('down_from_big(feat_2): ', feat_big.shape)

        feat_2 = self.down_2(feat_big)  # 32 128
        # print('feat_2: ', feat_2.shape)

        feat_4 = self.down_4(feat_2)  # 64 64
        # print('feat_4: ', feat_4.shape)

        feat_8 = self.down_8(feat_4)  # 128 32
        feat_8 = self.se_big_8(feat_big, feat_8)
        # print('feat_8: ', feat_8.shape)

        feat_16 = self.down_16(feat_8)  # 256 16
        feat_16 = self.se_2_16(feat_2, feat_16)
        # print('feat_16: ', feat_16.shape)

        feat_last = self.down_32(feat_16)  # 512 8
        feat_last = self.se_4_32(feat_4, feat_last)
        print('feat_last: ', feat_last.shape)

        rf_0 = self.rf_big(feat_last).view(-1)
        # print('rf_0 before shape: ', self.rf_big(feat_last).shape)
        # print('rf_0 shape: ', rf_0.shape)

        feat_small = self.down_from_small(imgs[1])  # 256 8
        # print('feat_small: ', feat_small.shape)

        # rf_1 = self.rf_small(feat_small)
        rf_1 = self.rf_small(feat_small).view(-1)
        # print('feat_small: ', feat_small.shape)
        # print('rf_1 before shape: ', self.rf_small(feat_small).shape)
        # print('rf_1 shape: ', rf_1.shape)

        if label == 'real':
            rec_img_big = self.decoder_big(feat_last)
            rec_img_small = self.decoder_small(feat_small)

            assert part is not None
            rec_img_part = None
            if part == 0:
                rec_img_part = self.decoder_part(feat_16[:, :, :8, :8])
            if part == 1:
                rec_img_part = self.decoder_part(feat_16[:, :, :8, 8:])
            if part == 2:
                rec_img_part = self.decoder_part(feat_16[:, :, 8:, :8])
            if part == 3:
                rec_img_part = self.decoder_part(feat_16[:, :, 8:, 8:])

            return torch.cat([rf_0, rf_1]), [rec_img_big, rec_img_small, rec_img_part]

        return torch.cat([rf_0, rf_1])


class DiscriminatorPlus(nn.Module):
    def __init__(self, image_size, n_colors):
        super(DiscriminatorPlus, self).__init__()
        self.image_size = image_size
        dim4Size = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4}
        # [bs, 3, 512, 512] -> [bs, 16, 128, 128]
        if image_size >= 512:
            self.down_from_big = nn.Sequential(
                conv2d(n_colors, 8, 4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(8, 16, 4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            )
        # [bs, 3, 256, 256] -> [bs, 16, 128, 128]
        elif image_size == 256:
            self.down_from_big = nn.Sequential(
                conv2d(n_colors, 16, 4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            )
        # [bs, 3, 128, 128] -> [bs, 16, 128, 128]
        elif image_size == 128:
            self.down_from_big = nn.Sequential(
                conv2d(n_colors, 16, 3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            )
        # [bs, 3, 64, 64] -> [bs, 64, 32, 32]
        elif image_size == 64:
            self.down_from_big = nn.Sequential(
                conv2d(n_colors, 64, 4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            )
        # [bs, 3, 32, 32] -> [bs, 64, 32, 32]
        else:
            self.down_from_big = nn.Sequential(
                conv2d(n_colors, 64, 3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            )

        if image_size >= 128:
            self.down_2 = DownBlockComp(16, 32)  # [bs, 16, 128, 128] -> [bs, 32, 64, 64]
            self.down_4 = DownBlockComp(32, 64)  # [bs, 32, 64, 64] -> [bs, 64, 32, 32]
            self.se_big_8 = SEBlock(16, 128)  # dim: 16, 128 -> 128, 16
            self.se_2_16 = SEBlock(32, 256)  # dim: 32, 64 -> 256, 8

        self.se_4_32 = SEBlock(64, 512)  # dim: 64, 32 -> 512, 4
        self.down_4_2 = DownBlockComp(64, 128)  # [bs, 64, 32, 32] -> [bs, 128, 16, 16]
        self.down_4_4 = DownBlockComp(128, 256)  # [bs, 128, 16, 16] -> [bs, 256, 8, 8]
        self.down_4_8 = DownBlockComp(256, 512)  # [bs, 256, 8, 8] -> [bs, 512, 4, 4]
        self.rf_big = nn.Sequential(
            # [bs, 512, 4, 4] -> [bs, 8, 4, 4]
            conv2d(512, 8, 1, 1, 0, bias=False),
            batchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flatten = conv2d(8, 1, 4, 1, 0, bias=False)  # [bs, 8, 4, 4] -> [bs, 1, 1, 1]
        # self.flatten = nn.Linear(128, 1)
        self.decoder_big = SimpleDecoder(512, n_colors, output_dim=image_size)
        self.decoder_part = SimpleDecoder(256, n_colors, output_dim=image_size)

    def forward(self, x_input, label='fake', part=None):
        if x_input.shape[2] != self.image_size:
            x_tf = transforms.Resize((self.image_size, self.image_size))
            x_input = x_tf(x_input)
        # print('x_input shape: ', x_input.shape, x_input.device)
        feat_big = self.down_from_big(x_input)  # [bs, 3, imageSize, imageSize] -> [bs, 16, 128, 128]
        # print('x_input shape: ', x_input.shape, x_input.device)
        # print('feat_big shape: ', feat_big.shape, feat_big.device)
        if self.image_size >= 128:
            feat_2 = self.down_2(feat_big)  # [bs, 16, 128, 128] -> [bs, 32, 64, 64]
            feat_4 = self.down_4(feat_2)  # [bs, 32, 64, 64] -> [bs, 64, 32, 32]

            feat_4_2 = self.down_4_2(feat_4)  # [bs, 64, 32, 32] -> [bs, 128, 16, 16]
            feat_4_2 = self.se_big_8(feat_big, feat_4_2)

            feat_4_4 = self.down_4_4(feat_4_2)  # [bs, 128, 16, 16] -> [bs, 256, 8, 8]
            feat_4_4 = self.se_2_16(feat_2, feat_4_4)

            feat_4_8 = self.down_4_8(feat_4_4)  # [bs, 256, 8, 8] -> [bs, 512, 4, 4]
            feat_4_8 = self.se_4_32(feat_4, feat_4_8)
        else:
            feat_4_2 = self.down_4_2(feat_big)  # [bs, 64, 32, 32] -> [bs, 128, 16, 16]
            feat_4_4 = self.down_4_4(feat_4_2)  # [bs, 128, 16, 16] -> [bs, 256, 8, 8]
            feat_4_8 = self.down_4_8(feat_4_4)  # [bs, 256, 8, 8] -> [bs, 512, 4, 4]
            feat_4_8 = self.se_4_32(feat_big, feat_4_8)
        feat_last = feat_4_8

        rf = self.rf_big(feat_last)  # [bs, 512, 4, 4] -> [bs, 8, 4, 4]
        # print('rf shape: ', rf.shape, rf.view(x.shape[0], -1).shape, rf.device)
        output = self.flatten(rf).view(-1)  # [bs, 8, 4, 4] -> [bs, 1, 1, 1] -> [bs]
        # output = self.flatten(rf.view(rf.shape[0], -1)).view(-1)  # [bs, 8, 4, 4] -> [bs, 1, 1, 1] -> [bs]
        # print('output shape: ', output, output.shape, output.device)
        if label == 'real':
            rec_img_big = self.decoder_big(feat_last)
            assert part is not None

            rec_img_part = None
            if part == 0:
                rec_img_part = self.decoder_part(crop_image_by_part(feat_4_4, part))
            if part == 1:
                rec_img_part = self.decoder_part(crop_image_by_part(feat_4_4, part))
            if part == 2:
                rec_img_part = self.decoder_part(crop_image_by_part(feat_4_4, part))
            if part == 3:
                rec_img_part = self.decoder_part(crop_image_by_part(feat_4_4, part))

            return output, [rf.view(x_input.shape[0], -1), rec_img_big, rec_img_part]

        return output, [rf.view(x_input.shape[0], -1)]


class DiscriminatorProPro(nn.Module):
    """ Discriminator of the GAN """

    def __init__(self, depth=7, feature_size=512,
                 use_eql=True, gpu_parallelize=False):
        """
        constructor for the class
        :param depth: total depth of the discriminator
                       (Must be equal to the Generator depth)
        :param feature_size: size of the deepest features extracted
                             (Must be equal to Generator latent_size)
        :param use_eql: whether to use the equalized learning rate or not
        :param gpu_parallelize: whether to use DataParallel on the discriminator
                                Note that the Last block contains StdDev layer
                                So, it is not parallelized.
        """
        from torch.nn import ModuleList
        from GAN_net.BaseNet2 import DisGeneralConvBlock, DisFinalBlock, _equalized_conv2d
        from torch.nn import Conv2d

        super().__init__()

        assert feature_size != 0 and ((feature_size & (feature_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert feature_size >= np.power(2, depth - 4), \
                "feature size cannot be produced"

        # create state of the object
        self.gpu_parallelize = gpu_parallelize
        self.use_eql = use_eql
        self.depth = depth
        self.feature_size = feature_size

        # create the fromRGB layers for various inputs:
        if self.use_eql:
            def from_rgb(out_channels):
                return _equalized_conv2d(3, out_channels, (1, 1), bias=True)
        else:
            def from_rgb(out_channels):
                return Conv2d(3, out_channels, (1, 1), bias=True)

        self.rgb_to_features = ModuleList()
        self.final_converter = from_rgb(self.feature_size // 2)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList()
        self.final_block = DisFinalBlock(self.feature_size, use_eql=self.use_eql)

        # create the remaining layers
        for i in range(self.depth - 1):
            if i > 2:
                layer = DisGeneralConvBlock(
                    int(self.feature_size // np.power(2, i - 2)),
                    int(self.feature_size // np.power(2, i - 2)),
                    use_eql=self.use_eql
                )
                rgb = from_rgb(int(self.feature_size // np.power(2, i - 1)))
            else:
                layer = DisGeneralConvBlock(self.feature_size, self.feature_size // 2,
                                            use_eql=self.use_eql)
                rgb = from_rgb(self.feature_size // 2)

            self.layers.append(layer)
            self.rgb_to_features.append(rgb)

        # just replace the last converter
        self.rgb_to_features[self.depth - 2] = \
            from_rgb(self.feature_size // np.power(2, i - 2))

        # parallelize the modules from the module-lists if asked to:
        if self.gpu_parallelize:
            for i in range(len(self.layers)):
                self.layers[i] = torch.nn.DataParallel(self.layers[i])
                self.rgb_to_features[i] = torch.nn.DataParallel(
                    self.rgb_to_features[i])

        # Note that since the FinalBlock contains the StdDev layer,
        # it cannot be parallelized so easily. It will have to be parallelized
        # from the Lower level (from CustomLayers). This much parallelism
        # seems enough for me.

    def forward(self, inputs):
        """
        forward pass of the discriminator
        :param inputs: (multi-scale input images) to the network list[Tensors]
        :return: out => raw prediction values
        """

        assert len(inputs) == self.depth, \
            "Mismatch between input and Network scales"

        y = self.rgb_to_features[self.depth - 2](inputs[self.depth - 1])
        y = self.layers[self.depth - 2](y)
        for x, block, converter in \
                zip(reversed(inputs[1:-1]),
                    reversed(self.layers[:-1]),
                    reversed(self.rgb_to_features[:-1])):
            input_part = converter(x)  # convert the input:
            y = torch.cat((input_part, y), dim=1)  # concatenate the inputs:
            y = block(y)  # apply the block

        # calculate the final block:
        input_part = self.final_converter(inputs[0])
        y = torch.cat((input_part, y), dim=1)
        y = self.final_block(y)

        # return calculated y
        return y


if __name__ == '__main__':
    # {4: 1024, 8: 1024, 16: 512, 32: 256, 64: 128, 128: 64, 256: 32, 512: 16, 1024: 8}
    x = torch.randn((8, 8, 4, 4), dtype=torch.float, device='cpu')
    print('x: ', x.shape, x.device)
    flatten = nn.Linear(128, 1)
    o = flatten(x.view(x.shape[0], -1))
    print('o: ', o, o.shape, o.device)
