# coding:utf-8
# @TIME         : 2022/3/26 21:05 
# @Author       : BTG
# @Project      : NBGAN
# @File Name    : Generator.py
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

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm  # 频谱归一化
from GAN_net.BaseNet import UpBlock, UpBlockComp, InitLayer, SEBlock, conv2d
import numpy as np


class Generator(nn.Module):

    def __init__(self, z_size, input_size, n_colors, spectral_G=False, Tanh_GD=False, no_batch_norm_G=False):
        """

        :param z_size:  噪声size
        :param input_size:  输入size
        :param n_colors:
        :param spectral_G:  (bool) default:　False, 用于判断生成器是否添加频谱归一化， 默认不添加
        :param Tanh_GD: (bool)  default:    False,  判断是否使用tanh激活函数
        :param no_batch_norm_G: (bool)  default: False, 用于判断生成器是否添加BN层，默认添加
        """
        super(Generator, self).__init__()

        self.z_size = z_size
        self.n_features = int(input_size / 8)
        # self.dense = nn.Linear(128 + self.z_size, 512 * self.n_features * self.n_features)
        self.dense = nn.Linear(2 * self.z_size, 512 * self.n_features * self.n_features)

        if spectral_G:
            model = [spectral_norm(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)),
                     nn.ReLU(True),
                     spectral_norm(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)),
                     nn.ReLU(True),
                     spectral_norm(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True)),
                     nn.ReLU(True),
                     spectral_norm(nn.ConvTranspose2d(64, n_colors, kernel_size=3, stride=1, padding=1, bias=True)),
                     nn.Tanh()]
        else:
            model = [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)]
            if not no_batch_norm_G:
                model += [nn.BatchNorm2d(256)]
            if Tanh_GD:
                model += [nn.Tanh()]
            else:
                model += [nn.ReLU(True)]
            model += [nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)]
            if not no_batch_norm_G:
                model += [nn.BatchNorm2d(128)]
            if Tanh_GD:
                model += [nn.Tanh()]
            else:
                model += [nn.ReLU(True)]
            model += [nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True)]
            if not no_batch_norm_G:
                model += [nn.BatchNorm2d(64)]
            if Tanh_GD:
                model += [nn.Tanh()]
            else:
                model += [nn.ReLU(True)]
            model += [nn.Conv2d(64, n_colors, kernel_size=3, stride=1, padding=1, bias=True)]

        self.model = nn.Sequential(*model)
        self.tanh = nn.Tanh()

    def forward(self, x, z):
        input = torch.cat((x, z), dim=1)
        # output = self.dense(input.view(-1, 128 + self.z_size))
        output = self.dense(input.view(-1, 2 * self.z_size))
        # print('output before view: ', output.shape)
        output = output.view(-1, 512, self.n_features, self.n_features)
        # print('output after view: ', output.shape)
        output = self.model(output)
        output = self.tanh(output)
        return output


class GeneratorPro(nn.Module):
    """
    输入为 batch_size x input_dim 的噪声，生成器借此生成图像 \n
    """

    def __init__(self, ngf=64, nz=100, nc=3, im_size=1024):
        super(GeneratorPro, self).__init__()
        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)
        # print(nfc)
        # {4: 1024, 8: 512, 16: 256, 32: 128, 64: 128,
        # 128: 64, 256: 32, 512: 16, 1024: 8}

        self.im_size = im_size
        # 将输入由 nz 升维到 nfc[4]，大小升到4x4
        self.init = InitLayer(nz, channel=nfc[4])
        # 将输入由 nfc[4] 降维到 nfc[8]，大小升到8x8
        self.feat_8 = UpBlockComp(nfc[4], nfc[8])
        # 将输入由 nfc[8] 降维到 nfc[16]，大小升到16x16
        self.feat_16 = UpBlock(nfc[8], nfc[16])
        # 将输入由 nfc[16] 降维到 nfc[32]，大小升到32x32
        self.feat_32 = UpBlockComp(nfc[16], nfc[32])

        # 将输入由 nfc[32] 降维到 nfc[64]，大小升到64x64
        if self.im_size >= 64:
            self.feat_64 = UpBlock(nfc[32], nfc[64])
            self.se_64 = SEBlock(nfc[4], nfc[64])
        # 将输入由 nfc[64] 降维到 nfc[128]，大小升到128x128
        if self.im_size >= 128:
            self.feat_128 = UpBlockComp(nfc[64], nfc[128])
            self.to_128 = conv2d(nfc[128], nc, 1, 1, 0, bias=False)
            self.se_128 = SEBlock(nfc[8], nfc[128])
        # 将输入由 nfc[128] 降维到 nfc[256]，大小升到256x256
        if self.im_size >= 256:
            self.feat_256 = UpBlock(nfc[128], nfc[256])
            self.feat_512 = UpBlockComp(nfc[256], nfc[512])
            self.se_256 = SEBlock(nfc[16], nfc[256])
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if im_size > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])

        self.to_big = conv2d(nfc[im_size], nc, 3, 1, 1, bias=False)

    def forward(self, x, z):
        input = torch.cat((x, z), dim=1)

        feat_4 = self.init(input)
        # print('feat_4: ', feat_4.shape)
        feat_8 = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        feat_32 = self.feat_32(feat_16)
        # 当生成图像大小为32时
        if self.im_size == 32:
            return self.to_big(feat_32)

        feat_64 = self.se_64(feat_4, self.feat_64(feat_32))
        # print('feat_64: ', feat_64.shape)
        # ==============================================================
        # 当生成图像大小为64时(自己加的)
        if self.im_size == 64:
            return self.to_big(feat_64)
        feat_128 = self.se_128(feat_8, self.feat_128(feat_64))  # 原代码就有的
        # 当生成图像大小为128时(自己加的)
        if self.im_size == 128:
            return [self.to_big(feat_128), self.to_128(feat_128)]
        # ==============================================================
        feat_256 = self.se_256(feat_16, self.feat_256(feat_128))

        # 当生成图像大小为256时
        if self.im_size == 256:
            return [self.to_big(feat_256), self.to_128(feat_128)]

        feat_512 = self.se_512(feat_32, self.feat_512(feat_256))
        # 当生成图像大小为512时
        if self.im_size == 512:
            return [self.to_big(feat_512), self.to_128(feat_128)]

        feat_1024 = self.feat_1024(feat_512)

        im_128 = torch.tanh(self.to_128(feat_128))
        im_1024 = torch.tanh(self.to_big(feat_1024))

        return [im_1024, im_128]


class GeneratorPlus(nn.Module):
    def __init__(self, z_size, image_size, n_colors):
        super(GeneratorPlus, self).__init__()
        dim4Size = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4}

        # self.z_size = z_size
        self.image_size = image_size
        self.init = InitLayer(nz=z_size, channel=512)  # 输入噪声[bs, z_size]转换为[bs, 512, 4, 4]
        self.feat_8 = UpBlockComp(512, 256)  # [bs, 512, 4, 4] -> [bs, 256, 8, 8]
        self.feat_16 = UpBlock(256, 128)  # [bs, 256, 8, 8] -> [bs, 128, 16, 16]
        self.feat_32 = UpBlockComp(128, 64)  # [bs, 128, 16, 16] -> [bs, 64, 32, 32]
        if image_size >= 64:
            self.feat_64 = UpBlock(64, 32)  # [bs, 64, 32, 32] -> [bs, 32, 64, 64]
            self.se_64 = SEBlock(512, 32)
        if image_size >= 128:
            self.feat_128 = UpBlockComp(32, 16)  # [bs, 32, 64, 64] -> [bs, 16, 128, 128]
            self.se_128 = SEBlock(256, 16)
        if image_size >= 256:
            self.feat_256 = UpBlock(16, 8)  # [bs, 16, 128, 128] -> [bs, 8, 256, 256]
            self.se_256 = SEBlock(128, 8)
        if image_size >= 512:
            self.feat_512 = UpBlockComp(8, 4)  # [bs, 8, 256, 256] -> [bs, 4, 512, 512]
            self.se_512 = SEBlock(64, 4)
        self.to_big = conv2d(dim4Size[image_size], n_colors, 3, 1, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x, z):
        input = torch.cat((x, z), dim=1)

        feat_4 = self.init(input)  # 输入噪声[bs, z_size] -> [bs, 512, 4, 4]
        # print('feat_4: ', feat_4.shape)
        feat_8 = self.feat_8(feat_4)  # [bs, 512, 4, 4] -> [bs, 256, 8, 8]
        # print('feat_8: ', feat_8.shape)
        feat_16 = self.feat_16(feat_8)  # [bs, 256, 8, 8] -> [bs, 128, 16, 16]
        # print('feat_16: ', feat_16.shape)
        feat_32 = self.feat_32(feat_16)  # [bs, 128, 16, 16] -> [bs, 64, 32, 32]
        # print('feat_32: ', feat_32.shape)
        if self.image_size == 32:
            feat_final = feat_32
        if self.image_size >= 64:
            feat_64 = self.feat_64(feat_32)  # [bs, 64, 32, 32] -> [bs, 32, 64, 64]
            feat_64 = self.se_64(feat_4, feat_64)
            # print('feat_64: ', feat_64.shape)
            feat_final = feat_64
        if self.image_size >= 128:
            feat_128 = self.feat_128(feat_64)  # [bs, 32, 64, 64] -> [bs, 16, 128, 128]
            feat_128 = self.se_128(feat_8, feat_128)
            # print('feat_128: ', feat_128.shape)
            feat_final = feat_128
        if self.image_size >= 256:
            feat_256 = self.feat_256(feat_128)  # [bs, 16, 128, 128] -> [bs, 8, 256, 256]
            feat_256 = self.se_256(feat_16, feat_256)
            feat_final = feat_256
        if self.image_size >= 512:
            feat_512 = self.feat_512(feat_256)  # [bs, 8, 256, 256] -> [bs, 4, 512, 512]
            feat_512 = self.se_512(feat_32, feat_512)
            feat_final = feat_512
        # feat_final = torch.clamp(feat_final, -1, 1)

        return self.tanh(self.to_big(feat_final))


class GeneratorProPro(nn.Module):
    """ Generator of the GAN network """

    def __init__(self, depth=7, latent_size=512, use_eql=True):
        """
        constructor for the Generator class
        :param depth: required depth of the Network
        :param latent_size: size of the latent manifold
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import ModuleList, Conv2d
        from GAN_net.BaseNet2 import GenGeneralConvBlock, GenInitialBlock, _equalized_conv2d

        super().__init__()

        assert latent_size != 0 and ((latent_size & (latent_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert latent_size >= np.power(2, depth - 4), "latent size will diminish to zero"

        # state of the generator:
        self.use_eql = use_eql
        self.depth = depth
        self.latent_size = latent_size

        # register the modules required for the Generator Below ...
        # create the ToRGB layers for various outputs:
        if self.use_eql:
            def to_rgb(in_channels):
                return _equalized_conv2d(in_channels, 3, (1, 1), bias=True)
        else:
            def to_rgb(in_channels):
                return Conv2d(in_channels, 3, (1, 1), bias=True)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([GenInitialBlock(self.latent_size, use_eql=self.use_eql)])
        self.rgb_converters = ModuleList([to_rgb(self.latent_size)])

        # create the remaining layers
        for i in range(self.depth - 1):
            if i <= 2:
                layer = GenGeneralConvBlock(self.latent_size, self.latent_size,
                                            use_eql=self.use_eql)
                rgb = to_rgb(self.latent_size)
            else:
                layer = GenGeneralConvBlock(
                    int(self.latent_size // np.power(2, i - 3)),
                    int(self.latent_size // np.power(2, i - 2)),
                    use_eql=self.use_eql
                )
                rgb = to_rgb(int(self.latent_size // np.power(2, i - 2)))
            self.layers.append(layer)
            self.rgb_converters.append(rgb)

    def forward(self, x, z):
        """
        forward pass of the Generator
        :param x: input noise
        :return: *y => output of the generator at various scales
        """
        outputs = []  # initialize to empty list
        input = torch.cat((x, z), dim=1)
        y = input  # start the computational pipeline
        for block, converter in zip(self.layers, self.rgb_converters):
            y = block(y)
            outputs.append(converter(y))

        return outputs

    @staticmethod
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


if __name__ == '__main__':
    z_shape = 128
    image_size = 128
    channels = 3
    device = 'cuda'
    g = GeneratorPlus(z_size=2 * z_shape, image_size=image_size, n_colors=channels).to(device)
    print(g)
