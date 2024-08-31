# coding:utf-8
# @TIME         : 2022/3/27 17:27 
# @Author       : BTG
# @Project      : NBGAN
# @File Name    : Encoder.py
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

from GAN_net.BaseNet import ResNet
import torchvision.models as models
from GAN_net.BaseNet import MLP


class Encoder(nn.Module):
    def __init__(self, input_size, n_colors, outDim=128):
        super(Encoder, self).__init__()
        self.n_features = int(input_size / 8)

        # resnet
        self.resnet = nn.Sequential(
            ResNet(n_colors, 64, downsample=False),  # 不带最大池化层
            ResNet(64, 128),
            ResNet(128, 256),
            ResNet(256, 512)
        )
        self.projection = nn.Linear(512 * self.n_features * self.n_features, outDim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.resnet(x)
        # 展平
        x = x.view(x.shape[0], -1)
        x = self.projection(x)
        x = self.tanh(x)

        return x


class EncoderNew(nn.Module):
    def __init__(self, input_size, n_colors, outDim=128):
        super(EncoderNew, self).__init__()
        # self.n_features = int(input_size / 8)
        self.n_features = int(input_size / 32)

        # resnet

        # [bs, 3, imageSize, imageSize] -> [bs, 512, 4, 4]
        self.resnet = nn.Sequential(
            ResNet(n_colors, 16, downsample=False),  # 不带最大池化层
            ResNet(16, 32),
            ResNet(32, 64),
            ResNet(64, 128),
            ResNet(128, 256),
            ResNet(256, 512)
        )
        self.projection = nn.Linear(512 * self.n_features * self.n_features, outDim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.resnet(x)
        # 展平
        x_f = x.view(x.shape[0], -1)
        x_f = self.projection(x_f)
        x_f = self.tanh(x_f)
        # print('x: ', x.shape)
        # print('x_f: ', x_f.shape)
        return [x, x_f]


# 预训练的残差网络，用作计算嵌入
class ResNetEmbedding(nn.Module):
    """
    预训练的残差网络
    """

    def __init__(self, input_size, n_colors, out_dim=128):
        super().__init__()
        self.input_size = input_size
        self.n_colors = n_colors
        # self.myResnet = models.resnet50(pretrained=False)
        # myResnet.load_state_dict(torch.load('./util/resnet50.pth'))
        self.myResnet = models.resnet18(pretrained=False)
        # myResnet.load_state_dict(torch.load('./BNN_Save/resnet18.pth'))
        self.updim = ResNet(1, 3, downsample=False)
        self.proj = nn.Linear(1000, out_dim)

    def forward(self, x):
        if self.n_colors == 1:
            x = self.updim(x)
        output = self.myResnet(x)
        output_proj = self.proj(output)
        return output_proj


# ==================================
class encoder(nn.Module):
    def __init__(self, input_size, n_colors):
        super(Encoder, self).__init__()

        self.n_features = int(input_size / 8)

        self.module1 = ResNet(n_colors, 64, downsample=False)
        self.module2 = ResNet(64, 128)
        self.module3 = ResNet(128, 256)
        self.module4 = ResNet(256, 512)

        self.dense = nn.Linear(512 * self.n_features * self.n_features, 128)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)

        return self.tanh(x)


class CNNEncoder_plus(nn.Module):
    """docstring for ClassName"""

    def __init__(self):
        super(CNNEncoder_plus, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = out.view(out.size(0),-1)
        return out  # 64


class CNNEncoder(nn.Module):
    """docstring for ClassName"""

    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = out.view(out.size(0),-1)
        return out  # 64


if __name__ == '__main__':
    imgSize = 32
    # myModel = ResNetEmbedding(3,1).cuda()
    # fuck = ResNet(1, 3, downsample=True).cuda()
    fuck = models.resnet18(pretrained=False).cuda()
    # fuck.load_state_dict(torch.load('../util/resnet18.pth'))
    print(fuck)

    test_input = torch.randn(3, 3, imgSize, imgSize).cuda()
    result = fuck(test_input)
    print(result.shape)
