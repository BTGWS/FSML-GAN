# coding:utf-8
# @TIME         : 2022/5/23 17:32 
# @Author       : BTG
# @Project      : NBGAN
# @File Name    : Decoder.py
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
from torch import nn
from GAN_net.BaseNet import UpBlock, conv2d, GLU


class SimpleDecoder(nn.Module):
    """docstring for CAN_SimpleDecoder"""

    def __init__(self, nfc_in=64, nc=3, output_dim=128):
        super(SimpleDecoder, self).__init__()

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        # {4: 1024, 8: 512, 16: 256, 32: 128, 64: 128, 128: 64, 256: 32, 512: 16, 1024: 8}

        nfc = {}
        model = []
        for k, v in nfc_multi.items():
            nfc[k] = int(v * 32)

        # self.main = nn.Sequential(nn.AdaptiveAvgPool2d(8),
        #                           UpBlock(nfc_in, nfc[16]),
        #                           UpBlock(nfc[16], nfc[32]),
        #                           UpBlock(nfc[32], nfc[64]),
        #                           UpBlock(nfc[64], nfc[128]),
        #                           conv2d(nfc[128], nc, 3, 1, 1, bias=False),
        #                           nn.Tanh())

        if output_dim >= 32:
            model = [nn.AdaptiveAvgPool2d(8),
                     UpBlock(nfc_in, nfc[16]),
                     UpBlock(nfc[16], nfc[32])]
        if output_dim >= 64:
            model += [UpBlock(nfc[32], nfc[64])]
        if output_dim >= 128:
            model += [UpBlock(nfc[64], nfc[128])]
        model += [conv2d(nfc[output_dim], nc, 3, 1, 1, bias=False),
                  nn.Tanh()]
        self.main = nn.Sequential(*model)

    def forward(self, input):
        # input shape: c x 4 x 4
        return self.main(input)


class testSimpleMLP(nn.Module):
    def __init__(self, nfc_in=64, nc=3, output_dim=128):
        super(testSimpleMLP, self).__init__()

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * 32)
        if output_dim >= 32:
            model = [nn.AdaptiveAvgPool2d(8),
                     UpBlock(nfc_in, nfc[16]),
                     UpBlock(nfc[16], nfc[32])]
        if output_dim >= 64:
            model += [UpBlock(nfc[32], nfc[64])]
        if output_dim >= 128:
            model += [UpBlock(nfc[64], nfc[128])]
        model += [conv2d(nfc[output_dim], nc, 3, 1, 1, bias=False),
                  nn.Tanh()]
        self.main = nn.Sequential(*model)

    def forward(self, x):
        out = self.main(x)
        return out


if __name__ == '__main__':
    z = torch.randn((4, 512, 4, 4), dtype=torch.float, device='cpu')
    # print(z.shape)
    de = SimpleDecoder(512, 3).to('cpu')
    # print(de)
    # a = de(z)
    # print('a: ', a.shape)
    ff = testSimpleMLP(512, 3, output_dim=32).to('cpu')
    aa = ff(z)
    print('aa: ', aa.shape)
    # print('#######################')
    # print(ff)
