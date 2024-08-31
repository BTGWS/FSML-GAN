# coding:utf-8
# @TIME         : 2022/9/19 16:44 
# @Author       : BTG
# @Project      : NBGAN
# @File Name    : createImblanceDatasets.py
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
"""Training script.
usage:
    train.py [options]
"""
import torch

from lib.plotImage import saveImages, saveImages_mutil
from util.Args_NEW import doc
from util.Args_BNN import doc as doc_bnn

from lib.getDatasets import MnistMetaEnv, OmniglotMetaEnv, ReadImageMetaEnv
from docopt import docopt
from tqdm import tqdm
from torch.utils.data import DataLoader

args = docopt(__doc__ + doc_bnn)
bs = int(args['--batch_size'])
nw = int(args['--num_workers'])
# device = args['--device']
device = 'cpu'
datasetName = args['--dataset']


def CreateImblanceDatasetsFun(dataset):
    print('start')
    dl = DataLoader(dataset, batch_size=bs, shuffle=False,
                    num_workers=nw, pin_memory=True)
    for i, (real_images, real_labels) in enumerate(dl):
        real_images, real_labels = real_images.to(device), real_labels.to(device)
        # print(real_labels)
        for j in range(real_images.shape[0]):
            saveImages_mutil(real_images[j], datasetName + '_real_imb', real_labels[j], bs * i + j)


if __name__ == '__main__':
    pass
