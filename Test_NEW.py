# coding:utf-8
# @TIME         : 2022/4/17 15:28 
# @Author       : BTG
# @Project      : NBGAN
# @File Name    : Test_NEW.py
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
"""Testing script.
usage:
    Test.py [options]
"""
import time

import torch
from docopt import docopt

# from NBGAN_NEW import NBGAN_NEW as NBGAN
from NBGAN_PRO import NBGAN_PRO as NBGAN
from lib.plotImage import plot_images
from util.Args_NEW import doc
from lib.getDatasets import VggFaceMetaEnv, ReadImageMetaEnv, CIFAR10MetaEnv

infoDict = {}
if __name__ == '__main__':
    args = docopt(__doc__ + doc)
    print('test Dataset: ', args['--dataset'])

    # path = 'NewGAN_gp_e&dcloss_d1g2_Omniglot_olr1e-05_ilr0.0002_bsize4_ie10_h32_l32'
    # path = 'NewGAN_gp_e&dcloss_d2g5_Omniglot_olr1e-05_ilr0.0002_bsize4_ie10_h32_l32'
    # path = 'NewGAN_gp_e&dcloss_d2g5_Omniglot_olr1e-05_ilr0.0002_bsize4_ie10_h64_l64'
    # path = 'VggFace_olr1e-05_ilr0.0002_bsize4_ie10_h64_l64'
    # path = 'NewGAN_gp_e&dcloss_d1g2_miniImageNet_olr1e-05_ilr0.0002_bsize4_ie10_h128_l128'
    path = '20220722_now_best_miniImageNet_olr1e-05_ilr0.0002_bsize4_ie10_h128_l128'
    # path = '20220825_proplus_miniImageNet_olr1e-05_ilr0.0002_bsize4_ie10_h128_l128'
    # path = 'miniImageNet_olr1e-05_ilr0.0002_bsize4_ie10_h128_l128'
    # path = 'maybe_not_bad_20220720_miniImageNet_olr1e-05_ilr0.0002_bsize4_ie10_h128_l128'
    # path = 'not_good_miniImageNet_olr1e-05_ilr0.0002_bsize4_ie10_h128_l128'
    # path = 'maybe_continue_miniImageNet_olr1e-05_ilr0.0002_bsize4_ie10_h128_l128'
    # path = 'NewGAN_gp_e&dcloss_d2g5_miniImageNet_olr1e-05_ilr0.0002_bsize4_ie10_h128_l128'
    # path = 'orgGen_VggFace_olr1e-05_ilr0.0002_bsize4_ie10_h64_l64'
    # path = 'orgGen_miniImageNet_olr1e-05_ilr0.0002_bsize4_ie10_h128_l128'
    # path = 'VggFace_olr1e-05_ilr0.0002_bsize4_ie10_h64_l64'
    # path = 'Mnist_olr1e-05_ilr0.0002_bsize4_ie10_h32_l32'
    # path = 'Omniglot_olr1e-05_ilr0.0002_bsize4_ie10_h32_l32'
    # path = 'd2g5_Omniglot_olr1e-05_ilr0.0002_bsize4_ie10_h32_l32'

    infoDict['save_folderPath'] = args['--outputSave_folder'] + '/' + path
    infoDict['Dataset'] = args['--dataset']
    epoch_list = [40, 50, 60, 80]

    # epoch_list = [80]

    name = '{}_100000_miniImageNet'
    # name = '{}_30000_cifar10'
    # name = '{}_VggFace'
    # name = '{}_MNIST'
    # name = '{}_Omniglot'

    infoDict['image_name'] = name
    infoDict['showImages'] = True
    # datasetsName = 'miniImageNet-sample'
    datasetsName = args['--dataset']
    cc = torch.load(args['--outputSave_folder'] + '/' + path + '/checkpoint')
    print('true dataset: ', cc['dataset'])
    print('true eps: ', cc['episode'])
    # func()

    # testEnv = VggFaceMetaEnv(height=args['--height'], length=args['--length'], modelType='TEST')
    testEnv = ReadImageMetaEnv(image_size=int(args['--height']), set_name='val', DatasetsName=datasetsName)
    # testEnv = None
    env = NBGAN(args, load_path=path, testEnv=testEnv)
    for taskIdx in range(10):
        # for taskIdx in range(10, 21):
        infoDict['taskIdx'] = taskIdx
        env.testing(inner_epoch=epoch_list, taskIdx=taskIdx, info_dict=infoDict)
    # for taskIdx in range(10):
    #     # for taskIdx in range(10, 21):
    #     infoDict['taskIdx'] = taskIdx
    #     for inepoch in epoch_list:
    #         img = env.testing(inner_epoch=inepoch, taskIdx=taskIdx)
    #         # img = env.testing2(meta_epoch=inepoch, taskIdx=taskIdx)
    #
    #         image_title = name.format(inepoch)
    #         plot_images(img, image_title, env.batch_size, info_dict=infoDict, showImages=True)
