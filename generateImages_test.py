# coding:utf-8
# @TIME         : 2022/8/5 15:36 
# @Author       : BTG
# @Project      : NBGAN
# @File Name    : generateImages_test.py
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
import numpy as np
import torch
from docopt import docopt

from NBGAN_PRO import NBGAN_PRO as NBGAN
from lib.getDatasets import ReadImageMetaEnv, CIFAR10MetaEnv
from lib.plotImage import plot_images, saveImages_mutil
from test_code.countfloderfnum import count_file_num
from util.Args_NEW import doc

infoDict = {}
if __name__ == '__main__':
    args = docopt(__doc__ + doc)
    print('test Dataset: ', args['--dataset'])
    # path = 'miniImageNet_olr1e-05_ilr0.0002_bsize4_ie10_h128_l128'

    # path = 'Mnist_olr1e-05_ilr0.0002_bsize4_ie10_h32_l32'
    # path = 'Omniglot_olr1e-05_ilr0.0002_bsize4_ie10_h32_l32'
    # path = 'd2g5_Omniglot_olr1e-05_ilr0.0002_bsize4_ie10_h32_l32'
    # path = '20220722_now_best_miniImageNet_olr1e-05_ilr0.0002_bsize4_ie10_h128_l128'
    path = '20220825_proplus_miniImageNet_olr1e-05_ilr0.0002_bsize4_ie10_h128_l128'

    infoDict['save_folderPath'] = args['--outputSave_folder'] + '/' + path
    infoDict['Dataset'] = args['--dataset']

    cc = torch.load(args['--outputSave_folder'] + '/' + path + '/checkpoint')
    print('true dataset: ', cc['dataset'])
    print('true eps: ', cc['episode'])
    # func()

    # DN = 'miniImageNet-sample'
    # DN = 'CIFAR10_real_imb_up1'
    DN = args['--dataset']
    # ========================================
    classNames, files = count_file_num(DN)
    files_arr = np.array(files)
    files_arr_log = np.log(files_arr)
    f_mul = (files_arr_log.max() - files_arr_log) * files_arr
    f_sub = list(map(lambda x: 0 if f_mul[x] < files_arr[x] else int(f_mul[x] - files_arr[x]), range(len(files_arr))))
    f_need_iterNum = [(x // 5) for x in f_sub]
    print('f_need_iterNum: ', f_need_iterNum)
    # ========================================

    # testEnv = VggFaceMetaEnv(height=args['--height'], length=args['--length'], modelType='TEST')
    # args['--dataset'] 'miniImageNet-sample'
    testEnv = ReadImageMetaEnv(image_size=int(args['--height']), set_name='val',
                               DatasetsName=DN)
    # testEnv = None
    env = NBGAN(args, load_path=path, testEnv=testEnv)
    from tqdm import tqdm

    # 方法1
    # for taskIdx in tqdm(range(20), desc='className', position=0):
    #     infoDict['taskIdx'] = int(taskIdx)
    #     img = env.GenerateImages(inner_epoch=80, taskIdx=int(taskIdx), imagesNum=5)
    # 方法2
    for taskIdx in tqdm(classNames, desc='className', position=0):
        if files[int(taskIdx)] >= max(files) or f_need_iterNum[int(taskIdx)] == 0:
            continue
        infoDict['taskIdx'] = taskIdx
        imgs = None
        # env.reset_meta_model()
        iters = 60
        for num in range(f_need_iterNum[int(taskIdx)]):
            # iters = int(iters // (np.log(num + 1) + 1))
            print('\n num: ', num, iters)
            img = env.GenerateImages_new(inner_epoch=iters, taskIdx=int(taskIdx))
            img = ((img - img.min()) / (img.max() - img.min()))
            plot_images(img, 'image_title', 5, info_dict=None, showImages=True)
            if imgs is None:
                imgs = img.clone()
            else:
                imgs = torch.cat((imgs, img), dim=0)
        for j in range(imgs.shape[0]):
            # x_tf = transforms.Resize((self.height, self.length))
            # x_input = F.interpolate(imgs[j], size=self.length)
            # x_input = x_tf(imgs[j])
            from lib.plotImage import adjust_dynamic_range

            # saveImages(adjust_dynamic_range(imgs[j]), self.dataset, taskIdx, j + 1)
            saveImages_mutil(adjust_dynamic_range(imgs[j]), DN + '_fuck', taskIdx, j + 1)
