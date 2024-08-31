# coding:utf-8
# @TIME         : 2022/9/22 17:25 
# @Author       : BTG
# @Project      : NBGAN_test
# @File Name    : countfloderfnum.py
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

root = '/home/hadoop/Datasets'


# root = 'E:/Datasets/'
# np.random.seed(3407)


def count_file_num(dataset):
    path = '/'.join([root, dataset])
    classNames = sorted(os.listdir(path))
    print(classNames)
    files = []
    for className in sorted(classNames):
        class_path = '/'.join([path, className])
        a = os.listdir(class_path)
        # print(len(a))
        files.append(len(a))
    print(files)

    return classNames, files


if __name__ == '__main__':
    path = 'CIFAR10_real_imb_up'
    classNames, files = count_file_num(path)
    files_arr = np.array(files)
    files_arr_log = np.log(files_arr)
    f_mul = (files_arr_log.max() - files_arr_log) * files_arr
    f_sub = list(map(lambda x: 0 if f_mul[x] < files_arr[x] else int(f_mul[x] - files_arr[x]), range(len(files_arr))))
    f_need_iterNum = [(x // 5) for x in f_sub]
    print('f_need_iterNum: ', f_need_iterNum)
    lam = np.random.beta(1.0, 1.0)
    print(lam)
