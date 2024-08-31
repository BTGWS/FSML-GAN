# coding:utf-8
# @TIME         : 2022/4/16 15:50 
# @Author       : BTG
# @Project      : NBGAN
# @File Name    : Args_NEW.py
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

doc = """
options:
    --inner_learning_rate=ilr   Learning rate of inner loop (2e-4) [default: 2e-4]
    --outer_learning_rate=olr   Learning rate of outer loop [default: 1e-5]
    --batch_size=bs             Size of task to train with [default: 4]
    --inner_epochs=ie           Amount of meta epochs in the inner loop [default: 10]
    --height=h                  Height of image [default: 128]
    --length=l                  Length of image [default: 128]
    --dataset=ds                Dataset name (Mnist, Omniglot, VggFace, miniImageNet, CIFAR10) [default: miniImageNet_real_4]
    --z_shape=zs                Dimension of latent code z [default: 128]
    --lambda_ms=lms             Lambda parameter of mode seeking regularization term [default: 2]
    --lambda_encoder=le         Lambda parameter of encoder loss term [default: 1]

    --outputSave_folder=osf     The save folder of output (Runs_NEW, Runs_NEW2) [default: Runs_NEW2]
    --all_epochs=epochs         Amount of meta epochs in the all loop [default: 50000]
    --Proj_D=PD                 If use proj on Discriminator [default: 1]
    --epochs_D=eD               Amount of Discriminator epochs in the meta training [default: 2]
    --epochs_G=eG               Amount of Generator epochs in the meta training [default: 5]

    -h, --help                  Show this help message and exit
"""
