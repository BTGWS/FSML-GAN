# coding:utf-8
# @TIME         : 2022/4/13 17:09 
# @Author       : BTG
# @Project      : NBGAN
# @File Name    : Args.py
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
    --inner_learning_rate=ilr   Learning rate of inner loop [default: 2e-4]
    --outer_learning_rate=olr   Learning rate of outer loop [default: 1e-5]
    --batch_size=bs             Size of task to train with [default: 4]
    --inner_epochs=ie           Amount of meta epochs in the inner loop [default: 10]
    --height=h                  Height of image [default: 64]
    --length=l                  Length of image [default: 64]
    --dataset=ds                Dataset name (Mnist, Omniglot, VggFace, miniImageNet) [default: Mnist]
    --z_shape=zs                Dimension of latent code z [default: 128]
    --lambda_ms=lms             Lambda parameter of mode seeking regularization term [default: 1]
    --lambda_encoder=le         Lambda parameter of encoder loss term [default: 1]

    --origin=org                The origin version of this loss  [default: 0]
    --moreCls=mc                Use different class  [default: 0]
    --only_contrastive_Loss=ocl Only use contrastive loss, don't use anymore  [default: 0]
    --hinge_loss=hl             Use hinge loss  [default: 0]
    --BCE_loss=bl               Use BCE loss  [default: 1]

    -h, --help                  Show this help message and exit
"""
