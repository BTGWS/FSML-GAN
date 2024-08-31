# coding:utf-8
# @TIME         : 2022/4/2 17:22 
# @Author       : BTG
# @Project      : NBGAN
# @File Name    : Train_NEW.py
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
from docopt import docopt

# from NBGAN_NEW import NBGAN_NEW as NBGAN
from NBGAN_PRO import NBGAN_PRO as NBGAN
from util.Args_NEW import doc

if __name__ == '__main__':
    args = docopt(__doc__ + doc)
    env = NBGAN(args)

    env.training()
