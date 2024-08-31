# coding:utf-8
# @TIME         : 2022/6/16 15:45 
# @Author       : BTG
# @Project      : NBGAN
# @File Name    : tt.py.py
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


# python 列表返回重复数据的下标
def searchRange(nums, target):
    """
    :type nums: Tensor
    :type target: int
    :rtype: Tensor or int
    """
    idx = (nums == target).nonzero().flatten()

    return idx


if __name__ == '__main__':
    # x = torch.tensor(3)
    x = 3

    test = torch.tensor([1, 2, 3, 3, 4, 5])
    res = searchRange(test, x)

    print('res: ', res)
