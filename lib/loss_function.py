# coding:utf-8
# @TIME         : 2022/3/26 21:12 
# @Author       : BTG
# @Project      : NBGAN
# @File Name    : loss_function.py
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
import time

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
# from lib.utils_FID.calc_FID_Func import load_inception_v3, extract_features, calc_fid_np, calc_fid_tensor_slow, \
#     calc_fid_tensor
from test_code.Index_of_duplicate_data_test import searchRange

cuda = True if torch.cuda.is_available() else False
Tensor = torch.FloatTensor
from torch.distributions import normal


# 生成器铰链损失
def hinge_loss_g(out_d_fake):
    """
    生成器铰链损失
    :param out_d_fake: D(G(z))
    :return: 损失结果 => Lg = -Ez~p(z) [D(G(z))]
    """
    loss = -torch.mean(out_d_fake)
    return loss


# 判别器铰链损失
def hinge_loss_d(out_d_real, out_d_fake):
    """
    判别器铰链损失 \n
    :param out_d_real: D(G(x))
    :param out_d_fake: D(G(z))
    :return: 损失结果 => Ld = - Ex~pdata [min(0, -1 + D(x))] - Ez~p(z) [min(0, -1 - D(G(z)))]
    """
    real_loss = torch.relu(1.0 - out_d_real)
    fake_loss = torch.relu(1.0 + out_d_fake)
    loss = torch.mean(real_loss + fake_loss)
    return loss


# 计算BCE
def culBCE(real, fake, real_labels, fake_labels):
    BCE_stable = nn.BCEWithLogitsLoss()
    result = (BCE_stable(real - torch.mean(fake), real_labels) + BCE_stable(fake - torch.mean(real), fake_labels)) / 2

    return result


# 计算交叉熵
def cross_entropy_loss_with_logits(labels, logits):
    logp = torch.log_softmax(logits, -1)
    # print('torch.multiply(labels, logp): \n', torch.multiply(labels, logp))
    loss = - torch.sum(torch.multiply(labels, logp), dim=-1)
    return loss


# loss fn
def loss_fn(x, y, flag=False):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    # x 和 y 越相似则结果越小
    if flag is False:
        result = 2 - 2 * (x * y).sum(dim=-1)
        # print('相似度：', result)
        return result.mean()
    # x 和 y 越相似则结果越大
    elif flag is True:
        result = 2 * (x * y).sum(dim=-1)
        # result = torch.arccos(result)
        # print('相似角度：', result)
        return result.mean()


# 没用上
def arccosLoss(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    temp = 2 * (x * y).sum(dim=-1)
    out = torch.arccos(temp)
    print('temp: ', temp)
    print('out: ', out)
    return out


def loss_fn_test(real, fake, real_feat, fake_feat):
    # 原图和生成图在特征上拉近，图像上拉远
    out1 = loss_fn(real_feat, fake_feat[0], flag=False)
    out2 = loss_fn(real_feat, fake_feat[1], flag=False)
    lz_1 = torch.mean(torch.abs(real - fake[0])) / out1
    lz_2 = torch.mean(torch.abs(real - fake[1])) / out2
    lz = lz_1 + lz_2
    # lz = (lz_1 + lz_2) / 2
    eps = 1 * 1e-5
    loss_lz = 1 / (lz + eps)
    return loss_lz


def loss_fn_test1(real, fake, real_feat, fake_feat):
    # 原图和生成图在特征上拉近，图像上拉远
    out1 = loss_fn(real_feat, fake_feat[0], flag=False)
    out2 = loss_fn(real_feat, fake_feat[1], flag=False)
    lz_1 = torch.mean(torch.abs(real - fake[0])) / out1
    lz_2 = torch.mean(torch.abs(real - fake[1])) / out2
    # lz = lz_1 + lz_2
    lz = (lz_1 + lz_2) / 2
    eps = 1 * 1e-5
    loss_lz = 1 / (lz + eps)
    return loss_lz


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))  # 超参数 温度
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())  # 主对角线为0，其余位置全为1的mask矩阵

    def forward(self, emb_i, emb_j):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)
        # print('z_i: \n', z_i)
        # print('z_j: \n', emb_j)

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim) 列不变，行增加
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)  # simi_mat: (2*bs, 2*bs)
        # print('similarity_matrix: \n', similarity_matrix)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs
        # print('positives: \n', positives)

        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)  # 2*bs, 2*bs

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


def Contrastive_loss(z_i, z_j, temperature=0.1, device='cuda'):
    """
    Calculates contrastive loss.\n
    :param device:
    :param z_i: 图像特征
    :param z_j:
    :param temperature: 温控超参数
    :return: 对比损失
    """

    image_feat = F.normalize(z_i, dim=-1)
    cond_feat = F.normalize(z_j, dim=-1)
    # print('image_feat: \n', image_feat)
    # print('cond_feat: \n', z_j)
    local_batch_size = image_feat.size(0)

    image_feat_large = image_feat
    cond_feat_large = cond_feat

    labels = F.one_hot(torch.arange(local_batch_size), local_batch_size).to(device)
    # print(labels)
    # print(image_feat.shape)
    logits_img2cond = torch.matmul(image_feat,
                                   cond_feat_large.permute(1, 0).contiguous()) / temperature
    logits_cond2img = torch.matmul(cond_feat,
                                   image_feat_large.permute(1, 0).contiguous()) / temperature
    # print('logits_img2cond: \n', logits_img2cond)

    loss_img2cond = cross_entropy_loss_with_logits(labels, logits_img2cond)
    loss_cond2img = cross_entropy_loss_with_logits(labels, logits_cond2img)
    # print('loss_img2cond: \n', loss_img2cond)

    # 图像到匹配对象的对比损失
    loss_img2cond = torch.mean(loss_img2cond)
    # 匹配对象到图像的对比损失
    loss_cond2img = torch.mean(loss_cond2img)
    print('CE_Contrastive_loss loss_img2cond: \n', loss_img2cond)
    print('CE_Contrastive_loss loss_cond2img: \n', loss_cond2img)
    loss = loss_img2cond + loss_cond2img

    return loss


# 计算原型对比损失
def Contrastive_loss_func(emb_a, emb_b, labels, temperature=0.1, device='cuda'):
    """
    Calculates contrastive loss.\n
    :param labels:
    :param device: 设备
    :param emb_a: 图像特征
    :param emb_b:
    :param temperature: 温控超参数
    :return: 对比损失
    """

    z_i = F.normalize(emb_a, dim=1)
    z_j = F.normalize(emb_b, dim=1)

    similarity_matrix = F.cosine_similarity(z_i.unsqueeze(1), z_j.unsqueeze(0), dim=2)
    positives_mask = torch.zeros(size=similarity_matrix.shape, dtype=torch.bool).to(device)
    positives_mask.scatter_(1, labels.unsqueeze(1), 1)

    negatives_mask = ~positives_mask
    positives = positives_mask * similarity_matrix

    nominator = torch.exp(torch.sum(positives, dim=1) / temperature)
    denominator = negatives_mask * torch.exp(similarity_matrix / temperature)

    loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
    loss_final = torch.sum(loss_partial) / (len(labels))
    return loss_final


def toOneHot(labels, rs):
    labels_one_hot = torch.zeros(size=rs.shape).to(rs.device)
    labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)

    return labels_one_hot


#  计算GP
def compute_gradient_penalty(D, real_samples, fake_samples):
    """
    Calculates the gradient penalty loss for WGAN GP \n
    :param D: 判别器模型
    :param real_samples: 真图
    :param fake_samples: 假图
    :return:
    """
    device = real_samples.device
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    # print('interpolates: ', interpolates.shape, interpolates.device)
    d_interpolates, _ = D(interpolates)
    d_interpolates = d_interpolates.view(real_samples.shape[0], 1)

    # print('d_interpolates: ', d_interpolates.shape, d_interpolates.device)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0).to(device), requires_grad=False)

    # Get gradient w.r.t. interpolates[
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    # print('gradients: ', gradients.shape, gradients.device)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    # print('gradient_penalty: ', gradient_penalty, gradient_penalty.shape, gradient_penalty.device)
    return gradient_penalty


# 计算FID 太慢
def compute_FID(real_images, fake_images, Device, flag='fast'):
    real = {}
    fake = {}
    inception = load_inception_v3().eval().to(Device)
    features_real = extract_features(real_images, inception, Device)  # (bs, 2048)
    features_fake = extract_features(fake_images, inception, Device)  # (bs, 2048)
    # print('features_real shape: ', features_real.shape)
    # print('features_fake shape: ', features_fake.shape)
    # print(f'extracted {features_real.shape[0]} features')
    # print(f'extracted {features_fake.shape[0]} features')

    real['real_mean'] = np.mean(features_real.cpu().numpy(), 0)
    real['real_cov'] = np.cov(features_real.cpu().numpy(), rowvar=False)
    # real['real_mean'] = torch.mean(features_real, 0)
    # real['real_cov'] = torch.cov(features_real.T)

    fake['fake_mean'] = np.mean(features_fake.cpu().numpy(), 0)
    fake['fake_cov'] = np.cov(features_fake.cpu().numpy(), rowvar=False)
    # fake['fake_mean'] = torch.mean(features_fake, 0)
    # fake['fake_cov'] = torch.cov(features_fake.T)
    # print(fake['fake_mean'], type(fake['fake_mean']), fake['fake_mean'].shape)
    # fid = calc_fid_np(real, fake)
    if flag == 'slow':
        fid = calc_fid_tensor_slow(real, fake)
    else:
        fid = calc_fid_tensor(real, fake)
    # print(' fid:', fid)
    return fid


# 未使用到
def update_lambda(eps, epochs: int, norm_idx: int = 0):
    if norm_idx == 2:
        temp = np.around(1 / np.power(epochs, 2) * (eps ** 2), 3)
        return min(temp, np.array(0.999))

    elif norm_idx == 1:
        temp = np.around(1 - 0.9 * np.exp(-eps / (epochs / np.log(eps + 10))), 3)
        return min(temp, np.array(0.999))

    else:
        return 0.99


def crop_image_by_part(image, part):
    """
    将原始图像分成四部分 \n
    :param image: 原始图像
    :param part: 不同部分（0：左上； 1：右上； 2：左下；3：右下）
    :return:
    """
    hw = image.shape[2] // 2
    if part == 0:
        return image[:, :, :hw, :hw]
    if part == 1:
        return image[:, :, :hw, hw:]
    if part == 2:
        return image[:, :, hw:, :hw]
    if part == 3:
        return image[:, :, hw:, hw:]


def compute_center_inBatch(real_feat, real_labels, center):
    for i in torch.unique(real_labels):
        index = searchRange(real_labels, i)
        res = torch.index_select(real_feat, 0, index)
        real_feat_sum = torch.mean(res, dim=0, keepdim=True)
        center_new = F.normalize(real_feat_sum, dim=1)
        # 可拓展到出现新类
        if center.shape[0] == 0 or (i not in range(center.shape[0])):
            center = torch.cat([center, center_new], dim=0)
        else:
            center[i] = 0.9 * center[i] + 0.1 * center_new

    return center


def compute_relation_score_fuck(relationNetwork, input_x, center):
    output = torch.zeros((input_x.shape[0], center.shape[0]), device=input_x.device)
    for i in range(len(input_x)):
        for j in range(len(center)):
            temp = torch.cat((input_x[i], center[j]), dim=0).view(-1, 4, 4)
            outTemp = relationNetwork(temp.unsqueeze(0)).view(-1)
            output[i][j] = outTemp

    return output


def compute_relation_score(relationNetwork, input_x, center):
    input_x = input_x.view(input_x.shape[0], -1, 4, 4)
    center = center.view(center.shape[0], -1, 4, 4)
    FEATURE_DIM = center.shape[1]
    CLASS_NUM = center.shape[0]
    input_x_re = input_x.unsqueeze(0).repeat(center.shape[0], 1, 1, 1, 1)
    center_re = center.unsqueeze(0).repeat(input_x.shape[0], 1, 1, 1, 1)
    input_x_re = torch.transpose(input_x_re, 0, 1)
    relation_pairs = torch.cat((center_re, input_x_re), dim=2).view(-1, 2 * FEATURE_DIM, 4, 4)
    relations = relationNetwork(relation_pairs).view(-1, CLASS_NUM)

    return relations


def balanced_softmax_loss(labels, logits, sample_per_class, reduction='mean'):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)  # 目标类概率
    loss = (1 - p.detach()) ** gamma * input_values
    return loss.mean()


class BalancedSoftmax(_Loss):
    """
    Balanced Softmax Loss
    """

    def __init__(self, sample_per_class):
        super(BalancedSoftmax, self).__init__()

        self.sample_per_class = torch.tensor(sample_per_class)

    def forward(self, input, label, reduction='mean'):
        return balanced_softmax_loss(label, input, self.sample_per_class, reduction)


class GCLLoss(nn.Module):

    def __init__(self, cls_num_list, m=0.5, weight=None, s=30, train_cls=False, noise_mul=1., gamma=0., device='cuda'):
        super(GCLLoss, self).__init__()
        cls_list = torch.FloatTensor(cls_num_list).to(device)
        m_list = torch.log(cls_list)
        m_list = m_list.max() - m_list
        self.m_list = m_list
        assert s > 0
        self.m = m
        self.s = s
        self.weight = weight
        self.simpler = normal.Normal(0, 1 / 3)
        self.train_cls = train_cls
        self.noise_mul = noise_mul
        self.gamma = gamma

    def forward(self, cosine, target):
        index = torch.zeros_like(cosine, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        noise = self.simpler.sample(cosine.shape).clamp(-1, 1).to(
            cosine.device)  # self.scale(torch.randn(cosine.shape).to(cosine.device))

        # cosine = cosine - self.noise_mul * noise/self.m_list.max() *self.m_list
        cosine = cosine - self.noise_mul * noise.abs() / self.m_list.max() * self.m_list
        output = torch.where(index, cosine - self.m, cosine)
        if self.train_cls:
            return focal_loss(F.cross_entropy(self.s * output, target, reduction='none', weight=self.weight),
                              self.gamma)
        else:
            return F.cross_entropy(self.s * output, target, weight=self.weight)


def smooth_label(batchSize: int, smoothing=0.0, device='cpu'):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    with torch.no_grad():
        true_label = torch.ones(batchSize, dtype=torch.float, device=device)
        fake_label = torch.zeros(batchSize, dtype=torch.float, device=device)
        true_label.fill_(confidence)
        fake_label.fill_(smoothing)

    return true_label, fake_label


if __name__ == '__main__':
    batch_size = 3
    dim = 3
    ContrastiveLoss_func = ContrastiveLoss(batch_size)

    emb_i = torch.rand(batch_size, dim).cuda()
    emb_j = torch.rand(batch_size, dim).cuda()
    # print('emb_i: ', emb_i)
    # print('emb_j: ', emb_j)
    s1 = time.time()
    loss_contra = ContrastiveLoss_func(emb_i, emb_j)
    e1 = time.time()
    print('ContrastiveLoss: ', loss_contra, e1 - s1)

    s2 = time.time()
    Contrastive_loss_func_ = Contrastive_loss(emb_i, emb_j)
    e2 = time.time()
    print('CE_Contrastive_loss: ', Contrastive_loss_func_, e2 - s2)

    s3 = time.time()
    o1 = loss_fn(emb_i, emb_j)
    e3 = time.time()
    o11 = loss_fn(emb_i, emb_j, flag=True)
    e4 = time.time()
    print('loss_fn1: ', o1, type(o1), e3 - s3)
    print('loss_fn2: ', o11, type(o11), e4 - e3)

    # o2 = loss_fn2(emb_i, emb_j)
    # print('loss_fn2: ', o2, type(o2))
    # print(o2.mean())
    a = np.array([[1, 2], [3, 4]])
    a1 = np.cov(a, rowvar=False)
    aa = torch.tensor(a, dtype=torch.float)
    aa = torch.from_numpy(a).type(torch.FloatTensor)
    aa1 = torch.cov(aa.T)

    print('a1: ', a1, type(a1))
    print('a: ', np.mean(a, 0))
    print('aa1: ', aa1, type(aa1))
    print('aa: ', torch.mean(aa, 0))
