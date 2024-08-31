# coding:utf-8
# @TIME         : 2022/7/4 22:14 
# @Author       : BTG
# @Project      : NBGAN
# @File Name    : NBGAN_PRO.py
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

from lib.imbalancel_fun.utils import mixup_data
from lib.plotImage import saveImages, plot_images, saveImages_mutil

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import random
import time
from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from torchvision import utils as vutils
from GAN_net.Discriminator import DiscriminatorPlus
from GAN_net.Encoder import EncoderNew as Encoder
# from GAN_net.Encoder import ResNetEmbedding as Encoder
from GAN_net.Generator import GeneratorPlus
from lib.loss_function import loss_fn as c_loss_fn, compute_FID, smooth_label
from lib.loss_function import compute_gradient_penalty, hinge_loss_g
from lib.trainGD import train_D_pro as train_D
from lib.loss_function import loss_fn_test
# from lib.getDatasets import OmniglotMetaEnv, MnistMetaEnv, miniImageNetMetaEnv
from lib.getDatasets import OmniglotMetaEnv, MnistMetaEnv, miniImageNetMetaEnv, VggFaceMetaEnv, YellowMetaEnv, \
    ReadImageMetaEnv, CIFAR10MetaEnv

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
criterionL1 = nn.L1Loss().to(device)

BCE_stable = nn.BCEWithLogitsLoss().to(device)

# Initialize weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight)
        # torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        # Estimated variance, must be around 1
        m.weight.data.normal_(1.0, 0.02)
        # Estimated mean, must be around 0
        m.bias.data.fill_(0)
    elif classname.find('ConvTranspose2d') != -1:
        torch.nn.init.xavier_normal_(m.weight)
        # torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()


class NBGAN_PRO:
    def __init__(self, args, load_path=None, testEnv=None):
        # 获取训练参数
        self.load_args(args)
        # 获取训练id
        self.id_string = self.get_id_string()
        # 创建检查点文件
        self.writer = SummaryWriter(self.osf + '/' + self.id_string)
        # 载入数据集环境
        if testEnv is None:
            # 正常训练使用时
            self.env = eval(self.dataset + 'MetaEnv(height=self.height, length=self.length)')
        else:
            # 测试时
            self.env = testEnv
        # 初始化gan
        self.initialize_gan()
        # 加载检查点
        self.load_checkpoint(load_path)

        self.lamda = 0.995
        self.lambda_gp = 10

    def load_args(self, args):
        """
        加载运行参数
        :param args: 参数字典
        :return:
        """
        self.outer_learning_rate = float(args['--outer_learning_rate'])
        self.inner_learning_rate = float(args['--inner_learning_rate'])
        self.batch_size = int(args['--batch_size'])
        self.inner_epochs = int(args['--inner_epochs'])
        self.height = int(args['--height'])
        self.length = int(args['--length'])
        self.dataset = args['--dataset']
        self.z_shape = int(args['--z_shape'])
        self.lms = float(args['--lambda_ms'])
        self.le = float(args['--lambda_encoder'])

        self.epochs = int(args['--all_epochs'])
        self.Proj_D = bool(int(args['--Proj_D']))
        self.eD = int(args['--epochs_D'])
        self.eG = int(args['--epochs_G'])
        self.osf = args['--outputSave_folder']

    def load_checkpoint(self, load_path=None):
        """
        加载检查点参数
        :return:
        """
        if load_path is None:
            path = self.id_string
        else:
            path = load_path
        if os.path.isfile(self.osf + '/' + path + '/checkpoint'):
            checkpoint = torch.load(self.osf + '/' + path + '/checkpoint')
            self.d.load_state_dict(checkpoint['discriminator'])
            self.g.load_state_dict(checkpoint['generator'])
            self.e.load_state_dict(checkpoint['encoder'])
            self.eps = checkpoint['episode']
            print("Loading model from episode: ", self.eps)
        else:
            self.eps = 0

    def checkpoint_model(self, saveName='checkpoint'):
        """
        保存检查点模型参数
        :return:
        """
        checkpoint = {
            'discriminator': self.d.state_dict(),
            'generator': self.g.state_dict(),
            'encoder': self.e.state_dict(),
            'episode': self.eps,
            'dataset': self.dataset}
        torch.save(checkpoint, self.osf + '/' + self.id_string + '/' + saveName)

    def get_id_string(self):
        """

        :return: 保存训练的id
        """

        base_str = '{}_olr{}_ilr{}_bsize{}_ie{}_h{}_l{}'

        return base_str.format(self.dataset,
                               str(self.outer_learning_rate),
                               str(self.inner_learning_rate),
                               str(self.batch_size),
                               str(self.inner_epochs),
                               str(self.height),
                               str(self.length))

    def initialize_gan(self):

        # D and G on CPU since they never do a feed forward operation

        self.g = GeneratorPlus(z_size=2 * self.z_shape, image_size=self.height, n_colors=self.env.channels)
        self.d = DiscriminatorPlus(image_size=self.height, n_colors=self.env.channels)
        # self.z_shape = 4 * self.z_shape

        self.e = Encoder(self.height, self.env.channels, outDim=self.z_shape)

        self.meta_g = GeneratorPlus(z_size=2 * self.z_shape, image_size=self.height, n_colors=self.env.channels).to(
            device)
        self.meta_d = DiscriminatorPlus(image_size=self.height, n_colors=self.env.channels).to(device)

        self.meta_e = Encoder(self.height, self.env.channels, outDim=self.z_shape).to(device)

        self.meta_g_optim = optim.Adam(params=self.meta_g.parameters(), lr=self.inner_learning_rate, betas=(0.5, 0.999))
        self.meta_d_optim = optim.Adam(params=self.meta_d.parameters(), lr=self.inner_learning_rate, betas=(0.5, 0.999))
        self.meta_e_optim = optim.Adam(params=self.meta_e.parameters(), lr=self.inner_learning_rate, betas=(0.5, 0.999))

        self.real_labels, self.fake_labels = smooth_label(self.batch_size, smoothing=0.05, device=device)


        decay = 0
        self.decayD = torch.optim.lr_scheduler.ExponentialLR(self.meta_d_optim, gamma=1 - decay)
        self.decayG = torch.optim.lr_scheduler.ExponentialLR(self.meta_g_optim, gamma=1 - decay)
        self.decayE = torch.optim.lr_scheduler.ExponentialLR(self.meta_e_optim, gamma=1 - decay)
        # 初始化
        self.meta_g.apply(weights_init)
        self.meta_d.apply(weights_init)
        self.meta_e.apply(weights_init)

    def reset_meta_model(self):
        """
        重设元生成器、元判别器、元编码器参数
        :return:
        """
        self.meta_g.train()
        self.meta_g.load_state_dict(self.g.state_dict())

        self.meta_d.train()
        self.meta_d.load_state_dict(self.d.state_dict())

        self.meta_e.train()
        self.meta_e.load_state_dict(self.e.state_dict())

    def inner_loop(self, real_batch):
        """
        # 内循环
        :param real_batch:  放入显存的单类别训练数据
        :return: 判别器，生成器，编码器的损失
        """
        x = real_batch
        for p in self.meta_d.parameters():
            p.requires_grad = True

        # 如果模型中用了dropout或bn，那么predict时必须使用eval
        # 不启用 BatchNormalization 和 Dropout
        self.meta_e.eval()

        # 更新判别器网络两次
        ################################################
        # (1) Update D network #
        ################################################
        for t in range(self.eD):
            # 判别器梯度置零
            self.meta_d.zero_grad(set_to_none=True)

            # 用编码器对输入进行编码
            enc_low, enc_x = self.meta_e(x)
            # 随即生成噪声z1, z2
            z1 = torch.randn((self.batch_size, self.z_shape), dtype=torch.float, device=device)
            z2 = torch.randn((self.batch_size, self.z_shape), dtype=torch.float, device=device)

            # 将生成的编码和两个噪声一起送入生成器中生成两个fake_image
            x_fake1, x_fake2 = self.meta_g(enc_x, z1), self.meta_g(enc_x, z2)


            fake1_feat_e, fake2_feat_e = self.meta_e(x_fake1)[1], self.meta_e(x_fake2)[1]

            errD, fuck = train_D(netD=self.meta_d,
                                 x_real=x,
                                 x_fake=[x_fake1, x_fake2],
                                 enc_x=[enc_x, fake1_feat_e, fake2_feat_e, enc_low],
                                 real_labels=self.real_labels,
                                 fake_labels=self.fake_labels,
                                 isCloss=True, isBCE=True, isHL=False, isGP=True, isConLoss=True)
            self.meta_d_optim.step()
            # torch.cuda.empty_cache()

        # Make it a tiny bit faster
        for p in self.meta_d.parameters():
            p.requires_grad = False

        ################################################
        # (2) Update G network #
        ################################################
        self.meta_e.train()
        # 更新生成器和编码器网络5次
        for t in range(self.eG):
            # 生成器和编码器梯度置零
            self.meta_g.zero_grad(set_to_none=True)
            self.meta_e.zero_grad(set_to_none=True)
            # 对原数据编码
            enc_low, enc_x = self.meta_e(x)

            z1 = torch.randn((self.batch_size, self.z_shape), dtype=torch.float, device=device)
            z2 = torch.randn((self.batch_size, self.z_shape), dtype=torch.float, device=device)
            x_fake1, x_fake2 = self.meta_g(enc_x, z1), self.meta_g(enc_x, z2)

            real_out, real_feat = self.meta_d(x)
            fake1_out, fake1_feat_d = self.meta_d(x_fake1)
            fake2_out, fake2_feat_d = self.meta_d(x_fake2)
            # errG1, errG2 = hinge_loss_g(fake1_out), hinge_loss_g(fake2_out)
            errG1 = (BCE_stable(real_out - torch.mean(fake1_out), self.fake_labels) + BCE_stable(
                fake1_out - torch.mean(real_out), self.real_labels)) / 2
            errG2 = (BCE_stable(real_out - torch.mean(fake2_out), self.fake_labels) + BCE_stable(
                fake2_out - torch.mean(real_out), self.real_labels)) / 2

            errG = errG1 + errG2

            #### Train encoder

            c_loss = loss_fn_test(real=x, fake=[x_fake1, x_fake2], real_feat=real_feat[0],
                                  fake_feat=[fake1_feat_d[0], fake2_feat_d[0]])

            l1_loss1 = criterionL1(x, torch.sigmoid(x_fake1))
            l1_loss2 = criterionL1(x, torch.sigmoid(x_fake2))

            errE = (l1_loss1 + l1_loss2) / 2

            # mode seeking loss
            # 将lz最大化，等价于将loss_lz最小化
            lz = torch.mean(torch.abs(x_fake2 - x_fake1)) / torch.mean(torch.abs(z2 - z1))
            eps = 1 * 1e-5
            loss_lz = 1 / (lz + eps)

            errG = errG + self.lms * loss_lz + 1 * self.le * errE + c_loss

            errG.backward()

            self.meta_g_optim.step()
            self.meta_e_optim.step()
            # torch.cuda.empty_cache()

            # 学习率更新
            self.decayD.step()
            self.decayG.step()
            self.decayE.step()

        return errD.item(), errG.item(), errE.item()

    def validation_run(self):
        data, task = self.env.sample_validation_task(self.batch_size)
        training_images = ((data - data.min()) / (data.max() - data.min())).cpu().numpy()
        training_images = np.concatenate([training_images[i] for i in range(self.batch_size)], axis=-1)
        real_batch = data.to(device)

        d_total_loss = 0
        g_total_loss = 0
        e_total_loss = 0

        for _ in range(self.inner_epochs):
            d_loss, g_loss, e_loss = self.inner_loop(real_batch)
            d_total_loss += d_loss
            g_total_loss += g_loss
            e_total_loss += e_loss
        # 不启用 BatchNormalization 和 Dropout
        self.meta_g.eval()
        with torch.no_grad():
            enc_x = torch.cat((self.meta_e(real_batch)[1], self.meta_e(real_batch)[1], self.meta_e(real_batch)[1]),
                              dim=0)
            z = torch.randn((self.batch_size * 3, self.z_shape), dtype=torch.float, device=device)
            # 生成图像
            img = self.meta_g(enc_x, z)
        img = img.cpu().numpy()
        img = ((img - img.min()) / (img.max() - img.min()))
        img = np.concatenate(
            [np.concatenate([img[i * 3 + j] for j in range(3)], axis=-2) for i in range(self.batch_size)], axis=-1)
        img = np.concatenate([training_images, img], axis=-2)
        self.writer.add_image('Validation_generated', img, self.eps)
        self.writer.add_scalar('Validation_d_loss', d_total_loss, self.eps)
        self.writer.add_scalar('Validation_g_loss', g_total_loss, self.eps)
        self.writer.add_scalar('Validation_e_loss', e_total_loss, self.eps)

        print("Episode: {:.2f}\tD Loss: {:.4f}\tG Loss: {:.4f}\tG Loss: {:.4f}".format(self.eps, d_total_loss,
                                                                                       g_total_loss, e_total_loss))

    def meta_training_loop(self, isWrite=True):
        data, task = self.env.sample_training_task(self.batch_size)
        real_batch = data.to(device)
        # task = random.sample(self.env.training_task, 1)[0]

        d_total_loss = 0
        g_total_loss = 0
        e_total_loss = 0

        # default：内循环10次
        for _ in range(self.inner_epochs):
            d_loss, g_loss, e_loss = self.inner_loop(real_batch)
            d_total_loss += d_loss
            g_total_loss += g_loss
            e_total_loss += e_loss
        # 写入d，g，e，训练损失
        print('Training_d_loss', d_total_loss, self.eps)
        print('Training_g_loss', g_total_loss, self.eps)
        print('Training_e_loss', e_total_loss, self.eps)
        if isWrite is True:
            self.writer.add_scalar('Training_d_loss', d_total_loss, self.eps)
            self.writer.add_scalar('Training_g_loss', g_total_loss, self.eps)
            self.writer.add_scalar('Training_e_loss', e_total_loss, self.eps)

        # Updating both generator and dicriminator
        # 更新生成器
        for p, meta_p in zip(self.g.parameters(), self.meta_g.parameters()):
            # diff = p - meta_p.cpu()
            # p.grad = diff
            p.data = p.data * self.lamda + meta_p.data.cpu() * (1 - self.lamda)
        # self.g_optim.step()
        # 更新判别器
        for p, meta_p in zip(self.d.parameters(), self.meta_d.parameters()):
            # diff = p - meta_p.cpu()
            # p.grad = diff
            p.data = p.data * self.lamda + meta_p.data.cpu() * (1 - self.lamda)
        # 更新编码器
        for p, meta_p in zip(self.e.parameters(), self.meta_e.parameters()):
            # diff = p - meta_p.cpu()
            # p.grad = diff
            p.data = p.data * self.lamda + meta_p.data.cpu() * (1 - self.lamda)
        # self.e_optim.step()

    def training(self):
        start1, end1 = 0, 0
        start2, end2 = 0, 0
        s, e = 0, 0

        # 100000
        torch.cuda.synchronize()
        s = time.time()
        while self.eps <= self.epochs:
            torch.cuda.synchronize()
            start1 = time.time()
            self.reset_meta_model()
            self.meta_training_loop()
            torch.cuda.synchronize()
            end1 = time.time()

            if self.eps % 100 == 0:
                self.checkpoint_model()
            # Validation run every 1000 training loop
            if self.eps % 1000 == 0:
                torch.cuda.synchronize()
                start2 = time.time()
                self.reset_meta_model()
                self.validation_run()
                self.checkpoint_model()
                torch.cuda.synchronize()
                end2 = time.time()

            if self.eps % 10000 == 0:
                self.checkpoint_model(saveName='checkpoint_' + str(self.eps))

            self.eps += 1
            print(("################### eps: {} ############################".format(self.eps)))
            print("once train: {:.2f}".format(end1 - start1))
            print("once validation_run: {:.2f}".format(end2 - start2))
            print('eps: {}, once eps time:　{:.2f}'.format(self.eps, (end1 - start1) + (end2 - start2)))
            print("#########################################################")
            start2, end2 = 0, 0
        torch.cuda.synchronize()
        e = time.time()

        print("all train: ", e - s)

    def testing(self, inner_epoch=None, taskIdx=None, info_dict=None):
        self.reset_meta_model()
        self.env.split_validation_and_training_task(task_idx=taskIdx)
        data, task = self.getData()
        training_images = ((data - data.min()) / (data.max() - data.min())).cpu().numpy()
        training_images = np.concatenate([training_images[i] for i in range(self.batch_size)], axis=-1)
        real_batch = data.to(device)

        d_total_loss = 0
        g_total_loss = 0
        e_total_loss = 0
        if inner_epoch is not None:
            inner_eps = inner_epoch[-1]
        else:
            inner_eps = self.inner_epochs
        # print(inner_eps)
        # info_dict['inner_epoch'] = inner_eps
        for iterNum in range(inner_eps + 1):
            y = torch.tensor(range(real_batch.shape[0])).to(device)
            real_batch_mix, _, _, _ = mixup_data(real_batch, y)
            d_loss, g_loss, e_loss = self.inner_loop(real_batch_mix)
            d_total_loss += d_loss
            g_total_loss += g_loss
            e_total_loss += e_loss
            if iterNum in inner_epoch:
                print(iterNum)

                img = self.createImages(real_batch)
                self.meta_g.train()
                name = info_dict['image_name']
                image_title = name.format(iterNum)
                imgTemp = np.concatenate([training_images, img], axis=-2)
                imgTemp = torch.tensor(imgTemp)
                plot_images(imgTemp, image_title, self.batch_size, info_dict=info_dict,
                            showImages=info_dict['showImages'])

        return None

    def createImages(self, real_batch):
        self.meta_g.eval()
        with torch.no_grad():
            e_o = self.meta_e(real_batch)[1]
            enc_x = torch.cat((e_o, e_o, e_o), dim=0)
            z = torch.randn((self.batch_size * 3, self.z_shape), dtype=torch.float, device=device)
            # 生成图像
            img = self.meta_g(enc_x, z)

            img = ((img - img.min()) / (img.max() - img.min())).cpu().numpy()
            img = np.concatenate(
                [np.concatenate([img[i * 3 + j] for j in range(3)], axis=-2) for i in range(self.batch_size)],
                axis=-1)

        return img

    def getData(self):
        if self.dataset in ['Mnist', 'Omniglot', 'CIFAR10']:
            data, task = self.env.sample_validation_task(self.batch_size)
        else:
            data, task = self.env.load_images(self.batch_size)
        return data, task

    def GenerateImages(self, inner_epoch=None, taskIdx=None, imagesNum=20):
        self.reset_meta_model()
        self.env.split_validation_and_training_task(task_idx=taskIdx)
        data, task = self.getData()
        real_batch = data.to(device)

        if inner_epoch is not None:
            inner_eps = inner_epoch
        else:
            inner_eps = self.inner_epochs
        from tqdm import tqdm
        for _ in tqdm(range(inner_eps), desc='training {}:'.format(taskIdx), position=0):
            y = torch.tensor(range(real_batch.shape[0])).to(device)
            real_mix, _, _, _ = mixup_data(real_batch, y)
            self.inner_loop(real_mix)
        self.meta_g.eval()
        self.meta_e.eval()
        data2, task = self.getData()
        real_batch2 = data2.to(device)

        # 不启用 BatchNormalization 和 Dropout
        imgs = None
        # x_tf = transforms.Resize((32, 32))
        with torch.no_grad():
            for i in range(imagesNum):
                enc_x = self.meta_e(real_batch2)[1]
                z = torch.randn((self.batch_size, self.z_shape), dtype=torch.float, device=device)
                # idx = torch.randperm(enc_x.shape[0])
                # z = enc_x[idx, :].view(enc_x.size())
                # 生成图像
                img = self.meta_g(enc_x, z)
                # img = x_tf(img)
                img = ((img - img.min()) / (img.max() - img.min()))

                if imgs is None:
                    imgs = img.clone()
                else:
                    imgs = torch.cat((imgs, img), dim=0)

            for j in range(imgs.shape[0]):
                from lib.plotImage import adjust_dynamic_range
                saveImages(adjust_dynamic_range(imgs[j]), self.dataset, taskIdx, j + 1)
                # saveImages_mutil(adjust_dynamic_range(imgs[j]), self.dataset + '_real_imb_up', taskIdx, j + 1)

    def GenerateImages_new(self, inner_epoch=None, taskIdx=None):
        self.reset_meta_model()
        self.env.split_validation_and_training_task(task_idx=taskIdx)
        data, task = self.getData()
        real_batch = data.to(device)
        if inner_epoch is not None:
            inner_eps = inner_epoch
        else:
            inner_eps = self.inner_epochs
        from tqdm import tqdm
        for _ in tqdm(range(inner_eps), desc='training {}:'.format(taskIdx), position=0):
            y = torch.tensor(range(real_batch.shape[0])).to(device)
            real_mix, _, _, _ = mixup_data(real_batch, y)
            self.inner_loop(real_mix)

        # 不启用 BatchNormalization 和 Dropout
        self.meta_g.eval()
        self.meta_e.eval()

        z1 = torch.randn((self.batch_size, self.env.channels, self.height, self.length), dtype=torch.float,
                         device=device)

        real_batch_aug = real_batch + z1
        with torch.no_grad():
            enc_x = self.meta_e(real_batch_aug)[1]
            z = torch.randn((self.batch_size, self.z_shape), dtype=torch.float, device=device)
            # 生成图像
            img = self.meta_g(enc_x, z)
        return img

    def testing2(self, inner_epoch=None, taskIdx=None, info_dict=None):
        self.reset_meta_model()
        self.env.split_validation_and_training_task(task_idx=taskIdx)
        if self.dataset in ['Mnist', 'Omniglot', 'CIFAR10']:
            data, task = self.env.sample_validation_task(self.batch_size)
        else:
            data, task = self.env.load_images(self.batch_size)

        training_images = ((data - data.min()) / (data.max() - data.min())).cpu().numpy()
        training_images = np.concatenate([training_images[i] for i in range(self.batch_size)], axis=-1)
        real_batch = data.to(device)

        if inner_epoch is None:
            inner_epoch = [self.inner_epochs]

        for inner_eps in inner_epoch:

            for iterNum in range(inner_eps):
                y = torch.tensor(range(real_batch.shape[0])).to(device)
                real_mix, _, _, _ = mixup_data(real_batch, y)
                d_loss, g_loss, e_loss = self.inner_loop(real_mix)

            if self.dataset in ['Mnist', 'Omniglot', 'CIFAR10']:
                data2, task = self.env.sample_validation_task(self.batch_size)
            else:
                data2, task = self.env.load_images(self.batch_size)

            print(inner_eps)

            z1 = torch.randn((self.batch_size, self.env.channels, self.height, self.length), dtype=torch.float,
                             device=device)

            real_batch_aug = real_batch + z1

            img = self.createImages(real_batch_aug)
            self.meta_g.train()
            name = info_dict['image_name']
            image_title = name.format(inner_eps)

            imgTemp = np.concatenate([training_images, img], axis=-2)
            imgTemp = torch.tensor(imgTemp)
            plot_images(imgTemp, image_title, self.batch_size, info_dict=info_dict,
                        showImages=info_dict['showImages'])
        return None
