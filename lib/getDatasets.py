# coding:utf-8
# @TIME         : 2022/3/26 21:07 
# @Author       : BTG
# @Project      : NBGAN
# @File Name    : getDatasets.py
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
import pickle as pkl
import random

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image

from lib.plotImage import plot_images

Datasets_root = 'E:/Datasets'


# Datasets_root = '/root/autodl-tmp/BTG/Datasets'

# Datasets_root = '/home/hadoop/Datasets'


# Datasets_root = '/home/t41/Datasets'


class MnistMetaEnv:
    # 初始化MnistMetaEnv
    def __init__(self, height=32, length=32):
        self.channels = 1
        self.height = height
        self.length = length
        self.data = datasets.MNIST(root=Datasets_root, train=True, download=False)
        self.make_tasks()
        self.split_validation_and_training_task()
        self.tf = transforms.Compose([
            transforms.Resize((self.height, self.length)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        # self.resize = transforms.Resize((self.height, self.length))
        # self.to_tensor = transforms.ToTensor()
        # self.norm = transforms.Normalize((0.5,), (0.5,))

    def make_tasks(self):
        # 存放每个类以及中的图像下标
        self.task_to_examples = {}
        # 保存所有类
        self.all_tasks = set()
        # 这里i为数据集中图像下标， _ 表示为图像， digit表示图像类别
        for i, (_, digit) in enumerate(self.data):
            self.all_tasks.update([digit])
            if str(digit) not in self.task_to_examples:
                self.task_to_examples[str(digit)] = []
            self.task_to_examples[str(digit)].append(i)

    def sample_training_task(self, batch_size=64):
        # 抽取训练样本任务
        task = str(random.sample(self.training_task, 1)[0])
        task_idx = random.sample(self.task_to_examples[task], batch_size)

        batch = torch.tensor(
            np.array([self.tf(self.data[idx][0]).numpy() for idx in task_idx]),
            dtype=torch.float)
        return batch, task

    def sample_validation_task(self, batch_size=64):
        # 获取验证集样本
        task = str(random.sample(self.validation_task, 1)[0])
        task_idx = random.sample(self.task_to_examples[task], batch_size)

        batch = torch.tensor(
            np.array([self.tf(self.data[idx][0]).numpy() for idx in task_idx]),
            dtype=torch.float)
        return batch, task

    def split_validation_and_training_task(self, task_idx=None):
        # 划分训练集和验证集
        # 将20个类分为验证集
        if task_idx is None:
            self.validation_task = set(random.sample(self.all_tasks, 1))
        else:
            self.validation_task = {task_idx}
        self.training_task = self.all_tasks - self.validation_task

    def sample_training_moreCls(self, bs=5):
        tasks = random.sample(self.training_task, bs)
        tasks_idx = [random.sample(self.task_to_examples[str(task)], 1)[0] for task in tasks]

        batch = torch.tensor(
            np.array([self.tf(self.data[idx][0]).numpy() for idx in tasks_idx]),
            dtype=torch.float)
        return batch, tasks

    def sample_validation_moreCls(self, bs=5):
        # 这里batch_size最大为256
        if bs > 256:
            bs = 256
            print('WARING: The batch_size maximum value is 20')
        tasks = random.sample(self.validation_task, bs)
        tasks_idx = [random.sample(self.task_to_examples[str(task)], 1)[0] for task in tasks]

        batch = torch.tensor(
            np.array([self.tf(self.data[idx][0]).numpy() for idx in tasks_idx]),
            dtype=torch.float)
        return batch, tasks


class OmniglotMetaEnv:
    # 初始化Omniglot
    def __init__(self, height=32, length=32):
        self.channels = 1
        self.height = height
        self.length = length
        self.data = datasets.Omniglot(root=Datasets_root, download=False)
        self.make_tasks()
        self.split_validation_and_training_task()
        self.tf = transforms.Compose([
            transforms.Resize((self.height, self.length)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def make_tasks(self):
        # 存放每个类以及中的图像下标
        self.task_to_examples = {}
        # 保存所有类
        self.all_tasks = set()
        # 这里i为数据集中图像下标， _ 表示为图像， digit表示图像类别
        for i, (_, digit) in enumerate(self.data):
            self.all_tasks.update([digit])
            if str(digit) not in self.task_to_examples:
                self.task_to_examples[str(digit)] = []
            self.task_to_examples[str(digit)].append(i)

    def sample_training_task(self, batch_size=4):
        # 抽取训练样本任务
        task = str(random.sample(self.training_task, 1)[0])
        task_idx = random.sample(self.task_to_examples[task], batch_size)

        batch = torch.tensor(
            np.array([self.tf(self.data[idx][0]).numpy() for idx in task_idx]),
            dtype=torch.float)
        return batch, task

    def sample_validation_task(self, batch_size=64):
        # 获取验证集样本
        task = str(random.sample(self.validation_task, 1)[0])
        task_idx = random.sample(self.task_to_examples[task], batch_size)

        batch = torch.tensor(
            np.array([self.tf(self.data[idx][0]).numpy() for idx in task_idx]),
            dtype=torch.float)
        return batch, task

    def split_validation_and_training_task(self, task_idx=None):
        # 划分训练集和验证集
        # 将20个类分为验证集
        if task_idx is None:
            self.validation_task = set(random.sample(self.all_tasks, 20))
        else:
            self.validation_task = {task_idx}

        self.training_task = self.all_tasks - self.validation_task

    def sample_training_moreCls(self, bs=5):
        tasks = random.sample(self.training_task, bs)
        tasks_idx = [random.sample(self.task_to_examples[str(task)], 1)[0] for task in tasks]

        batch = torch.tensor(
            np.array([self.tf(self.data[idx][0]).numpy() for idx in tasks_idx]),
            dtype=torch.float)
        return batch, tasks

    def sample_validation_moreCls(self, bs=5):
        # 这里batch_size最大为256
        if bs > 256:
            bs = 256
            print('WARING: The batch_size maximum value is 20')
        tasks = random.sample(self.validation_task, bs)
        tasks_idx = [random.sample(self.task_to_examples[str(task)], 1)[0] for task in tasks]

        batch = torch.tensor(
            np.array([self.tf(self.data[idx][0]).numpy() for idx in tasks_idx]),
            dtype=torch.float)
        return batch, tasks


class CIFAR10MetaEnv:
    # 初始化Omniglot
    def __init__(self, height=32, length=32):
        self.channels = 3
        self.height = height
        self.length = length
        self.data = datasets.CIFAR10(root=Datasets_root, download=False)
        self.make_tasks()
        self.split_validation_and_training_task()
        self.tf = transforms.Compose([
            transforms.Resize((self.height, self.length)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomGrayscale(0.2),  # 随机灰度
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                 std=[0.202, 0.199, 0.201])
        ])
        # self.resize = transforms.Resize((self.height, self.length))
        # self.to_tensor = transforms.ToTensor()
        # self.norm = transforms.Normalize((0.5,), (0.5,))

    def make_tasks(self):
        # 存放每个类以及中的图像下标
        self.task_to_examples = {}
        # 保存所有类
        self.all_tasks = set()
        # 这里i为数据集中图像下标， _ 表示为图像， digit表示图像类别
        for i, (_, digit) in enumerate(self.data):
            self.all_tasks.update([digit])
            if str(digit) not in self.task_to_examples:
                self.task_to_examples[str(digit)] = []
            self.task_to_examples[str(digit)].append(i)
        # print('task_to_examples: ', len(self.task_to_examples['9']))

    def sample_training_task(self, batch_size=4):
        # 抽取训练样本任务
        task = str(random.sample(self.training_task, 1)[0])
        task_idx = random.sample(self.task_to_examples[task], batch_size)

        batch = torch.tensor(
            np.array([self.tf(self.data[idx][0]).numpy() for idx in task_idx]),
            dtype=torch.float)
        return batch, task

    def sample_validation_task(self, batch_size=64):
        # 获取验证集样本
        task = str(random.sample(self.validation_task, 1)[0])
        task_idx = random.sample(self.task_to_examples[task], batch_size)

        batch = torch.tensor(
            np.array([self.tf(self.data[idx][0]).numpy() for idx in task_idx]),
            dtype=torch.float)
        return batch, task

    def split_validation_and_training_task(self, task_idx=None, val_num=2):
        # 划分训练集和验证集
        # 将2个类分为验证集
        if task_idx is None:
            self.validation_task = set(random.sample(self.all_tasks, val_num))
        else:
            self.validation_task = {task_idx}

        self.training_task = self.all_tasks - self.validation_task

    def sample_training_moreCls(self, bs=5):
        tasks = random.sample(self.training_task, bs)
        tasks_idx = [random.sample(self.task_to_examples[str(task)], 1)[0] for task in tasks]

        batch = torch.tensor(
            np.array([self.tf(self.data[idx][0]).numpy() for idx in tasks_idx]),
            dtype=torch.float)
        return batch, tasks

    def sample_validation_moreCls(self, bs=5):
        # 这里batch_size最大为256
        if bs > 256:
            bs = 256
            print('WARING: The batch_size maximum value is 20')
        tasks = random.sample(self.validation_task, bs)
        tasks_idx = [random.sample(self.task_to_examples[str(task)], 1)[0] for task in tasks]

        batch = torch.tensor(
            np.array([self.tf(self.data[idx][0]).numpy() for idx in tasks_idx]),
            dtype=torch.float)
        return batch, tasks


class miniImageNetMetaEnv(Dataset):
    def __init__(self, height=32, length=32):
        self.channels = 3
        self.height = height
        self.length = length
        self.tf = transforms.Compose([
            transforms.Resize((self.height, self.length)),
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataPath = Datasets_root + f"/miniimagenet/"

        if os.path.exists(self.dataPath + f"mini_tasks_{self.height}_norm_0point5.pkl"):
            self.tasks = pkl.load(open(self.dataPath + f"mini_tasks_{self.height}_norm_0point5.pkl", "rb"))
            self.validation_task = sorted(
                pkl.load(open(self.dataPath + f"mini_validation_tasks_{self.height}.pkl", "rb")))
            self.training_task = sorted(
                pkl.load(open(self.dataPath + f"mini_training_tasks_{self.height}.pkl", "rb")))
            print("Pickle data exits")
        else:
            print("Pickle data not found")
            self.tasks = self.get_tasks()
            self.all_tasks = set(self.tasks)
            self.split_validation_and_training_task()

    def get_tasks(self):

        tasks = dict()
        # path = '/home/hadoop/Datasets/mini-imagenet/data/images'
        path = Datasets_root + '/mini-imagenet/data/images'
        for task in os.listdir(path):
            tasks[task] = []
            task_path = '/'.join([path, task])
            for imgs in os.listdir(task_path):
                img = Image.open(os.path.join(task_path, imgs))
                tasks[task].append(np.array(self.tf(img)))
            tasks[task] = np.array(tasks[task])
            print(len(tasks))
        pkl.dump(tasks, open(self.dataPath + f"mini_tasks_{self.height}_norm_0point5.pkl", 'wb'))
        return tasks

    def split_validation_and_training_task(self):
        self.validation_task = set(random.sample(self.all_tasks, 20))
        self.training_task = self.all_tasks - self.validation_task

        pkl.dump(self.validation_task, open(self.dataPath + f"mini_validation_tasks_{self.height}.pkl", 'wb'))
        pkl.dump(self.training_task, open(self.dataPath + f"mini_training_tasks_{self.height}.pkl", 'wb'))

    def sample_training_task(self, bs=4, task=None):
        if task is None:
            task = random.sample(self.training_task, 1)[0]
        task_idx = random.sample([i for i in range(self.tasks[task].shape[0])], bs)
        batch = self.tasks[task][task_idx]
        batch = torch.tensor(batch, dtype=torch.float)
        return batch, task

    def sample_validation_task(self, bs=4, task=None):
        if task is None:
            task = random.sample(self.validation_task, 1)[0]
        task_idx = random.sample([i for i in range(self.tasks[task].shape[0])], bs)
        batch = self.tasks[task][task_idx]
        batch = torch.tensor(batch, dtype=torch.float)
        return batch, task

    def __getitem__(self, index):
        task = random.sample(self.training_task, 1)[0]
        task_idx = index
        batch = self.tasks[task][task_idx]
        batch = torch.tensor(batch, dtype=torch.float)
        return batch, task

    def __len__(self):
        return len(self.files)


class ReadImageMetaEnv(Dataset):
    def __init__(self, image_size, set_name='train', DatasetsName=None, val_task_idx=None, tfm=None):
        super(ReadImageMetaEnv, self).__init__()
        self.channels = 3
        self.image_size = image_size
        self.set_name = set_name
        self.datasetName = DatasetsName
        assert self.set_name == 'train' or self.set_name == 'val' or self.set_name == 'all', "Only have 'train' or 'val'."
        val_num = 20

        if tfm is None:
            self.tf = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.473, 0.449, 0.403],
                                     std=[0.234, 0.230, 0.230])
            ])
        else:
            self.tf = tfm
        if self.datasetName == 'mini-imagenet' or self.datasetName == 'miniImageNet':
                self.dataPath = Datasets_root + f"/mini-imagenet/data/images"
        elif self.datasetName == 'vggFace' or self.datasetName == 'VggFace':
            self.dataPath = Datasets_root + f"/vggFace/vgg_face_large"
        elif self.datasetName == 'Yellow' or self.datasetName == 'yellow':
            self.dataPath = f"D:/BTG/fuck/桜井宁宁"
        elif self.datasetName == 'miniImageNet-sample':
            self.dataPath = Datasets_root + f"/mini-imagenet/miniImageNet_real_sample"
        elif self.datasetName in ['cifar10_real_imb_up', 'CIFAR10_real_imb_up1']:
            val_num = 2
            self.dataPath = Datasets_root + f"/CIFAR10_real_imb_up1"
        elif self.datasetName in ['CIFAR10_real_4']:
            val_num = 2
            self.dataPath = Datasets_root + f"/CIFAR10_real_4"
        elif self.datasetName in ['VggFace_real_4']:
            self.dataPath = Datasets_root + f"/VggFace_real_4"
        elif self.datasetName in ['miniImageNet_real_4']:
            self.dataPath = Datasets_root + f"/miniImageNet_real_4"
        self.tasks = self._get_tasks()
        self.all_tasks = set(self.tasks)
        self.split_validation_and_training_task(val_task_idx, val_num)

    def _get_tasks(self):
        tasks = dict()
        self._imgClass_names = os.listdir(self.dataPath)
        self._imgClass_names.sort()

        for task in range(len(self._imgClass_names)):
            tasks[task] = []
            task_path = '/'.join([self.dataPath, self._imgClass_names[task]])
            for imgName in os.listdir(task_path):
                if imgName[-4:] == '.jpg' or imgName[-4:] == '.png' or imgName[-5:] == '.jpeg':
                    tasks[task].append(imgName)
            tasks[task] = np.array(tasks[task])
            # print(len(tasks))
        return tasks

    def split_validation_and_training_task(self, task_idx=None, val_num=20):
        # 划分训练集和验证集
        # 将20个类分为验证集
        if task_idx is None:
            self.validation_task = set(random.sample(self.all_tasks, val_num))
        else:
            self.validation_task = {task_idx}

        self.training_task = self.all_tasks - self.validation_task

    def _sample_training_task(self, bs=4, task=None):
        if task is None:
            task = random.sample(self.training_task, 1)[0]
        task_idx = random.sample([i for i in range(len(self.tasks[task]))], bs)
        batch = self.tasks[task][task_idx]
        # batch = torch.tensor(batch, dtype=torch.float)
        return batch, task

    def _sample_validation_task(self, bs=4, task=None):
        if task is None:
            task = random.sample(self.validation_task, 1)[0]
        task_idx = random.sample([i for i in range(len(self.tasks[task]))], bs)
        batch = self.tasks[task][task_idx]
        # batch = torch.tensor(batch, dtype=torch.float)
        return batch, task

    def load_images(self, bs=4, task=None, modelType=None):
        imgs = []
        taskName = 'training'
        if modelType is None:
            if self.set_name == 'train':
                taskName = 'training'
            elif self.set_name == 'val':
                taskName = 'validation'
        else:
            if modelType == 'train':
                taskName = 'training'
            elif modelType == 'val':
                taskName = 'validation'

        imageNames, imageClassNum = eval('self._sample_{}_task(bs, task)'.format(taskName))
        for imageName in imageNames:
            imagePath = '{}/{}/{}'.format(self.dataPath, self._imgClass_names[imageClassNum], imageName)
            try:
                img = Image.open(imagePath).convert('RGB')
                # img.show()
                imgs.append(self.tf(img))
            except:
                continue
        imgs = torch.stack(imgs, 0)
        return imgs, imageClassNum

    # 未完成
    def __getitem__(self, index):
        task_idx = index

        if self.set_name == 'train':
            print('train')
            batch = self.tasks[task_idx]
            print(batch, task_idx)

        elif self.set_name == 'val':
            print('val')
            batch = self.tasks[task_idx][task_idx]
            print(batch, task_idx)
        else:
            # print('all')
            task = random.sample(self.all_tasks, 2)
            # print(task)
            self.tasks = np.array(self.tasks)
            batch = self.tasks[task][task_idx]
            print(batch)
            # print(task_idx)

        return batch

        # batch = torch.tensor(batch, dtype=torch.float)

    def __len__(self):
        if self.set_name == 'train':
            return len(self.training_task)
        elif self.set_name == 'val':
            return len(self.validation_task)
        else:
            return len(self.all_tasks)


class VggFaceMetaEnv(Dataset):
    def __init__(self, height=32, length=32, modelType='TRAIN'):
        self.channels = 3
        self.height = height
        self.length = length
        self.tf = transforms.Compose([
            transforms.Resize((self.height, self.length)),
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # self.dataPath = f"./data/vgg/"
        # self.dataPath = Datasets_root + f"/miniimagenet/"     # hadoop
        self.dataPath = Datasets_root + f"/vgg/"  # hadoop

        if os.path.exists(self.dataPath + f"vgg_tasks_{self.height}_norm_0point5.pkl"):
            if modelType == 'TRAIN':
                self.tasks = pkl.load(open(self.dataPath + f"vgg_tasks_{self.height}_norm_0point5.pkl", "rb"))
            self.validation_task = sorted(
                pkl.load(open(self.dataPath + f"vgg_validation_tasks_{self.height}.pkl", "rb")))
            self.training_task = sorted(
                pkl.load(open(self.dataPath + f"vgg_training_tasks_{self.height}.pkl", "rb")))
            print("Pickle data exits")

        else:
            print("Pickle data not found")
            self.tasks = self.get_tasks()
            # self.tasks = pkl.load(open(self.dataPath + f"vgg_tasks_{self.height}_norm_0point5.pkl", "rb"))

            self.all_tasks = set(self.tasks)
            self.split_validation_and_training_task()

    def get_tasks(self):

        tasks = dict()

        path = Datasets_root + '/vggFace/vgg_face_large'
        for task in os.listdir(path):
            tasks[task] = []
            task_path = '/'.join([path, task])
            for imgs in os.listdir(task_path):
                try:
                    img = Image.open(os.path.join(task_path, imgs))
                    tasks[task].append(np.array(self.tf(img)))
                except:
                    continue
            tasks[task] = np.array(tasks[task])
            print(len(tasks))
        pkl.dump(tasks, open(self.dataPath + f"vgg_tasks_{self.height}_norm_0point5.pkl", 'wb'))
        return tasks

    def split_validation_and_training_task(self, task_idx=None, val_num=20):
        if task_idx is None:
            self.validation_task = set(random.sample(self.all_tasks, val_num))
            self.training_task = self.all_tasks - self.validation_task
            pkl.dump(self.validation_task, open(self.dataPath + f"vgg_validation_tasks_{self.height}.pkl", 'wb'))
            pkl.dump(self.training_task, open(self.dataPath + f"vgg_training_tasks_{self.height}.pkl", 'wb'))
        else:
            self.validation_task = {list(self.validation_task)[0]}
            self.tasks = pkl.load(open(self.dataPath + f"vgg_tasks_{self.height}_norm_0point5.pkl", "rb"))
            # print(self.validation_task)
            # fuck()
            # self.validation_task = {task_idx}

    def sample_training_task(self, bs=4, task=None):
        if task is None:
            task = random.sample(self.training_task, 1)[0]
        task_idx = random.sample([i for i in range(self.tasks[task].shape[0])], bs)
        batch = self.tasks[task][task_idx]
        batch = torch.tensor(batch, dtype=torch.float)
        return batch, task

    def sample_validation_task(self, bs=4, task=None):
        if task is None:
            task = random.sample(self.validation_task, 1)[0]
        task_idx = random.sample([i for i in range(self.tasks[task].shape[0])], bs)
        batch = self.tasks[task][task_idx]
        batch = torch.tensor(batch, dtype=torch.float)
        return batch, task

    def __len__(self):
        return len(self.files)


class YellowMetaEnv(Dataset):
    def __init__(self, height=128, length=128):
        self.channels = 3
        self.height = height
        self.length = length
        self.tf = transforms.Compose([
            transforms.Resize((self.height, self.length)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # self.dataPath = f"./data/miniimagenet/"
        self.dataPath = Datasets_root + f"/yellow_pkl/"

        if os.path.exists(self.dataPath + f"yellow_tasks_{self.height}_norm_0point5.pkl"):
            self.tasks = pkl.load(open(self.dataPath + f"yellow_tasks_{self.height}_norm_0point5.pkl", "rb"))
            self.validation_task = sorted(
                pkl.load(open(self.dataPath + f"yellow_validation_tasks_{self.height}.pkl", "rb")))
            self.training_task = sorted(
                pkl.load(open(self.dataPath + f"yellow_training_tasks_{self.height}.pkl", "rb")))
            print("Pickle data exits")
        else:
            print("Pickle data not found")
            self.tasks = self.get_tasks()
            self.all_tasks = set(self.tasks)
            self.split_validation_and_training_task()

    def get_tasks(self):

        tasks = dict()
        # path = '/home/hadoop/Datasets/yellow-imagenet/data/images'
        path = Datasets_root + '/yellow/'
        for task in os.listdir(path):
            tasks[task] = []
            task_path = '/'.join([path, task])
            for imgs in os.listdir(task_path):
                img = Image.open(os.path.join(task_path, imgs))
                tasks[task].append(np.array(self.tf(img)))
            tasks[task] = np.array(tasks[task])
            print(len(tasks))
        pkl.dump(tasks, open(self.dataPath + f"yellow_tasks_{self.height}_norm_0point5.pkl", 'wb'))
        return tasks

    def split_validation_and_training_task(self):
        self.validation_task = set(random.sample(self.all_tasks, 10))
        self.training_task = self.all_tasks - self.validation_task

        pkl.dump(self.validation_task, open(self.dataPath + f"yellow_validation_tasks_{self.height}.pkl", 'wb'))
        pkl.dump(self.training_task, open(self.dataPath + f"yellow_training_tasks_{self.height}.pkl", 'wb'))

    def sample_training_task(self, batch_size=4):
        task = random.sample(self.training_task, 1)[0]
        task_idx = random.sample([i for i in range(self.tasks[task].shape[0])], batch_size)
        batch = self.tasks[task][task_idx]
        batch = torch.tensor(batch, dtype=torch.float)
        return batch, task

    def sample_validation_task(self, batch_size=4):
        task = random.sample(self.validation_task, 1)[0]
        task_idx = random.sample([i for i in range(self.tasks[task].shape[0])], batch_size)
        batch = self.tasks[task][task_idx]
        batch = torch.tensor(batch, dtype=torch.float)
        return batch, task

    def __getitem__(self, index):
        task = random.sample(self.training_task, 1)[0]
        task_idx = index
        batch = self.tasks[task][task_idx]
        batch = torch.tensor(batch, dtype=torch.float)
        return batch, task

    def __len__(self):
        return len(self.files)


# 只能读取一个文件夹下的那种数据集
class imageFolder(Dataset):
    """docstring for ArtDataset"""

    def __init__(self, root, transform=None):
        super(imageFolder, self).__init__()
        self.root = root

        self.frame = self._parse_frame()
        self.transform = transform

    def _parse_frame(self):
        frame = []
        img_names = os.listdir(self.root)
        img_names.sort()

        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx]
        img = Image.open(file).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img


if __name__ == '__main__':
    img_size = 64
    batch_size = 2
    a = MnistMetaEnv(height=img_size, length=img_size)
    # a = OmniglotMetaEnv(height=he, length=le)
    # a = miniImageNetMetaEnv(height=he, length=le)
    # a = VggFaceMetaEnv(height=he, length=le)
    # a = YellowMetaEnv(height=img_size, length=img_size)

    # a = ReadImageMetaEnv(image_size=img_size, set_name='train', DatasetsName='mini-imagenet')
    # a = ReadImageMetaEnv(image_size=img_size, set_name='val', DatasetsName='vggFace')
    # a.split_validation_and_training_task(task_idx=9)
    # # print('val len: ', len(a))
    #
    # batch, classId = a.load_images(batch_size)
    # batch, classId = a.sample_validation_task(batch_size)
    # print(batch.shape, type(batch))
    # print(classId)
    # plot_images(batch, 'real_image', batch_size)
    # print('data len: ', len(a.tasks))
    # print('all_tasks len: ', a.all_tasks.__len__())
    # print('training_task len: ', a.training_task.__len__())
    # print('task_to_examples len: ', a.task_to_examples.__len__())
    # print(a.validation_task)

    tf = transforms.Compose([
        transforms.Resize((128, 128)),
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        # ([0.4730396, 0.44857216, 0.4027566],
        # [0.23441759, 0.23003672, 0.2302052])
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.473, 0.449, 0.403],
                             std=[0.234, 0.230, 0.230])
    ])
    # dataset_mini_imagenet = ImageFolder(root=Datasets_root + f"/mini-imagenet/data/images", transform=tf)
    dataset_vggFace = ImageFolder(root=Datasets_root + f"/vggFace/vgg_face_large", transform=tf)
    print('dataset len: ', len(dataset_vggFace))
    dataloader = DataLoader(dataset_vggFace,
                            batch_size=batch_size,
                            shuffle=True,
                            # sampler=InfiniteSamplerWrapper(dataset),
                            num_workers=2,
                            pin_memory=True)
    print('dataloader length: ', len(dataloader))
    image, label = iter(dataloader).next()
    print('dataset classNum: ', len(dataset_vggFace.classes))
    # print(dataset_vggFace.imgs)
    # print(dataset.class_to_idx)
    # print(dataset[0][0].size())
    #
    print('image shape: ', image.shape)
    print('label shape: ', label, label.shape)
    # plot_images(image, 'real_image', batch_size)
    # plot_images(image, 'fuck you', batch_size, info_dict=None)
    testEnv = ReadImageMetaEnv(image_size=128, set_name='val', DatasetsName='miniImageNet')
    # testEnv = ReadImageMetaEnv(image_size=128, set_name='val', DatasetsName='miniImageNet-sample')

    # print('testEnv val: ', len(testEnv))
    image, label = testEnv.load_images(batch_size)
    print('image shape: ', image.shape)
    print('label: ', label)
    plot_images(image, 'fuck you 0', batch_size, info_dict=None)
    x_tf = transforms.Resize((128, 128))
    x_input = x_tf(image)
    plot_images(x_input, 'fuck you 1', batch_size, info_dict=None)
    import torch.nn.functional as F

    x_input = F.interpolate(image, size=128)
    plot_images(x_input, 'fuck you 2', batch_size, info_dict=None)
