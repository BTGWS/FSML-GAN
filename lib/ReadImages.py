# coding:utf-8
# @TIME         : 2022/5/22 15:35 
# @Author       : BTG
# @Project      : FastGAN-pytorch-main
# @File Name    : ReadImages.py
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

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
# from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from lib.plotImage import plot_images
import torch.nn.functional as F

class ImageFolder(Dataset):
    """docstring for ArtDataset"""

    def __init__(self, root, transform=None, set_name='train'):
        super(ImageFolder, self).__init__()
        self.root = root

        # self.imagePathDict = self._load_ImagePath()
        self.frame = self._parse_frame()
        self.transform = transform

    # 加载图像路径， 字典形式，key是数字化的类别。
    def _load_ImagePath(self):
        imageClassPath = dict()
        imgClass_names = os.listdir(self.root)
        imgClass_names.sort()
        for className in range(len(imgClass_names)):
            imageClassPath[className] = []
            imageClass_path = '/'.join([self.root, imgClass_names[className]])
            for imgName in os.listdir(imageClass_path):
                if imgName[-4:] == '.jpg' or imgName[-4:] == '.png' or imgName[-5:] == '.jpeg':
                    imageClassPath[className].append(imgName)

        return imageClassPath

    # 加载图片
    def _load_Image(self, image_index):
        pass

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
        print(idx)
        file = self.frame[idx]
        img = Image.open(file).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img


policy = 'color,translation'
im_size = 1024
batch_size = 2
dataloader_workers = 1
transform_list = [
    transforms.Resize((int(im_size), int(im_size))),
    # transforms.RandomHorizontalFlip(),  # 随机图片水平翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]
trans = transforms.Compose(transform_list)
data_root = 'D:/BTG/fuck/桜井宁宁/Coser@桜井宁宁 Vol.001 小恶魔'
# data_root = 'E:/Datasets/mini-imagenet/data/images/n01532829'
# data_root = 'E:/Datasets/omniglot-py/images_background/Alphabet_of_the_Magi/character02'
# data_root = 'D:/BTG/fuck/桜井宁宁'
if __name__ == '__main__':
    dataset = ImageFolder(root=data_root, transform=trans)
    dataloader = iter(DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 # sampler=InfiniteSamplerWrapper(dataset),
                                 num_workers=dataloader_workers,
                                 pin_memory=True))
    print('dataloader length: ', len(dataloader))
    # for idx, (images, fuck) in enumerate(dataloader):
    #     print('idx: ', idx)
    #     print('images shape: ', images.shape)
    #     print('fuck: ', fuck)
    #     break
    real_image = next(dataloader)

    print('real_image shape: ', real_image.shape)
    plot_images(real_image, 'real_image', batch_size=batch_size)
    # fuck_resize = transforms.Resize(64)
    # new_real_image = fuck_resize(real_image)

    new_real_image = F.interpolate(real_image, size=64)
    plot_images(new_real_image, 'new_real_image', batch_size=batch_size)
    print('new_real_image shape: ', new_real_image.shape)

    # real_image = DiffAugment(real_image, policy=policy)
    # print(real_image.shape)
    # plot_images(real_image, 'real_image_up', batch_size=batch_size)
