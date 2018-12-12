# crop size 448
# mean 109.973, 127.336, 123.883
import os
import cv2
import numpy as np
import torch
import random
from torch.utils.data import Dataset
import torch.utils.data as data
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image

# 默认使用PIL读图
def default_loader(path):
    # return Image.open(path)
    return Image.open(path).convert('RGB')

# 数据增强：在给定角度中随机进行旋转
class FixedRotation(object):
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        return fixed_rotate(img, self.angles)

def fixed_rotate(img, angles):
    angles = list(angles)
    angles_num = len(angles)
    index = random.randint(0, angles_num - 1)
    return img.rotate(angles[index])


# 训练集图片读取
class TrainDataset(Dataset):
    def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        for index, row in label_list.iterrows():
            imgs.append((row['img_path'], row['label']))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        filename, label = self.imgs[index]
        img = self.loader(filename)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

# 验证集图片读取
class ValDataset(Dataset):
    def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        for index, row in label_list.iterrows():
            imgs.append((row['img_path'], row['label']))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        filename, label = self.imgs[index]
        img = self.loader(filename)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

# 测试集图片读取
class TestDataset(Dataset):
    def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        for index, row in label_list.iterrows():
            imgs.append((row['img_path']))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = self.loader(filename)
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)

def CUB_collate(batch):
    imgs = []
    cls = []
    for sample in batch:
        imgs.append(sample[0])
        cls.append(sample[1])
    imgs = torch.stack(imgs, 0)
    cls = torch.LongTensor(cls)
    return imgs, cls

# if __name__ == '__main__':
#
#     # 随机种子
#     np.random.seed(666)
#     torch.manual_seed(666)
#     torch.cuda.manual_seed_all(666)
#     random.seed(666)
#
#     std = 1. / 255.
#     means = [109.97 / 255., 127.34 / 255., 123.88 / 255.]
#
#     train_data_list = pd.read_csv('./AgriculturalDisease_trainingset/label_Tomato_all.csv')
#     #
#     val_data_list = pd.read_csv('./AgriculturalDisease_validationset/label_Tomato_all.csv')
#     # train_data_list = pd.read_csv('./AgriculturalDisease_trainingset/label_delete.csv')
#
#     # val_data_list = pd.read_csv('./AgriculturalDisease_validationset/label_delete.csv')
#     # 读取测试图片列表
#     test_data_list = pd.read_csv('./AgriculturalDisease_testA/test.csv')
#
#     # 图片归一化，由于采用ImageNet预训练网络，因此这里直接采用ImageNet网络的参数
#     # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#
#     # 训练集图片变换，输入网络的尺寸为384*384
#     train_data = TrainDataset(train_data_list,
#                                     transform=transforms.Compose([
#                                         transforms.Resize((384, 384)),
#                                         transforms.ColorJitter(0.15, 0.15, 0.15, 0.075),
#                                         transforms.RandomHorizontalFlip(p=0.5),
#                                         transforms.RandomGrayscale(p=0.1),
#                                         transforms.RandomVerticalFlip(p=0.5),
#                                         # transforms.RandomRotation(20),
#                                         FixedRotation([0, 90, 180, 270]),
#                                         transforms.RandomCrop(384),
#                                         transforms.ToTensor(),
#                                         normalize,
#                                     ]))
#
#     # 验证集图片变换
#     val_data = ValDataset(val_data_list,
#                           transform=transforms.Compose([
#                               transforms.Resize((384, 384)),
#                               # transforms.ColorJitter(0.15, 0.15, 0.15, 0.075),
#                               # transforms.RandomHorizontalFlip(p=0.5),
#                               # transforms.RandomGrayscale(p=0.1),
#                               # # transforms.RandomRotation(20),
#                               # FixedRotation([0, 90, 180, 270]),
#                               transforms.CenterCrop(384),
#                               transforms.ToTensor(),
#                               normalize,
#                           ]))
#
#     # 测试集图片变换
#     test_data = TestDataset(test_data_list,
#                             transform=transforms.Compose([
#                                 transforms.Resize((384, 384)),
#                                 transforms.CenterCrop(384),
#                                 transforms.ToTensor(),
#                                 normalize,
#                             ]))
#     # trainset = CUB200_loader(os.getcwd() + '/CUB_200_2011')
#     trainloader = data.DataLoader(train_data, batch_size = 32,
#             shuffle = False, collate_fn = CUB_collate, num_workers = 1)
#     for img, cls in trainloader:
#         print(' [*] train images:', img.size())
#         print(' [*] train class:', cls.size())
#         break
#
#     # testset = CUB200_loader(os.getcwd() + '/CUB_200_2011')
#     valloader = data.DataLoader(val_data, batch_size = 32,
#             shuffle = False, collate_fn = CUB_collate, num_workers = 1)
#
#     for img, cls in valloader:
#         # print(img)
#         print(' [*] test images:', img.size())
#         print(' [*] test class:', cls.size())
#         break
