import os 
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# import seaborn as sns
import pandas as pd
import numpy as np
import random

train_path = './AgriculturalDisease_trainingset/'
valid_path = './AgriculturalDisease_validationset/'


def genImage(gpath, datatype):

    

    if datatype == 'train':
        gen_number = 0  # 统计生成的图片数量
        if not os.path.exists(gpath+'delete'):
            os.makedirs(gpath+'delete')

        label = pd.read_csv(gpath + 'label.csv')
        label_gen_dict = {'img_path':[], 'label':[]}  # 生成图片label
        for i in range(61):
            li = label[label['label'] == i]
            imagenum = li['label'].count()
            print('第%d个，总共有有%d个图片'%(i, imagenum))
            imagelist = np.array(li['img_path']).tolist()
            img_path_gen, label_gen = [], []

            # for imagefile in imagelist:
            for aa in range(len(imagelist)):
                if aa <= 40:
                    print(aa)
                    path, imagename = os.path.split(imagelist[aa])

                    im = Image.open(imagelist[aa])
                    im = im.convert('RGB')
                    im_detail = im.transpose(Image.ROTATE_180)
                    # im_detail = im.filter(ImageFilter.DETAIL)  # 细节增强

                    img_path_gen.append(gpath + 'delete/' +'idetail_'+imagename)

                    label_gen.extend([int(i)])


                    im_detail.save(gpath + 'delete/' +'idetail_'+imagename)

                    gen_number += 1

            label_dict = {'img_path':img_path_gen, 'label':label_gen}
            label_gen_dict['img_path'].extend(img_path_gen)
            label_gen_dict['label'].extend(label_gen)
            label_gen_pd = pd.DataFrame(label_dict)
            # label = label.append(label_gen_pd)  # 将生成的图片label加入原先的label
            # label['label'] = label[['label']].astype('int64')  # 转化为int64
            # print(label)
        label_gen_p = pd.DataFrame(label_gen_dict)
        label_gen_p.to_csv(gpath + 'label_delete.csv', index=False)

        # label_gen_p = pd.DataFrame(label_gen_dict)
        # label_gen_p.to_csv(gpath + 'label_gen.csv', index=False)

        print('训练集总共生成%d个图片'%gen_number)


    if datatype == 'valid':
        gen_number = 0
        if not os.path.exists(gpath+'delete'):
            os.makedirs(gpath+'delete')
        label = pd.read_csv(gpath + 'label.csv')
        label_gen_dict = {'img_path':[], 'label':[]}
        for i in range(61):
            li = label[label['label'] == i]
            imagenum = li['label'].count()
            print('第%d个，总共有有%d个图片'%(i, imagenum))
            imagelist = np.array(li['img_path']).tolist()
            img_path_gen, label_gen = [], []
            # for imagefile in imagelist:
            for aa in range(len(imagelist)):
                if aa <= 20:
                    print(aa)
                    path, imagename = os.path.split(imagelist[aa])

                    im = Image.open(imagelist[aa])
                    im = im.convert('RGB')
                    im_detail = im.transpose(Image.ROTATE_180)

                    #im_detail = im.filter(ImageFilter.DETAIL)  # 细节增强

                    img_path_gen.append(gpath + 'delete/' + 'idetail_' + imagename)

                    label_gen.extend([int(i)])

                    im_detail.save(gpath + 'delete/' + 'idetail_' + imagename)

                    gen_number += 1

            label_dict = {'img_path': img_path_gen, 'label': label_gen}
            label_gen_dict['img_path'].extend(img_path_gen)
            label_gen_dict['label'].extend(label_gen)
            label_gen_pd = pd.DataFrame(label_dict)
            # label = label.append(label_gen_pd)  # 将生成的图片label加入原先的label
            # label['label'] = label[['label']].astype('int64')  # 转化为int64
            # print(label)
        label_gen_p = pd.DataFrame(label_gen_dict)

        label_gen_p.to_csv(gpath + 'label_delete.csv', index=False)

        print('验证集总共生成%d个图片'%gen_number)
if __name__ == '__main__':
    genImage(train_path, 'train')
    genImage(valid_path, 'valid')
