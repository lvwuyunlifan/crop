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
# acc < 60
accuracy_lower = [2, 8, 11, 13, 22, 42, 44, 45, 53, 57]

def genImage(gpath, datatype):

    

    if datatype == 'train':
        gen_number = 0  # 统计生成的图片数量
        if not os.path.exists(gpath+'acc_gen'):
            os.makedirs(gpath+'acc_gen')

        label = pd.read_csv(gpath + 'label.csv')
        label_g = pd.read_csv(gpath + 'label_all.csv')
        label_gen_dict = {'img_path':[], 'label':[]}  # 生成图片label
        for i in accuracy_lower:
            li = label[label['label'] == i]
            imagenum = li['label'].count()
            print('第%d个，总共有有%d个图片'%(i, imagenum))
            imagelist = np.array(li['img_path']).tolist()
            img_path_gen, label_gen = [], []

            if i == 22:
                for imagefile in imagelist:
                    path, imagename = os.path.split(imagefile)
                    im = Image.open(imagefile)
                    im = im.convert('RGB')
        #             im_blur = im.filter(ImageFilter.GaussianBlur)  # 模糊化
        #             im_sharp = im.filter(ImageFilter.UnsharpMask)  # 锐化增强
                    # im_detail = im.filter(ImageFilter.DETAIL)  # 细节增强
        #             im_smooth = im.filter(ImageFilter.SMOOTH)  # 平滑滤波
                    im_copy = im.copy()
                    im_retate = im.rotate(60)
                    im_enhance = im.filter(ImageFilter.EDGE_ENHANCE)  # 边缘增强滤波

        #             img_path_gen.append(gpath + 'acc_gen/' + 'iblur_'+imagename)
        #             img_path_gen.append(gpath + 'acc_gen/' + 'isharp_'+imagename)
        #             img_path_gen.append(gpath + 'acc_gen/' + 'idetail_'+imagename)
        #             img_path_gen.append(gpath + 'acc_gen/' + 'ismooth_'+imagename)
                    img_path_gen.append(gpath + 'acc_gen/' + 'icopy_' + imagename)
                    img_path_gen.append(gpath + 'acc_gen/' + 'irotate_'+imagename)
                    img_path_gen.append(gpath + 'acc_gen/' + 'ienhance_'+imagename)

                    label_gen.extend([int(i), int(i), int(i)])



        #             im_blur.save(gpath + 'acc_gen/' + 'iblur_'+imagename)
        #             im_sharp.save(gpath + 'acc_gen/' + 'isharp_'+imagename)
        #             im_detail.save(gpath + 'acc_gen/' + 'idetail_'+imagename)
                    im_copy.save(gpath + 'acc_gen/' + 'icopy_'+imagename)
                    im_retate.save(gpath + 'acc_gen/' + 'irotate_'+imagename)
                    im_enhance.save(gpath + 'acc_gen/' + 'ienhance_'+imagename)
                    gen_number += 3

            elif i == 44 or i == 45:
                for imagefile in imagelist:
                    path, imagename = os.path.split(imagefile)
                    im = Image.open(imagefile)
                    im = im.convert('RGB')
                    #             im_blur = im.filter(ImageFilter.GaussianBlur)  # 模糊化
                    #             im_sharp = im.filter(ImageFilter.UnsharpMask)  # 锐化增强
                    # im_detail = im.filter(ImageFilter.DETAIL)  # 细节增强
                    #             im_smooth = im.filter(ImageFilter.SMOOTH)  # 平滑滤波
                    im_copy = im.copy()
                    # im_retate = im.rotate(60)
                    im_enhance = im.filter(ImageFilter.EDGE_ENHANCE)  # 边缘增强滤波

                    #             img_path_gen.append(gpath + 'acc_gen/' + 'iblur_'+imagename)
                    #             img_path_gen.append(gpath + 'acc_gen/' + 'isharp_'+imagename)
                    #             img_path_gen.append(gpath + 'acc_gen/' + 'idetail_'+imagename)
                    #             img_path_gen.append(gpath + 'acc_gen/' + 'ismooth_'+imagename)
                    img_path_gen.append(gpath + 'acc_gen/' + 'icopy_' + imagename)
                    # img_path_gen.append(gpath + 'geacc_gen/' + 'irotate_' + imagename)
                    img_path_gen.append(gpath + 'acc_gen/' + 'ienhance_' + imagename)

                    label_gen.extend([int(i), int(i)])

                    #             im_blur.save(gpath + 'acc_gen/' + 'iblur_'+imagename)
                    #             im_sharp.save(gpath + 'acc_gen/' + 'isharp_'+imagename)
                    #             im_detail.save(gpath + 'acc_gen/' + 'idetail_'+imagename)
                    im_copy.save(gpath + 'acc_gen/' + 'icopy_' + imagename)
                    # im_retate.save(gpath + 'acc_gen/' + 'irotate_' + imagename)
                    im_enhance.save(gpath + 'acc_gen/' + 'ienhance_' + imagename)
                    gen_number += 2

            else:

                for imagefile in imagelist:
                    path, imagename = os.path.split(imagefile)
                    im = Image.open(imagefile)
                    im = im.convert('RGB')
                    #             im_blur = im.filter(ImageFilter.GaussianBlur)  # 模糊化
                    #             im_sharp = im.filter(ImageFilter.UnsharpMask)  # 锐化增强
                    # im_detail = im.filter(ImageFilter.DETAIL)  # 细节增强
                    #             im_smooth = im.filter(ImageFilter.SMOOTH)  # 平滑滤波
                    #             im_retate = im.rotate(30)
                    im_enhance = im.filter(ImageFilter.EDGE_ENHANCE)  # 边缘增强滤波

                    #             img_path_gen.append(gpath + 'acc_gen/' + 'iblur_'+imagename)
                    #             img_path_gen.append(gpath + 'acc_gen/' + 'isharp_'+imagename)
                    #             img_path_gen.append(gpath + 'acc_gen/' + 'idetail_'+imagename)
                    #             img_path_gen.append(gpath + 'acc_gen/' + 'ismooth_'+imagename)
                    #             img_path_gen.append(gpath + 'geacc_gen/' + 'irotate_'+imagename)
                    img_path_gen.append(gpath + 'acc_gen/' + 'ienhance_' + imagename)

                    label_gen.extend([int(i)])

                    #             im_blur.save(gpath + 'acc_gen/' + 'iblur_'+imagename)
                    #             im_sharp.save(gpath + 'acc_gen/' + 'isharp_'+imagename)
                    #             im_detail.save(gpath + 'acc_gen/' + 'idetail_'+imagename)
                    #             im_smooth.save(gpath + 'acc_gen/' + 'ismooth_'+imagename)
                    im_enhance.save(gpath + 'acc_gen/' + 'ienhance_' + imagename)
                    gen_number += 1


            label_dict = {'img_path':img_path_gen, 'label':label_gen}
            label_gen_dict['img_path'].extend(img_path_gen)
            label_gen_dict['label'].extend(label_gen)

            label_gen_pd = pd.DataFrame(label_dict)

            label = label.append(label_gen_pd)
            label['label'] = label[['label']].astype('int64')

            label_g = label_g.append(label_gen_pd)
            label_g['label'] = label_g[['label']].astype('int64')

        label_g.to_csv(gpath + 'label_acc_gen_all.csv', index=False)  # acc+gen+label
        label.to_csv(gpath + 'label_acc_all.csv', index=False)  # acc+label

        label_gen_p = pd.DataFrame(label_gen_dict)
        label_gen_p.to_csv(gpath + 'label_acc.csv', index=False)  # acc

        print('训练集总共生成%d个图片'%gen_number)


    if datatype == 'valid':
        gen_number = 0
        if not os.path.exists(gpath+'acc_gen'):
            os.makedirs(gpath+'acc_gen')
        label = pd.read_csv(gpath + 'label.csv')
        label_g = pd.read_csv(gpath + 'label_all.csv')
        label_gen_dict = {'img_path':[], 'label':[]}
        for i in accuracy_lower:
            li = label[label['label'] == i]
            imagenum = li['label'].count()
            print('第%d个，总共有有%d个图片'%(i, imagenum))
            imagelist = np.array(li['img_path']).tolist()
            img_path_gen, label_gen = [], []

            if i == 22:
                for imagefile in imagelist:
                    path, imagename = os.path.split(imagefile)
                    im = Image.open(imagefile)
                    im = im.convert('RGB')
                    #             im_blur = im.filter(ImageFilter.GaussianBlur)  # 模糊化
                    #             im_sharp = im.filter(ImageFilter.UnsharpMask)  # 锐化增强
                    # im_detail = im.filter(ImageFilter.DETAIL)  # 细节增强
                    #             im_smooth = im.filter(ImageFilter.SMOOTH)  # 平滑滤波
                    im_copy = im.copy()
                    im_retate = im.rotate(60)
                    im_enhance = im.filter(ImageFilter.EDGE_ENHANCE)  # 边缘增强滤波

                    #             img_path_gen.append(gpath + 'acc_gen/' + 'iblur_'+imagename)
                    #             img_path_gen.append(gpath + 'acc_gen/' + 'isharp_'+imagename)
                    #             img_path_gen.append(gpath + 'acc_gen/' + 'idetail_'+imagename)
                    #             img_path_gen.append(gpath + 'acc_gen/' + 'ismooth_'+imagename)
                    img_path_gen.append(gpath + 'acc_gen/' + 'icopy_' + imagename)
                    img_path_gen.append(gpath + 'acc_gen/' + 'irotate_' + imagename)
                    img_path_gen.append(gpath + 'acc_gen/' + 'ienhance_' + imagename)

                    label_gen.extend([int(i), int(i), int(i)])

                    #             im_blur.save(gpath + 'acc_gen/' + 'iblur_'+imagename)
                    #             im_sharp.save(gpath + 'acc_gen/' + 'isharp_'+imagename)
                    #             im_detail.save(gpath + 'acc_gen/' + 'idetail_'+imagename)
                    im_copy.save(gpath + 'acc_gen/' + 'icopy_' + imagename)
                    im_retate.save(gpath + 'acc_gen/' + 'irotate_' + imagename)
                    im_enhance.save(gpath + 'acc_gen/' + 'ienhance_' + imagename)
                    gen_number += 3

            elif i == 44 or i == 45:
                for imagefile in imagelist:
                    path, imagename = os.path.split(imagefile)
                    im = Image.open(imagefile)
                    im = im.convert('RGB')
                    #             im_blur = im.filter(ImageFilter.GaussianBlur)  # 模糊化
                    #             im_sharp = im.filter(ImageFilter.UnsharpMask)  # 锐化增强
                    # im_detail = im.filter(ImageFilter.DETAIL)  # 细节增强
                    #             im_smooth = im.filter(ImageFilter.SMOOTH)  # 平滑滤波
                    im_copy = im.copy()
                    # im_retate = im.rotate(60)
                    im_enhance = im.filter(ImageFilter.EDGE_ENHANCE)  # 边缘增强滤波

                    #             img_path_gen.append(gpath + 'acc_gen/' + 'iblur_'+imagename)
                    #             img_path_gen.append(gpath + 'acc_gen/' + 'isharp_'+imagename)
                    #             img_path_gen.append(gpath + 'acc_gen/' + 'idetail_'+imagename)
                    #             img_path_gen.append(gpath + 'acc_gen/' + 'ismooth_'+imagename)
                    img_path_gen.append(gpath + 'acc_gen/' + 'icopy_' + imagename)
                    # img_path_gen.append(gpath + 'geacc_gen/' + 'irotate_' + imagename)
                    img_path_gen.append(gpath + 'acc_gen/' + 'ienhance_' + imagename)

                    label_gen.extend([int(i), int(i)])

                    #             im_blur.save(gpath + 'acc_gen/' + 'iblur_'+imagename)
                    #             im_sharp.save(gpath + 'acc_gen/' + 'isharp_'+imagename)
                    #             im_detail.save(gpath + 'acc_gen/' + 'idetail_'+imagename)
                    im_copy.save(gpath + 'acc_gen/' + 'icopy_' + imagename)
                    # im_retate.save(gpath + 'acc_gen/' + 'irotate_' + imagename)
                    im_enhance.save(gpath + 'acc_gen/' + 'ienhance_' + imagename)
                    gen_number += 2

            else:

                for imagefile in imagelist:
                    path, imagename = os.path.split(imagefile)
                    im = Image.open(imagefile)
                    im = im.convert('RGB')
                    #             im_blur = im.filter(ImageFilter.GaussianBlur)  # 模糊化
                    #             im_sharp = im.filter(ImageFilter.UnsharpMask)  # 锐化增强
                    # im_detail = im.filter(ImageFilter.DETAIL)  # 细节增强
                    #             im_smooth = im.filter(ImageFilter.SMOOTH)  # 平滑滤波
                    #             im_retate = im.rotate(30)
                    im_enhance = im.filter(ImageFilter.EDGE_ENHANCE)  # 边缘增强滤波

                    #             img_path_gen.append(gpath + 'acc_gen/' + 'iblur_'+imagename)
                    #             img_path_gen.append(gpath + 'acc_gen/' + 'isharp_'+imagename)
                    #             img_path_gen.append(gpath + 'acc_gen/' + 'idetail_'+imagename)
                    #             img_path_gen.append(gpath + 'acc_gen/' + 'ismooth_'+imagename)
                    #             img_path_gen.append(gpath + 'geacc_gen/' + 'irotate_'+imagename)
                    img_path_gen.append(gpath + 'acc_gen/' + 'ienhance_' + imagename)

                    label_gen.extend([int(i)])

                    #             im_blur.save(gpath + 'acc_gen/' + 'iblur_'+imagename)
                    #             im_sharp.save(gpath + 'acc_gen/' + 'isharp_'+imagename)
                    #             im_detail.save(gpath + 'acc_gen/' + 'idetail_'+imagename)
                    #             im_smooth.save(gpath + 'acc_gen/' + 'ismooth_'+imagename)
                    im_enhance.save(gpath + 'acc_gen/' + 'ienhance_' + imagename)
                    gen_number += 1

            label_dict = {'img_path':img_path_gen, 'label':label_gen}

            label_gen_dict['img_path'].extend(img_path_gen)
            label_gen_dict['label'].extend(label_gen)

            label_gen_pd = pd.DataFrame(label_dict)

            label = label.append(label_gen_pd)
            label['label'] = label[['label']].astype('int64')

            label_g = label_g.append(label_gen_pd)
            label_g['label'] = label_g[['label']].astype('int64')

        label_g.to_csv(gpath + 'label_acc_gen_all.csv', index=False)
        label.to_csv(gpath + 'label_acc_all.csv', index=False)


        label_gen_p = pd.DataFrame(label_gen_dict)
        label_gen_p.to_csv(gpath + 'label_acc.csv', index=False)


        print('验证集总共生成%d个图片'%gen_number)
if __name__ == '__main__':
    genImage(train_path, 'train')
    genImage(valid_path, 'valid')
