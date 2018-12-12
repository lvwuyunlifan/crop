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
        if not os.path.exists(gpath+'gen'):
            os.makedirs(gpath+'gen')

        label = pd.read_csv(gpath + 'label.csv')
        label_gen_dict = {'img_path':[], 'label':[]}  # 生成图片label
        for i in range(61):
            li = label[label['label'] == i]
            imagenum = li['label'].count()
            print('第%d个，总共有有%d个图片'%(i, imagenum))
            imagelist = np.array(li['img_path']).tolist()
            img_path_gen, label_gen = [], []
            if imagenum < 100:
                for imagefile in imagelist:
                    path, imagename = os.path.split(imagefile)
                    for count in range(0,2):
                        im = Image.open(imagefile)
                        im = im.convert('RGB')
                        im_blur = im.filter(ImageFilter.GaussianBlur)  # 模糊化
                        im_sharp = im.filter(ImageFilter.UnsharpMask)  # 锐化增强
                        im_detail = im.filter(ImageFilter.DETAIL)  # 细节增强
                        im_smooth = im.filter(ImageFilter.SMOOTH)  # 平滑滤波
                        if count == 0:
                            # im_retate = im.rotate(30, Image.BICUBIC,1)  # 逆旋转30
                            im_retate = im.rotate(30)  # 逆旋转30

                        if count == 1:
                            im_retate = im.rotate(60)  # 逆旋转60
                            
                        img_path_gen.append(gpath + 'gen/' + str(count)+'iblur_'+imagename)
                        img_path_gen.append(gpath + 'gen/' + str(count)+'isharp_'+imagename)
                        img_path_gen.append(gpath + 'gen/' + str(count)+'idetail_'+imagename)
                        img_path_gen.append(gpath + 'gen/' + str(count)+'ismooth_'+imagename)
                        img_path_gen.append(gpath + 'gen/' + str(count)+'irotate_'+imagename)
                        label_gen.extend([int(i),int(i),int(i),int(i),int(i)])
                        
                        
                        im_blur.save(gpath + 'gen/' + str(count)+'iblur_'+imagename)        
                        im_sharp.save(gpath + 'gen/' + str(count)+'isharp_'+imagename)  
                        im_detail.save(gpath + 'gen/' + str(count)+'idetail_'+imagename)            
                        im_smooth.save(gpath + 'gen/' + str(count)+'ismooth_'+imagename)            
                        im_retate.save(gpath + 'gen/' + str(count)+'irotate_'+imagename)

                        gen_number += 5

                label_dict = {'img_path':img_path_gen, 'label':label_gen}
                label_gen_dict['img_path'].extend(img_path_gen)
                label_gen_dict['label'].extend(label_gen)
                label_gen_pd = pd.DataFrame(label_dict)
                label = label.append(label_gen_pd)  # 将生成的图片label加入原先的label
                label['label'] = label[['label']].astype('int64')  # 转化为int64
                # print(label)
                
            if 100 <= imagenum <= 200:
                for imagefile in imagelist:
                    path, imagename = os.path.split(imagefile)
                    im = Image.open(imagefile)
                    im = im.convert('RGB')
        #             im_blur = im.filter(ImageFilter.GaussianBlur)  # 模糊化
                    im_sharp = im.filter(ImageFilter.UnsharpMask)  # 锐化增强
                    im_detail = im.filter(ImageFilter.DETAIL)  # 细节增强
        #             im_smooth = im.filter(ImageFilter.SMOOTH)  # 平滑滤波
                    im_retate = im.rotate(30)
                    
        #             img_path_gen.append(gpath + 'gen/' + 'iblur_'+imagename)
                    img_path_gen.append(gpath + 'gen/' + 'isharp_'+imagename)
                    img_path_gen.append(gpath + 'gen/' + 'idetail_'+imagename)
        #             img_path_gen.append(gpath + 'gen/' + 'ismooth_'+imagename)
                    img_path_gen.append(gpath + 'gen/' + 'irotate_'+imagename)
                    label_gen.extend([int(i),int(i),int(i)])


        #             im_blur.save(gpath + 'gen/' + 'iblur_'+imagename)        
                    im_sharp.save(gpath + 'gen/' + 'isharp_'+imagename)  
                    im_detail.save(gpath + 'gen/' + 'idetail_'+imagename)            
        #             im_smooth.save(gpath + 'gen/' + 'ismooth_'+imagename)            
                    im_retate.save(gpath + 'gen/' + 'irotate_'+imagename)
                    gen_number += 3

                label_dict = {'img_path':img_path_gen, 'label':label_gen}
                label_gen_dict['img_path'].extend(img_path_gen)
                label_gen_dict['label'].extend(label_gen)
                label_gen_pd = pd.DataFrame(label_dict)
                label = label.append(label_gen_pd)
                label['label'] = label[['label']].astype('int64')
                
            if 200 <= imagenum <= 300:
                for imagefile in imagelist:
                    path, imagename = os.path.split(imagefile)
                    im = Image.open(imagefile)
                    im = im.convert('RGB')
        #             im_blur = im.filter(ImageFilter.GaussianBlur)  # 模糊化
                    im_sharp = im.filter(ImageFilter.UnsharpMask)  # 锐化增强
                    im_detail = im.filter(ImageFilter.DETAIL)  # 细节增强
        #             im_smooth = im.filter(ImageFilter.SMOOTH)  # 平滑滤波
                    
        #             img_path_gen.append(gpath + 'gen/' + 'iblur_'+imagename)
                    img_path_gen.append(gpath + 'gen/' + 'isharp_'+imagename)
                    img_path_gen.append(gpath + 'gen/' + 'idetail_'+imagename)
        #             img_path_gen.append(gpath + 'gen/' + 'ismooth_'+imagename)
        #             img_path_gen.append(gpath + 'gen/' + 'irotate_'+imagename)
                    
                    label_gen.extend([int(i),int(i)])


        #             im_blur.save(gpath + 'gen/' + 'iblur_'+imagename)        
                    im_sharp.save(gpath + 'gen/' + 'isharp_'+imagename)  
                    im_detail.save(gpath + 'gen/' + 'idetail_'+imagename)            
        #             im_smooth.save(gpath + 'gen/' + 'ismooth_'+imagename)            
                    # im_retate.save(gpath + 'gen/' + 'irotate_'+imagename)
                    gen_number += 2
                label_dict = {'img_path':img_path_gen, 'label':label_gen}
                label_gen_dict['img_path'].extend(img_path_gen)
                label_gen_dict['label'].extend(label_gen)
                label_gen_pd = pd.DataFrame(label_dict)
                label = label.append(label_gen_pd)
                label['label'] = label[['label']].astype('int64')
            
            if 300 <= imagenum <= 400:
                print('300-400')
                for imagefile in imagelist:
                    # print(imagefile)
                    path, imagename = os.path.split(imagefile)
                    # print('0')
                    im = Image.open(imagefile)
                    # print('000')
                    im = im.convert('RGB')
                    # im_blur = im.filter(ImageFilter.GaussianBlur)  # 模糊化
                    # print('1111')
                    # im_sharp = im.filter(ImageFilter.UnsharpMask)  # 锐化增强
                    im_detail = im.filter(ImageFilter.DETAIL)  # 细节增强
        #             im_smooth = im.filter(ImageFilter.SMOOTH)  # 平滑滤波
                    # print('1')
                    # img_path_gen.append(gpath + 'gen/' + 'iblur_'+imagename)
                    # img_path_gen.append(gpath + 'gen/' + 'isharp_' + imagename)
                    img_path_gen.append(gpath + 'gen/' + 'idetail_' + imagename)
        #             img_path_gen.append(gpath + 'gen/' + 'ismooth_'+imagename)
                    # img_path_gen.append(gpath + 'gen/' + 'irotate_'+imagename)
                    label_gen.extend([int(i)])

                    # print('2')
                    # im_blur.save(gpath + 'gen/' + 'iblur_'+imagename)        
                    # im_sharp.save(gpath + 'gen/' + 'isharp_' + imagename)  
                    im_detail.save(gpath + 'gen/' + 'idetail_'+imagename)            
        #             im_smooth.save(gpath + 'gen/' + 'ismooth_'+imagename)            
                    # im_retate.save(gpath + 'gen/' + 'irotate_'+imagename)
                    # print('3')
                    gen_number += 1
                    # print('4')

                label_dict = {'img_path':img_path_gen, 'label':label_gen}
                label_gen_dict['img_path'].extend(img_path_gen)
                label_gen_dict['label'].extend(label_gen)
                # print(6)
                label_gen_pd = pd.DataFrame(label_dict)
                label = label.append(label_gen_pd)
                label['label'] = label[['label']].astype('int64')
                print('5')

            # 数据大于1600的，随机删除多余的数据
            if imagenum > 1600:
                li2=li.sample(frac=0.5)
                rowlist=[]
                li2_index = li2.index
                for indexs in range(len(li2_index)):
                    if indexs <=(imagenum - 1600):
                        rowlist.append(li2_index[indexs])
                label=label.drop(rowlist,axis=0)
        label.to_csv(gpath + 'label_all.csv', index=False)

        label_gen_p = pd.DataFrame(label_gen_dict)
        label_gen_p.to_csv(gpath + 'label_gen.csv', index=False)

        print('训练集总共生成%d个图片'%gen_number)


    if datatype == 'valid':
        gen_number = 0
        if not os.path.exists(gpath+'gen'):
            os.makedirs(gpath+'gen')
        label = pd.read_csv(gpath + 'label.csv')
        label_gen_dict = {'img_path':[], 'label':[]}
        for i in range(61):
            li = label[label['label'] == i]
            imagenum = li['label'].count()
            print('第%d个，总共有有%d个图片'%(i, imagenum))
            imagelist = np.array(li['img_path']).tolist()
            img_path_gen, label_gen = [], []
            if imagenum < 10:
                for imagefile in imagelist:
                    path, imagename = os.path.split(imagefile)
                    for count in range(0,2):
                        im = Image.open(imagefile)
                        im = im.convert('RGB')
                        im_blur = im.filter(ImageFilter.GaussianBlur)  # 模糊化
                        im_sharp = im.filter(ImageFilter.UnsharpMask)  # 锐化增强
                        im_detail = im.filter(ImageFilter.DETAIL)  # 细节增强
                        im_smooth = im.filter(ImageFilter.SMOOTH)  # 平滑滤波
                        if count == 0:
                            im_retate = im.rotate(30)  # 逆旋转30
                        if count == 1:
                            im_retate = im.rotate(60)  # 逆旋转60
                            
                        img_path_gen.append(gpath + 'gen/' + str(count)+'iblur_'+imagename)
                        img_path_gen.append(gpath + 'gen/' + str(count)+'isharp_'+imagename)
                        img_path_gen.append(gpath + 'gen/' + str(count)+'idetail_'+imagename)
                        img_path_gen.append(gpath + 'gen/' + str(count)+'ismooth_'+imagename)
                        img_path_gen.append(gpath + 'gen/' + str(count)+'irotate_'+imagename)
                        label_gen.extend([int(i),int(i),int(i),int(i),int(i)])
                        
                        
                        im_blur.save(gpath + 'gen/' + str(count)+'iblur_'+imagename)        
                        im_sharp.save(gpath + 'gen/' + str(count)+'isharp_'+imagename)  
                        im_detail.save(gpath + 'gen/' + str(count)+'idetail_'+imagename)            
                        im_smooth.save(gpath + 'gen/' + str(count)+'ismooth_'+imagename)            
                        im_retate.save(gpath + 'gen/' + str(count)+'irotate_'+imagename)

                        gen_number += 5

                label_dict = {'img_path':img_path_gen, 'label':label_gen}  
                label_gen_dict['img_path'].extend(img_path_gen)
                label_gen_dict['label'].extend(label_gen)
                label_gen_pd = pd.DataFrame(label_dict)
                label = label.append(label_gen_pd)  # 将生成的图片label加入原先的label
                label['label'] = label[['label']].astype('int64')  # 转化为int64
                # print(label)
                
            if 10 <= imagenum <= 20:
                for imagefile in imagelist:
                    path, imagename = os.path.split(imagefile)
                    im = Image.open(imagefile)
                    im = im.convert('RGB')
        #             im_blur = im.filter(ImageFilter.GaussianBlur)  # 模糊化
                    im_sharp = im.filter(ImageFilter.UnsharpMask)  # 锐化增强
                    im_detail = im.filter(ImageFilter.DETAIL)  # 细节增强
        #             im_smooth = im.filter(ImageFilter.SMOOTH)  # 平滑滤波
                    im_retate = im.rotate(30)
                    
        #             img_path_gen.append(gpath + 'gen/' + 'iblur_'+imagename)
                    img_path_gen.append(gpath + 'gen/' + 'isharp_'+imagename)
                    img_path_gen.append(gpath + 'gen/' + 'idetail_'+imagename)
        #             img_path_gen.append(gpath + 'gen/' + 'ismooth_'+imagename)
                    img_path_gen.append(gpath + 'gen/' + 'irotate_'+imagename)
                    label_gen.extend([int(i),int(i),int(i)])


        #             im_blur.save(gpath + 'gen/' + 'iblur_'+imagename)        
                    im_sharp.save(gpath + 'gen/' + 'isharp_'+imagename)  
                    im_detail.save(gpath + 'gen/' + 'idetail_'+imagename)            
        #             im_smooth.save(gpath + 'gen/' + 'ismooth_'+imagename)            
                    im_retate.save(gpath + 'gen/' + 'irotate_'+imagename)
                    gen_number += 3

                label_dict = {'img_path':img_path_gen, 'label':label_gen}
                label_gen_dict['img_path'].extend(img_path_gen)
                label_gen_dict['label'].extend(label_gen)
                label_gen_pd = pd.DataFrame(label_dict)
                label = label.append(label_gen_pd)
                label['label'] = label[['label']].astype('int64')
                
            if 20 <= imagenum <= 30:
                for imagefile in imagelist:
                    path, imagename = os.path.split(imagefile)
                    im = Image.open(imagefile)
                    im = im.convert('RGB')
        #             im_blur = im.filter(ImageFilter.GaussianBlur)  # 模糊化
                    im_sharp = im.filter(ImageFilter.UnsharpMask)  # 锐化增强
                    im_detail = im.filter(ImageFilter.DETAIL)  # 细节增强
        #             im_smooth = im.filter(ImageFilter.SMOOTH)  # 平滑滤波
                    
        #             img_path_gen.append(gpath + 'gen/' + 'iblur_'+imagename)
                    img_path_gen.append(gpath + 'gen/' + 'isharp_'+imagename)
                    img_path_gen.append(gpath + 'gen/' + 'idetail_'+imagename)
        #             img_path_gen.append(gpath + 'gen/' + 'ismooth_'+imagename)
        #             img_path_gen.append(gpath + 'gen/' + 'irotate_'+imagename)
                    
                    label_gen.extend([int(i),int(i)])


        #             im_blur.save(gpath + 'gen/' + 'iblur_'+imagename)        
                    im_sharp.save(gpath + 'gen/' + 'isharp_'+imagename)  
                    im_detail.save(gpath + 'gen/' + 'idetail_'+imagename)            
        #             im_smooth.save(gpath + 'gen/' + 'ismooth_'+imagename)            
                    # im_retate.save(gpath + 'gen/' + 'irotate_'+imagename)
                    gen_number += 2
                label_dict = {'img_path':img_path_gen, 'label':label_gen}
                label_gen_dict['img_path'].extend(img_path_gen)
                label_gen_dict['label'].extend(label_gen)
                label_gen_pd = pd.DataFrame(label_dict)
                label = label.append(label_gen_pd)
                label['label'] = label[['label']].astype('int64')
            
            if 30 <= imagenum <= 40:
                for imagefile in imagelist:
                    path, imagename = os.path.split(imagefile)
                    im = Image.open(imagefile)
                    im = im.convert('RGB')
                    # im_blur = im.filter(ImageFilter.GaussianBlur)  # 模糊化
                    im_sharp = im.filter(ImageFilter.UnsharpMask)  # 锐化增强
        #             im_detail = im.filter(ImageFilter.DETAIL)  # 细节增强
        #             im_smooth = im.filter(ImageFilter.SMOOTH)  # 平滑滤波
                    
                    # img_path_gen.append(gpath + 'gen/' + 'iblur_'+imagename)
                    img_path_gen.append(gpath + 'gen/' + 'isharp_' + imagename)
        #             img_path_gen.append(gpath + 'gen/' + 'idetail_'+imagename)
        #             img_path_gen.append(gpath + 'gen/' + 'ismooth_'+imagename)
                    # img_path_gen.append(gpath + 'gen/' + 'irotate_'+imagename)
                    label_gen.extend([int(i)])


                    # im_blur.save(gpath + 'gen/' + 'iblur_'+imagename)        
                    im_sharp.save(gpath + 'gen/' + 'isharp_' + imagename)  
        #             im_detail.save(gpath + 'gen/' + 'idetail_'+imagename)            
        #             im_smooth.save(gpath + 'gen/' + 'ismooth_'+imagename)            
                    # im_retate.save(gpath + 'gen/' + 'irotate_'+imagename)

                    gen_number += 1

                label_dict = {'img_path':img_path_gen, 'label':label_gen}
                label_gen_dict['img_path'].extend(img_path_gen)
                label_gen_dict['label'].extend(label_gen)
                label_gen_pd = pd.DataFrame(label_dict)
                label = label.append(label_gen_pd)
                label['label'] = label[['label']].astype('int64')

            if 40 <= imagenum <= 50:
                for imagefile in imagelist:
                    rd = random.choice([0, 1])
                    if rd:
                        path, imagename = os.path.split(imagefile)
                        im = Image.open(imagefile)
                        im = im.convert('RGB')
                        # im_blur = im.filter(ImageFilter.GaussianBlur)  # 模糊化
                        # im_sharp = im.filter(ImageFilter.UnsharpMask)  # 锐化增强
                        im_detail = im.filter(ImageFilter.DETAIL)  # 细节增强
            #             im_smooth = im.filter(ImageFilter.SMOOTH)  # 平滑滤波
                        
                        # img_path_gen.append(gpath + 'gen/' + 'iblur_'+imagename)
                        # img_path_gen.append(gpath + 'gen/' + 'isharp_' + imagename)
                        img_path_gen.append(gpath + 'gen/' + 'idetail_'+imagename)
            #             img_path_gen.append(gpath + 'gen/' + 'ismooth_'+imagename)
                        # img_path_gen.append(gpath + 'gen/' + 'irotate_'+imagename)
                        label_gen.extend([int(i)])


                        # im_blur.save(gpath + 'gen/' + 'iblur_'+imagename)        
                        # im_sharp.save(gpath + 'gen/' + 'isharp_' + imagename)  
                        im_detail.save(gpath + 'gen/' + 'idetail_'+imagename)            
            #             im_smooth.save(gpath + 'gen/' + 'ismooth_'+imagename)            
                        # im_retate.save(gpath + 'gen/' + 'irotate_'+imagename)

                        gen_number += 1

                label_dict = {'img_path':img_path_gen, 'label':label_gen}
                label_gen_dict['img_path'].extend(img_path_gen)
                label_gen_dict['label'].extend(label_gen)
                label_gen_pd = pd.DataFrame(label_dict)
                label = label.append(label_gen_pd)
                label['label'] = label[['label']].astype('int64')


            # 数据大于1600的，随机删除多余的数据
            if imagenum > 250:
                li2=li.sample(frac=0.5)
                rowlist=[]
                li2_index = li2.index
                for indexs in range(len(li2_index)):
                    if indexs <=(imagenum - 250):
                        rowlist.append(li2_index[indexs])
                label=label.drop(rowlist,axis=0)

        label.to_csv(gpath + 'label_all.csv', index=False)

        label_gen_p = pd.DataFrame(label_gen_dict)
        label_gen_p.to_csv(gpath + 'label_gen.csv', index=False)

        print('验证集总共生成%d个图片'%gen_number)
if __name__ == '__main__':
    genImage(train_path, 'train')
    genImage(valid_path, 'valid')
