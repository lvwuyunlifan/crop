import os
import math
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
import json

dataset = {'Apple': '0-5', 'Cherry': '6-8', 'Corn': '9-16', 'Grape': '17-23', 'Citrus': '24-26', 'Peach': '27-29', 'Pepper': '30-32', 'Potato': '33-37',
           'Strawberry': '38-40', 'Tomato': '41-60'}

def json_convert_to_csv():

    print('train data...')
    # train data
    traindata_path = './AgriculturalDisease_trainingset/images'
    img_path, label = [], []
    with open(osp.join('./AgriculturalDisease_trainingset', 'AgriculturalDisease_train_annotations.json')) as f:
        traindata_json = json.load(f)
    # print(traindata_json[0])

    #for key in dataset:

        #split = dataset[key].split('-')
        #a = int(split[0])
        #b = int(split[1])

        for i in traindata_json:
            img_p = osp.join(traindata_path, i['image_id'])
            img_path.append(img_p)
            label.append(i['disease_class'])


    label_file = pd.DataFrame({'img_path': img_path, 'label': label})

    # label_file['label'] = label_file['label'].map(label_warp)

    label_file.to_csv('./AgriculturalDisease_trainingset/label.csv', index=False)


    print('valid data...')
    # validate data
    validdata_pathv = './AgriculturalDisease_validationset/images'
    img_pathv, labelv = [], []
    with open(osp.join('./AgriculturalDisease_validationset', 'AgriculturalDisease_validation_annotations.json')) as fv:
        traindata_jsonv = json.load(fv)
    # print(traindata_jsonv[0])

    #for key in dataset:
        #splitv = dataset[key].split('-')
        #c = int(splitv[0])
        #d = int(splitv[1])

        for iv in traindata_jsonv:
            # for key in dataset:
            #     split = dataset[key].split('-')
            #     a = int(split[0])
            #     b = int(split[1])
            #     print(a, b)
            #if c <= iv['disease_class'] <= d:
            img_pv = osp.join(validdata_pathv, iv['image_id'])
            img_pathv.append(img_pv)
            labelv.append(iv['disease_class'])

    label_filev = pd.DataFrame({'img_path': img_pathv, 'label': labelv})

        # label_file['label'] = label_file['label'].map(label_warp)
    label_filev.to_csv('./AgriculturalDisease_validationset/label.csv', index=False)

    print('test data...')
    # test data
    test_data_path = './AgriculturalDisease_testA/images'
    all_test_img = os.listdir(test_data_path)
    test_img_path = []

    for img in all_test_img:
        # if osp.splitext(img)[1] == '.jpg':
        test_img_path.append(osp.join(test_data_path, img))

    test_file = pd.DataFrame({'img_path': test_img_path})
    test_file.to_csv('./AgriculturalDisease_testA/test.csv', index=False)

# def csv_convert_to_json():
#     try:
#         sub = pd.read_csv('./result/main_inception_v4/submission.csv')


if __name__ == '__main__':
    # parser = argpase.ArgumentParser()
    # parser.add_argument('--mode', type=str, default='getcsv', help='get train/valid/test label.csv')
    # # parser.add_argument('--getjson', type=str, default='csv_convert_to_json', help='get submit.json')
    #
    # FLAGS = parser.parse_args()
    # if FLAGS.mode == 'getcsv':
    #     json_convert_to_csv()
    # elif:
    #     csv_convert_to_json()
    # else:
    #     raise Exception('error mode')
    json_convert_to_csv()
    print('done...')