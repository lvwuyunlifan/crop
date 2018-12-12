import os
# 2 gpus
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# path to train and test labels

TRAIN_LABEL_DIR = './AgriculturalDisease_trainingset/label_all.csv'
VAL_LABEL_DIR = './AgriculturalDisease_validationset/label_all.csv'
TEST_LABEL_DIR = './AgriculturalDisease_testA/test.csv'

# path to train and test images
# TRAIN_IMG_DIR = 'datasets/train/'
# TEST_IMG_DIR =  'datasets/final-rank/'

# path to trianed models
MODEL_INCEPTIONV4 =  'models_kares/length_inceptionv4_480_12.h5'
MODEL_INCEPTIONRESNETV2 = 'models_kares/length_inceptionresnet_480_12.h5'


task_list1 = {
    'Apple': 6,
    'Cherry': 3,
    'Corn': 8,
    'Grape': 7,
    'Citrus': 3,
    'Peach': 3,
    'Pepper': 3,
    'Potato': 5,
    'Strawberry': 3,
    'Tomato': 20
}
task_list = {
    'Apples': 24,  #apple-grape
    'Citrus': 17,   # citrus-Strawberry
    'Tomato': 20
}
# input size
width = 480
model_name = 'inceptionv4'
