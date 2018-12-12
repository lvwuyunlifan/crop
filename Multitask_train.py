#-- coding: utf-8 --

import gc
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.utils import multi_gpu_model
from keras.applications.inception_resnet_v2 import preprocess_input
import inception_v4
from tqdm import tqdm
from dataset import *
from config import *
from keras.utils import plot_model
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score



# load data into memory
def getX(type):

    if type == 'train':
        X = np.zeros((n, width, width, 3), dtype=np.uint8)
        for i in tqdm(range(n)):
            # print(df['img_path'][i])
            img = cv2.resize(cv2.imread(df['img_path'][i]), (width, width))
    if type == 'valid':
        X = np.zeros((m, width, width, 3), dtype=np.uint8)
        for i in tqdm(range(m)):
            img = cv2.resize(cv2.imread(val_df['img_path'][i]), (width, width))
    X[i] = img[:, :, ::-1]
    return X

# calculate the accuracy on validation set
def acc(y_true, y_pred):
    print('true: ', y_true)
    index = tf.reduce_any(y_true > 0.5, axis=-1)
    res = tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))
    index = tf.cast(index, tf.float32)
    print('index: ', index)
    res = tf.cast(res, tf.float32)
    print('res: ', res)
    accuracy = tf.reduce_sum(res * index) / (tf.reduce_sum(index) + 1e-7)
    print("accracy: ", accuracy)

    return accuracy




class Metrics(Callback):
    def on_train_begin(self, logs={}):
        # self.val_f1s = []
        # self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (numpy.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        # _val_f1 = f1_score(val_targ, val_predict)
        # _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        # self.val_f1s.append(_val_f1)
        # self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        return

def train():
    # base_model = inception_v4.create_model(weights='imagenet', width=width, include_top=False)
    # plot_model(base_model, to_file='base_model.png')
    # input_tensor = Input((width, width, 3))
    # x = input_tensor
    # x = Lambda(preprocess_input, name='preprocessing')(x)
    # x = base_model(x)
    # x = GlobalAveragePooling2D()(x)
    # x = Dropout(0.5)(x)
    # x = [Dense(count, activation='softmax', name=name)(x) for name, count in task_list.items()]
    #
    # model2 = Model(input_tensor, x)


    # if (task == 'train'):
    #     task_list = task_list
    # else:
    #     task_list = task_list_length
    labelname = {'Apple': '0-5', 'Cherry': '6-8', 'Corn': '9-16', 'Grape': '17-23', 'Citrus': '24-26', 'Peach': '27-29',
               'Pepper': '30-32', 'Potato': '33-37',
               'Strawberry': '38-40', 'Tomato': '41-60'}
    # label_names = list(task_list.keys())
    dfs = [n, m]
    for d in dfs:
        y = [np.zeros((d, task_list[x])) for x in task_list.keys()]
        for i in range(d):
            # label_name = df.label_name[i]
            if d ==n:
                label = df.label[i]
            else:
                label = val_df.label[i]
            # print(i, label)
            if 0<=label<=23:

                # label = label - 0
                y[0][i, label] = 1

            if 24<=label<=40:
                label1 = label - 24
                y[1][i, label1] = 1

            if 41<=label<=60:
                label2 = label - 41
                y[2][i, label2] = 1
            # if 0<=label<=5:
            #
            #     # label = label - 0
            #     y[0][i, label] = 1
            # if 6<=label<=8:
            #     label1 = label - 6
            #     y[1][i, label1] = 1
            # if 9<=label<=16:
            #     label2 = label - 9
            #     y[2][i, label2] = 1
            # if 17<=label<=23:
            #     label3 = label - 17
            #     y[3][i, label3] = 1
            # if 24<=label<=26:
            #     label4 = label - 24
            #     y[4][i, label4] = 1
            # if 27<=label<=29:
            #     label5 = label - 27
            #     y[5][i, label5] = 1
            # if 30<=label<=32:
            #     label6 = label - 30
            #     y[6][i, label6] = 1
            # if 33<=label<=37:
            #     label7 = label - 33
            #     y[7][i, label7] = 1
            # if 38<=label<=40:
            #     label8 = label - 38
            #     y[8][i, label8] = 1
            # if 41<=label<=60:
            #     label9 = label - 41
            #     y[9][i, label9] = 1
        if d == n:
            y_train = y
        elif d == m:
            y_valid = y

        # y[label_names.index(label_name)][i, label.find('y')] = 1

    # print('train: ', y_train[8].tolist())
    # print('valid: ', y_valid[5].tolist())
    # print('valid: ', y_valid[0].shape)
    # for j in range(10):
    #     with open('./yvalid%d.txt'%j,"w") as f:
    #         for i in y_valid[j].tolist():
    #             f.write('%s'%i)

    X_train = getX('train')
    X_valid = getX('valid')

    tb = TensorBoard(log_dir='./keras_logs',# log 目录
                     histogram_freq=1, # 按照何等频率（epoch）来计算直方图，0为不计算
                     batch_size=12, # 用多大量的数据计算直方图
                     write_graph=True, # 是否存储网络结构图
                     write_grads=False, # 是否可视化梯度直方图
                     write_images=False,# 是否可视化参数
                     embeddings_freq=0,
                     embeddings_layer_names=None,
                     embeddings_metadata=None)
    callbacks = [tb]
    metrics = Metrics()
    print('train: ', X_train.shape)
    print('valid: ', X_valid.shape)

    gen_train = Generator(X_train, y_train, batch_size=12, aug=False)

    base_model = inception_v4.create_model(weights='imagenet', width=width, include_top=False)
    input_tensor = Input((width, width, 3))
    x = input_tensor
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = [Dense(count, activation='softmax', name=name)(x) for name, count in task_list.items()]

    model2 = Model(input_tensor, x)
    print('finish...', model2.summary())
    # model.load_weights('models/base.h5',by_name=True)
    model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    # model2 = multi_gpu_model(model, 2)
    plot_model(model2, to_file='model_3.png')

    model2.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=[acc])
    model2.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=10, validation_data=(X_valid, y_valid), callbacks=callbacks)

    model2.compile(optimizer=Adam(0.00025), loss='categorical_crossentropy', metrics=[acc])
    model2.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=10, validation_data=(X_valid, y_valid))

    model2.compile(optimizer=Adam(0.0000625), loss='categorical_crossentropy', metrics=[acc])
    model2.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=10, validation_data=(X_valid, y_valid))

    model2.compile(optimizer=Adam(0.0000425), loss='categorical_crossentropy', metrics=[acc])
    model2.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=5, validation_data=(X_valid, y_valid))

    model2.compile(optimizer=Adam(0.000001), loss='categorical_crossentropy', metrics=[acc])
    model2.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=5, validation_data=(X_valid, y_valid))
    model2.save_weights('models_kares/%s.h5' % model_name)


    del model
    gc.collect()

# load the label file and split it into two portions
def csv_loader():
    df_test = pd.read_csv(TRAIN_LABEL_DIR, header=None)
    df_test.columns = ['filename', 'label_name', 'label']

    df_test_length = df_test[(df_test.label_name == 'skirt_length_labels') | (df_test.label_name == 'sleeve_length_labels')
                          |(df_test.label_name == 'coat_length_labels')|(df_test.label_name == 'pant_length_labels')]

    df_test_design = df_test[(df_test.label_name == 'collar_design_labels') | (df_test.label_name == 'lapel_design_labels')
                          | (df_test.label_name == 'neckline_design_labels') | (df_test.label_name == 'neck_design_labels')]
    df_test_length.to_csv(TRAIN_LENGTH_LABEL_DIR, index=False, header=None)
    df_test_design.to_csv(TRAIN_DESIGN_LABEL_DIR, index=False, header=None)


if __name__ == "__main__":

    # csv_loader()

    df = pd.read_csv(TRAIN_LABEL_DIR)#, header=None)
    val_df = pd.read_csv(VAL_LABEL_DIR)
    test_df = pd.read_csv(TEST_LABEL_DIR)
    # df.columns = ['filename', 'label_name', 'label']
    # df.columns = ['img_path', 'label']
    # df = df.sample(frac=1).reset_index(drop=True)
    n = len(df)
    m = len(val_df)
    train()
    # del df, n