from visual import Logger

import os
import os.path as osp
import time
import numpy as np
import random
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms

from CUB_loader import TrainDataset, ValDataset, TestDataset
import cv2

from models_old import RACNN, pairwise_ranking_loss, multitask_loss

import argparse

parser = argparse.ArgumentParser(description = 'Training arguments')
# parser.add_argument('--cuda', default = True, type = bool, help = "use cuda to train")
# parser.add_argument('--lr', default = 0.01, type = float, help = "initial learning rate")
parser.add_argument('--cuda', action = "store_true", help = "use cuda to train")
parser.add_argument('--lr', default = 0.01, type = float, help = "initial learning rate (default = 0.01)")
parser.add_argument('--batch_size', default = 2, type = int, help = "batch size (default = 4)")
parser.add_argument('--num_workers', default = 2, type = int, help = "number of worker to parralle read data (default = 8")

args = parser.parse_args()
decay_steps = [20, 40] # based on epoch

net = RACNN(num_classes = 61)
if args.cuda or torch.cuda.is_available():
    print(" [*] Set cuda: True")
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
else:
    print(" [*] Set cuda: False")

logger = Logger('./visual/' + 'RACNN_CUB200_9')
# cls_params = list(net.module.b1.parameters()) + list(net.module.b2.parameters()) + list(net.module.b3.parameters()) + list(net.module.classifier1.parameters()) + list(net.module.classifier2.parameters()) + list(net.module.classifier3.parameters())
if args.cuda:
    cls_params = list(net.module.b1.parameters()) + list(net.module.b2.parameters()) + list(net.module.b3.parameters()) + list(net.module.classifier1.parameters()) + list(net.module.classifier2.parameters()) + list(net.module.classifier3.parameters())
else:
    cls_params = list(net.b1.parameters()) + list(net.b2.parameters()) + list(net.b3.parameters()) + list(net.classifier1.parameters()) + list(net.classifier2.parameters()) + list(net.classifier3.parameters())
opt1 = optim.SGD(cls_params, lr = args.lr, momentum = 0.9, weight_decay = 0.0005)

# apn_params = list(net.module.apn1.parameters()) + list(net.module.apn2.parameters())
if args.cuda:
    apn_params = list(net.module.apn1.parameters()) + list(net.module.apn2.parameters())
else:
    apn_params = list(net.apn1.parameters()) + list(net.apn2.parameters())
opt2 = optim.SGD(apn_params, lr = args.lr, momentum = 0.9, weight_decay = 0.0005)
#for param in apn_params:
#    param.register_hook(print)


# 获取当前文件名，用于创建模型及结果文件的目录
file_name = os.path.basename(__file__).split('.')[0]

# 创建保存模型和结果的文件夹
if not os.path.exists('./racnn_models'):
    os.makedirs('./racnn_models')
if not os.path.exists('./result_racnn/%s' % file_name):
    os.makedirs('./result_racnn/%s' % file_name)
# 创建日志文件
if not os.path.exists('./result_racnn/%s.txt' % file_name):
    with open('./result_racnn/%s.txt' % file_name, 'w') as acc_file:
        pass
with open('./result_racnn/%s.txt' % file_name, 'a') as acc_file:
    acc_file.write('\n%s %s\n' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), file_name))

def CUB_collate(batch):
    imgs = []
    cls = []
    for sample in batch:
        imgs.append(sample[0])
        cls.append(sample[1])
    imgs = torch.stack(imgs, 0)
    cls = torch.LongTensor(cls)
    return imgs, cls


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

# 保存最新模型以及最优模型
def save_checkpoint(type, state):
    filename = './racnn_models/checkpoint_%s.pth.tar' % type
    if type == 'pre_APN':
        torch.save(state, filename)

    else:
        torch.save(state, filename)


def train(trainset, trainloader, valloader):
    net.train()

    conf_loss = 0
    loc_loss = 0

    print(" [*] Loading dataset...")
    batch_iterator = None

    resume_apn = True

    # pretrain APN
    if resume_apn:
        checkpoint_path = './racnn_models/checkpoint_pre_APN.pth.tar'
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            apn_iter = checkpoint['apn_iter']
            apn_epoch = checkpoint['apn_epoch']
            apn_steps = checkpoint['apn_steps']
            net.load_state_dict(checkpoint['state_dict'])
    else:
        apn_iter, apn_epoch, apn_steps = pretrainAPN(trainset, trainloader)

    cls_iter, cls_epoch, cls_steps = 0, 0, 1
    switch_step = 0
    old_cls_loss, new_cls_loss = 2, 1
    old_apn_loss, new_apn_loss = 2, 1
    iteration = 0 # count the both of iteration
    epoch_size = len(trainset) // 4
    cls_tol = 0
    apn_tol = 0
    batch_iterator = iter(trainloader)

    # while ((old_cls_loss - new_cls_loss)**2 > 1e-7) and ((old_apn_loss - new_apn_loss)**2 > 1e-7) and (iteration < 500000):
    while ((old_cls_loss - new_cls_loss)**2 > 1e-7) and ((old_apn_loss - new_apn_loss)**2 > 1e-7) and (iteration < 50000):
        # until the two type of losses no longer change
        print(' [*] Swtich optimize parameters to Class')
        while ((cls_tol < 10) and (cls_iter % 500 != 0)):
            # print('cls_tol=%d, cls_iter=%d'%(cls_tol, cls_iter))
            if (not batch_iterator) or (iteration % epoch_size == 0):
                batch_iterator = iter(trainloader)

            if cls_iter % epoch_size == 0:
                cls_epoch += 1
                if cls_epoch in decay_steps:
                    cls_steps += 1
                    adjust_learning_rate(opt1, 0.1, cls_steps, args.lr)

            old_cls_loss = new_cls_loss

            images, labels = next(batch_iterator)
            # images, labels = Variable(images, requires_grad = True), Variable(labels)
            images.requires_grad_()
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()

            t0 = time.time()
            logits, _, _ = net(images)
            
            opt1.zero_grad()
            new_cls_losses = multitask_loss(logits, labels)
            new_cls_loss = sum(new_cls_losses)
            #new_cls_loss = new_cls_losses[0]
            new_cls_loss.backward()
            opt1.step()
            t1 = time.time()

            if (old_cls_loss - new_cls_loss)**2 < 1e-6:
                cls_tol += 1
            else:
                cls_tol = 0


            logger.scalar_summary('cls_loss', new_cls_loss.item(), iteration + 1)
            logger.scalar_summary('cls_loss1', new_cls_losses[0].item(), iteration + 1)
            logger.scalar_summary('cls_loss12', new_cls_losses[1].item(), iteration + 1)
            logger.scalar_summary('cls_loss123', new_cls_losses[2].item(), iteration + 1)
            iteration += 1
            cls_iter += 1
            if (cls_iter % 10) == 0:
                print(" [*] cls_epoch[%d], Iter %d || cls_iter %d || cls_loss: %.4f || Timer: %.4fsec"%(cls_epoch, iteration, cls_iter, new_cls_loss.item(), (t1 - t0)))
                # 在日志文件中记录每个epoch的精度和loss
                with open('./result_racnn/%s.txt' % 'cls', 'a') as acc_file:
                    acc_file.write(" [*] cls_epoch[%d], Iter %d || cls_iter %d || cls_loss: %.4f || Timer: %.4fsec\n"%
                        (cls_epoch, iteration, cls_iter, new_cls_loss.item(), (t1 - t0)))

        images, labels = next(batch_iterator)
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        logits, _, _ = net(images)
        preds = []
        for i in range(len(labels)):
            pred = [logit[i][labels[i]] for logit in logits]
            preds.append(pred)
        new_apn_loss = pairwise_ranking_loss(preds)
        logger.scalar_summary('rank_loss', new_apn_loss.item(), iteration + 1)
        iteration += 1
        #cls_iter += 1
        print('cls validate class......')
        accuracy_list = validate(valloader, iteration, type='cls')


        #continue
        print(' [*] Swtich optimize parameters to APN')
        switch_step += 1
        while ((apn_tol < 10) and apn_iter % 500 != 0):
            if (not batch_iterator) or (iteration % epoch_size == 0):
                batch_iterator = iter(trainloader)

            if apn_iter % epoch_size == 0:
                apn_epoch += 1
                if apn_epoch in decay_steps:
                    apn_steps += 1
                    adjust_learning_rate(opt2, 0.1, apn_steps, args.lr)

            old_apn_loss = new_apn_loss

            images, labels = next(batch_iterator)
            # images, labels = Variable(images, requires_grad = True), Variable(labels)
            images.requires_grad_()
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()

            t0 = time.time()
            logits, _, _ = net(images)

            opt2.zero_grad()
            preds = []
            for i in range(len(labels)):
                pred = [logit[i][labels[i]] for logit in logits]
                preds.append(pred)
            new_apn_loss = pairwise_ranking_loss(preds)
            new_apn_loss.backward()
            opt2.step()
            t1 = time.time()

            if (old_apn_loss - new_apn_loss)**2 < 1e-6:
                apn_tol += 1
            else:
                apn_tol = 0

            logger.scalar_summary('rank_loss', new_apn_loss.item(), iteration + 1)
            iteration += 1
            apn_iter += 1
            if (apn_iter % 20) == 0:
                print(" [*] apn_epoch[%d], Iter %d || apn_iter %d || apn_loss: %.4f || Timer: %.4fsec"%
                      (apn_epoch, iteration, apn_iter, new_apn_loss.item(), (t1 - t0)))
                # 在日志文件中记录每个epoch的精度和loss
                with open('./result_racnn/%s.txt' % 'APN', 'a') as acc_file:
                    acc_file.write(" [*] apn_epoch[%d], Iter %d || apn_iter %d || apn_loss: %.4f || Timer: %.4fsec\n"%
                        (apn_epoch, iteration, apn_iter, new_apn_loss.item(), (t1 - t0)))

        switch_step += 1

        images, labels = next(batch_iterator)
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        new_cls_losses = multitask_loss(logits, labels)
        new_cls_loss = sum(new_cls_losses)
        logger.scalar_summary('cls_loss', new_cls_loss.item(), iteration + 1)
        iteration += 1
        cls_iter += 1
        apn_iter += 1

        print('apn validate apn......')

        accuracy_list = validate(valloader, iteration, type='apn')
        print("accuracy_list: ", accuracy_list)
        # save model
        # torch.save(net.state_dict(), 'racnn_1_2_3.pth.tar')
        number = 0
        for i, acc in enumerate(accuracy_list):
            accuracy_max = accuracy_list[0]

            if acc > accuracy_max:
                number = i
                # 在日志文件中记录每个epoch的精度和loss
        with open('./result_racnn/%s.txt' % 'accuracy', 'a') as acc_file:
            acc_file.write(" accuracy1 = %d || accuracy2 =  %d || accuracy3 = %d || accuracy_best_number = %d \n" %
                (accuracy_list[0], accuracy_list[1], accuracy_list[2], accuracy_list[number]))
        state = net.state_dict()
        save_checkpoint(type='cls_and_rank', state=state)


def pretrainAPN(trainset, trainloader):
    epoch_size = len(trainset) // 4
    print('epoch_size: ', epoch_size)
    apn_steps, apn_epoch = 1, -1

    batch_iterator = iter(trainloader)
    for _iter in range(0, 10000):
        iteration = _iter
        if (not batch_iterator) or (iteration % epoch_size == 0):
            batch_iterator = iter(trainloader)
        
        if iteration % epoch_size == 0:
            apn_epoch += 1
            if apn_epoch in decay_steps:
                apn_steps += 1
                adjust_learning_rate(opt2, 0.1, apn_steps, args.lr)

        images, labels = next(batch_iterator)
        # images, labels = Variable(images, requires_grad = True), Variable(labels)
        images.requires_grad_()
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()

        t0 = time.time()
        _, conv5s, attens = net(images)
            
        opt2.zero_grad()
        # search regions with the highest response value in conv5
        weak_loc = []
        for i in range(len(conv5s)):
            loc_label = torch.ones([images.size(0),3]) * 0.33 # tl = 0.25, fixed
            resize = 448
            if i >= 1:
                resize = 224
            if args.cuda:
                loc_label = loc_label.cuda()
            for j in range(images.size(0)):
                response_map = conv5s[i][j]
                response_map = F.interpolate(response_map, size=resize)
                # response_map = F.upsample(response_map, size = [resize, resize])

                response_map = response_map.mean(0)
                rawmaxidx = response_map.view(-1).max(0)[1]
                idx = []
                for d in list(response_map.size())[::-1]:
                    idx.append(rawmaxidx % d)
                    rawmaxidx = rawmaxidx / d
                loc_label[j, 0] = (idx[1].float() + 0.5) / response_map.size(0)
                loc_label[j, 1] = (idx[0].float() + 0.5) / response_map.size(1)
            weak_loc.append(loc_label)
        weak_loss1 = F.smooth_l1_loss(attens[0], weak_loc[0])
        weak_loss2 = F.smooth_l1_loss(attens[1], weak_loc[1])
        apn_loss = weak_loss1 + weak_loss2
        apn_loss.backward()
        opt2.step()
        t1 = time.time()
        
        if (iteration % 20) == 0:
            print(" [*] pre_apn_epoch[%d], || pre_apn_iter %d || pre_apn_loss: %.4f || Timer: %.4fsec"%
                  (apn_epoch, iteration, apn_loss.item(), (t1 - t0)))
            with open('./result_racnn/%s.txt' % 'pre_apn', 'a') as acc_file:
                acc_file.write(" [*] pre_apn_epoch[%d], || pre_apn_iter %d || pre_apn_loss: %.4f || Timer: %.4fsec\n"%
                    (apn_epoch, iteration, apn_loss.item(), (t1 - t0)))
        logger.scalar_summary('pre_apn_loss', apn_loss.item(), iteration + 1)

    state = {
            'apn_iter': 10000,
            'state_dict': net.state_dict(),
            'apn_epoch': apn_epoch,
            'apn_steps': apn_steps
        }
    save_checkpoint(type='pre_APN', state=state)

    #20000
    return 10000, apn_epoch, apn_steps

def validate(valloader, iteration, type):
    with torch.no_grad():
        net.eval()
        corrects1 = 0
        corrects2 = 0
        corrects3 = 0
        cnt = 0
        val_cls_losses = []
        val_apn_losses = []

        t0 = time.time()
        for j, (val_images, val_labels) in enumerate(valloader):
            accuracy_list = []   
            if args.cuda:
                val_images = val_images.cuda()
                val_labels = val_labels.cuda()
            cnt += val_labels.size(0)

            logits, _, _ = net(val_images)
            preds = []
            for i in range(len(val_labels)):
                pred = [logit[i][val_labels[i]] for logit in logits]
                preds.append(pred)

            # val_cls_losses = multitask_loss(logits, val_labels)
            val_cls_loss = multitask_loss(logits, val_labels)
            val_apn_loss = pairwise_ranking_loss(preds)

            # val_cls_losses.append(sum(val_cls_los))
            val_cls_losses.append(sum(val_cls_loss))
            val_apn_losses.append(val_apn_loss)

            _, predicted1 = torch.max(logits[0], 1)
            correct1 = (predicted1 == val_labels).sum()
            corrects1 += correct1

            _, predicted2 = torch.max(logits[1], 1)
            correct2 = (predicted2 == val_labels).sum()
            corrects2 += correct2

            _, predicted3 = torch.max(logits[2], 1)
            correct3 = (predicted3 == val_labels).sum()
            corrects3 += correct3
            if j % 20 == 0:
                print(" || validate %s [*] Iter[%d] %d || " %(type, len(valloader), j))
        t1 = time.time()
        val_cls_losses = torch.stack(val_cls_losses).mean()
        val_apn_losses = torch.stack(val_apn_losses).mean()
        accuracy1 = corrects1.float() / cnt
        accuracy2 = corrects2.float() / cnt
        accuracy3 = corrects3.float() / cnt
        accuracy_list.append(accuracy1.item())
        accuracy_list.append(accuracy2.item())
        accuracy_list.append(accuracy3.item())


        logger.scalar_summary('val_cls_loss', val_cls_losses.item(), iteration + 1)
        logger.scalar_summary('val_rank_loss', val_apn_losses.item(), iteration + 1)
        logger.scalar_summary('val_acc1', accuracy1.item(), iteration + 1)
        logger.scalar_summary('val_acc2', accuracy2.item(), iteration + 1)
        logger.scalar_summary('val_acc3', accuracy3.item(), iteration + 1)
        if type == 'cls':
            print(" cls: [*] Iter %d || Val accuracy1: %.2f || Val accuracy2: %.2f || Val accuracy3: %.2f || Timer: %.4fsec"%
                  (iteration, accuracy1.item(), accuracy2.item(), accuracy3.item(), (t1-t0)))
            with open('./result_racnn/%s.txt' % 'cls_val_accuracy', 'a') as acc_file:
                acc_file.write(" cls: [*] Iter %d || Val accuracy1: %.2f || Val accuracy2: %.2f || Val accuracy3: %.2f || Timer: %.4fsec\n"%
                  (iteration, accuracy1.item(), accuracy2.item(), accuracy3.item(), (t1-t0)))
        else:
            print(" apn: [*] Iter %d || Val accuracy1: %.2f, Val accuracy2: %.2f, Val accuracy3: %.2f || Timer: %.4fsec"%
                  (iteration, accuracy1.item(), accuracy2.item(), accuracy3.item(), (t1-t0)))
            with open('./result_racnn/%s.txt' % 'apn_val_accuracy', 'a') as acc_file:
                acc_file.write(" cls: [*] Iter %d || Val accuracy1: %.2f || Val accuracy2: %.2f || Val accuracy3: %.2f || Timer: %.4fsec\n"%
                  (iteration, accuracy1.item(), accuracy2.item(), accuracy3.item(), (t1-t0)))
    net.train()
    return accuracy_list
# 测试函数
def test(test_loader, net):
    csv_map = OrderedDict({'filename1': [], 'probability1': [], 'filename2': [], 'probability2': [],
                           'filename3': [], 'probability3': []})
    # switch to evaluate mode
    net.eval()
    for i, (images, filepath) in enumerate(tqdm(test_loader)):
        # bs, ncrops, c, h, w = images.size()
        filepath = [os.path.basename(i) for i in filepath]
        image_var = torch.tensor(images, requires_grad=False)  # for pytorch 0.4

        with torch.no_grad():
            # y_pred = net(image_var)
            # 使用softmax函数将图片预测结果转换成类别概率
            # smax = nn.Softmax(1)
            # smax_out = smax(y_pred)
            logits, _, _ = net(image_var)
            smax1 = nn.Softmax(1)
            smax_out1 = smax(smax1)
            smax2 = nn.Softmax(1)
            smax_out2 = smax(smax2)
            smax3 = nn.Softmax(1)
            smax_out3 = smax(smax3)




        # 保存图片名称与预测概率
        csv_map['filenam1e1'].extend(filepath)
        for output in smax_out1:
            prob = ';'.join([str(i) for i in output.data.tolist()])
            csv_map['probability1'].append(prob)

        # 保存图片名称与预测概率
        csv_map['filenam1e2'].extend(filepath)
        for output in smax_out2:
            prob = ';'.join([str(i) for i in output.data.tolist()])
            csv_map['probability2'].append(prob)

        # 保存图片名称与预测概率
        csv_map['filenam1e3'].extend(filepath)
        for output in smax_out3:
            prob = ';'.join([str(i) for i in output.data.tolist()])
            csv_map['probability3'].append(prob)

    result = pd.DataFrame(csv_map)
    result.to_csv('result.csv', index=False)
    result['probability1'] = result['probability1'].map(lambda x: [float(i) for i in x.split(';')])
    result['probability2'] = result['probability2'].map(lambda x: [float(i) for i in x.split(';')])
    result['probability3'] = result['probability3'].map(lambda x: [float(i) for i in x.split(';')])

    # 转换成提交样例中的格式
    sub_filename1, sub_label1 = [], []
    sub_filename2, sub_label2 = [], []
    sub_filename3, sub_label3 = [], []

    for index, row in result.iterrows():
        # print(index, row)
        sub_filename1.append(row['filename1'])
        pred_label1 = np.argmax(row['probability1'])
        sub_label1.append(pred_label1)

        sub_filename2.append(row['filename2'])
        pred_label2 = np.argmax(row['probability2'])
        sub_label2.append(pred_label2)

        sub_filename3.append(row['filename3'])
        pred_label3 = np.argmax(row['probability3'])
        sub_label3.append(pred_label3)

    sub_list1 = []
    ll = len(sub_filename1)
    for l in range(ll):
        sub_dict = {}
        sub_dict['image_id'] = sub_filename1[l]
        sub_dict['disease_class'] = int(sub_label1[l])
        sub_list1.append(sub_dict)
    # print('saa', sub_list[0:10])
    with open('./result_racnn/submit1.json', "w") as f:
        json.dump(sub_list1, f)

    sub_list2 = []
    ll = len(sub_filename2)
    for l in range(ll):
        sub_dict = {}
        sub_dict['image_id'] = sub_filename2[l]
        sub_dict['disease_class'] = int(sub_label2[l])
        sub_list2.append(sub_dict)
    # print('saa', sub_list[0:10])
    with open('./result_racnn/submit2.json', "w") as f:
        json.dump(sub_list2, f)

    sub_list3 = []
    ll = len(sub_filename3)
    for l in range(ll):
        sub_dict = {}
        sub_dict['image_id'] = sub_filename3[l]
        sub_dict['disease_class'] = int(sub_label1[l])
        sub_list13.append(sub_dict)
    # print('saa', sub_list[0:10])
    with open('./result_racnn/submit3.json', "w") as f:
        json.dump(sub_list3, f)


    print("加载入文件完成...")

    # 生成结果文件，保存在result文件夹中，可用于直接提交
    # submission = pd.DataFrame({'filename': sub_filename, 'label': sub_label})
    # submission.to_csv('./result/%s/submission.csv' % file_name, header=None, index=False)

    return

def adjust_learning_rate(optimizer, gamma, steps, _lr):
    lr = _lr * (gamma ** (steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    # ----------------load data-----------------------
    # 随机种子
    np.random.seed(666)
    torch.manual_seed(666)
    torch.cuda.manual_seed_all(666)
    random.seed(666)

    std = 1. / 255.
    means = [109.97 / 255., 127.34 / 255., 123.88 / 255.]
    normalize = transforms.Normalize(mean=means, std=[std] * 3)

    # 读取测试图片列表
    test_data_list = pd.read_csv('./AgriculturalDisease_testA/test.csv')

    # 测试集图片变换
    testset = TestDataset(test_data_list,
                          transform=transforms.Compose([
                              transforms.Resize((448, 448)),
                              transforms.CenterCrop(448),
                              transforms.ToTensor(),
                              normalize,
                          ]))

    testloader = data.DataLoader(testset, batch_size=int(args.batch_size*2),
                                 shuffle=True, collate_fn=CUB_collate, num_workers=args.num_workers)


    # 读取最佳模型，预测测试集，并生成可直接提交的结果文件
    print('test--------------------------------')
    best_model = torch.load('./racnn_models/checkpoint_%s.pth.tar'%'cls_and_rank')
    net.load_state_dict(best_model['state_dict'])
    test(test_loader=testloader, model=net)
    print(" [*] Train done")            

