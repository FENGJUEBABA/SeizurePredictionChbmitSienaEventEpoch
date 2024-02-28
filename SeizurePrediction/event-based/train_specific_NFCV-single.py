# -*-coding:utf-8 -*-
import math
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from thop import profile

from evaluation import get_Confusion_Matrix, get_accuracy, get_auc, get_f1score, get_fpr, get_mcc, \
    get_sensitivity, get_specificity, nFoldCrossVal
from model import DACNN

data_path = 'D:/learn/dataset/preprocess/chbmitmul10_event'
# PATIENTS = ['01']
C, H, W = 22, 80, 32

lr = 1e-4  # 学习率
epoch = 1  # 训练轮数
batch_size = 128
n_fold = 5  # 定义交叉验证次数
print_per_epoch = 100  # 多少个epoch输出一次，用于控制输出的频率
dataset_name = 'chb'  # ‘chb’或'PN'
# 'time_domain','frequency_domain','timefrequency_domain','mfcc_domain','ht_domain','hht_domain'
domain_name = 'time_domain'
result_path = './result_NFCV'  # 保存结果的路径
checkpoint_dir = "./checkpoint_NFCV"  # 保存训练好的模型的路径
result_filename = 'model-specific-time'  # 结果名称
PATIENTS = []
if dataset_name == 'chb':
    PATIENTS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18',
                '19', '20', '21', '22', '23']
elif dataset_name == 'PN':
    PATIENTS = ['01', '03', '05', '06', '07', '09', '10', '11', '12', '13', '14', '16', '17']
else:
    print('不支持的数据集名称')
PATIENTS = ['01']

if not os.path.lexists(result_path):
    os.makedirs(result_path)
if not os.path.lexists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
# 加标签
all_start = 0
result_sum = open(os.path.join(result_path, result_filename + '-sum.txt'), 'w')
for patient in PATIENTS:
    result_one = open(os.path.join(result_path, '{}{}-{}.txt'.format(dataset_name, patient, result_filename)), 'w')
    XZ = []
    XF = []
    path = os.path.join(data_path, dataset_name + patient)
    if os.path.lexists(path):
        clas = os.listdir(path)
        for cla in clas:
            if cla == 'interictal':
                npys_path = os.path.join(path, cla, domain_name)
                npys = os.listdir(npys_path)
                for npy in npys:
                    XF.append(os.path.join(npys_path, npy))
            if cla == 'preictal':
                npys_path = os.path.join(path, cla, domain_name)
                npys = os.listdir(npys_path)
                for npy in npys:
                    XZ.append(os.path.join(npys_path, npy))
    # nums.update({patient: [len(XZ), len(XF)]})
    print('病人{}{}：正样本个数={}，负样本个数={}'.format(dataset_name, patient, len(XZ), len(XF)))
    print('病人{}{}：正样本个数={}，负样本个数={}'.format(dataset_name, patient, len(XZ), len(XF)), file=result_one)
    # print(num)
    X = []
    Y = []
    num = min(len(XZ), len(XF))
    print(num * 2)
    for i in range(num):
        X.append(XZ[i])
        Y.append([0, 1])
        X.append(XF[i])
        Y.append([1, 0])

    # 打乱数据，同时打乱数据和标签
    state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(state)
    np.random.shuffle(Y)
    # 创建网络模型
    model = DACNN().to('cuda:0')
    #
    input_data = torch.randn(batch_size, C, H, W).to('cuda:0')
    FLOPs, Params = profile(model, inputs=(input_data,))  # 运算量FLOPs,网络参数量 Params
    print('FLOPs= {}M, Params= {}K'.format(FLOPs / pow(1000, 2), Params / pow(1000, 1)))
    print('FLOPs= {}M, Params= {}K'.format(FLOPs / pow(1000, 2), Params / pow(1000, 1)), file=result_one)
    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 优化器

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_train_step, total_test_step = 0, 0  # 训练次数
    print('lr={},epoch={},batch_size={},n_fold={}'.format(lr, epoch, batch_size, n_fold), file=result_one)
    n_fold_acc, n_fold_spe, n_fold_sen, n_fold_auc, n_fold_mcc, n_fold_f1, n_fold_fpr = 0, 0, 0, 0, 0, 0, 0
    total_fold_start = time.time()
    for nf in range(n_fold):  # 交叉验证
        every_fold_start = time.time()
        x_train_filename, y_train, x_test_filename, y_test = nFoldCrossVal(X, Y, n_fold, nf + 1)
        print('=============================病人{}第{}折交叉验证开始========================'.format(patient, nf + 1))
        print('=============================病人{}第{}折交叉验证开始========================'.format(patient, nf + 1),
              file=result_one)
        # 训练开始
        total_epoch_start = time.time()
        for i in range(epoch):
            y_train_score = []  # 记录全部训练样本的预测值，用于求总的roc
            every_epoch_start = time.time()
            train_matrix = torch.Tensor([[0, 0], [0, 0]])
            print('============================第{}轮训练开始============================='.format(i + 1))
            print('============================第{}轮训练开始============================='.format(i + 1),
                  file=result_one)
            # 训练开始i
            batch_num = math.ceil(len(x_train_filename) / batch_size)  # batch的个数 ceil向上取整
            batch_remainder = len(x_train_filename) % batch_size  # 最后一个batch的大小
            true_batch_size = []
            if batch_remainder == 0:
                for num in range(batch_num):
                    true_batch_size.append(batch_size)
            else:
                for num in range(batch_num - 1):
                    true_batch_size.append(batch_size)
                true_batch_size.append(batch_remainder)
            for j in range(batch_num):
                total_train_step += 1
                data_batch = np.zeros((true_batch_size[j], C, H, W))
                for k in range(true_batch_size[j]):
                    data_batch[k, :] = (np.load(x_train_filename[j * true_batch_size[j] + k])).reshape((C, H, W))
                x_train_batch = torch.Tensor(data_batch).to('cuda:0')
                y_train_batch = torch.Tensor(y_train[j * batch_size:j * batch_size + true_batch_size[j]]).to(
                    'cuda:0')
                y_predict_batch = model(x_train_batch)
                loss = loss_fn(y_predict_batch, y_train_batch)

                # 优化器优化模型
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (j + 1) % print_per_epoch == 0:
                    # print('X_train.size==', x_train_batch.size())
                    # print('Y_train.size==', y_train_batch.size())
                    print(
                        '{}-{}:{},Loss:{}'.format(j * batch_size, j * batch_size + true_batch_size[j], total_train_step,
                                                  loss.item()))
                    print(
                        '{}-{}:{},Loss:{}'.format(j * batch_size, j * batch_size + true_batch_size[j], total_train_step,
                                                  loss.item()), file=result_one)

                # 评价指标
                train_matrix += get_Confusion_Matrix(y_predict_batch, y_train_batch)
                train_batch_acc = get_accuracy(train_matrix)
                train_batch_spe = get_specificity(train_matrix)
                train_batch_sen = get_sensitivity(train_matrix)

                train_batch_auc = get_auc(y_true=y_train_batch, y_score=y_predict_batch)
                for b in y_predict_batch.to('cpu').detach().numpy():
                    y_train_score.append(b)

                if (j + 1) % print_per_epoch == 0:
                    print('batch {} : acc={} , spe={} , sen={} , auc={}'.format(j + 1, train_batch_acc, train_batch_spe,
                                                                                train_batch_sen, train_batch_auc))
                    print('batch {} : acc={} , spe={} , sen={}'.format(j + 1, train_batch_acc, train_batch_spe,
                                                                       train_batch_sen), file=result_one)
            train_total_acc = get_accuracy(train_matrix)
            train_total_spe = get_specificity(train_matrix)
            train_total_sen = get_sensitivity(train_matrix)
            train_total_auc = roc_auc_score(y_true=y_train, y_score=y_train_score)
            every_epoch_end = time.time()
            print(
                'epoch {} : acc={} , spe={} , sen={} , time={}s, auc={}'.format(i + 1, train_total_acc, train_total_spe,
                                                                                train_total_sen,
                                                                                int(every_epoch_end - every_epoch_start),
                                                                                train_total_auc))
            print(
                'epoch {} : acc={} , spe={} , sen={} , time={}s, auc={}'.format(i + 1, train_total_acc, train_total_spe,
                                                                                train_total_sen,
                                                                                int(every_epoch_end - every_epoch_start),
                                                                                train_total_auc),
                file=result_one)
        checkpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir,
                                            '{}_{}_{}-FCV_epoch{}_lr{:.0e}_bs{}.pth'.format(str(dataset_name + patient),
                                                                                            result_filename,
                                                                                            int(nf + 1),
                                                                                            int(epoch), lr,
                                                                                            int(batch_size))))
        total_epoch_end = time.time()
        print('总训练时间为：', int(total_epoch_end - total_epoch_start), '秒')
        print('总训练时间为：', int(total_epoch_end - total_epoch_start), '秒', file=result_one)

        # 测试开始
        with torch.no_grad():
            y_test_score = []  # 记录全部测试样本的预测值，用于求总的roc
            test_start = time.time()
            test_matrix = torch.Tensor([[0, 0], [0, 0]])
            # correct_num = 0  # 统计预测正确的个数
            print('==================================验证开始===================================')
            print('==================================验证开始===================================', file=result_one)
            batch_num = math.ceil(len(x_test_filename) / batch_size)  # batch的个数 ceil向上取整
            batch_remainder = len(x_test_filename) % batch_size  # 最后一个batch的大小
            true_batch_size = []
            if batch_remainder == 0:
                for num in range(batch_num):
                    true_batch_size.append(batch_size)
            else:
                for num in range(batch_num - 1):
                    true_batch_size.append(batch_size)
                true_batch_size.append(batch_remainder)
            for j in range(batch_num):
                total_test_step += 1
                data_batch = np.zeros((true_batch_size[j], C, H, W))
                for k in range(true_batch_size[j]):
                    data_batch[k, :] = (np.load(x_test_filename[j * true_batch_size[j] + k])).reshape((C, H, W))
                x_test_batch = torch.Tensor(data_batch).to('cuda:0')
                y_test_batch = torch.Tensor(y_test[j * batch_size:j * batch_size + true_batch_size[j]]).to('cuda:0')
                y_predict_batch = model(x_test_batch)
                loss = loss_fn(y_predict_batch, y_test_batch)

                if (j + 1) % print_per_epoch == 0:
                    # print('X_test.size==', x_test_batch.size())
                    # print('Y_test.size==', y_test_batch.size())
                    print(
                        '{}-{}:{},Loss:{}'.format(j * batch_size, j * batch_size + true_batch_size[j], total_test_step,
                                                  loss.item()))
                    print(
                        '{}-{}:{},Loss:{}'.format(j * batch_size, j * batch_size + true_batch_size[j], total_test_step,
                                                  loss.item()), file=result_one)
                # 计算准确度
                test_matrix += get_Confusion_Matrix(y_predict_batch, y_test_batch)
                test_batch_acc = get_accuracy(test_matrix)
                test_batch_spe = get_specificity(test_matrix)
                test_batch_sen = get_sensitivity(test_matrix)
                test_batch_auc = get_auc(y_true=y_test_batch, y_score=y_predict_batch)
                for b in y_predict_batch.to('cpu').detach().numpy():
                    y_test_score.append(b)

                if (j + 1) % print_per_epoch == 0:
                    print('batch {} : acc={} , spe={} , sen={}'.format(j + 1, test_batch_acc, test_batch_spe,
                                                                       test_batch_sen))
                    print(
                        'batch {} : acc={} , spe={} , sen={}'.format(j + 1, test_batch_acc, test_batch_spe,
                                                                     test_batch_sen),
                        file=result_one)
            test_total_acc = get_accuracy(test_matrix)
            test_total_spe = get_specificity(test_matrix)
            test_total_sen = get_sensitivity(test_matrix)
            test_total_auc = roc_auc_score(y_true=y_test, y_score=y_test_score)
            test_total_mcc = get_mcc(test_matrix)
            test_total_f1 = get_f1score(test_matrix)
            test_total_fpr = get_fpr(test_matrix)
            test_end = time.time()
            print('total acc={}, spe={}, sen={}, test_time={}s, auc={}, mcc={},f1={},fpr={}'.format(test_total_acc,
                                                                                                    test_total_spe,
                                                                                                    test_total_sen,
                                                                                                    int(test_end - test_start),
                                                                                                    test_total_auc,
                                                                                                    test_total_mcc,
                                                                                                    test_total_f1,
                                                                                                    test_total_fpr))
            print('total acc={}, spe={}, sen={}, test_time={}s, auc={}, mcc={},f1={},fpr={}'.format(test_total_acc,
                                                                                                    test_total_spe,
                                                                                                    test_total_sen,
                                                                                                    int(test_end - test_start),
                                                                                                    test_total_auc,
                                                                                                    test_total_mcc,
                                                                                                    test_total_f1,
                                                                                                    test_total_fpr),
                  file=result_one)
            print('test matrix: ', test_matrix)
            print('test matrix: ', test_matrix, file=result_one)
        every_fold_end = time.time()
        print('第{}折所用的时间是{}秒'.format(nf + 1, int(every_fold_end - every_fold_start)))
        print('第{}折所用的时间是{}秒'.format(nf + 1, int(every_fold_end - every_fold_start)), file=result_one)
        n_fold_acc += test_total_acc
        n_fold_spe += test_total_spe
        n_fold_sen += test_total_sen
        n_fold_auc += test_total_auc
        n_fold_mcc += test_total_mcc
        n_fold_f1 += test_total_f1
        n_fold_fpr += test_total_fpr
    average_acc = n_fold_acc / n_fold
    average_spe = n_fold_spe / n_fold
    average_sen = n_fold_sen / n_fold
    average_auc = n_fold_auc / n_fold
    average_mcc = n_fold_mcc / n_fold
    average_f1 = n_fold_f1 / n_fold
    average_fpr = n_fold_fpr / n_fold
    total_fold_end = time.time()
    print('{}{}:{}折交叉验证的平均acc={}, spe={}, sen={}, time={}s, auc={}, mcc={},f1={},fpr={}' \
          .format(dataset_name, patient, n_fold, average_acc, average_spe, average_sen,
                  int(total_fold_end - total_fold_start), average_auc, average_mcc, average_f1, average_fpr))
    print('{}{}:{}折交叉验证的平均acc={}, spe={}, sen={}, time={}s, auc={}, mcc={},f1={},fpr={}' \
          .format(dataset_name, patient, n_fold, average_acc, average_spe, average_sen,
                  int(total_fold_end - total_fold_start), average_auc, average_mcc, average_f1, average_fpr),
          file=result_one)
    print('{}{},{},{},{},{},{},{},{}'.format(dataset_name, patient, average_acc, average_spe, average_sen, average_auc,
                                             average_mcc, average_f1, average_fpr), file=result_one)
    print('{}{},{},{},{},{},{},{},{}'.format(dataset_name, patient, average_acc, average_spe, average_sen, average_auc,
                                             average_mcc, average_f1, average_fpr), file=result_sum)
    result_one.close()
all_end = time.time()
print('实验总共用时:', int(all_end - all_start), file=result_sum)
result_sum.close()
