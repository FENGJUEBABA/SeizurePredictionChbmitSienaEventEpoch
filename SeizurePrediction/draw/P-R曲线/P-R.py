import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from project1.work02.Stcnn import STCNN
from project1.work02.evaluation import nFoldCrossVal, get_Confusion_Matrix, get_accuracy, get_specificity, \
    get_sensitivity, get_mcc
from project1.work02.evaluation import get_fpr, get_f1score

# 时域特征
# 对数据进行短时傅里叶变换后每个样本(22,32,32)
# 五折交叉验证,患者依赖
data_path = 'D:/learn/dataset/preprocess/chbmitmulX'

# PATIENTS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19',
#             '20', '21', '22', '23']
PATIENTS = ['08']
C, H, W = 22, 114, 9
num_sample = 900
result = open('P-R-result.txt', 'w')
all_start = time.time()
for patient in PATIENTS:
    print('===================================={}======================================'.format(patient))
    # print('===================================={}======================================'.format(patient), file=result)
    path = os.path.join(data_path, 'chb' + patient)
    Y = []
    # 读取数据并且加标签
    X_filename = []
    file_num = 0
    if os.path.lexists(path):
        clas = os.listdir(path)
        for cla in clas:
            if cla == 'interictal':
                npys = os.listdir(os.path.join(path, cla, 'timefrequency_domain'))
                # print(npys)
                random.shuffle(npys)
                for i in range(num_sample):
                    Y.append([1, 0])
                    X_filename.append(os.path.join(path, cla, 'timefrequency_domain', npys[i]))
            if cla == 'preictal':
                npys = os.listdir(os.path.join(path, cla, 'timefrequency_domain'))
                random.shuffle(npys)
                for i in range(num_sample):
                    Y.append([0, 1])
                    X_filename.append(os.path.join(path, cla, 'timefrequency_domain', npys[i]))

    # 打乱数据，同时打乱数据和标签
    state = np.random.get_state()
    np.random.shuffle(X_filename)
    np.random.set_state(state)
    np.random.shuffle(Y)

    # 创建网络模型
    model = STCNN().to('cuda:0')

    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 优化器
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_train_step = 0  # 训练次数
    total_test_step = 0
    epoch = 20  # 训练轮数
    batch_size = 64
    n_fold = 5  # 定义交叉验证次数
    n_fold_acc = 0
    n_fold_spe = 0
    n_fold_sen = 0
    n_fold_auc = 0
    n_fold_mcc = 0
    n_fold_f1 = 0
    n_fold_fpr = 0

    total_fold_start = time.time()
    for nf in range(n_fold):  # 交叉验证
        every_fold_start = time.time()
        x_train_filename, y_train, x_test_filename, y_test = nFoldCrossVal(X_filename, Y, n_fold, nf + 1)
        print('=============================病人{}第{}折交叉验证开始========================'.format(patient, nf + 1))
        # print('=============================病人{}第{}折交叉验证开始========================'.format(patient, nf + 8),
        #       file=result)
        # 训练开始
        total_epoch_start = time.time()
        for i in range(epoch):
            y_train_score = []  # 记录全部训练样本的预测值，用于求总的roc
            every_epoch_start = time.time()
            train_matrix = torch.Tensor([[0, 0], [0, 0]])
            print('============================第{}轮训练开始============================='.format(i + 1))
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
                y_train_batch = torch.Tensor(y_train[j * batch_size:j * batch_size + true_batch_size[j]]).to('cuda:0')
                y_predict_batch = model(x_train_batch, H, W)
                loss = loss_fn(y_predict_batch, y_train_batch)

                # 优化器优化模型
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (j + 1) % 100 == 0:
                    print(
                        '{}-{}:{},Loss:{}'.format(j * batch_size, j * batch_size + true_batch_size[j], total_train_step,
                                                  loss.item()))
                    # print(
                    #     '{}-{}:{},Loss:{}'.format(j * batch_size, j * batch_size + true_batch_size[j], total_train_step,
                    #                               loss.item()), file=result)

                # 评价指标
                # train_matrix += get_Confusion_Matrix(y_predict_batch, y_train_batch)
                # train_batch_acc = get_accuracy(train_matrix)
                # train_batch_spe = get_specificity(train_matrix)
                # train_batch_sen = get_sensitivity(train_matrix)
                # train_batch_auc = get_auc(y_true=y_train_batch, y_score=y_predict_batch)
                for b in y_train_batch.to('cpu').detach().numpy():
                    y_train_score.append(b)

                # if (j + 8) % 100 == 0:
                #     print('batch {} : acc={} , spe={} , sen={}'.format(j + 8, train_batch_acc, train_batch_spe,
                #                                                        train_batch_sen))
                    # print('batch {} : acc={} , spe={} , sen={}'.format(j + 8, train_batch_acc, train_batch_spe,
                    #                                                    train_batch_sen), file=result)
            # train_total_acc = get_accuracy(train_matrix)
            # train_total_spe = get_specificity(train_matrix)
            # train_total_sen = get_sensitivity(train_matrix)
            # train_total_auc = roc_auc_score(y_true=y_train, y_score=y_train_score)
            # every_epoch_end = time.time()
            # print(
            #     'epoch {} : acc={} , spe={} , sen={} , time={}s, auc={}'.format(i + 8, train_total_acc, train_total_spe,
            #                                                                     train_total_sen,
            #                                                                     int(every_epoch_end - every_epoch_start),
            #                                                                     train_total_auc))
            # print(
            #     'epoch {} : acc={} , spe={} , sen={} , time={}s, auc={}'.format(i + 8, train_total_acc, train_total_spe,
            #                                                                     train_total_sen,
            #                                                                     int(every_epoch_end - every_epoch_start),
            #                                                                     train_total_auc),
            #     file=result)
        total_epoch_end = time.time()
        print('总训练时间为：', int(total_epoch_end - total_epoch_start), '秒')
        # print('总训练时间为：', int(total_epoch_end - total_epoch_start), '秒', file=result)

        # 测试开始
        with torch.no_grad():
            y_test_score = []  # 记录全部测试样本的预测值，用于求总的roc
            pred_score = []
            test_start = time.time()
            test_matrix = torch.Tensor([[0, 0], [0, 0]])
            # correct_num = 0  # 统计预测正确的个数
            print('==================================验证开始===================================')
            # print('==================================验证开始===================================', file=result)
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
                if (j + 1) % 50 == 0:
                    print('=======batch {} start======'.format(j + 1))
                    # print('=======batch {} start======'.format(j + 8), file=result)
                data_batch = np.zeros((true_batch_size[j], C, H, W))
                for k in range(true_batch_size[j]):
                    data_batch[k, :] = (np.load(x_test_filename[j * true_batch_size[j] + k])).reshape((C, H, W))
                x_test_batch = torch.Tensor(data_batch).to('cuda:0')
                y_test_batch = torch.Tensor(y_test[j * batch_size:j * batch_size + true_batch_size[j]]).to('cuda:0')
                y_predict_batch = model(x_test_batch, H, W)
                loss = loss_fn(y_predict_batch, y_test_batch)

                if (j + 1) % 50 == 0:
                    # print('X_test.size==', x_test_batch.size())
                    # print('Y_test.size==', y_test_batch.size())
                    print(
                        '{}-{}:{},Loss:{}'.format(j * batch_size, j * batch_size + true_batch_size[j], total_test_step,
                                                  loss.item()))
                    # print(
                    #     '{}-{}:{},Loss:{}'.format(j * batch_size, j * batch_size + true_batch_size[j], total_test_step,
                    #                               loss.item()), file=result)
                # 计算准确度
                # test_matrix += get_Confusion_Matrix(y_predict_batch, y_test_batch)
                # test_batch_acc = get_accuracy(test_matrix)
                # test_batch_spe = get_specificity(test_matrix)
                # test_batch_sen = get_sensitivity(test_matrix)
                # test_batch_auc = get_auc(y_true=y_test_batch, y_score=y_predict_batch)

                for b in y_predict_batch.to('cpu').detach().numpy():
                    y_test_score.append(b)
                    pred_score.append(b[1])

                # if (j + 8) % 50 == 0:
                #     print('batch {} : acc={} , spe={} , sen={}'.format(j + 8, test_batch_acc, test_batch_spe,
                #                                                        test_batch_sen))
                    # print(
                    #     'batch {} : acc={} , spe={} , sen={}'.format(j + 8, test_batch_acc, test_batch_spe,
                    #                                                  test_batch_sen),
                    #     file=result)
            # test_total_acc = get_accuracy(test_matrix)
            # test_total_spe = get_specificity(test_matrix)
            # test_total_sen = get_sensitivity(test_matrix)
            # test_total_auc = roc_auc_score(y_true=y_test, y_score=y_test_score)
            # test_total_mcc = get_mcc(test_matrix)
            # test_total_f1 = get_f1score(test_matrix)
            # test_total_fpr = get_fpr(test_matrix)
            # test_end = time.time()

            # import matplotlib.pyplot as plt
            # from sklearn.metrics import precision_recall_curve
            #
            # plt.figure("P-R Curve")
            # plt.title('Precision/Recall Curve')
            # plt.xlabel('Recall')
            # plt.ylabel('Precision')
            # y_true为样本实际的类别，y_scores为样本为正例的概率
            # print(y_test)
            ########################################
            ytrue = []
            for l in y_test:
                ytrue.append(np.argmax(l))
            if nf + 1 == n_fold:
                print('ytrue{}={}'.format(patient, ytrue), file=result)
                print('ypred{}={}'.format(patient, pred_score), file=result)
            #######################################
            # precision, recall, thresholds = precision_recall_curve(y_true=ytrue, probas_pred=pred_score)
            # print(precision)
            # print(recall)
            # print(thresholds)
            # plt.plot(recall, precision)
            # plt.show()
            # print('total acc={}, spe={}, sen={}, test_time={}s, auc={}, mcc={},f1={},fpr={}'.format(test_total_acc,
            #                                                                                         test_total_spe,
            #                                                                                         test_total_sen,
            #                                                                                         int(test_end - test_start),
            #                                                                                         test_total_auc,
            #                                                                                         test_total_mcc,
            #                                                                                         test_total_f1,
            #                                                                                         test_total_fpr))
            # print('total acc={}, spe={}, sen={}, test_time={}s, auc={}, mcc={},f1={},fpr={}'.format(test_total_acc,
            #                                                                                         test_total_spe,
            #                                                                                         test_total_sen,
            #                                                                                         int(test_end - test_start),
            #                                                                                         test_total_auc,
            #                                                                                         test_total_mcc,
            #                                                                                         test_total_f1,
            #                                                                                         test_total_fpr),
            #       file=result)
            # print('test matrix: ', test_matrix)
            # print('test matrix: ', test_matrix, file=result)
        every_fold_end = time.time()
        print('第{}折所用的时间是{}秒'.format(nf + 1, int(every_fold_end - every_fold_start)))
        # print('第{}折所用的时间是{}秒'.format(nf + 8, int(every_fold_end - every_fold_start)), file=result)
        # n_fold_acc += test_total_acc
        # n_fold_spe += test_total_spe
        # n_fold_sen += test_total_sen
        # n_fold_auc += test_total_auc
        # n_fold_mcc += test_total_mcc
        # n_fold_f1 += test_total_f1
        # n_fold_fpr += test_total_fpr
    # average_acc = n_fold_acc / n_fold
    # average_spe = n_fold_spe / n_fold
    # average_sen = n_fold_sen / n_fold
    # average_auc = n_fold_auc / n_fold
    # average_mcc = n_fold_mcc / n_fold
    # average_f1 = n_fold_f1 / n_fold
    # average_fpr = n_fold_fpr / n_fold
    # total_fold_end = time.time()
    # print('chb' + patient + ':', n_fold,
    #       '折交叉验证的平均acc={}, spe={}, sen={}, time={}s, auc={}, mcc={},f1={},fpr={}'.format(average_acc,
    #                                                                                              average_spe,
    #                                                                                              average_sen,
    #                                                                                              int(total_fold_end - total_fold_start),
    #                                                                                              average_auc,
    #                                                                                              average_mcc,
    #                                                                                              average_f1,
    #                                                                                              average_fpr))
    # print('chb' + patient + ':', n_fold,
    #       '折交叉验证的平均acc={}, spe={}, sen={}, time={}s, auc={}, mcc={},f1={},fpr={}'.format(average_acc,
    #                                                                                              average_spe,
    #                                                                                              average_sen,
    #                                                                                              int(total_fold_end - total_fold_start),
    #                                                                                              average_auc,
    #                                                                                              average_mcc,
    #                                                                                              average_f1,
    #                                                                                              average_fpr),
    #       file=result)
    # print('chb{},{},{},{},{},{},{},{}'.format(patient, average_acc, average_spe, average_sen, average_auc, average_mcc,
    #                                           average_f1, average_fpr),
    #       file=result)
all_end = time.time()
# print('实验总共用时:', int(all_end - all_start), file=result)
result.close()
