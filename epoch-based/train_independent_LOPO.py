import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from evaluation import get_Confusion_Matrix, get_accuracy, get_specificity, get_sensitivity, get_mcc, get_f1score, \
    get_fpr
from project1.work02.Stcnn import STCNN

'''跨病人'''
# 取出其中一个病人的数据作为验证集,用余下其他病人的数据作为训练集 LOPO
data_path = 'E:/dataset/preprocess/chbmitmulX'

C, H, W = 22, 32, 32
lr = 1e-4  # 学习率
epoch = 1  # 训练轮数
batch_size = 128
print_per_epoch = 100  # 多少个epoch输出一次，用于控制输出的频率
dataset_name = 'chb'  # ‘chb’或'PN'
# 'time_domain','frequency_domain','timefrequency_domain','mfcc_domain','ht_domain','hht_domain'
domain_name = 'time_domain'
result_path = './result_LOPO'  # 保存结果的路径
checkpoint_dir = "./checkpoint_LOPO"  # 保存训练好的模型的路径
result_filename = 'model-lopo-time'  # 结果名称
PATIENTS = []
if dataset_name == 'chb':
    PATIENTS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18',
                '19', '20', '21', '22', '23']
elif dataset_name == 'PN':
    PATIENTS = ['01', '03', '05', '06', '07', '09', '10', '11', '12', '13', '14', '16', '17']
else:
    print('不支持的数据集名称')
# 创建目录
if not os.path.lexists(result_path):
    os.makedirs(result_path)
if not os.path.lexists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
# 读取数据并且加标签
X_dict = {}
result_sum = open(os.path.join(result_path, result_filename + '-sum.txt'), 'w')
Y = []
X = []
# 加标签
for patient in PATIENTS:
    path = os.path.join(data_path, dataset_name + patient)
    if os.path.lexists(path):
        clas = os.listdir(path)
        for cla in clas:
            if cla == 'interictal':
                npys_path = os.path.join(path, cla, domain_name)
                npys = os.listdir(npys_path)
                for npy in npys:
                    Y.append([1, 0])
                    X.append(os.path.join(npys_path, npy))
            if cla == 'preictal':
                npys_path = os.path.join(path, cla, domain_name)
                npys = os.listdir(npys_path)
                for npy in npys:
                    Y.append([0, 1])
                    X.append(os.path.join(npys_path, npy))

all_acc, all_spe, all_sen, all_auc, all_mcc, all_f1, all_fpr = 0, 0, 0, 0, 0, 0, 0
start = time.time()
for patient in PATIENTS:
    result_one = open(os.path.join(result_path, '{}{}-{}.txt'.format(dataset_name, patient, result_filename)), 'w')
    print(
        '===============================================病人{}开始==================================='.format(patient))
    print(
        '===============================================病人{}开始==================================='.format(patient),
        file=result_one)
    # 打乱数据，同时打乱数据和标签
    state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(state)
    np.random.shuffle(Y)
    X_dict.update({patient: [X, Y]})
    # 划分训练集和测试集
    x_test, y_test, x_train, y_train = [], [], [], []
    for i in range(len(X)):
        if (dataset_name + patient) in X[i]:  # 选出一个病人的全部数据作为验证集
            x_test.append(X[i])
            y_test.append(Y[i])
        else:
            x_train.append(X[i])
            y_train.append(Y[i])

    # 创建网络模型
    model = STCNN().to('cuda:0')
    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_train_step = 0  # 训练次数

    # 训练开始
    total_epoch_start = time.time()
    for i in range(epoch):
        y_train_score = []  # 记录全部训练样本的预测值，用于求总的roc
        every_epoch_start = time.time()
        train_matrix = torch.Tensor([[0, 0], [0, 0]])
        print('============================第{}轮训练开始============================='.format(i + 1))
        print('============================第{}轮训练开始============================='.format(i + 1), file=result_one)
        # 训练开始i
        batch_num = math.ceil(len(x_train) / batch_size)  # batch的个数 ceil向上取整
        batch_remainder = len(x_train) % batch_size  # 最后一个batch的大小
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
            if (j + 1) % print_per_epoch == 0:
                print('=======batch {} start======'.format(j + 1))
                print('=======batch {} start======'.format(j + 1), file=result_one)
            data_batch = np.zeros((true_batch_size[j], C, H, W))
            for k in range(true_batch_size[j]):
                data_batch[k, :] = (np.load(x_train[j * true_batch_size[j] + k])).reshape((C, H, W))
            x_train_batch = torch.Tensor(data_batch).to('cuda:0')
            y_train_batch = torch.Tensor(y_train[j * batch_size:j * batch_size + true_batch_size[j]]).to('cuda:0')
            y_predict_batch = model(x_train_batch, H, W)
            loss = loss_fn(y_predict_batch, y_train_batch)

            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (j + 1) % print_per_epoch == 0:
                print('{}-{}:{},Loss:{}'.format(j * batch_size, j * batch_size + true_batch_size[j], total_train_step,
                                                loss.item()))
                print('{}-{}:{},Loss:{}'.format(j * batch_size, j * batch_size + true_batch_size[j], total_train_step,
                                                loss.item()), file=result_one)

            # 评价指标
            train_matrix += get_Confusion_Matrix(y_predict_batch, y_train_batch)
            train_batch_acc = get_accuracy(train_matrix)
            train_batch_spe = get_specificity(train_matrix)
            train_batch_sen = get_sensitivity(train_matrix)
            # train_batch_auc = get_auc(y_true=y_train_batch, y_score=y_predict_batch)
            for b in y_predict_batch.to('cpu').detach().numpy():
                y_train_score.append(b)

            if (j + 1) % print_per_epoch == 0:
                print('batch {} : acc={} , spe={} , sen={}'.format(j + 1, train_batch_acc, train_batch_spe,
                                                                   train_batch_sen))
                print(
                    'batch {} : acc={} , spe={} , sen={}'.format(j + 1, train_batch_acc, train_batch_spe,
                                                                 train_batch_sen), file=result_one)
        train_total_acc = get_accuracy(train_matrix)
        train_total_spe = get_specificity(train_matrix)
        train_total_sen = get_sensitivity(train_matrix)
        train_total_auc = roc_auc_score(y_true=y_train, y_score=y_train_score)
        # train_total_f1 = get_f1score(train_matrix)
        # train_total_fpr = get_fpr(train_matrix)
        every_epoch_end = time.time()
        print('epoch {} : acc={} , spe={} , sen={} , time={}s, auc={}'.format(i + 1, train_total_acc, train_total_spe,
                                                                              train_total_sen,
                                                                              int(every_epoch_end - every_epoch_start),
                                                                              train_total_auc))
        print('epoch {} : acc={} , spe={} , sen={} , time={}s, auc={}'.format(i + 1, train_total_acc, train_total_spe,
                                                                              train_total_sen,
                                                                              int(every_epoch_end - every_epoch_start),
                                                                              train_total_auc), file=result_one)
    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir,
                                        '{}_{}_epoch{}_lr{:.0e}_bs{}.pth'.format(str(dataset_name + patient),
                                                                                 result_filename, int(epoch), lr,
                                                                                 int(batch_size))))
    total_epoch_end = time.time()
    print('总训练时间为：', int(total_epoch_end - total_epoch_start), '秒')
    print('总训练时间为：', int(total_epoch_end - total_epoch_start), '秒', file=result_one)
    # 测试开始

    with torch.no_grad():
        print('==================================验证病人{}开始==================================='.format(patient))
        print('==================================验证病人{}开始==================================='.format(patient),
              file=result_one)
        total_test_step = 0
        y_test_score = []  # 记录全部测试样本的预测值，用于求总的roc
        test_start = time.time()
        test_matrix = torch.Tensor([[0, 0], [0, 0]])

        batch_num = math.ceil(len(x_test) / batch_size)  # batch的个数 ceil向上取整
        batch_remainder = len(x_test) % batch_size  # 最后一个batch的大小
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
            if (j + 1) % print_per_epoch == 0:
                print('=======batch {} start======'.format(j + 1))
                print('=======batch {} start======'.format(j + 1), file=result_one)
            data_batch = np.zeros((true_batch_size[j], C, H, W))
            for k in range(true_batch_size[j]):
                data_batch[k, :] = (np.load(x_test[j * true_batch_size[j] + k])).reshape((C, H, W))
            x_test_batch = torch.Tensor(data_batch).to('cuda:0')
            y_test_batch = torch.Tensor(
                y_test[j * batch_size:j * batch_size + true_batch_size[j]]).to('cuda:0')
            y_predict_batch = model(x_test_batch, H, W)
            loss = loss_fn(y_predict_batch, y_test_batch)

            if (j + 1) % print_per_epoch == 0:
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
            # test_batch_auc = get_auc(y_true=y_test_batch, y_score=y_predict_batch)
            for b in y_predict_batch.to('cpu').detach().numpy():
                y_test_score.append(b)

            if (j + 1) % print_per_epoch == 0:
                print(
                    'batch {} : acc={} , spe={} , sen={}'.format(j + 1, test_batch_acc, test_batch_spe,
                                                                 test_batch_sen))
                print(
                    'batch {} : acc={} , spe={} , sen={}'.format(j + 1, test_batch_acc, test_batch_spe,
                                                                 test_batch_sen), file=result_one)
        test_total_acc = get_accuracy(test_matrix)
        test_total_spe = get_specificity(test_matrix)
        test_total_sen = get_sensitivity(test_matrix)
        test_total_auc = roc_auc_score(y_true=y_test, y_score=y_test_score)
        test_total_mcc = get_mcc(test_matrix)
        test_total_f1 = get_f1score(test_matrix)
        test_total_fpr = get_fpr(test_matrix)
        test_end = time.time()
        print(
            '==========================================病人{}的测试结果========================================='.format(
                patient))
        print(
            '==========================================病人{}的测试结果========================================='.format(
                patient), file=result_one)
        print('total acc={}, spe={}, sen={}, test_time={}s, auc={}, mcc={}'.format(test_total_acc, test_total_spe,
                                                                                   test_total_sen,
                                                                                   int(test_end - test_start),
                                                                                   test_total_auc,
                                                                                   test_total_mcc))
        print('total acc={}, spe={}, sen={}, test_time={}s, auc={}, mcc={}'.format(test_total_acc, test_total_spe,
                                                                                   test_total_sen,
                                                                                   int(test_end - test_start),
                                                                                   test_total_auc,
                                                                                   test_total_mcc), file=result_one)
        print('test matrix: ', test_matrix)
        print('test matrix: ', test_matrix, file=result_one)
        all_acc += test_total_acc
        all_spe += test_total_spe
        all_sen += test_total_sen
        all_auc += test_total_auc
        all_mcc += test_total_mcc
        all_f1 += test_total_f1
        all_fpr += test_total_fpr

        print(
            '{}{}:acc={}, spe={}, sen={}, time={}s, auc={}, mcc={},f1={},fpr={}'.format(dataset_name, patient, all_acc,
                                                                                        all_spe, all_sen,
                                                                                        int(test_end - test_start),
                                                                                        all_auc, all_mcc, all_f1,
                                                                                        all_f1))
        print(
            '{}{}:acc={}, spe={}, sen={}, time={}s, auc={}, mcc={},f1={},fpr={}'.format(dataset_name, patient, all_acc,
                                                                                        all_spe, all_sen,
                                                                                        int(test_end - test_start),
                                                                                        all_auc, all_mcc, all_f1,
                                                                                        all_f1),
            file=result_one)
        print('{}{},{},{},{},{},{},{},{}'.format(dataset_name, patient, all_acc, all_spe, all_sen, all_auc, all_mcc,
                                                 all_f1, all_fpr), file=result_one)
        print('{}{},{},{},{},{},{},{},{}'.format(dataset_name, patient, all_acc, all_spe, all_sen, all_auc, all_mcc,
                                                 all_f1, all_fpr), file=result_sum)
        result_one.close()
average_acc = all_acc / len(PATIENTS)
average_spe = all_spe / len(PATIENTS)
average_sen = all_sen / len(PATIENTS)
average_auc = all_auc / len(PATIENTS)
average_mcc = all_mcc / len(PATIENTS)
average_f1 = all_f1 / len(PATIENTS)
average_fpr = all_fpr / len(PATIENTS)
end = time.time()
print('所有病人总的结果:acc={}, spe={}, sen={}, auc={}, mcc={}, time={}s'.format(average_acc, average_spe, average_sen,
                                                                                 average_auc, average_mcc,
                                                                                 int(end - start)))
print('总用时：{}s = {}h'.format(int(end - start), (end - start) / 3600))
result_sum.close()
