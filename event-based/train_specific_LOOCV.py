# -*-coding:utf-8 -*-
import math
import os
import time

import numpy as np
import pandas as pd
import scipy
import torch
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from thop import profile

from evaluation import get_Confusion_Matrix, get_accuracy, \
    get_sensitivity, get_specificity
from model import DACNN

data_path = 'D:/learn/dataset/preprocess/chbmitmul10_event'
# PATIENTS = ['01']
C, H, W = 22, 13, 196

threshold = 10  # 连续预测为前期的阈值
lr = 1e-4  # 学习率
epoch = 10  # 训练轮数
batch_size = 128
print_per_epoch = 20  # 多少个epoch输出一次，用于控制输出的频率
DATABASE_NAME = 'chb'  # ‘chb’或'PN'
# 'time_domain','frequency_domain','timefrequency_domain','mfcc_domain','ht_domain','hht_domain'
OVERLOP = 0.5

domain_name = 'mfcc_domain'
result_path = './result_LOOCV'  # 保存结果的路径
checkpoint_dir = "./checkpoint_LOOCV"  # 保存训练好的模型的路径
result_filename = 'model-specific-mfcc'  # 结果名称
PATIENTS = []
if DATABASE_NAME == 'chb':
    PATIENTS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18',
                '19', '20', '21', '22', '23']
elif DATABASE_NAME == 'PN':
    PATIENTS = ['01', '03', '05', '06', '07', '09', '10', '11', '12', '13', '14', '16', '17']
else:
    print('不支持的数据集名称')
PATIENTS = ['15']
# 创建网络模型
model = DACNN().to('cuda:0')
# 读取配置信息
with open('data-summary/configure.txt', "r") as f:
    lines = f.readlines()
    for line in lines:
        if "SPH" in line:
            SPH = int(line.split("=")[1].strip()) * 60
        if "SOP" in line:
            SOP = int(line.split("=")[1].strip()) * 60
        if 'WINDOWS' in line:  # 样本采样频率
            WINDOWS = int(line.split("=")[1].strip())
f.close()
if not os.path.lexists(result_path):
    os.makedirs(result_path)
if not os.path.lexists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
# 加标签
all_start = 0
seizures = pd.read_csv('data-summary/chbmit_seizure_summary_generated.csv')
times = pd.read_csv('data-summary/time_len_sum.csv')
result_sum = open(os.path.join(result_path, result_filename + '-sum.txt'), 'w', encoding='utf-8')
print('Patient,Sen(%),FPR(/h),Pw(%),p-value,AUC', file=result_sum)
# 计算运算量FLOPs,网络参数量 Params
input_data = torch.randn(batch_size, C, H, W).to('cuda:0')
FLOPs, Params = profile(model, inputs=(input_data,), verbose=False)  # 运算量FLOPs,网络参数量 Params
print('FLOPs= {}M, Params= {}K'.format(FLOPs / pow(1000, 2), Params / pow(1000, 1)))
for patient in PATIENTS:
    path = os.path.join(data_path, DATABASE_NAME + patient)
    # 统计发作次数
    preictal_path = os.path.join(path, 'preictal', domain_name)
    seizures_include = []  # 记录所有满足条件的发作序号
    for sl in os.listdir(preictal_path):
        seizures_include.append(int(sl.split('_')[-1].split('.')[0]))
    seizures_include = list(set(seizures_include))
    seizures_include.sort()
    result_one = open(os.path.join(result_path, '{}{}-{}.txt'.format(DATABASE_NAME, patient, result_filename)), 'w',
                      encoding='utf-8')
    print('FLOPs= {}M, Params= {}K'.format(FLOPs / pow(1000, 2), Params / pow(1000, 1)), file=result_one)
    XZ = []
    XF = []
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
    XF.sort()
    XZ.sort()
    # nums.update({patient: [len(XZ), len(XF)]})
    print(
        '############################################################################################################')
    print(
        '############################################################################################################',
        file=result_one)
    print('## 病人{}{}：正样本个数={}，负样本个数={}'.format(DATABASE_NAME, patient, len(XZ), len(XF)))
    print('病人{}{}：正样本个数={}，负样本个数={}'.format(DATABASE_NAME, patient, len(XZ), len(XF)), file=result_one)
    print('## 满足实验条件的发作序号包括：', seizures_include)
    print('满足实验条件的发作序号包括：', seizures_include, file=result_one)
    num_alarm = 0  # 发出的所有预警次数
    correct_num = 0  # 在正确的时间内发出的正确预警
    alarm_interictal = 0  # 间期发出的预警
    alarm_false_time = 0  # 在错误时间发出的预警
    num_correct_seizure = 0  # 正确预测出的发作次数
    sum_alarm_length = 0  # 报警的总时长
    for seizure_no in seizures_include:  # LOOCV开始
        alarm_length = 0  # 单词发作报警时长
        X_train, X_test, Y_train, Y_test = [], [], [], []
        XZ_train, XF_train, XZ_test, XF_test = [], [], [], []
        for sp in XZ:
            if sp.split('_')[-1].split('.')[0] == str(seizure_no):
                XZ_test.append(sp)  # 将同一次发作的所有前期数据作为测试数据
            else:
                XZ_train.append(sp)  # 剩余的前期数据作为训练数据
        # XF_train = random.sample(XF, len(XZ_train))
        # XF_test = list(set(XF) - set(XF_train))
        XF_test = XF[0:len(XZ_test)]
        XF_train = XF[len(XZ_test):]
        X_train = XF_train + XZ_train  # 等多的正负样本构成训练集
        Y_train = [[1, 0]] * len(XF_train) + [[0, 1]] * len(XZ_train)
        XZ_test.sort()  # 升序排序，测试时前期数据要保留时间顺序
        XF_test.sort()
        X_test = XF_test + XZ_test
        Y_test = [[1, 0]] * len(XF_test) + [[0, 1]] * len(XZ_test)
        XZ_test_time_one = []  # 记录测试集发作前期每个数据的的开始和结束时间 用于统计报警持续时间
        XF_test_time_one = []  # 记录测试集发作前期每个数据的的开始和结束时间 用于统计报警持续时间
        for xzi in XZ_test:
            stop = int(xzi.split('_')[-2])
            start = int(xzi.split('_')[-3])
            XZ_test_time_one.append([start, stop])
        for xfi in XF_test:
            stop = int(xfi.split('_')[-2])
            start = int(xfi.split('_')[-3])
            XF_test_time_one.append([start, stop])
        interictal_flag = True  # 只有当间期采样不连续时为False
        preictal_flag = True  # 只有当前期采样不连续时为False
        for xzii in range(len(XZ_test_time_one) - 1):
            if XZ_test_time_one[xzii][1] < XZ_test_time_one[xzii + 1][0]:
                preictal_flag = False  # 不连续
        for xfii in range(len(XF_test_time_one) - 1):
            if XF_test_time_one[xfii][1] < XF_test_time_one[xfii + 1][0]:
                interictal_flag = False  # 不连续
        X_test_time_one = XF_test_time_one + XZ_test_time_one
        print(
            '====================================Seizure {}======================================='.format(seizure_no))
        print(
            '====================================Seizure {}======================================='.format(seizure_no),
            file=result_one)
        print(
            'Seizure {} : XZ_test={},XZ_train={},XF_test={},XF_train={},X_test={},X_train={},Y_test={},Y_train={}'.format(
                seizure_no, len(XZ_test), len(XZ_train), len(XF_test), len(XF_train), len(X_test), len(X_train),
                len(Y_test), len(Y_train)))
        print(
            'Seizure {} : XZ_test={},XZ_train={},XF_test={},XF_train={},X_test={},X_train={},Y_test={},Y_train={}'.format(
                seizure_no, len(XZ_test), len(XZ_train), len(XF_test), len(XF_train), len(X_test), len(X_train),
                len(Y_test), len(Y_train)), file=result_one)
        endtime = []
        # seo = []
        for xt in X_test[len(XF_test):]:
            endtime.append(xt.split('_')[-2])
            # seo.append(xt.split('_')[-1].split('.')[0])
        print('endtime', endtime)
        print('endtime', endtime, file=result_one)
        # print('seo:', set(seo))
        # print(len(set(X_test[len(XF_test):])))
        # print(Y_test[len(XF_test):] == [[0, 1]] * len(XZ_test))
        # print(Y_test[:len(XF_test)] == [[1, 0]] * len(XF_test))

        # 打乱训练数据，同时打乱数据和标签,测试集保留原来的顺序
        state = np.random.get_state()
        np.random.shuffle(X_train)
        np.random.set_state(state)
        np.random.shuffle(Y_train)

        # 损失函数
        loss_fn = nn.CrossEntropyLoss()
        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        total_train_step, total_test_step = 0, 0  # 训练次数
        print('lr={},epoch={},batch_size={}'.format(lr, epoch, batch_size), file=result_one)
        # print(1 / 0)
        # 训练开始
        total_epoch_start = time.time()
        for i in range(epoch):
            y_train_score = []  # 记录全部训练样本的预测值，用于求总的roc
            every_epoch_start = time.time()
            train_matrix = torch.Tensor([[0, 0], [0, 0]])
            print('...................................TRAINING {}...................................'.format(i + 1))
            print('...................................TRAINING {}...................................'.format(i + 1),
                  file=result_one)
            # 训练开始
            batch_num = math.ceil(len(X_train) / batch_size)  # batch的个数 ceil向上取整
            batch_remainder = len(X_train) % batch_size  # 最后一个batch的大小
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
                    data_batch[k, :] = (np.load(X_train[j * true_batch_size[j] + k])).reshape((C, H, W))
                x_train_batch = torch.Tensor(data_batch).to('cuda:0')
                y_train_batch = torch.Tensor(Y_train[j * batch_size:j * batch_size + true_batch_size[j]]).to(
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

                for b in y_predict_batch.to('cpu').detach().numpy():
                    y_train_score.append(b)

                if (j + 1) % print_per_epoch == 0:
                    print('batch {} : acc={} , spe={} , sen={}'.format(j + 1, train_batch_acc, train_batch_spe,
                                                                       train_batch_sen))
                    print('batch {} : acc={} , spe={} , sen={}'.format(j + 1, train_batch_acc, train_batch_spe,
                                                                       train_batch_sen), file=result_one)
            train_total_acc = get_accuracy(train_matrix)
            train_total_spe = get_specificity(train_matrix)
            train_total_sen = get_sensitivity(train_matrix)
            train_total_auc = roc_auc_score(y_true=Y_train, y_score=y_train_score)
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
                                            '{}_{}_LOOCV_Seizure{}_epoch{}_lr{:.0e}_bs{}.pth'.format(
                                                str(DATABASE_NAME + patient),
                                                result_filename,
                                                seizure_no,
                                                int(epoch), lr,
                                                int(batch_size))))
        total_epoch_end = time.time()
        print('总训练时间为：', int(total_epoch_end - total_epoch_start), '秒')
        print('总训练时间为：', int(total_epoch_end - total_epoch_start), '秒', file=result_one)

        # 测试开始
        # 加载发作开始和结束绝对时间
        print(
            '.................................TEST SEIZURE {}.................................'.format(seizure_no))
        print(
            '.................................TEST SEIZURE {}.................................'.format(seizure_no),
            file=result_one)
        condition1 = seizures['Seizure_filename'].str.startswith(DATABASE_NAME + patient)
        condition2 = seizures['Seizure_no'] == int(seizure_no)
        seizure_start_abs_one = seizures[condition1][condition2]['Seizure_start_abs'].tolist()[0]
        seizure_stop_abs_one = seizures[condition1][condition2]['Seizure_stop_abs'].tolist()[0]
        print('Seizure {} : 真实发作时间=[{},{}], 合理时间=[{},{}]'.format(seizure_no, seizure_start_abs_one,
                                                                           seizure_stop_abs_one,
                                                                           seizure_stop_abs_one - SOP,
                                                                           seizure_start_abs_one - SPH))
        print('Seizure {} : 真实发作时间=[{},{}], 合理时间=[{},{}]'.format(seizure_no, seizure_start_abs_one,
                                                                           seizure_stop_abs_one,
                                                                           seizure_stop_abs_one - SOP,
                                                                           seizure_start_abs_one - SPH),
              file=result_one)
        with torch.no_grad():
            test_start = time.time()
            num_preictal = 0

            batch_num = math.ceil(len(X_test) / batch_size)  # batch的个数 ceil向上取整
            batch_remainder = len(X_test) % batch_size  # 最后一个batch的大小
            true_batch_size = []
            if batch_remainder == 0:
                for num in range(batch_num):
                    true_batch_size.append(batch_size)
            else:
                for num in range(batch_num - 1):
                    true_batch_size.append(batch_size)
                true_batch_size.append(batch_remainder)
            # print('batch_num:', batch_num)
            # print('batch_num:', batch_num, file=result_one)

            predict_result = []  # onehot编码逆转换 存储所有预测结果
            y_score = []  # 存储所有预测值
            for j in range(batch_num):
                total_test_step += 1
                data_batch = np.zeros((true_batch_size[j], C, H, W))
                for k in range(true_batch_size[j]):
                    data_batch[k, :] = (np.load(X_test[j * true_batch_size[j] + k])).reshape((C, H, W))
                x_test_batch = torch.Tensor(data_batch).to('cuda:0')
                y_test_batch = torch.Tensor(Y_test[j * batch_size:j * batch_size + true_batch_size[j]]).to('cuda:0')
                y_predict_batch = model(x_test_batch)

                # predict_result_batch = []  # 记录一个batch的预测结果
                for item in y_predict_batch.cpu().detach().numpy():
                    # predict_result_batch.append(np.argmax(item))
                    predict_result.append(np.argmax(item))
                    y_score.append(item)

            num_one = 0  # 统计 1 连续出现的次数
            num_zero = 0  # 统计 0 连续出现的次数
            over_threshold = 0  # 记录连续报警的次数
            alarm_start = 0  # 记录每次报警开始的时间
            alarm_end = 0  # 记录每次报警结束的时间
            num_alarm_one = 0  # 记录每次测试报警次数
            predict_bool = False  # 是否正确预测本次发作
            index_alarm_start, index_alarm_stop = 0, 0
            alarm_one_time = [] # 记录连续发作的文件开始和结束时间
            for i, c in enumerate(predict_result):
                if c == 1:
                    num_one += 1
                    if num_one == 1:
                        num_zero = 0
                else:
                    num_zero += 1
                    if num_zero == 1:
                        num_one = 0
                if over_threshold > 0 and (num_zero == 1 or i == len(predict_result) - 1):
                    alarm_end = int(X_test[i].split('_')[-2])
                    index_alarm_stop = i
                    if X_test[index_alarm_stop].split('_')[-4] == 'interictal':
                        if interictal_flag:
                            alarm_length = alarm_end - alarm_start
                        if not interictal_flag:
                            alarm_length = 0
                            for ii in range(index_alarm_stop - index_alarm_start):
                                alarm_length += X_test_time_one[ii + index_alarm_start + 1][1] - \
                                                X_test_time_one[ii + index_alarm_start + 1][0]
                    if X_test[index_alarm_start].split('_')[-4] == 'preictal':
                        if preictal_flag:
                            alarm_length = alarm_end - alarm_start
                        if not preictal_flag:
                            alarm_length = 0
                            for ii in range(index_alarm_stop - index_alarm_start):
                                alarm_length += X_test_time_one[ii + index_alarm_start + 1][1] - \
                                                X_test_time_one[ii + index_alarm_start + 1][0]
                    if X_test[index_alarm_stop].split('_')[-4] == 'preictal' and X_test[index_alarm_start].split('_')[
                        -4] == 'interictal':
                        end_i = len(XF_test) - 1  # 记录间期结束序号
                        interictal_end_time = int(XF_test[-1].split('_')[-2])
                        preictal_start_time = int(XZ_test[0].split('_')[-3])
                        if interictal_flag == True and preictal_flag == True:
                            alarm_length = (interictal_end_time - alarm_start) + (alarm_end - preictal_start_time)
                        if interictal_flag == True and preictal_flag == False:
                            alarm_length = interictal_end_time - alarm_start
                            for ii in range(index_alarm_stop - end_i):
                                alarm_length += X_test_time_one[ii + index_alarm_start + 1][1] - \
                                                X_test_time_one[ii + index_alarm_start + 1][0]
                        if interictal_flag == False and preictal_flag == True:
                            alarm_length = 0
                            for ii in range(end_i - index_alarm_start):
                                alarm_length += X_test_time_one[ii + index_alarm_start + 1][1] - \
                                                X_test_time_one[ii + index_alarm_start + 1][0]
                            alarm_length += alarm_end - preictal_start_time
                        if interictal_flag == False and preictal_flag == False:
                            alarm_length = 0
                            for ii in range(index_alarm_stop - index_alarm_start):
                                alarm_length += X_test_time_one[ii + index_alarm_start + 1][1] - \
                                                X_test_time_one[ii + index_alarm_start + 1][0]
                    sum_alarm_length += alarm_length
                    print('[{}-{}]报警结束,本次报警开始时间[{}],结束时间[{}],持续时间[{}]秒'.format(num_alarm,
                                                                                                    num_alarm_one,
                                                                                                    alarm_start,
                                                                                                    alarm_end,
                                                                                                    alarm_length))
                    print('[{}-{}]报警结束,本次报警开始时间[{}],结束时间[{}],持续时间[{}]秒'.format(num_alarm,
                                                                                                    num_alarm_one,
                                                                                                    alarm_start,
                                                                                                    alarm_end,
                                                                                                    alarm_length),
                          file=result_one)
                if num_one < threshold:
                    over_threshold = 0
                    if i == len(predict_result) - 1 and num_alarm == 0:
                        print('NO SEIZURE.')
                        print('NO SEIZURE.', file=result_one)
                elif num_one == threshold:
                    over_threshold = 1  # 报警持续次数置零
                    num_alarm += 1
                    num_alarm_one += 1
                    predict_bool = True
                    alarm_start = int(X_test[i].split('_')[-2])
                    index_alarm_start = i
                    print('[{}-{}]!!!!!!!!!!!!!!!!!!!报警开始!!!!!!!!!!!!!!!!!!!!!'.format(num_alarm, num_alarm_one))
                    print('预测发作的文件名称为[{}]'.format(X_test[i]))
                    print('[{}-{}]!!!!!!!!!!!!!!!!!!!报警开始!!!!!!!!!!!!!!!!!!!!!'.format(num_alarm, num_alarm_one),
                          file=result_one)
                    print('预测发作的文件名称为[{}]'.format(X_test[i]), file=result_one)

                    if X_test[i].split('_')[-4] != "preictal":  # 如果是间期发出的预警
                        alarm_interictal += 1
                        print('错误的预警')
                        print('错误的预警', file=result_one)
                    elif seizure_start_abs_one - SOP < alarm_start < seizure_start_abs_one - SPH:
                        # 发出报警的时间是否在合理范围内，Seizure_start_abs-SOP<alarm_time<Seizure_start_abs-SPH
                        correct_num += 1
                        print('报警正确,距离发作还有 {} 秒。'.format(seizure_start_abs_one - alarm_start))
                        print('报警正确,距离发作还有 {} 秒。'.format(seizure_start_abs_one - alarm_start),
                              file=result_one)
                        # break  # 在发作前期的预警只发生一次 持续报警直到前期结束
                    else:
                        alarm_false_time += 1
                        print('报警时间不在合理时间内')
                        print('报警时间不在合理时间内', file=result_one)
                else:  # 统计报警持续时间
                    over_threshold += 1
                    # for i in range(5):
                    #     time.sleep(1)
                    #     sys.stdout.write("\r now is :{0}".format(i))
                    #     sys.stdout.flush()
        if predict_bool:
            num_correct_seizure += 1

    # 评估
    interictal_len = int(times[times['Patient'] == DATABASE_NAME + patient]['Interictal_len'])
    preictal_len = int(times[times['Patient'] == DATABASE_NAME + patient]['Preictal_len'])

    print(
        '正确预测出的发作次数 {}。\n总共发出的预警次数 {}。\n在正确的时间内发出的正确预警 {}。\n间期发出的预警 {}。\n前期未在正确时间内发出的预警 {}。\n间期总时长 {}。\n前期总时长 {}。\n所有报警持续的总时间 {}。\n'.format(
            num_correct_seizure, num_alarm, correct_num, alarm_interictal, alarm_false_time, interictal_len,
            preictal_len, sum_alarm_length))
    print(
        '正确预测出的发作次数 {}。\n总共发出的预警次数 {}。\n在正确的时间内发出的正确预警 {}。\n间期发出的预警 {}。\n前期未在正确时间内发出的预警 {}。\n间期总时长 {}。\n前期总时长 {}。\n所有报警持续的总时间 {}。\n'.format(
            num_correct_seizure, num_alarm, correct_num, alarm_interictal, alarm_false_time, interictal_len,
            preictal_len, sum_alarm_length), file=result_one)
    sen = num_correct_seizure / len(seizures_include)  # 预测正确次数/发作总次数
    fpr = alarm_interictal / (interictal_len/3600)  # 错误的警报次数(单指将间期发出的预警)/间期时间
    pw = sum_alarm_length / (interictal_len + preictal_len)  # 用于警告的时间与总时间的比率, 它是预测的总持续时间
    _, pvalue = scipy.stats.pearsonr((torch.Tensor(Y_test)[:, 1]).cpu().detach().numpy(),
                                     (torch.Tensor(y_score)[:, 1]).cpu().detach().numpy())  # 返回双边p值，单边p值=2*双边p值
    auc = roc_auc_score(y_true=Y_test, y_score=y_score)
    print('SEN= {}, FPR= {}, Pw= {}, p-value= {}, AUC= {}'.format(sen, fpr, pw, 2 * pvalue, auc))
    print('SEN= {}, FPR= {}, Pw= {}, p-value= {}, AUC= {}'.format(sen, fpr, pw, 2 * pvalue, auc), file=result_one)

    result_one.close()
    print('{}{},{},{},{},{},{}'.format(DATABASE_NAME, patient, sen * 100, fpr, pw * 100, 2 * pvalue, auc),
          file=result_sum)
all_end = time.time()
print('实验总共用时:', int(all_end - all_start), file=result_sum)
result_sum.close()
