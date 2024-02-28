# -*-coding:utf-8 -*-
import math
import os
import random
import time

import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from thop import profile

from model import DACNN
from evaluation import get_Confusion_Matrix, get_accuracy, get_auc, get_sensitivity, get_specificity

data_path = 'D:\learn\dataset\preprocess\chbmitmul10_event'
C, H, W = 22, 13, 198

n_samples = 200  # 训练数据中每个病人每类样本的个数
threshold = 10  # 连续预测为前期的阈值
lr = 1e-4  # 学习率
epoch = 10  # 训练轮数
batch_size = 128
print_per_epoch = 50  # 多少个epoch输出一次，用于控制输出的频率
DATABASE_NAME = 'chb'  # ‘chb’或'PN'
# 'time_domain','frequency_domain','timefrequency_domain','mfcc_domain','ht_domain','hht_domain'
OVERLOP = 0.5
# SEED = 2023  # 随机数种子
domain_name = 'mfcc_domain'
result_path = './result_LOPO'  # 保存结果的路径
checkpoint_dir = "./checkpoint_LOPO"  # 保存训练好的模型的路径
result_filename = 'train-LOPO-mfcc'  # 结果名称
PATIENTS = []
if DATABASE_NAME == 'chb':
    PATIENTS = ['01', '03', '04', '05', '06', '07', '09', '10', '11', '13', '14', '16', '17', '18', '19', '20', '21',
                '22', '23']
elif DATABASE_NAME == 'PN':
    PATIENTS = ['01', '03', '05', '06', '07', '09', '10', '11', '12', '13', '14', '16', '17']
else:
    print('不支持的数据集名称')
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
# 固定随机数
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)
# 加标签
all_start = 0
seizures = pd.read_csv('data-summary/chbmit_seizure_summary_generated.csv')
times = pd.read_csv('data-summary/time_len_sum.csv')
result_sum = open(os.path.join(result_path, result_filename + '-sum.txt'), 'w', encoding='utf-8')
print('Patient,Sen(%),FPR(/h),Pw(%),p-value', file=result_sum)
# 计算运算量FLOPs,网络参数量 Params
input_data = torch.randn(batch_size, C, H, W).to('cuda:0')
FLOPs, Params = profile(model, inputs=(input_data,), verbose=False)  # 运算量FLOPs,网络参数量 Params
print('FLOPs= {}M, Params= {}K'.format(FLOPs / pow(1000, 2), Params / pow(1000, 1)))
PATIENTS_TEST = ['01', '03', '04', '05', '06', '07', '09', '10', '11', '13', '14', '16', '17', '18', '19', '20', '21',
                 '22', '23']
for p_test in PATIENTS_TEST:
    result_one = open(os.path.join(result_path, '{}{}-{}.txt'.format(DATABASE_NAME, p_test, result_filename)), 'w',
                      encoding='utf-8')
    path_test = os.path.join(data_path, DATABASE_NAME + p_test)
    XZ_test, XF_test, YZ_test, YF_test = [], [], [], []
    if os.path.lexists(path_test):
        clas = os.listdir(path_test)
        for cla in clas:
            if cla == 'interictal':
                npys_path = os.path.join(path_test, cla, domain_name)
                npys = os.listdir(npys_path)
                for npy in npys:
                    XF_test.append(os.path.join(npys_path, npy))
                    YF_test.append([0, 1])
            if cla == 'preictal':
                npys_path = os.path.join(path_test, cla, domain_name)
                npys = os.listdir(npys_path)
                for npy in npys:
                    XZ_test.append(os.path.join(npys_path, npy))
                    YZ_test.append([1, 0])
    XZ_test.sort()
    XF_test.sort()
    # 统计发作次数
    preictal_path = os.path.join(path_test, 'preictal', domain_name)
    seizures_include = []  # 记录所有满足条件的发作序号
    for sl in os.listdir(preictal_path):
        seizures_include.append(int(sl.split('_')[-1].split('.')[0]))
    seizures_include = list(set(seizures_include))
    seizures_include.sort()
    print(
        '############################################################################################################')
    print(
        '############################################################################################################',
        file=result_one)
    print(
        '## 测试病人{}{}：正样本个数(PREICTAL)={}，负样本个数(INTERICTAL)={}'.format(DATABASE_NAME, p_test, len(XZ_test),
                                                                                   len(XF_test)))
    print('测试病人{}{}：正样本个数(PREICTAL)={}，负样本个数(INTERICTAL)={}'.format(DATABASE_NAME, p_test, len(XZ_test),
                                                                                  len(XF_test)),
          file=result_one)
    print('## 满足实验条件的发作序号包括：', seizures_include)
    print('满足实验条件的发作序号包括：', seizures_include, file=result_one)
    X_train, Y_train = [], []
    for p_train in PATIENTS:
        if p_train != p_test:
            train_path = os.path.join(data_path, DATABASE_NAME + p_train)
            if os.path.lexists(train_path):
                clas = os.listdir(train_path)
                for cla in clas:
                    if cla == 'interictal':
                        npys_path = os.path.join(train_path, cla, domain_name)
                        npys = os.listdir(npys_path)
                        npys_part = random.sample(npys, n_samples)
                        for npy in npys_part:
                            X_train.append(os.path.join(npys_path, npy))
                            Y_train.append([1, 0])
                    if cla == 'preictal':
                        npys_path = os.path.join(train_path, cla, domain_name)
                        npys = os.listdir(npys_path)
                        npys_part = random.sample(npys, n_samples)
                        for npy in npys_part:
                            X_train.append(os.path.join(npys_path, npy))
                            Y_train.append([0, 1])
    print('## 训练病人数={}：样本个数={}'.format(len(PATIENTS) - 1, len(X_train)))
    print('## 训练病人数={}：样本个数={}'.format(len(PATIENTS) - 1, len(X_train)), file=result_one)
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
                                        '{}_{}_LOPO_epoch{}_lr{:.0e}_bs{}.pth'.format(
                                            str(DATABASE_NAME + p_test),
                                            result_filename,
                                            int(epoch), lr,
                                            int(batch_size))))
    total_epoch_end = time.time()
    print('总训练时间为：', int(total_epoch_end - total_epoch_start), '秒')
    print('总训练时间为：', int(total_epoch_end - total_epoch_start), '秒', file=result_one)

    num_alarm = 0  # 发出的所有预警次数
    correct_num = 0  # 在正确的时间内发出的正确预警
    alarm_interictal = 0  # 间期发出的预警
    alarm_false_time = 0  # 在错误时间发出的预警
    num_correct_seizure = 0  # 正确预测出的发作次数
    alarm_length = 0  # 报警的总时长
    XF_one_len = len(XF_test) // len(seizures_include)
    for si, seizure_no in enumerate(seizures_include):
        X_test_one, Y_test_one = [], []
        XZ_test_one, XF_test_one = [], []
        for sp in XZ_test:
            if sp.split('_')[-1].split('.')[0] == str(seizure_no):
                XZ_test_one.append(sp)  # 将同一次发作的所有前期数据作为测试数据
        XZ_test_one.sort()
        XF_test_one = XF_test[si * XF_one_len:(si + 1) * XF_one_len]
        X_test_one = XF_test_one + XZ_test_one  # 等多的正负样本构成训练集
        Y_test_one = [[1, 0]] * len(XF_test_one) + [[0, 1]] * len(XZ_test_one)
        print('Seizure {} : XZ_test={},XF_test={}'.format(seizure_no, len(XZ_test_one), len(XF_test_one)))
        print('Seizure {} : XZ_test={},XF_test={}'.format(seizure_no, len(XZ_test_one), len(XF_test_one)),
              file=result_one)
        # 测试开始
        # 加载发作开始和结束绝对时间
        print(
            '.................................TEST SEIZURE {}.................................'.format(seizure_no))
        print(
            '.................................TEST SEIZURE {}.................................'.format(seizure_no),
            file=result_one)
        condition1 = seizures['Seizure_filename'].str.startswith(DATABASE_NAME + p_test)
        condition2 = seizures['Seizure_no'] == seizure_no
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
            batch_num = math.ceil(len(X_test_one) / batch_size)  # batch的个数 ceil向上取整
            batch_remainder = len(X_test_one) % batch_size  # 最后一个batch的大小
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
                    data_batch[k, :] = (np.load(X_test_one[j * true_batch_size[j] + k])).reshape((C, H, W))
                x_test_batch = torch.Tensor(data_batch).to('cuda:0')
                y_test_batch = torch.Tensor(Y_test_one[j * batch_size:j * batch_size + true_batch_size[j]]).to('cuda:0')
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
                    alarm_end = int(X_test_one[i].split('_')[-2])
                    alarm_length += (alarm_end - alarm_start)
                    print('[{}-{}]报警结束,本次报警开始时间[{}],结束时间[{}],持续时间[{}]秒'.format(num_alarm,
                                                                                                    num_alarm_one,
                                                                                                    alarm_start,
                                                                                                    alarm_end,
                                                                                                    alarm_end - alarm_start))
                    print('[{}-{}]报警结束,本次报警开始时间[{}],结束时间[{}],持续时间[{}]秒'.format(num_alarm,
                                                                                                    alarm_start,
                                                                                                    num_alarm_one,
                                                                                                    alarm_end,
                                                                                                    alarm_end - alarm_start),
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
                    alarm_start = int(X_test_one[i].split('_')[-2])
                    print('[{}-{}]!!!!!!!!!!!!!!!!!!!报警开始!!!!!!!!!!!!!!!!!!!!!'.format(num_alarm, num_alarm_one))
                    print('预测发作的文件名称为[{}]'.format(X_test_one[i]))
                    print('[{}-{}]!!!!!!!!!!!!!!!!!!!报警开始!!!!!!!!!!!!!!!!!!!!!'.format(num_alarm, num_alarm_one),
                          file=result_one)
                    print('预测发作的文件名称为[{}]'.format(X_test_one[i]), file=result_one)

                    if X_test_one[i].split('_')[-4] != "preictal":  # 如果是间期发出的预警
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
    interictal_len = int(times[times['Patient'] == DATABASE_NAME + p_test]['Interictal_len'])
    preictal_len = int(times[times['Patient'] == DATABASE_NAME + p_test]['Preictal_len'])

    print(
        '正确预测出的发作次数 {}。\n总共发出的预警次数 {}。\n在正确的时间内发出的正确预警 {}。\n间期发出的预警 {}。\n前期未在正确时间内发出的预警 {}。\n间期总时长 {}。\n前期总时长 {}。\n所有报警持续的总时间 {}。\n'.format(
            num_correct_seizure, num_alarm, correct_num, alarm_interictal, alarm_false_time, interictal_len,
            preictal_len, alarm_length))
    print(
        '正确预测出的发作次数 {}。\n总共发出的预警次数 {}。\n在正确的时间内发出的正确预警 {}。\n间期发出的预警 {}。\n前期未在正确时间内发出的预警 {}。\n间期总时长 {}。\n前期总时长 {}。\n所有报警持续的总时间 {}。\n'.format(
            num_correct_seizure, num_alarm, correct_num, alarm_interictal, alarm_false_time, interictal_len,
            preictal_len, alarm_length), file=result_one)
    sen = num_correct_seizure / len(seizures_include)  # 预测正确次数/发作总次数
    fpr = alarm_interictal / (interictal_len/3600)  # 错误的警报次数(单指将间期发出的预警)/间期时间
    pw = alarm_length / (interictal_len + preictal_len)  # 用于警告的时间与总时间的比率, 它是预测的总持续时间
    _, pvalue = scipy.stats.pearsonr((torch.Tensor(Y_test_one)[:, 1]).cpu().detach().numpy(),
                                     (torch.Tensor(y_score)[:, 1]).cpu().detach().numpy())  # 返回双边p值，单边p值=2*双边p值
    auc = roc_auc_score(y_true=Y_test_one, y_score=y_score)
    print('SEN= {}, FPR= {}, Pw= {}, p-value= {}, AUC= {}'.format(sen, fpr, pw, 2 * pvalue, auc))
    print('SEN= {}, FPR= {}, Pw= {}, p-value= {}, AUC= {}'.format(sen, fpr, pw, 2 * pvalue, auc), file=result_one)

    result_one.close()
    print('{}{},{},{},{},{}'.format(DATABASE_NAME, p_test, sen * 100, fpr, pw * 100, 2 * pvalue), file=result_sum)
all_end = time.time()
print('实验总共用时:', int(all_end - all_start), file=result_sum)
result_sum.close()
