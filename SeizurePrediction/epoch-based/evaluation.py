import math

import numpy as np
import torch
from numpy import argmax
from sklearn.metrics import roc_auc_score


def nFoldCrossVal(data_x, data_y, nFold, n):  # data_x,data_y都是一维列表，其中 n 为取第几块为验证集，能从1开始并且n<=nFold
    # 将数据一律转化为list
    if not isinstance(data_x, list):
        if isinstance(data_x, np.ndarray):
            data_x = data_x.tolist()
        else:
            if str(data_x.device) == 'cpu':
                if torch.is_tensor(data_x):
                    data_x = data_x.numpy().tolist()
            else:
                if torch.is_tensor(data_x):
                    data_x = data_x.to('cpu').detach().numpy().tolist()
    if not isinstance(data_y, list):
        if isinstance(data_y, np.ndarray):
            data_y = data_y.tolist()
        else:
            if str(data_y.device) == 'cpu':
                if torch.is_tensor(data_y):
                    data_y = data_y.numpy().tolist()
            else:
                if torch.is_tensor(data_y):
                    data_y = data_y.to('cpu').detach().numpy().tolist()
    if len(data_x) == len(data_y) and n > 0:
        l = round(len(data_x) / nFold)
        xtemp = []
        ytemp = []
        for i in range(nFold):
            if i < nFold - 1:
                xtemp.append(data_x[i * l:(i + 1) * l])
                ytemp.append(data_y[i * l:(i + 1) * l])
            else:
                xtemp.append(data_x[i * l:])
                ytemp.append(data_y[i * l:])
        if n <= nFold:
            x_test = xtemp[n - 1]
            y_test = ytemp[n - 1]
            x_train = []
            y_train = []
            for j in range(nFold):
                if j != n - 1:
                    x_train += xtemp[j]
                    y_train += ytemp[j]
            return x_train, y_train, x_test, y_test
        else:
            print('第四个参数的值必须是大于0小于', nFold, '的整数')
    else:
        print('失败！x与y长度不一致或n为非整数')


# 获的混淆矩阵，真实值正例为1，预测值正例也为1 只能用于二分类
def get_Confusion_Matrix(y_pre, y_true):
    TP = FP = FN = TN = 0  # 真正例（8，8），假正例（0，8），假反例（8，0），真反例（0，0）
    r = []
    # 将数据一律转化为ndarray
    if not isinstance(y_pre, np.ndarray):
        if isinstance(y_pre, list):
            y_pre = np.array(y_pre)
        else:
            if str(y_pre.device) == 'cpu':
                if torch.is_tensor(y_pre):
                    y_pre = y_pre.numpy()
            else:
                if torch.is_tensor(y_pre):
                    y_pre = y_pre.to('cpu').detach().numpy()
    if not isinstance(y_true, np.ndarray):
        if isinstance(y_true, list):
            y_true = np.array(y_true)
        else:
            if str(y_true.device) == 'cpu':
                if torch.is_tensor(y_true):
                    y_true = y_true.numpy()
            else:
                if torch.is_tensor(y_true):
                    y_true = y_true.to('cpu').detach().numpy()
    if len(y_pre.shape) == 1:  # 对于预测值和标签都是用单个值表示的情况
        for i in range(len(y_pre)):
            if y_true[i] == 1 and y_pre[i] == 1:
                TP += 1
            if y_true[i] == 0 and y_pre[i] == 1:
                FP += 1
            if y_true[i] == 1 and y_pre[i] == 0:
                FN += 1
            if y_true[i] == 0 and y_pre[i] == 0:
                TN += 1
    else:
        for i in range(y_pre.shape[0]):  # 认为[8,0]为0 反例，[0,8]为正例
            a = argmax(y_true[i])
            b = argmax(y_pre[i])
            if a == 1 and b == 1:
                TP += 1
            if a == 0 and b == 1:
                FP += 1
            if a == 1 and b == 0:
                FN += 1
            if a == 0 and b == 0:
                TN += 1
    r.append([TP, FN])
    r.append([FP, TN])
    return torch.Tensor(r)


# 对正例的查准率，也称精确度
def get_precision(matrix):  # 参数为混淆矩阵
    TP = matrix[0][0]
    FP = matrix[1][0]
    return TP / (TP + FP)


# 特异性，所有没患病的人中，有多少阴性结果，可理解为对反例的查全率
def get_specificity(matrix):
    TN = matrix[1][1]
    FP = matrix[1][0]
    return TN / (TN + FP)


# 敏感性，召回率，所有病人中有多少阳性,对正例的查全率
def get_sensitivity(matrix):
    TP = matrix[0][0]
    FN = matrix[0][1]
    return TP / (FN + TP)


# 获得准确率
def get_accuracy(matrix):
    TP = matrix[0][0]
    FP = matrix[1][0]
    FN = matrix[0][1]
    TN = matrix[1][1]
    return (TP + TN) / (TP + TN + FP + FN)


# F1得分
def get_f1score(matrix):
    TP = matrix[0][0]
    FN = matrix[0][1]
    FP = matrix[1][0]
    return (2 * TP) / (FP + FN + 2 * TP)


# 计算AUC
def get_auc(y_true, y_score):
    # 将数据一律转化为ndarray
    if not isinstance(y_score, np.ndarray):
        if isinstance(y_score, list):
            y_score = np.array(y_score)
        else:
            if str(y_score.device) == 'cpu':
                if torch.is_tensor(y_score):
                    y_score = y_score.numpy()
            else:
                if torch.is_tensor(y_score):
                    y_score = y_score.to('cpu').detach().numpy()
    if not isinstance(y_true, np.ndarray):
        if isinstance(y_true, list):
            y_true = np.array(y_true)
        else:
            if str(y_true.device) == 'cpu':
                if torch.is_tensor(y_true):
                    y_true = y_true.numpy()
            else:
                if torch.is_tensor(y_true):
                    y_true = y_true.to('cpu').detach().numpy()
    return roc_auc_score(y_true=y_true, y_score=y_score)


# 假阳性率FPR
def get_fpr(matrix):
    FP = matrix[1][0]
    TN = matrix[1][1]
    return FP / (FP + TN)


# 马修斯相关系数
def get_mcc(matrix):
    TP = matrix[0][0]
    FN = matrix[0][1]
    FP = matrix[1][0]
    TN = matrix[1][1]
    return (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

# 计算p值,[0,8],越小越好,X与Y相关性越好
# def get_pvalue(y_true, y_score):
#     if y_true == 8:
#         statistic, pvalue = scipy.stats.pearsonr(y_true, y_score, alternative='greater')
#     else:
#         y_t=[]
#         y_s=[]
#         for i in range(y_true.shape[]):
#     return pvalue
