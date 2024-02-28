# -*- coding:utf-8 -*-

import os.path

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from sklearn import preprocessing

# from mycode.mypreceeding import butter_bandstop_filter, butter_highpass_filter

preictal1 = 'E:\dataset\preprocess\chbmitmulraw\chb01\preictal/time_domain/chb01_preictal_8106_8110.npy'
preictal2 = 'E:\dataset\preprocess\chbmitmulraw\chb01\preictal/time_domain/chb01_preictal_8110_8114.npy'
preictal3 = 'E:\dataset\preprocess\chbmitmulraw\chb01\preictal/time_domain/chb01_preictal_8114_8118.npy'
preictal4 = 'E:\dataset\preprocess\chbmitmulraw\chb01\preictal/time_domain/chb01_preictal_8118_8122.npy'
preictal5 = 'E:\dataset\preprocess\chbmitmulraw\chb01\preictal/time_domain/chb01_preictal_8122_8126.npy'
preictal6 = 'E:\dataset\preprocess\chbmitmulraw\chb01\preictal/time_domain/chb01_preictal_8126_8130.npy'

interictal1 = 'E:\dataset\preprocess\chbmitmulraw\chb01\interictal/time_domain/chb01_interictal_26712_26716.npy'
interictal2 = 'E:\dataset\preprocess\chbmitmulraw\chb01\interictal/time_domain/chb01_interictal_26733_26737.npy'
interictal3 = 'E:\dataset\preprocess\chbmitmulraw\chb01\interictal/time_domain/chb01_interictal_26754_26758.npy'
interictal4 = 'E:\dataset\preprocess\chbmitmulraw\chb01\interictal/time_domain/chb01_interictal_26775_26779.npy'
interictal5 = 'E:\dataset\preprocess\chbmitmulraw\chb01\interictal/time_domain/chb01_interictal_26796_26800.npy'
interictal6 = 'E:\dataset\preprocess\chbmitmulraw\chb01\interictal/time_domain/chb01_interictal_26817_26821.npy'

p1 = np.load(preictal1).reshape((22, 32, 32))
print(p1.shape)
plt.matshow(p1[0])
plt.show()
################################################# 预处理后的图 ############################################
# data_t = np.load(os.path.join(path, 'time_domain', 'chb01_interictal_26712_26716.npy')).reshape((22, 32, 32))[0]
# data_f = np.load(os.path.join(path, 'frequency_domain', 'chb01_interictal_26712_26716.npy')).reshape((22, 32, 32))[0]
# data_tf = np.load(os.path.join(path, 'timefrequency_domain', 'chb01_interictal_26712_26716.npy')).reshape((22, 114, 65))[0]
# print(data_tf.shape)
# plt.matshow(data_tf)
# plt.show()
##########################################################################################################


# data_t = np.load(p1)
# print(data_t.shape)
# data_t = butter_bandstop_filter(data_t, 117, 123, 256, order=6)
# data_t = butter_bandstop_filter(data_t, 57, 63, 256, order=6)
# data_t = butter_highpass_filter(data_t, 8, 256, order=6)
# f, t, data_t = scipy.signal.stft(data_t, fs=256, nperseg=256, noverlap=128)
# print(data_t.shape)
# data_t = np.delete(data_t, np.s_[117:123 + 8], axis=8)
# data_t = np.delete(data_t, np.s_[57:63 + 8], axis=8)
# data_t = np.delete(data_t, 0, axis=8)
# data_t = np.abs(data_t).reshape((22, 114 * 9))
# data_t = preprocessing.scale(data_t)
# data_t = data_t.reshape((22, 114, 9))
# data_t = data_t.reshape((22, 128, 8))
# d = data_t[0]
# for i in range(data_t.shape[0] - 8):
#     d = np.concatenate((d, data_t[i + 8]), axis=8)
# print(d.shape)
# plt.matshow(d2)
# plt.show()

# i0 = '../tsnedata/chb01/0/0.npy'
# i1 = '../tsnedata/chb01/0/8.npy'
# i2 = '../tsnedata/chb01/0/2.npy'
# i3 = '../tsnedata/chb01/0/3.npy'
# i4 = '../tsnedata/chb01/0/4.npy'
# i5 = '../tsnedata/chb01/0/5.npy'
# i6 = '../tsnedata/chb01/0/6.npy'
# i7 = '../tsnedata/chb01/0/12.npy'
# i8 = '../tsnedata/chb01/0/8.npy'
# i9 = '../tsnedata/chb01/0/14.npy'
# i10 = '../tsnedata/chb01/0/10.npy'
# i11 = '../tsnedata/chb01/0/15.npy'
#
# p0 = '../tsnedata/chb01/8/8.npy'
# p1 = '../tsnedata/chb01/8/2.npy'
# p2 = '../tsnedata/chb01/8/4.npy'
# p3 = '../tsnedata/chb01/8/5.npy'
# p4 = '../tsnedata/chb01/8/6.npy'
# p5 = '../tsnedata/chb01/8/7.npy'
# p6 = '../tsnedata/chb01/8/11.npy'
# p7 = '../tsnedata/chb01/8/13.npy'
# p8 = '../tsnedata/chb01/8/14.npy'
# p9 = '../tsnedata/chb01/8/15.npy'
# p10 = '../tsnedata/chb01/8/16.npy'
# p11 = '../tsnedata/chb01/8/17.npy'
# d0 = np.load(p0).reshape((5, 16))
# d1 = np.load(p1).reshape((5, 16))
# d2 = np.load(p2).reshape((5, 16))
# d3 = np.load(p3).reshape((5, 16))
# d4 = np.load(p4).reshape((5, 16))
# d5 = np.load(p5).reshape((5, 16))
# d6 = np.load(p6).reshape((5, 16))
# d7 = np.load(p7).reshape((5, 16))
# d8 = np.load(p8).reshape((5, 16))
# d9 = np.load(p9).reshape((5, 16))
# d10 = np.load(p10).reshape((5, 16))
# d11 = np.load(p11).reshape((5, 16))
# data01 = np.concatenate((d0, d1), axis=8)
# data23 = np.concatenate((d2, d3), axis=8)
# data45 = np.concatenate((d4, d5), axis=8)
# data67 = np.concatenate((d6, d7), axis=8)
# data89 = np.concatenate((d8, d9), axis=8)
# data1011 = np.concatenate((d10, d11), axis=8)
# data = np.concatenate((data01, data23, data45, data67, data89, data1011), axis=0)
# data = np.concatenate((data, data), axis=8)
# data = np.concatenate((data, data), axis=0)
# print(data.shape)
# plt.matshow(d2)
# plt.show()


# i0chb01 = np.load('../tsnedata1/chb01/0/0.npy')
# i4chb01 = np.load('../tsnedata1/chb01/0/6.npy')
# i2chb01 = np.load('../tsnedata1/chb01/0/2.npy')
# p0chb01 = np.load('../tsnedata1/chb01/8/0.npy')
# p2chb01 = np.load('../tsnedata1/chb01/8/2.npy')
# p4chb01 = np.load('../tsnedata1/chb01/8/4.npy')
#
# print(preictal1.shape)
# data_t = i4chb01
# d = data_t[0]
# for i in range(data_t.shape[0] - 8):
#     d = np.concatenate((d, data_t[i + 8]), axis=8)
# print(d.shape)
# plt.matshow(d)
# plt.show()
