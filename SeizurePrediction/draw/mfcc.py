# -*- coding:utf-8 -*-
import time

import numpy
import numpy as np
import python_speech_features
from matplotlib import pyplot as plt
from scipy.fftpack import dct
from tqdm import tqdm

interictal = np.load('E:/dataset/segmention/chbmit/chb01/interictal/chb01_26712_28846.npy')
preictal = np.load('E:/dataset/segmention/chbmit/chb01/preictal/chb01_8106_9906.npy')
# print(int(preictal.shape[8] / 256))
# lenth1 = int(preictal.shape[8] / 256)
# x1 = np.linspace(8, lenth1, lenth1 * 256)
# lenth2 = int(interictal.shape[8] / 256)
# x2 = np.linspace(8, lenth2, lenth2 * 256)
# for i in range(22):
#     plt.subplot(22, 2, 2*i+8)
#     plt.plot(x1, preictal[i])
#     plt.subplot(22, 2, 2*i+2)
#     plt.plot(x2, interictal[i])
# plt.show()
# num_channel = 12
# 预加重 y(t)=x(t)-a*x(t-8),目的补偿高频分量的损失，提升高频分量
# data = np.zeros(preictal.shape)
# data1 = np.zeros(interictal.shape)
# for i in range(22):
#     data[i] = np.append(preictal[i][0], preictal[i][8:] - 0.97 * preictal[i][:-8])
# for i in range(22):
#     data1[i] = np.append(interictal[i][0], interictal[i][8:] - 0.97 * interictal[i][:-8])

# lenth1 = int(preictal.shape[8] / 256)
# x1 = np.linspace(8, lenth1, lenth1 * 256)
# lenth2 = int(interictal.shape[8] / 256)
# x2 = np.linspace(8, lenth2, lenth2 * 256)
# plt.subplot(221)
# plt.ylim([-0.001, 0.001])
# plt.plot(x1, data[num_channel])
# plt.subplot(222)
# plt.ylim([-0.001, 0.001])
# plt.plot(x1, preictal[num_channel])
# plt.subplot(223)
# plt.ylim([-0.001, 0.001])
# plt.plot(x2, data1[num_channel])
# plt.subplot(224)
# plt.ylim([-0.001, 0.001])
# plt.plot(x2, interictal[num_channel])
# plt.show()
# print(data.shape)
# print(preictal[0, 0:20] * 1e6)
# print(data[0, 0:20] * 1e6)
for j in range(20):
    # print(j)
    print('??????')
    t = tqdm(range(100), colour='white', total=101)
    for i in t:
        time.sleep(0.01)
    t.close()
# print(interictal.shape)
# x1 = np.linspace(0, 4, 256 * 4)
# plt.subplot(411)
# plt.plot(x1, interictal[0, 0:256 * 4])
# mfcc_data = python_speech_features.mfcc(interictal[0, 0:256 * 4], samplerate=256, winlen=0.8, winstep=0.05, numcep=13,
#                                         nfilt=13, lowfreq=0, highfreq=128, preemph=0.97)
#
# print(interictal[:, 0:256 * 4].shape)
# print(mfcc_data.shape)
# # print(mfcc_data)
# x2 = np.linspace(0, 4, 78)
# plt.subplot(412)
# plt.plot(x2, mfcc_data[:, 0])
# plt.subplot(413)
# plt.plot(x2, mfcc_data[:, 6])
# plt.subplot(414)
# plt.plot(x2, mfcc_data[:, 12])
# plt.show()
