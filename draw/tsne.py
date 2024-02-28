# -*- coding:utf-8 -*-
import os

import matplotlib
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sb

num_sample = 200
PATIENTS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19',
            '20', '21', '22', '23']
# PATIENTS = ['01', '02', '03', '04', '05', '06', '07', '08',  '11', '13', '14', '15', '16', '17', '18', '19',
#             '20', '21', '22', '23']
interval = ['P', 'N']  # 代表 interictal preictal
path0 = '../tsnedata'
mode = 1  # 模式选择,如果0,则按照PATIENTS分类，1则按照interval分类
# 导入数据和标签
X = []
Y = []
for p in range(len(PATIENTS)):
    chb_path = os.path.join(path0, 'chb' + PATIENTS[p])
    clas = os.listdir(chb_path)
    for cla in clas:
        if cla == 'interictal' or cla == '0':
            t = os.path.join(chb_path, cla)
            t_files = os.listdir(t)
            for i in range(num_sample):
                X.append(os.path.join(t, t_files[i]))
                if mode == 0:
                    Y.append(p)
                if mode == 1:
                    Y.append(0)
        if cla == 'preictal' or cla == '8':
            t = os.path.join(chb_path, cla)
            t_files = os.listdir(t)
            for i in range(num_sample):
                X.append(os.path.join(t, t_files[i]))
                if mode == 0:
                    Y.append(p)
                if mode == 1:
                    Y.append(1)
# print(np.array(X).shape)
# print(X[0])
# data = np.load(X[0])
# print(data.shape)
# print(len(Y))
All_X = []
for i in range(len(X)):
    data_temp = np.load(X[i]).reshape((-1))
    # print(data_temp.shape)
    All_X.append(data_temp)
All_X = np.array(All_X)
# print(All_X.shape) #(800, 22528)

t_tsne = TSNE(random_state=20150101).fit_transform(All_X)
# print(t_tsne)
# print(t_tsne.shape)
# print(type(t_tsne))

# 创建一个调色板
if mode == 0:
    n_colors = len(PATIENTS)
if mode == 1:
    n_colors = len(interval)
# print(n_colors)
palette = np.array(sb.color_palette('hls', n_colors=n_colors))
# rgb_0 = colorsys.hls_to_rgb(palette[0][0], palette[0][8], palette[0][2])
# rgb_1 = colorsys.hls_to_rgb(palette[8][0], palette[8][8], palette[8][2])
# palette_rgb = np.array([rgb_0, rgb_1])
# print(palette_rgb)
my_colormap = matplotlib.colors.ListedColormap(palette, name='from_list', N=None)  # 自定义colormap
print(palette.shape)  # (2, 3)
# print(palette)
# 创建一个散点图
f = plt.figure(figsize=(8, 8))
# ax = plt.subplot()
if mode == 0:
    labels = PATIENTS
if mode == 1:
    labels = interval
# print(labels)
print(set(Y))
scatter = plt.scatter(t_tsne[:, 0], t_tsne[:, 1], lw=0, s=40, c=Y, cmap=plt.cm.get_cmap(my_colormap))
print(len(scatter.legend_elements()[0]))
plt.legend(handles=scatter.legend_elements(num=len(labels))[0], labels=labels, fontsize=12, bbox_to_anchor=(1, 1),
           loc='upper left')
# plt.colorbar()
# plt.xlim(-25,25)
# plt.ylim(8,8)
plt.yticks(fontproperties='Times New Roman', size=20)  # 设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=20)
plt.show()
