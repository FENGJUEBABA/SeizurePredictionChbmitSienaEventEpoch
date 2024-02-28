# -*-coding:utf-8 -*-
# 本文件用于保存测试中间数据
import os

import numpy as np
import torch

from model00 import SingleNetwork

data_path = 'D:/learn/dataset/preprocess/chbmitmul10'
PATIENTS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19',
            '20', '21', '22', '23']
# PATIENTS = ['01', '03', '05', '06', '07', '09', '10', '11', '12', '13', '14', '16', '17']
# PATIENTS = ['01', '02']
C, H, W = 22, 114, 21
dataset_name = 'chb'
model_name = 'siamese'
domain_name = 'timefrequency_domain'
# if not os.path.lexists('tsnedata'):
#     os.makedirs('tsnedata')

model = SingleNetwork(C)
# print(model.parameters())

for patient in PATIENTS:
    da = torch.load('weight' +
                    '/' + model_name + '/' + dataset_name + patient + '_Fold5_epoch20.pth')  # 加载训练好的模型 dict_keys(['net', 'optimizer', 'epoch'])
    model.load_state_dict(da['net'], strict=False)
    model.eval()
    save_root = 'tsnedata/' + dataset_name + patient
    chb_dir = os.path.join(data_path, dataset_name + patient)
    for cla in os.listdir(chb_dir):
        cla_path = os.path.join(chb_dir, cla)
        if cla == 'interictal':
            save_path = save_root + '/' + cla
            if not os.path.lexists(save_path):
                os.makedirs(save_path)
            npys = os.path.join(cla_path, domain_name)
            for i, npy in enumerate(os.listdir(npys)):
                data = torch.Tensor(np.load(os.path.join(npys, npy)).reshape((1, C, H, W)))
                y, medium = model(data)
                np.save(save_path + '/' + dataset_name + patient + '-' + str(i) + '.npy',
                        medium.detach().numpy().squeeze())
        if cla == 'preictal':
            save_path = save_root + '/' + cla
            if not os.path.lexists(save_path):
                os.makedirs(save_path)
            npys = os.path.join(cla_path, domain_name)
            for i, npy in enumerate(os.listdir(npys)):
                data = torch.Tensor(np.load(os.path.join(npys, npy)).reshape((1, C, H, W)))
                y, medium = model(data)
                np.save(save_path + '/' + dataset_name + patient + '-' + str(i) + '.npy',
                        medium.detach().numpy().squeeze())
