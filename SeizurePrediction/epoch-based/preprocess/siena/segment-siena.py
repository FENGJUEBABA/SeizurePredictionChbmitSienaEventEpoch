# -*- coding:utf-8 -*-
import mne.io
import numpy as np
import os
import pandas as pd
from scipy.signal import butter
from scipy.signal import lfilter

DATA_ROOT_PATH = 'D:\learn\dataset\\rawdata\siena'  # siena数据根目录
DATA_SAVE_PATH = 'D:\learn/dataset/segmention/siena'  # 处理后的数据保存路径
PATIENTS = ['00', '01', '03', '05', '06', '07', '09', '10', '11', '12', '13', '14', '16', '17']
PATIENTS_PATH = []  # 保存每个病人数据目录绝对路径(所有病人)
SEIZURES_FILENAMES = []  # 保存有发作记录的edf文件名称
NOSEIZURES_FILENAMES = []  # 保存未发作记录的edf文件名称
RECORD_FILENAMES = []  # 记录所有的记录文件名字
Seizure_start = []  # 保存发作开始时间
Seizure_stop = []  # 保存发作结束时间
INTERICTAL_BETWEEN_ICTAL = 60 * 60  # 定义发作间期距离发作期的长度,单位秒，siena数据集为1小时
PREICTAL = 30 * 60  # 发作前期的长度,单位秒
SPH = 5 * 60  # seizure prediction horizon(SPH)一般为5分钟,
POSTICTAL = 30 * 60  # 发作后期
SAMPLE_RATE = 512  # 样本采样频率
SEPARATER = '_'  # 分隔符
CHANNELS = ['EEG T4', 'EEG F9', 'EEG F4', 'EEG C4', 'EEG Fc6', 'EEG Cp5', 'EEG Pz', 'EEG P3', 'EEG O1', 'EEG Fz',
            'EEG Cp1', 'EEG Cp6',
            'EEG T3', 'EEG P4', 'EEG F7', 'EEG F3', 'EEG C3', 'EEG T6', 'EEG Fc5', 'EEG Fc2', 'EEG F10', 'EEG Fc1',
            'EEG F8', 'EEG T5',
            'EEG O2', 'EEG Fp1', 'EEG Cp2']

filetime = pd.read_csv('data-summary/siena_filetime_summary_generated.csv')
# print(filetime)
File_name = filetime['File_name'].tolist()
File_start_abs = filetime['File_start_abs'].tolist()
File_end_abs = filetime['File_end_abs'].tolist()
File_length = filetime['File_length'].tolist()
File_start_rel = filetime['File_start_rel'].tolist()
File_end_rel = filetime['File_end_rel'].tolist()
Seizure_or_not = filetime['Seizure_or_not'].tolist()
seizures = pd.read_csv('data-summary/siena_seizure_summary_generated.csv')
Seizure_filename = seizures['Seizure_filename']
Before_interictal_end = []
Preictal_start = []
Preictal_end = []
Sph_start = []
Ictal_start = seizures['Seizure_start_abs'].tolist()
Ictal_end = seizures['Seizure_stop_abs'].tolist()
Postictal_end = []
After_interictal_start = []
for si in range(len(seizures['Seizure_filename'])):
    before_interictal_end = seizures['Seizure_start_abs'][si] - INTERICTAL_BETWEEN_ICTAL
    preictal_start = seizures['Seizure_start_abs'][si] - SPH - PREICTAL
    preictal_end = seizures['Seizure_start_abs'][si] - SPH
    sph_start = seizures['Seizure_start_abs'][si] - SPH
    postictal_end = seizures['Seizure_stop_abs'][si] + POSTICTAL
    after_interictal_start = seizures['Seizure_stop_abs'][si] + INTERICTAL_BETWEEN_ICTAL
    Before_interictal_end.append(before_interictal_end)
    Preictal_start.append(preictal_start)
    Preictal_end.append(preictal_end)
    Sph_start.append(sph_start)
    Postictal_end.append(postictal_end)
    After_interictal_start.append(after_interictal_start)

file1 = pd.DataFrame(
    {'Seizure_filename': Seizure_filename, 'Before_interictal_end': Before_interictal_end,
     'Preictal_start': Preictal_start,
     'Preictal_end': Preictal_end, 'Sph_start': Sph_start, 'Ictal_start': Ictal_start, 'Ictal_end': Ictal_end,
     'Postictal_end': Postictal_end, 'After_interictal_start': After_interictal_start})
file1.to_csv('data-summary/siena_interval_division_generated.csv', encoding='utf-8', index=None, sep=',')
during = pd.read_csv('data-summary/siena_interval_division_generated.csv')
filesum = pd.read_csv('data-summary/siena_filetime_summary_generated.csv')
File_length = filesum['File_length']
# ns = 0  # 记录总的记录中第几次发作
interictal_sum = {}  # 记录所有可截取的间期记录
preictal_sum = {}  # 记录所有可截取的前期记录
sph_sum = {}  # 记录所有可截取的SPH记录
ictal_sum = {}
postictal_sum = {}
sum_temp = {}  # 记录所有的间期和前期时间信息

txt_file = open('data-summary/siena_segmentfinal_summary_generated.txt', 'w')
Seizure_filename = during['Seizure_filename']
# Seizure_filename,Before_interictal_end,Preictal_start,Preictal_end,Sph_start,Ictal_start,Ictal_end,Postictal_end,After_interictal_start
Before_interictal_end = during['Before_interictal_end']
Preictal_start = during['Preictal_start']
Preictal_end = during['Preictal_end']
Sph_start = during['Sph_start']
Ictal_start = during['Ictal_start']
Ictal_end = during['Ictal_end']
Postictal_end = during['Postictal_end']
After_interictal_start = during['After_interictal_start']
num_seizure = 1  # 记录一个文件中有几次发作记录
for sf_i, sf in enumerate(Seizure_filename):
    patient_name = sf.split('-')[0]
    print('========================= {} ========================'.format(sf), file=txt_file)
    print('开始处理文件[{}]....'.format(sf), file=txt_file)
    if sf_i > 0 and sf == Seizure_filename[sf_i - 1]:
        num_seizure += 1
    else:
        num_seizure = 1

    bie = Before_interictal_end[sf_i]
    ps = Preictal_start[sf_i]
    pe = Preictal_end[sf_i]
    ss = Sph_start[sf_i]
    si = Ictal_start[sf_i]
    ie = Ictal_end[sf_i]
    poe = Postictal_end[sf_i]
    ais = After_interictal_start[sf_i]
    file_length = int(File_length[filesum['File_name'] == sf])
    print('文件长度为 {} 秒，这是本文件第 {} 次发作。'.format(file_length, num_seizure), file=txt_file)
    if num_seizure == 1:
        pre_ais = 0
    else:
        pre_ais = After_interictal_start[sf_i - 1]

    interictal_temp = []
    preictal_temp = []
    sph_temp = []
    ictal_temp = []
    postictal_temp = []

    # 间期
    if pre_ais <= bie:
        interictal_temp = [pre_ais, bie]
    else:
        interictal_temp = []

    # 前期
    if pre_ais < ps:
        preictal_temp = [ps, pe]
    elif ps <= pre_ais < ss:
        preictal_temp = [pre_ais, pe]
    else:
        preictal_temp = []

    # SPH
    if pre_ais < ss:
        sph_temp = [ss, si]
    elif ss < pre_ais < si:
        sph = [pre_ais, si]
    else:
        sph_temp = []
    # 发作期
    ictal_temp = [si, ie]

    if len(interictal_temp) == 0:
        print('no interictal', file=txt_file)
    else:
        print('interictal:', interictal_temp, file=txt_file)
        interictal_sum[sf] = interictal_temp

    if len(preictal_temp) == 0:
        print('no preictal', file=txt_file)
    else:
        print('preictal:', preictal_temp, file=txt_file)
        preictal_sum[sf] = preictal_temp

    if len(sph_temp) == 0:
        print('no SPH', file=txt_file)
    else:
        print('SPH:', sph_temp, file=txt_file)
        sph_sum[sf] = sph_temp

    if len(ictal_temp) == 0:
        print('no ictal', file=txt_file)
    else:
        print('ictal:', ictal_temp, file=txt_file)
        ictal_sum[sf] = ictal_temp

    if sf_i == len(Seizure_filename) - 1:
        # 文件最后是否有间期
        if ais < file_length:
            interictal_temp = [ais, file_length]
        else:
            interictal_temp = []
        # 发作后期
        if poe <= file_length:
            postictal_temp = [ie, poe]
        elif file_length < poe:
            postictal_temp = [ie, file_length]
        else:
            postictal_temp = []
    else:
        # 文件后间期
        if sf != Seizure_filename[sf_i + 1] and ais < file_length:
            interictal_temp = [ais, file_length]
        else:
            interictal_temp = []
        # 发作后期
        if sf == Seizure_filename[sf_i + 1]:
            next_ss = Sph_start[sf_i + 1]
        else:
            next_ss = file_length
        if poe <= next_ss:
            postictal_temp = [ie, poe]
        elif ie < next_ss < poe:
            postictal_temp = [ie, next_ss]
        else:
            postictal_temp = []

    if len(postictal_temp) == 0:
        print('no postictal', file=txt_file)
    else:
        print('postictal:', postictal_temp, file=txt_file)
        postictal_sum[sf] = postictal_temp

    if len(interictal_temp) == 0:
        print('no interictal', file=txt_file)
    else:
        print('interictal:', interictal_temp, file=txt_file)
        interictal_sum[sf] = interictal_temp
sum_temp = {'interictal': interictal_sum, 'preictal': preictal_sum, 'sph': sph_sum, 'ictal': ictal_sum,
            'postictal': postictal_sum}
print(sum_temp, file=txt_file)
print(sum_temp)
print('******************** 切 片 开 始 ***********************')


# 带阻滤波器：允许特定频段通过并且屏蔽掉其他频段，相反的有带通滤波器
def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs  # 奈奎斯特频率，为防止信号混叠需要定义最小采样频率
    low = lowcut / nyq  # 截至频率下限/奈奎斯特频率，即 Wn的下限
    high = highcut / nyq  # 截止频率上限/奈奎斯特频率，即Wn的上限
    b, a = butter(order, [low, high], btype='bandstop')
    y = lfilter(b, a, data)
    return y


# 带通滤波器：允许特定频段通过并且屏蔽掉其他频段，相反的有带阻滤波器
def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs  # 奈奎斯特频率，为防止信号混叠需要定义最小采样频率
    low = lowcut / nyq  # 截至频率下限/奈奎斯特频率，即 Wn的下限
    high = highcut / nyq  # 截止频率上限/奈奎斯特频率，即Wn的上限
    b, a = butter(order, [low, high], btype='bandpass')
    y = lfilter(b, a, data)
    return y


# 高通滤波器，让某个频段以上的频率通过
def butter_highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='highpass')
    y = lfilter(b, a, data)
    return y


# 低通滤波器，让某个频段以下的频率通过
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='lowpass')
    y = filter(b, a, data)  # 沿着一维进行滤波
    return y  # 返回类型是数组


# 切片
def createclips(clip_dict, type, patients=None):
    # filetime = pd.read_csv('data-summary/chbmit_filetime_summary_generated.csv')
    if patients is None:
        patients = PATIENTS
    values = list(clip_dict[type].values())
    keys = list(clip_dict[type].keys())
    print(keys, values)
    for ii in range(len(keys)):
        if keys[ii][2:4] in patients:
            edf_path = os.path.join(DATA_ROOT_PATH, keys[ii][0:4], keys[ii])
            print('edf_path', edf_path)
            rawEEG = mne.io.read_raw_edf(edf_path, preload=False)
            if set(rawEEG.ch_names) >= set(CHANNELS):  # 当前edf包含所有指定的通道
                rawEEG = rawEEG.pick_channels(CHANNELS)  # 选择需要的通道
                rawEEG = rawEEG.reorder_channels(CHANNELS)  # 对通道重新排序
                # print(rawEEG.ch_names)
                # chschange = {'T8-P8-0': 'T8-P8'}
                # rawEEG = rawEEG.rename_channels(chschange)  # 更改通道名称
                # rawEEG = rawEEG.filter(LOW, HIGH, method='iir')  # 滤波器 癫痫预测不需要，检测需要
                fsr = int(filetime[filetime['File_name'] == keys[ii]]['File_start_rel'])
                start = values[ii][0]
                end = values[ii][1]
                print(keys[ii], ':::', type, ':::', start, ':::', end)
                print(keys[ii], ':::', type, ':::', start, ':::', end, file=txt_file)
                rawEEG_num = rawEEG.get_data()
                print('处理前:', rawEEG_num.shape)
                fsr_is = start - fsr
                fsr_ie = end - fsr
                temp = rawEEG_num[:, fsr_is * SAMPLE_RATE:fsr_ie * SAMPLE_RATE]
                print('处理后:', temp.shape)
                save_path = os.path.join(DATA_SAVE_PATH, keys[ii][0:4], type)
                if not os.path.lexists(save_path):
                    os.makedirs(save_path, mode=511, exist_ok=False)
                np.save(
                    os.path.join(save_path, keys[ii][0:-4] + SEPARATER + str(start) + SEPARATER + str(end) + '.npy'),
                    temp)
            else:
                print('通道数不足,丢弃（', keys[ii], ')')


if os.path.lexists(DATA_ROOT_PATH):
    # createclips(sum_temp, 'sph')
    createclips(sum_temp, 'interictal', PATIENTS)
    createclips(sum_temp, 'preictal', PATIENTS)
    # createclips(sum_temp, 'ictal')
    # createclips(sum_temp, 'postictal')
else:
    print('不存在路径(', DATA_ROOT_PATH, ')')
print('____________________________结束_____________________________')
txt_file.close()
