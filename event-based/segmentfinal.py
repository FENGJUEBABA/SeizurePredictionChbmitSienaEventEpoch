'''
发作前期时间是一小时
'''
import mne.io
import numpy as np
import os
import pandas as pd
from scipy.signal import butter
from scipy.signal import lfilter

# 读取配置信息
with open('data-summary/configure.txt', "r") as f:
    lines = f.readlines()
    for line in lines:
        if "RAW_DATA_PATH" in line:
            RAW_DATA_PATH = line.split("=")[1].strip()  # 原始数据集路径
        if "SEGMENTATION_SAVE_PATH" in line:
            SEGMENTATION_SAVE_PATH = line.split("=")[1].strip()  # segmentation处理后的数据保存路径
        if "INTERICTAL_BETWEEN_ICTAL" in line:  # # 定义发作间期距离发作期的长度,单位小时，一般4小时
            INTERICTAL_BETWEEN_ICTAL = int(line.split("=")[1].strip()) * 3600
        if "PREICTAL" in line:  # 发作前期的长度,单位分钟
            PREICTAL = int(line.split("=")[1].strip()) * 60
        if "SPH" in line:  # seizure prediction horizon(SPH)一般为5分钟,单位分钟
            SPH = int(line.split("=")[1].strip()) * 60
        if "POSTICTAL" in line:  # 发作后期,单位分钟
            POSTICTAL = int(line.split("=")[1].strip()) * 60
        if 'SAMPLE_RATE' in line:  # 样本采样频率
            SAMPLE_RATE = int(line.split("=")[1].strip())
f.close()
PATIENTS = ['01', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18',
            '19', '20', '21', '22', '23']
# PATIENTS = ['12', '24']  # 12没有合适的间期记录，24号病人缺少文件开始结束时间。
# LOW = 0.8  # 滤波器下边界频率
# HIGH = 60  # 滤波器上边界频率
SEPARATER = '_'  # 分隔符
CHANNELS = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4',
            u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'T8-P8-0', u'P8-O2', u'FZ-CZ', u'CZ-PZ', u'P7-T7', u'T7-FT9',
            u'FT9-FT10', u'FT10-T8']  # 重复通道u'T8-P8-0',u'T8-P8-8'

filetime = pd.read_csv('data-summary/chbmit_filetime_summary_generated.csv')
seizures = pd.read_csv('data-summary/chbmit_seizure_summary_generated.csv')
# print(filetime)
File_name = filetime['File_name'].tolist()
File_start_abs = filetime['File_start_abs'].tolist()
File_end_abs = filetime['File_end_abs'].tolist()
File_length = filetime['File_length'].tolist()
File_start_rel = filetime['File_start_rel'].tolist()
File_end_rel = filetime['File_end_rel'].tolist()
Seizure_or_not = filetime['Seizure_or_not'].tolist()
Seizure_filename = seizures['Seizure_filename']
Seizure_no = seizures['Seizure_no']
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
file1.to_csv('data-summary/chbmit_interval_division_generated.csv', encoding='utf-8', index=None, sep=',')
during = pd.read_csv('data-summary/chbmit_interval_division_generated.csv')
ns = 0  # 记录总的记录中第几次发作
interictal_sum = {}  # 记录所有可截取的间期记录
preictal_sum = {}  # 记录所有可截取的前期记录
sph_sum = {}  # 记录所有可截取的SPH记录
ictal_sum = {}
postictal_sum = {}
sum_temp = {}  # 记录所有的间期和前期时间信息
txt_file = open('data-summary/chbmit_segmentfinal_summary_generated.txt', 'w', encoding='utf-8')
for p in range(len(PATIENTS)):
    print('开始处理病人chb', PATIENTS[p], '...', file=txt_file)
    patient_one = []  # 记录特定病人的所有记录的文件名
    pa = 'chb' + PATIENTS[p]
    for fi in range(len(File_name)):
        if pa in File_name[fi]:
            patient_one.append(File_name[fi])
    # 记录特定病人发作记录的文件
    seizure_one = pd.DataFrame(
        {'Seizure_filename': [], 'Before_interictal_end': [], 'Preictal_start': [], 'Preictal_end': [],
         'Sph_start': [], 'Ictal_start': [], 'Ictal_end': [], 'Postictal_end': [], 'After_interictal_start': []})
    for i in range(during.shape[0]):
        if pa in during['Seizure_filename'][i]:
            row_sf = during['Seizure_filename'][i]
            row_bie = during['Before_interictal_end'][i]
            row_ps = during['Preictal_start'][i]
            row_pe = during['Preictal_end'][i]
            row_ss = during['Sph_start'][i]
            row_is = during['Ictal_start'][i]
            row_ie = during['Ictal_end'][i]
            row_poe = during['Postictal_end'][i]
            row_ais = during['After_interictal_start'][i]
            a = [row_sf, row_bie, row_ps, row_pe, row_ss, row_is, row_ie, row_poe, row_ais]
            seizure_one.loc[len(seizure_one)] = a
    print('patient_one:', patient_one, file=txt_file)
    print('seizure_one:', seizure_one, file=txt_file)
    for j in range(seizure_one.shape[0]):
        sf = seizure_one['Seizure_filename'][j]
        # print('t1=', t, 'seizure_one:', seizure_one[j])
        bie = seizure_one['Before_interictal_end'][j]
        ps = seizure_one['Preictal_start'][j]
        pe = seizure_one['Preictal_end'][j]
        ss = seizure_one['Sph_start'][j]
        si = seizure_one['Ictal_start'][j]
        ie = seizure_one['Ictal_end'][j]
        poe = seizure_one['Postictal_end'][j]
        ais = seizure_one['After_interictal_start'][j]
        # print('发作记录：', Seizure_filename[j])
        if j > 0:
            pre_edf = seizure_one['Seizure_filename'][j - 1]
            pre_ais = seizure_one['After_interictal_start'][j - 1]
            pre_poe = seizure_one['Postictal_end'][j - 1]
        else:
            pre_edf = ''
            pre_ais = 0
            pre_poe = 0

        if j < seizure_one.shape[0] - 1:
            next_ss = seizure_one['Sph_start'][j + 1]
        else:
            next_ss = None

        for pt in patient_one:
            if pre_edf < pt <= sf:
                print('=====================================', file=txt_file)
                print(
                    'pre_edf:{},pre_ais:{},bie:{},ps:{},pe:{},si={},ie={},poe={}'.format(pre_edf, pre_ais, bie, ps, pe,
                                                                                         si, ie, poe), file=txt_file)
                fsr = int(filetime[filetime['File_name'] == pt]['File_start_rel'])
                fer = int(filetime[filetime['File_name'] == pt]['File_end_rel'])
                print(pt, ':', fsr, ',', fer, file=txt_file)
                interictal_temp = []
                preictal_temp = []
                sph_temp = []
                ictal_temp = []
                postictal_temp = []
                # 处理发作期数据
                if fsr <= si < ie <= fer:
                    ictal_temp = [si, ie]
                else:
                    ictal_temp = []
                # 处理发作后期
                if next_ss:
                    # 处理发作后期
                    if fsr <= ie < poe <= fer and poe <= next_ss:
                        postictal_temp = [ie, poe]
                    elif fsr <= ie < fer <= poe <= next_ss:
                        postictal_temp = [ie, fer]
                    elif ie <= fsr < poe <= fer and poe <= next_ss:
                        postictal_temp = [fsr, poe]
                    elif ie <= fsr < fer <= poe <= next_ss:
                        postictal_temp = [fsr, fer]
                    else:
                        postictal_temp = []
                else:
                    if fsr <= ie < poe <= fer:
                        postictal_temp = [ie, poe]
                    elif ie < fsr < poe <= fer:
                        postictal_temp = [fsr, poe]
                    elif fsr < ie < fer < poe:
                        postictal_temp = [ie, fer]
                    elif ie < fsr < fer < poe:
                        postictal_temp = [fsr, fer]
                    else:
                        postictal_temp = []

                if len(pre_edf) == 0:
                    # 处理发作间期
                    if bie >= fer:
                        interictal_temp = [fsr, fer]
                    elif fsr < bie < fer:
                        interictal_temp = [fsr, bie]
                    else:
                        interictal_temp = []
                    # 处理发作前期
                    if fsr <= ps < pe <= fer:
                        preictal_temp = [ps, pe]
                    elif ps < fsr < pe < fer:
                        preictal_temp = [fsr, pe]
                    elif ps < fsr < fer < pe:
                        preictal_temp = [fsr, fer]
                    elif fsr < ps < fer < pe:
                        preictal_temp = [ps, fer]
                    else:
                        preictal_temp = []

                    # 处理sph
                    if fsr <= ss and si <= fer:
                        sph_temp = [ss, si]
                    elif ss < fsr < si < fer:
                        sph_temp = [fsr, si]
                    elif ss < fsr < fer < si:
                        sph_temp = [fsr, fer]
                    elif fsr < ss < fer < si:
                        sph_temp = [ss, fer]
                    else:
                        sph_temp = []


                else:
                    # 处理发作间期
                    if pre_ais < fsr < bie < fer:
                        interictal_temp = [fsr, bie]
                    elif fsr <= pre_ais < bie <= fer:
                        interictal_temp = [pre_ais, bie]
                    elif fsr < pre_ais < fer < bie:
                        interictal_temp = [pre_ais, fer]
                    elif pre_ais < fsr < fer < bie:
                        interictal_temp = [fsr, fer]
                    else:
                        interictal_temp = []

                    # 处理发作前期
                    if fsr <= ps < pe <= fer and pre_poe <= ps:
                        preictal_temp = [ps, pe]
                    elif fsr < ps < fer < pe and pre_poe <= ps:
                        preictal_temp = [ps, fer]
                    elif ps < fsr < pe < fer and pre_poe <= fsr:
                        preictal_temp = [fsr, pe]
                    elif ps < fsr < fer < pe and pre_poe <= fsr:
                        preictal_temp = [fsr, fer]
                    elif ps < pre_poe < pe <= fer and pre_poe >= fsr:
                        preictal_temp = [pre_poe, pe]
                    elif ps < pre_poe < fer < pe and pre_poe >= fsr:
                        preictal_temp = [pre_poe, fer]
                    else:
                        preictal_temp = []

                    # 处理sph
                    if fsr <= ss < si <= fer and pre_poe <= ss:
                        sph_temp = [ss, si]
                    elif fsr < ss < fer < si and pre_poe <= ss:
                        sph_temp = [ss, fer]
                    elif ss < fsr < si < fer and pre_poe <= fsr:
                        sph_temp = [fsr, si]
                    elif ss < fsr < fer < si and pre_poe <= fsr:
                        sph_temp = [fsr, fer]
                    else:
                        sph_temp = []

                if len(interictal_temp) == 0:
                    print('no interictal', file=txt_file)
                else:
                    print('interictal:', interictal_temp, file=txt_file)
                    interictal_sum[pt] = interictal_temp

                if len(preictal_temp) == 0:
                    print('no preictal', file=txt_file)
                else:
                    print('preictal:', preictal_temp, file=txt_file)
                    preictal_sum[pt] = preictal_temp

                if len(sph_temp) == 0:
                    print('no SPH', file=txt_file)
                else:
                    print('SPH:', sph_temp, file=txt_file)
                    sph_sum[pt] = sph_temp

                if len(ictal_temp) == 0:
                    print('no ictal', file=txt_file)
                else:
                    print('ictal:', ictal_temp, file=txt_file)
                    ictal_sum[pt] = ictal_temp

                if len(postictal_temp) == 0:
                    print('no postictal', file=txt_file)
                else:
                    print('postictal:', postictal_temp, file=txt_file)
                    postictal_sum[pt] = postictal_temp

            one_patient_len = len(seizure_one['Seizure_filename'])
            one_patient_last_seizuresfilename = seizure_one['Seizure_filename'][one_patient_len - 1]
            one_patient_last_ais = seizure_one['After_interictal_start'][one_patient_len - 1]
            one_patient_last_bie = seizure_one['Before_interictal_end'][one_patient_len - 1]
            one_patient_last_ps = seizure_one['Preictal_start'][one_patient_len - 1]
            one_patient_last_pe = seizure_one['Preictal_end'][one_patient_len - 1]
            one_patient_last_si = seizure_one['Ictal_start'][one_patient_len - 1]
            one_patient_last_ie = seizure_one['Ictal_end'][one_patient_len - 1]
            one_patient_last_poe = seizure_one['Postictal_end'][one_patient_len - 1]
            if pt > one_patient_last_seizuresfilename and sf == one_patient_last_seizuresfilename:
                print('=====================================', file=txt_file)
                pre_edf = one_patient_last_seizuresfilename
                pre_ais = one_patient_last_ais
                pre_bie = one_patient_last_bie
                pre_ps = one_patient_last_ps
                pre_pe = one_patient_last_pe
                pre_si = one_patient_last_si
                pre_ie = one_patient_last_ie
                pre_poe = one_patient_last_poe
                print('pre_edf:{},pre_ais:{},pre_bie:{},pre_ps:{},pre_pe:{},pre_si={},pre_ie={},pre_poe={}'.format(
                    pre_edf, pre_ais, pre_bie, pre_ps, pre_pe, pre_si, pre_ie, pre_poe), file=txt_file)
                fsr = int(filetime[filetime['File_name'] == pt]['File_start_rel'])
                fer = int(filetime[filetime['File_name'] == pt]['File_end_rel'])
                print(pt, ':', fsr, ',', fer, file=txt_file)
                interictal_temp = []
                postictal_temp = []
                fsr = int(filetime[filetime['File_name'] == pt]['File_start_rel'])
                fer = int(filetime[filetime['File_name'] == pt]['File_end_rel'])
                last_ais = during['After_interictal_start'].tolist()[j]
                last_poe = during['Postictal_end'].tolist()[j]
                last_ie = during['Ictal_end'].tolist()[j]
                if last_ais <= fsr:
                    interictal_temp = [fsr, fer]
                elif fsr < last_ais < fer:
                    interictal_temp = [last_ais, fer]
                else:
                    interictal_temp = []

                if last_ie < fsr < last_poe <= fer:
                    postictal_temp = [fsr, last_poe]
                elif fsr <= last_ie < last_poe < fer:
                    postictal_temp = [last_ie, last_poe]
                elif fsr < last_ie < fer < last_poe:
                    postictal_temp = [last_ie, fer]
                elif last_ie < fsr < fer < last_poe:
                    postictal_temp = [fsr, fer]
                else:
                    postictal_temp = []

                print('no preictal', file=txt_file)
                print('no SPH', file=txt_file)
                print('no ictal', file=txt_file)

                if len(interictal_temp) == 0:
                    print('no interictal', file=txt_file)
                else:
                    print('interictal:', interictal_temp, file=txt_file)
                    interictal_sum[pt] = interictal_temp
                if len(postictal_temp) == 0:
                    print('no postictal', file=txt_file)
                else:
                    print('postictal:', postictal_temp, file=txt_file)
                    postictal_sum[pt] = postictal_temp

sum_temp = {'interictal': interictal_sum, 'preictal': preictal_sum, 'sph': sph_sum, 'ictal': ictal_sum,
            'postictal': postictal_sum}
print(sum_temp, file=txt_file)
print(sum_temp)
# print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
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
        if keys[ii][3:5] in patients:
            edf_path = os.path.join(RAW_DATA_PATH, keys[ii][0:5], keys[ii])
            print('edf_path', edf_path)
            rawEEG = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            if set(rawEEG.ch_names) >= set(CHANNELS):  # 当前edf包含所有指定的通道
                rawEEG = rawEEG.pick_channels(CHANNELS)  # 选择需要的通道
                rawEEG = rawEEG.reorder_channels(CHANNELS)  # 对通道重新排序
                # print(rawEEG.ch_names)
                chschange = {'T8-P8-0': 'T8-P8'}
                rawEEG = rawEEG.rename_channels(chschange)  # 更改通道名称
                # rawEEG = rawEEG.filter(LOW, HIGH, method='iir')  # 滤波器 癫痫预测不需要，检测需要
                fsr = int(filetime[filetime['File_name'] == keys[ii]]['File_start_rel'])
                start = values[ii][0]
                end = values[ii][1]
                print(keys[ii], ':::', type, ':::', start, ':::', end)
                print(keys[ii], ':::', type, ':::', start, ':::', end, file=txt_file)
                rawEEG_num = rawEEG.get_data()
                rawEEG_num = butter_bandstop_filter(rawEEG_num, 117, 123, 256, order=6)
                rawEEG_num = butter_bandstop_filter(rawEEG_num, 57, 63, 256, order=6)
                rawEEG_num = butter_highpass_filter(rawEEG_num, 1, 256, order=6)
                print('处理前:', rawEEG_num.shape)
                fsr_is = start - fsr
                fsr_ie = end - fsr
                temp = rawEEG_num[:, fsr_is * SAMPLE_RATE:fsr_ie * SAMPLE_RATE]
                print('处理后:', temp.shape)
                save_path = os.path.join(SEGMENTATION_SAVE_PATH, keys[ii][0:5], type)
                if not os.path.lexists(save_path):
                    os.makedirs(save_path, mode=511, exist_ok=False)

                np.save(os.path.join(save_path, keys[ii][0:5] + SEPARATER + str(start) + SEPARATER + str(end) + '.npy'),
                        temp)
            else:
                print('通道数不足,丢弃（', keys[ii], ')')


if os.path.lexists(RAW_DATA_PATH):
    # createclips(sum_temp, 'sph')
    createclips(sum_temp, 'interictal', PATIENTS)
    createclips(sum_temp, 'preictal', PATIENTS)
    # createclips(sum_temp, 'ictal')
    # createclips(sum_temp, 'postictal')
else:
    print('不存在路径(', RAW_DATA_PATH, ')')
print('____________________________结束_____________________________')
txt_file.close()
