"""
本程序用于获取chbmit数据集每个文件的时间信息
生成chbmit_filetime_summary_generated.csv和生成chbmit_seizure_summary_generated.csv
"""
import datetime
import os

import mne
import pandas as pd

CHB_PATIENTS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
                '18', '19', '20', '21', '22', '23']

'''
chbmit_segementation_raw_essential.csv(包含24个病人全部文件)

必须的文件，24号病人,没有时间信息不用
chbmit_segementation_selective_essential.csv(手工选择要使用的edf数据文件,未使用24号病人的数据,记录文件名和文件中是否包含发作记录)
chbmit_seizure_summary_raw_essential.csv(记录24个病人全部发作记录和发作期的起始时间)
'''

"""
生成chbmit_filetime_summary_generated.csv ,记录文件开始结束的绝对时间和相对时间,以及每个文件记录的时长和是否有发作记录
"""
# 读取配置信息
with open('data-summary/configure.txt', "r") as f:
    line = f.readline()
    if line.split("=")[0] == "RAW_DATA_PATH":
        RAW_DATA_PATH = line.split("=")[1].strip()  # 原始数据集路径
f.close()
segment = pd.read_csv('data-summary/chbmit_segementation_selective_essential.csv', header=None)
File_name = segment[0].tolist()
File_start_abs = []  # 记录文件开始的绝对时间
File_end_abs = []  # 记录文件结束的绝对时间
File_start_rel = []  # 记录文件开始的相对时间
File_end_rel = []  # 记录文件结束的相对时间
File_length = []  # 记录文件记录的时长
Seizure_or_not = segment[1].tolist()  # 记录文件是否含有发作记录
Seizure_n = []  # 文件含有发作记录个数
for edfname in File_name:
    temp = mne.io.read_raw_edf(os.path.join(RAW_DATA_PATH, edfname.split('_')[0][0:5], edfname), verbose=False)
    File_start_abs.append(str(temp.info['meas_date'])[:-6])
    date_temp = str(temp.info['meas_date'])[0:10]
    chbn = edfname.split('_')[0][0:5]
    summary_file = os.path.join(RAW_DATA_PATH, chbn, chbn + '-summary.txt')
    with open(summary_file, 'r', encoding='utf-8') as file:
        temp = []
        for num, line in enumerate(file):
            temp.append(line)
        for i, t in enumerate(temp):
            if edfname in t:
                if 'File End Time' in temp[i + 2]:
                    if '24' in temp[i + 2]:
                        temp[i + 2][-9:-1].replace('24', '00')
                    File_end_abs.append(date_temp + ' ' + temp[i + 2][-9:-1].replace(' ', '0'))
for i in range(len(File_end_abs)):
    sa = datetime.datetime.strptime(File_start_abs[i], '%Y-%m-%d %H:%M:%S')
    if int(File_end_abs[i][11:13]) == 24:
        time_temp = File_end_abs[i].split(' ')[0] + ' ' + File_end_abs[i].split(' ')[1][0:2].replace('24', '00') + \
                    File_end_abs[i][-6:]
        new = datetime.datetime.strptime(time_temp, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(days=1)
        File_end_abs[i] = str(new)
    if int(File_end_abs[i][11:13]) == 25:
        time_temp = File_end_abs[i].split(' ')[0] + ' ' + File_end_abs[i].split(' ')[1][0:2].replace('25', '01') + \
                    File_end_abs[i][-6:]
        new = datetime.datetime.strptime(time_temp, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(days=1)
        File_end_abs[i] = str(new)
    if int(File_end_abs[i][11:13]) == 26:
        time_temp = File_end_abs[i].split(' ')[0] + ' ' + File_end_abs[i].split(' ')[1][0:2].replace('26', '02') + \
                    File_end_abs[i][-6:]
        new = datetime.datetime.strptime(time_temp, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(days=1)
        File_end_abs[i] = str(new)
    if int(File_end_abs[i][11:13]) == 27:
        time_temp = File_end_abs[i].split(' ')[0] + ' ' + File_end_abs[i].split(' ')[1][0:2].replace('27', '03') + \
                    File_end_abs[i][-6:]
        new = datetime.datetime.strptime(time_temp, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(days=1)
        File_end_abs[i] = str(new)
    ea = datetime.datetime.strptime(File_end_abs[i], '%Y-%m-%d %H:%M:%S')
    interval = ea - sa
    File_length.append(interval.seconds)
# print(File_length)

pfirst = []  # 记录每个病人的第一个记录文件名称
for n in PATIENTS:
    for f in File_name:
        if ('chb' + n) in f:
            pfirst.append(f)
            break
for j in range(len(File_name)):
    if File_name[j] in pfirst:
        File_start_rel.append(0)
        File_end_rel.append(File_start_rel[j] + File_length[j])
    else:
        time = datetime.datetime.strptime(File_start_abs[j], '%Y-%m-%d %H:%M:%S') - \
               datetime.datetime.strptime(File_end_abs[j - 1], '%Y-%m-%d %H:%M:%S')
        File_start_rel.append(File_end_rel[j - 1] + time.seconds)
        File_end_rel.append(File_start_rel[j] + File_length[j])

file_csv = pd.DataFrame(
    {'File_name': File_name, 'File_start_abs': File_start_abs, 'File_end_abs': File_end_abs, 'File_length': File_length,
     'File_start_rel': File_start_rel, 'File_end_rel': File_end_rel, 'Seizure_or_not': Seizure_or_not})
file_csv.to_csv('data-summary/chbmit_filetime_summary_generated.csv', encoding='utf-8', index=None, sep=',')

"""
生成chbmit_seizure_summary_generated.csv,记录所有发作记录开始和结束的绝对时间和相对时间,及距离上一次发作的时间和发作时长
"""
filetime = pd.read_csv('data-summary/chbmit_filetime_summary_generated.csv')
seizures = pd.read_csv('data-summary/chbmit_seizure_summary_raw_essential.csv')
Seizure_filename = []
Seizure_start_rel = []
Seizure_stop_rel = []
for i in range(len(seizures['File_name'].tolist())):  # 被选择的文件
    if seizures['File_name'][i] in File_name:
        Seizure_filename.append(seizures['File_name'][i])
        Seizure_start_rel.append(seizures['Seizure_start'][i])
        Seizure_stop_rel.append(seizures['Seizure_stop'][i])
Seizure_start_abs = []  # 记录从第一个文件记录开始到发作的时间，单位秒数
Seizure_stop_abs = []  # 记录从第一个文件记录发作到结束的时间，单位秒数
Seizure_length = []  # 记录每次发作的时长
After_previous_seizure = []  # 记录相邻两次发作之间的时间间隔
Seizure_no = []  # 记录每个病人第几次发作
for patient in PATIENTS:
    s_no = 0  # 记录单个病人第几次发作
    for j in range(len(Seizure_filename)):
        if ('chb' + patient) in Seizure_filename[j]:
            fstart = filetime[filetime['File_name'] == Seizure_filename[j]]['File_start_rel'].tolist()[0]
            sstart = Seizure_start_rel[j] + fstart
            sstop = Seizure_stop_rel[j] + fstart
            if j > 0 and ('chb' + patient) in Seizure_filename[j - 1]:
                pfstart = filetime[filetime['File_name'] == Seizure_filename[j - 1]]['File_start_rel'].tolist()[0]
                aps = sstart - (Seizure_stop_rel[j - 1] + pfstart)
                s_no += 1
            else:
                aps = 0
                s_no = 0
            Seizure_start_abs.append(sstart)
            Seizure_stop_abs.append(sstop)
            Seizure_length.append(sstop - sstart)
            After_previous_seizure.append(aps)
            Seizure_no.append(s_no)
file_csv1 = pd.DataFrame(
    {'Seizure_filename': Seizure_filename, 'Seizure_start_abs': Seizure_start_abs, 'Seizure_stop_abs': Seizure_stop_abs,
     'Seizure_start_rel': Seizure_start_rel, 'Seizure_stop_rel': Seizure_stop_rel, 'Seizure_length': Seizure_length,
     'After_previous_seizure': After_previous_seizure, 'Seizure_no': Seizure_no})
file_csv1.to_csv('data-summary/chbmit_seizure_summary_generated.csv', encoding='utf-8', index=None, sep=',')
