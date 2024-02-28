"""
本程序用于获取每个文件的时间信息
生成seizure-sum.csv，chbmit_filetime_summary_generated.csv
"""
import datetime
import os

import mne
import pandas as pd

data_path = 'D:\learn\dataset\\rawdata\siena'
PATIENTS = ['00', '01', '03', '05', '06', '07', '09', '10', '11', '12', '13', '14', '16', '17']

"""
生成siena_filetime_summary_generated.csv ,记录文件开始结束的绝对时间和相对时间,以及每个文件记录的时长和是否有发作记录
"""
segment = pd.read_csv('data-summary/siena_segementation_selective_essential.csv', header=None)  # <class 'pandas.core.frame.DataFrame'>
File_name = segment[0].tolist()
File_start_abs = []  # 记录文件开始的绝对时间
File_end_abs = []  # 记录文件结束的绝对时间
File_start_rel = []  # 记录文件开始的相对时间
File_end_rel = []  # 记录文件结束的相对时间
File_length = []  # 记录文件记录的时长
Seizure_or_not = segment[1].tolist()  # 记录文件是否含有发作记录
Seizure_n = []  # 文件含有发作记录个数
for edf_i, edfname in enumerate(File_name):
    # print(edfname)
    dtemp = mne.io.read_raw_edf(os.path.join(data_path, edfname.split('-')[0], edfname))
    File_start_abs.append(str(dtemp.info['meas_date'])[:-6])
    pn = edfname.split('-')[0]
    # print(pn)
    summary_file = os.path.join(data_path, pn, 'Seizures-list-' + pn + '.txt')
    # print(summary_file)
    with open(summary_file, 'r', encoding='utf-8') as file:
        temp = []
        for num, line in enumerate(file):
            temp.append(line)
        for i, t in enumerate(temp):
            if edfname in t:
                if 'Registration end time' in temp[i + 2]:
                    endtime = datetime.datetime.strptime(
                        File_start_abs[len(File_end_abs)][0:10] + ' ' + temp[i + 2][-9:-1].replace('.', ':'), '%Y-%m-%d %H:%M:%S')
                    starttime = datetime.datetime.strptime(File_start_abs[len(File_end_abs)], '%Y-%m-%d %H:%M:%S')
                    if endtime <= starttime:
                        endtime += datetime.timedelta(days=1)
                    File_end_abs.append(str(endtime))
                    File_length.append(int(datetime.timedelta.total_seconds(endtime - starttime)))
print(File_start_abs)
print(File_end_abs)
print(File_length)
# print(len(File_start_abs))
# print(len(File_end_abs))
# print(len(File_length))

for i in range(len(File_length)):
    File_start_rel.append(0)
    File_end_rel.append(File_length[i])
file_csv = pd.DataFrame(
    {'File_name': File_name, 'File_start_abs': File_start_abs, 'File_end_abs': File_end_abs, 'File_length': File_length,
     'File_start_rel': File_start_rel, 'File_end_rel': File_end_rel, 'Seizure_or_not': Seizure_or_not})
file_csv.to_csv('data-summary/siena_filetime_summary_generated.csv', encoding='utf-8', index=None, sep=',')

"""
生成siena_seizure_summary_generated.csv,记录所有发作记录开始和结束的绝对时间和相对时间,及距离上一次发作的时间和发作时长
"""
filetime = pd.read_csv('data-summary/siena_filetime_summary_generated.csv')
seizures = pd.read_csv('data-summary/siena_seizure_summary_raw_essential.csv')
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

print(Seizure_filename)
for j in range(len(Seizure_filename)):
    nn = 1  # 记录文件名连续出现次数，即同一个文件发作几次
    fstart = filetime[filetime['File_name'] == Seizure_filename[j]]['File_start_abs'].tolist()[0]
    sstartabs = datetime.datetime.strptime(fstart, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(seconds=int(Seizure_start_rel[j]))
    sstopabs = datetime.datetime.strptime(fstart, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(seconds=int(Seizure_stop_rel[j]))
    if Seizure_filename[j] == Seizure_filename[j - 1]:
        nn += 1
    if nn == 1:
        After_previous_seizure.append(0)
    else:
        aps = Seizure_start_rel[j] - Seizure_stop_rel[j - 1]
        After_previous_seizure.append(aps)
    Seizure_start_abs = Seizure_start_rel
    Seizure_stop_abs = Seizure_stop_rel
    Seizure_length.append(int(datetime.timedelta.total_seconds(sstopabs - sstartabs)))

file_csv1 = pd.DataFrame(
    {'Seizure_filename': Seizure_filename, 'Seizure_start_abs': Seizure_start_abs, 'Seizure_stop_abs': Seizure_stop_abs,
     'Seizure_start_rel': Seizure_start_rel, 'Seizure_stop_rel': Seizure_stop_rel, 'Seizure_length': Seizure_length,
     'After_previous_seizure': After_previous_seizure})
file_csv1.to_csv('data-summary/siena_seizure_summary_generated.csv', encoding='utf-8', index=None, sep=',')
