"""
提取时域,频域(fft),时频域(STFT)数据等
不平衡处理,主要是对数据较多的进行欠采样,间隔采样
"""
import math
import os.path

import emd
import numpy as np
import python_speech_features
import scipy
from scipy import signal
from scipy.fft import fft
from scipy.signal import hilbert2
from sklearn import preprocessing
from tqdm import tqdm

DATA_ROOT_PATH = 'D:/learn/dataset/segmention/chbmit'
DATA_SAVE_PATH = 'D:/learn/dataset/preprocess/chbmitmul10'
WINDOWS = 10
SAMPLE_RATE = 256  # 样本采样频率
DATABASE_NAME = 'chb'
# 获得数据,可以是‘time’ (时域),'fre' (频域，fft),'tf' (时频域,stft),'mfcc','ht'(希尔伯特变化),'hht'(希尔伯特黄变换)
DOMAIN = ['time']
SEPARATER = '_'  # 分隔符
CHANNELS = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4',
            u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'T8-P8-0', u'P8-O2', u'FZ-CZ', u'CZ-PZ', u'P7-T7', u'T7-FT9',
            u'FT9-FT10', u'FT10-T8']  #
PATIENTS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18',
            '19', '20', '21', '22', '23']

# 提取多视角信息
def mypreprocess(patientname, over_lop, dur_type):
    print('====================开始处理{}--{} 数据========================'.format(patientname, dur_type))
    overlop = over_lop
    npy_path = os.path.join(DATA_ROOT_PATH, patientname, dur_type)
    npys = os.listdir(npy_path)
    for npy in npys:
        npy_start = int(npy.split(SEPARATER)[1])
        npy_end = int(npy.split(SEPARATER)[2][0:-4])
        edf_name = npy.split(SEPARATER)[0]
        data_one = np.load(os.path.join(npy_path, npy))
        n = math.floor((data_one.shape[1] - WINDOWS * SAMPLE_RATE) / math.floor(overlop * WINDOWS * SAMPLE_RATE)) + 1
        print(npy)
        for i in tqdm(range(n)):
            # print('{}--{} 进度[{:.4}%]'.format(patientname, dur_type, ((i + 8) / n) * 100))
            if int(npy_start + (i * overlop + 1) * WINDOWS) <= npy_end:
                data_temp = data_one[:,
                            round(i * overlop * WINDOWS * SAMPLE_RATE):round((i * overlop + 1) * WINDOWS * SAMPLE_RATE)]
                ns = str(int(npy_start + i * overlop * WINDOWS))
                ne = str(int(npy_start + (i * overlop + 1) * WINDOWS))
                savepath = os.path.join(DATA_SAVE_PATH, patientname, dur_type)
                if 'time' in DOMAIN:
                    # 获取时域
                    time_temp = preprocessing.scale(data_temp)  # z-score标准化，关键
                    if not os.path.lexists(os.path.join(savepath, 'time_domain')):
                        os.makedirs(os.path.join(savepath, 'time_domain'))
                    time_file = os.path.join(os.path.join(savepath, 'time_domain'),
                                             edf_name + SEPARATER + dur_type + SEPARATER + ns + SEPARATER + ne + '.npy')
                    np.save(time_file, time_temp)
                if 'fre' in DOMAIN:
                    # 获取频域
                    frequency_temp = []
                    for ch in range(data_temp.shape[0]):
                        data_fre = fft(data_temp[ch, :])
                        frequency_temp.append(data_fre)
                    frequency_temp = preprocessing.scale(np.abs(data_temp))
                    if not os.path.lexists(os.path.join(savepath, 'frequency_domain')):
                        os.makedirs(os.path.join(savepath, 'frequency_domain'))
                    frequency_file = os.path.join(os.path.join(savepath, 'frequency_domain'),
                                                  edf_name + SEPARATER + dur_type + SEPARATER + ns + SEPARATER + ne + '.npy')
                    np.save(frequency_file, frequency_temp)
                if 'tf' in DOMAIN:
                    # 获取时频信息
                    f, t, pxx = scipy.signal.stft(data_temp, fs=SAMPLE_RATE, nperseg=256, noverlap=128)
                    pxx = np.delete(pxx, np.s_[117:123 + 1], axis=1)
                    pxx = np.delete(pxx, np.s_[57:63 + 1], axis=1)
                    pxx = np.delete(pxx, 0, axis=1)
                    pxx = preprocessing.scale(np.abs(pxx).reshape((22, 114 * 21)))
                    if not os.path.lexists(os.path.join(savepath, 'timefrequency_domain')):
                        os.makedirs(os.path.join(savepath, 'timefrequency_domain'))
                    timefrequency_file = os.path.join(os.path.join(savepath, 'timefrequency_domain'),
                                                      edf_name + SEPARATER + dur_type + SEPARATER + ns + SEPARATER + ne + '.npy')
                    np.save(timefrequency_file, pxx)
                if 'mfcc' in DOMAIN:
                    # MFCC
                    mfcc_data = []
                    for n_ch in range(len(CHANNELS)):
                        mfcc_data_temp = python_speech_features.mfcc(data_temp[n_ch], samplerate=256, winlen=0.1,
                                                                     winstep=0.05, numcep=13, nfilt=13,
                                                                     lowfreq=0, highfreq=128, preemph=0.97)
                        mfcc_data.append(mfcc_data_temp.T)
                    mfcc_data = np.array(mfcc_data)
                    mfcc_data = preprocessing.scale(np.abs(mfcc_data).reshape((22, 13 * 78)))
                    if not os.path.lexists(os.path.join(savepath, 'mfcc_domain')):
                        os.makedirs(os.path.join(savepath, 'mfcc_domain'))
                    mfcc_file = os.path.join(os.path.join(savepath, 'mfcc_domain'),
                                             edf_name + SEPARATER + dur_type + SEPARATER + ns + SEPARATER + ne + '.npy')
                    np.save(mfcc_file, mfcc_data)
                if 'ht' in DOMAIN:
                    # 希尔伯特变换,频谱图
                    ht_temp = hilbert2(data_temp)
                    ht_temp = preprocessing.scale(np.abs(ht_temp))  # z-score标准化，关键
                    if not os.path.lexists(os.path.join(savepath, 'ht_domain')):
                        os.makedirs(os.path.join(savepath, 'ht_domain'))
                    ht_file = os.path.join(os.path.join(savepath, 'ht_domain'),
                                           edf_name + SEPARATER + dur_type + SEPARATER + ns + SEPARATER + ne + '.npy')
                    np.save(ht_file, ht_temp)
                if 'hht' in DOMAIN:
                    # 希尔伯特黄变换
                    hht_temp = []  # (22, 78, 1024)
                    for n_ch1 in range(len(CHANNELS)):
                        imf = emd.sift.sift(data_temp[n_ch1], max_imfs=1)  # (1024, 6) emd分解
                        IP, IF, IA = emd.spectra.frequency_transform(imf, 256, 'hilbert')  # (1024, 6)  瞬时相位/幅值/频率
                        _, hht = emd.spectra.hilberthuang(IF, IA, sample_rate=256, sum_time=False)  # HHT 谱(78, 1024)
                        hht_temp.append(hht)
                    hht_temp = np.array(hht_temp).reshape((22, 32 * 1024))
                    hht_temp = preprocessing.scale(hht_temp)  # z-score标准化，关键
                    if not os.path.lexists(os.path.join(savepath, 'hht_domain')):
                        os.makedirs(os.path.join(savepath, 'hht_domain'))
                    hht_file = os.path.join(os.path.join(savepath, 'hht_domain'),
                                            edf_name + SEPARATER + dur_type + SEPARATER + ns + SEPARATER + ne + '.npy')
                    np.save(hht_file, hht_temp)


def balance_feature_extraction(patients_list):
    for patient in patients_list:
        preictal_size, interictal_size = 0, 0  # 记录发作前期和发作后期数据大小
        patient_path = os.path.join(DATA_ROOT_PATH, DATABASE_NAME + patient)
        sub_dirs = os.listdir(patient_path)
        for cla in sub_dirs:
            if cla == 'interictal':
                npys = os.listdir(os.path.join(patient_path, cla))
                for npy in npys:
                    interictal_size += os.path.getsize(os.path.join(patient_path, cla, npy))
            if cla == 'preictal':
                npys = os.listdir(os.path.join(patient_path, cla))
                for npy in npys:
                    preictal_size += os.path.getsize(os.path.join(patient_path, cla, npy))
        print(interictal_size / (1024 * 1024))
        print(preictal_size / (1024 * 1024))
        if interictal_size >= preictal_size:
            r = interictal_size / preictal_size
            rr = round(WINDOWS * r) / WINDOWS  # 为了取整秒
            print('间期多', patient, '   ', rr)
            mypreprocess(patientname=DATABASE_NAME + patient, over_lop=rr, dur_type='interictal')
            mypreprocess(patientname=DATABASE_NAME + patient, over_lop=1, dur_type='preictal')
        else:
            r = preictal_size / interictal_size
            rr = round(WINDOWS * r) / WINDOWS
            print('前期多', patient, '   ', rr)
            mypreprocess(patientname=DATABASE_NAME + patient, over_lop=1, dur_type='interictal')
            mypreprocess(patientname=DATABASE_NAME + patient, over_lop=rr, dur_type='preictal')
    print('ALL END')


if __name__ == '__main__':
    balance_feature_extraction(PATIENTS)
