import matplotlib
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as ss


# 带通滤波器
def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    i, u = ss.butter(order, [low, high], btype='bandstop')
    y = ss.lfilter(i, u, data)
    return y


# 高通滤波器
def butter_highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    b, a = ss.butter(order, normal_cutoff, btype='high', analog=False)
    y = ss.lfilter(b, a, data)
    return y


def createSpec(data):
    fs = 256
    lowcut = 117
    highcut = 123

    y = butter_bandstop_filter(data, lowcut, highcut, fs, order=6)
    lowcut = 57
    highcut = 63
    y = butter_bandstop_filter(y, lowcut, highcut, fs, order=6)

    cutoff = 1
    y = butter_highpass_filter(y, cutoff, fs, order=6)

    Pxx = ss.spectrogram(y, nfft=256, fs=256, return_onesided=True, noverlap=128)[2]
    Pxx = np.delete(Pxx, np.s_[117:123 + 1], axis=0)
    Pxx = np.delete(Pxx, np.s_[57:63 + 1], axis=0)
    Pxx = np.delete(Pxx, 0, axis=0)

    result = (10 * np.log10(np.transpose(Pxx)) - (10 * np.log10(np.transpose(Pxx))).min()) / (
            10 * np.log10(np.transpose(Pxx))).ptp()
    return result


# 使用matplotlib库创建波谱图和可视化
def createSpecAndPlot(data):
    freqs, bins, Pxx = ss.spectrogram(data, nfft=256, fs=256, return_onesided=True, noverlap=128)

    print("Original")
    plt.pcolormesh(freqs, bins, 10 * np.log10(np.transpose(Pxx)), cmap=plt.cm.jet)
    plt.colorbar()
    plt.ylabel('sec')
    plt.xlabel('Hz')
    plt.title('Spettrogramma')
    plt.show()
    plt.close()

    fs = 256
    lowcut = 117
    highcut = 123

    y = butter_bandstop_filter(data, lowcut, highcut, fs, order=6)
    lowcut = 57
    highcut = 63
    y = butter_bandstop_filter(y, lowcut, highcut, fs, order=6)

    cutoff = 1
    y = butter_highpass_filter(y, cutoff, fs, order=6)

    # Pxx=signal.spectrogram(y, nfft=256, fs=256, return_onesided=True, noverlap=128)[2]
    freqs, bins, Pxx = ss.spectrogram(y, nfft=256, fs=256, return_onesided=True, noverlap=128)

    print("Filtered")
    plt.pcolormesh(freqs, bins, 10 * np.log10(np.transpose(Pxx)), cmap=plt.cm.jet)
    plt.colorbar()
    plt.ylabel('sec')
    plt.xlabel('Hz')
    plt.title('Spettrogramma')
    plt.show()
    plt.close()

    Pxx = np.delete(Pxx, np.s_[117:123 + 1], axis=0)
    Pxx = np.delete(Pxx, np.s_[57:63 + 1], axis=0)
    Pxx = np.delete(Pxx, 0, axis=0)

    print("Cleaned but not standard")
    freqs = np.arange(Pxx.shape[0])
    plt.pcolormesh(freqs, bins, 10 * np.log10(np.transpose(Pxx)), cmap=plt.cm.jet)
    plt.colorbar()
    plt.ylabel('sec')
    plt.xlabel('Hz')
    plt.title('Spettrogramma')
    plt.show()
    plt.close()

    result = (10 * np.log10(np.transpose(Pxx)) - (10 * np.log10(np.transpose(Pxx))).min()) / (
            10 * np.log10(np.transpose(Pxx))).ptp()

    print("Standard")
    freqs = np.arange(result.shape[1])
    plt.pcolormesh(freqs, bins, result, cmap=plt.cm.jet)
    plt.colorbar()
    plt.ylabel('sec')
    plt.xlabel('Hz')
    plt.title('Spettrogramma')
    plt.show()
    plt.close()

    return result


raw_data_path = "D:\learn\dataset\\rawdata\chb-mit\chb02/chb02_16.edf"
# raw_data = mne.io.read_raw_edf(raw_data_path)  # 返回rawEDF对象
# print(raw_data)  # <RawEDF | chb02_16.edf, 23 x 245504 (959.0 s), ~27 kB, data not loaded>
# print(raw_data.info) # 获取信息
# <Info | 7 non-empty values
#  bads: []
#  ch_names: FP1-F7, F7-T7, T7-P7, P7-O1, FP1-F3, F3-C3, C3-P3, P3-O1, ...
#  chs: 23 EEG
#  custom_ref_applied: False
#  highpass: 0.0 Hz
#  lowpass: 128.0 Hz
#  meas_date: 2074-07-25 09:31:46 UTC
#  nchan: 23
#  projs: []
#  sfreq: 256.0 Hz>
# print(raw_data)

channels = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4',
            u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'T8-P8', u'P8-O2', u'FZ-CZ', u'CZ-PZ', u'P7-T7', u'T7-FT9',
            u'FT9-FT10', u'FT10-T8']  # 22
cfchan = 'T8-P8'

raw_data = mne.io.read_raw_edf(raw_data_path, preload=True)
# print(raw_data)  # <RawEDF | chb02_16.edf, 23 x 245504 (959.0 s), ~43.8 MB, data loaded>
# print(raw_data.info)
# print(raw_data.info['bads'])  # 输出坏的通道  []
# print(raw_data.info['ch_names'])  # 输出通道名字
# ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
# 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2',
# 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8-8']
# 其实T8-P8，这里是rawEDF对重复通道重新标记
# print(raw_data.info['chs'])  #
# print(raw_data.info['nchan']) # 输出通道数，23
# print(raw_data.info['sfreq']) # 输出样本采样率，256.0
# print(mne.channel_type(raw_data.info, 22))  # eeg 获取通道类型，索引从0开始
## 可视化

# raw_data.plot_psd()  # 输出功率谱图，横轴是频率，纵轴是功率谱密度(psd)
raw_data.plot(duration=5, n_channels=22)  # 画各个通道的脑电图
plt.show()  # 防止窗口闪退

# picks = mne.pick_types(raw_data, eeg=True)
ndarr_data = raw_data.get_data()  # 将通道数据转化为 ndarray
# print(type(ndarr_data))  # <class 'numpy.ndarray'>
# print(ndarr_data.shape)  # (23, 245504)
uni_ndarr_data = ndarr_data[0:22, :]  # 删除重复通道 [0,22)
# print(uni_ndarr_data.shape)  # (22, 245504)
# mne.viz.plot_topomap(ndarr_data, raw_data.info)
# plt.show()  # 防止窗口闪退

# fft_uni_data = scipy.fft.fft(uni_ndarr_data)  # 傅里叶变换
# 滤波

# uni_data = mne.io.RawArray(uni_ndarr_data, mne.create_info(22, 256, 'eeg'))  # 从ndarray读取脑电信号
# import scipy.signal as ss

# f, t, sxx = ss.spectrogram(uni_ndarr_data, )
# print('采样频率:', f.shape)
# print('段时间数组:', t.shape)
# print('sxx:', sxx.shape)

# plt.pcolormesh(t, f, sxx, shading='gouraud')

# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
# uni_data.plot(color='blue')
# print(uni_data.info)
# plt.show()

# import librosa.feature as lf

# mfcc_data = lf.mfcc(uni_ndarr_data, sr=256, n_mfcc=22, dct_type=2)
# print(mfcc_data.shape)
# mfcc_uni_data = mne.io.RawArray(mfcc_data, mne.create_info(22, 256, 'eeg'), )
# mfcc_uni_data.plot()
# plt.show()

# tiny_uni_ndarr_data = uni_ndarr_data[0]
# print('=====================================')
# print(tiny_uni_ndarr_data.shape)
# f, t, zxx = ss.stft(tiny_uni_ndarr_data, nperseg=256)
# print('stft:::', type(f))  # <class 'numpy.ndarray'>
# print('stft:::', type(t))  # <class 'numpy.ndarray'>
# print('stft:::', type(zxx))  # <class 'numpy.ndarray'>
# print('stft::', f.shape)  # (129,)
# print('stft::', t.shape)  # (1919,)
# print('stft::', zxx.shape)  # (22, 129, 1919)
# # tx=t[0:129]
# # plt.plot(f,tx)
# plt.ylim([0, 0.05])
# plt.pcolormesh(np.abs(t), np.abs(f), np.abs(zxx), shading='gouraud')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()


# spec = createSpec(uni_ndarr_data)
# print(spec.shape)
# createSpecAndPlot(uni_ndarr_data[8])
