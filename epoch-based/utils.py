import torch
from scipy.signal import butter
from scipy.signal import lfilter


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


# 定义MyDataset类，继承Dataset方法，并重写__getitem__()和__len__()方法
class MyDataset(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)