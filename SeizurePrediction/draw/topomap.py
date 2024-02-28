# -*-coding:utf-8 -*-
import mne
import numpy as np
from matplotlib import pyplot as plt

# path = "D:\learn\dataset\\rawdata\chb-mit\chb01\chb01_03.edf"

preictal_path = "D:\learn\dataset\segmention\siena\PN01\preictal\PN01-1_44253_46053.npy"
interictal_path = "D:\learn\dataset\segmention\siena\PN01\interictal\PN01-1_13872_42753.npy"
# CHANNELS = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4',
#             u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'T8-P8-0', u'P8-O2', u'FZ-CZ', u'CZ-PZ', u'P7-T7', u'T7-FT9',
#             u'FT9-FT10', u'FT10-T8']
# c_type = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
#           'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']
CHS = ['T4', 'F9', 'F4', 'C4', 'FC6', 'CP5', 'Pz', 'P3', 'O1', 'Fz',
       'CP1', 'CP6', 'T3', 'P4', 'F7', 'F3', 'C3', 'T6', 'FC5', 'FC2',
       'F10', 'FC1', 'F8', 'T5', 'O2', 'Fp1', 'CP2']
CHANNELS = ['EEG T4', 'EEG F9', 'EEG F4', 'EEG C4', 'EEG Fc6', 'EEG Cp5', 'EEG Pz', 'EEG P3', 'EEG O1', 'EEG Fz',
            'EEG Cp1', 'EEG Cp6', 'EEG T3', 'EEG P4', 'EEG F7', 'EEG F3', 'EEG C3', 'EEG T6', 'EEG Fc5', 'EEG Fc2',
            'EEG F10', 'EEG Fc1', 'EEG F8', 'EEG T5', 'EEG O2', 'EEG Fp1', 'EEG Cp2']

# 从EDF文件中读取数据
# raw_data = mne.io.read_raw_edf(path, preload=True)
# sfreq = raw_data.info['sfreq']
# info = mne.create_info(ch_names=CHS, sfreq=sfreq, ch_types='eeg')
# raw_data = raw_data.pick_channels(CHANNELS)  # 选择需要的通道
# raw_data = raw_data.reorder_channels(CHANNELS)  # 对通道重新排序
# raw_data = raw_data.rename_channels({'T8-P8-0': 'T8-P8'})  # 更改通道名称
# epochs = mne.Epochs(raw_data, events=events_from_annot, event_id=event_dict, proj=True, baseline=(None, 0),
#                     preload=True, tmin=-0.2, tmax=0.5)
# print(epochs)
# npy_data = raw_data.get_data()
# raw_data
sfreq = 512
info = mne.create_info(ch_names=CHS, sfreq=sfreq, ch_types='eeg')
npy_data = np.load(interictal_path)
print(npy_data.shape)
# DAN = mne.io.RawArray(npy_data, info)
# print(type(DAN))
# DAN.plot(n_channels=22,
#               scalings='auto',
#               title='Data from arrays',
#               show=True, block=True)
# raw_data.plot_psd_topo()
# DAN.plot_sensors()
# plt.show()

montage = mne.channels.make_standard_montage('standard_1020')
# montage.plot()
# plt.show()
# print(montage)
# 创建evokeds对象
evoked = mne.EvokedArray(npy_data, info)
# evoked.set_channel_types()

# evokeds设置通道
evoked.set_montage(montage)
print(evoked.data.shape)

# times = np.arange(0.05, 0.151, 0.02)
# evoked.plot_topomap(times, ch_type="eeg")

# evoked.plot_sensors()
times = np.arange(0, 1, 0.1)
# evoked.animate_topomap(times=times, ch_type="eeg", frame_rate=8, blit=False)
# plt.ylim([-3e6, 3e6])
# Define a threshold and create the mask
# print(evoked.data)
mask = evoked.data > 1e-5
# Select times and plot
mask_params = dict(markersize=10, markerfacecolor="y")
evoked.plot_topomap(times=times, ch_type="eeg", mask=mask, mask_params=mask_params)
plt.show()
# fig, ax = plt.subplots(ncols=2, figsize=(8, 4), gridspec_kw=dict(top=0.9),
#                        sharex=True, sharey=True)
# mne.viz.plot_topomap(raw_data, raw_data.info, axes=ax[0],
#                      show=False)
# mne.viz.plot_topomap(data2, info, axes=ax[8],
#                      show=False)

# add titles
# ax[0].set_title('MNE', fontweight='bold')
# ax[8].set_title('EEGLAB', fontweight='bold')
