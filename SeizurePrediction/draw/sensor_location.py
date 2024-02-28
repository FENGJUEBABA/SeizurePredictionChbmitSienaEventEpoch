# -*-coding:utf-8 -*-
# 绘制电极位置图
# 详情见 https://mne.tools/dev/auto_tutorials/intro/40_sensor_locations.html#sphx-glr-auto-tutorials-intro-40-sensor-locations-py
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
# the following import is required for matplotlib < 3.2:
from mpl_toolkits.mplot3d import Axes3D  # noqa

import mne

# 使用内置蒙太奇
builtin_montages = mne.channels.get_builtin_montages(descriptions=True)
for montage_name, montage_description in builtin_montages:
    print(f'{montage_name}: {montage_description}')  # 输出所有内置蒙太奇
# 获取10-20系统蒙太奇电极位置，
# 支持的所有蒙太奇见网页https://www.nmr.mgh.harvard.edu/mne/0.14/generated/mne.channels.read_montage.html
easycap_montage = mne.channels.make_standard_montage('standard_primed')
print(easycap_montage)

easycap_montage.plot()  # 2D

# fig = easycap_montage.plot(kind='3d', show=False)  # 3D
# fig = fig.gca().view_init(azim=70, elev=15)  # 设置3D视图的视角

# ssvep_folder = mne.datasets.ssvep.data_path()
# ssvep_data_raw_path = (ssvep_folder / 'sub-02' / 'ses-01' / 'eeg' /
#                        'sub-02_ses-01_task-ssvep_eeg.vhdr')
# ssvep_raw = mne.io.read_raw_brainvision(ssvep_data_raw_path, verbose=False)

# Use the preloaded montage
# ssvep_raw.set_montage(easycap_montage)
# fig = ssvep_raw.plot_sensors(show_names=True)

# Apply a template montage directly, without preloading
# ssvep_raw.set_montage('easycap-M1')
# fig = ssvep_raw.plot_sensors(show_names=True)

# ssvep_raw.set_montage('easycap-M1')
# fig = ssvep_raw.plot_sensors(show_names=True, sphere='eeglab')
plt.show()
