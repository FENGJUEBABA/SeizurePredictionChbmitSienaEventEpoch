# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

results = [{"acc": 0.9282, "spe": 0.927, "sen": 0.9294, "AUC": 0.9556, "MCC": 0.8576, "8-FPR": 1 - 0.073},
           {"acc": 0.9087, "spe": 0.9078, "sen": 0.9094, "AUC": 0.9472, "MCC": 0.8201, "8-FPR": 1 - 0.0922}]
data_length = len(results[0])
# 将极坐标根据数据长度进行等分
angles = np.linspace(0, 2 * np.pi, data_length, endpoint=False)
labels = [key for key in results[0].keys()]
score = [[v for v in result.values()] for result in results]
# 使雷达图数据封闭
score_a = np.concatenate((score[0], [score[0][0]]))
score_b = np.concatenate((score[1], [score[1][0]]))
angles = np.concatenate((angles, [angles[0]]))
labels = np.concatenate((labels, [labels[0]]))
# 设置图形的大小
fig = plt.figure(figsize=(4, 4), dpi=100)

# 新建一个子图
ax = plt.subplot(111, polar=True)
# 绘制雷达图
ax.plot(angles, score_a, linewidth=2, color='g')
ax.plot(angles, score_b, linewidth=2, color='b')
# 设置雷达图中每一项的标签显示
ax.set_thetagrids(angles * 180 / np.pi, labels)
# 设置雷达图的0度起始位置
ax.set_theta_zero_location('N')
# 设置雷达图的坐标刻度范围
ax.set_rlim(0.8, 1)
# 设置雷达图的坐标值显示角度，相对于起始角度的偏移量
ax.set_rlabel_position(180)
# ax.set_title("end-to-end experiment")
ax.grid(True)
plt.yticks([0.8, 0.85, 0.9, 0.95, 1], fontsize=12, family='Times New Roman')
plt.xticks(fontsize=14, family='Times New Roman')
plt.legend(["STFT data", "DAN data"], loc='best', fontsize=9)

# plt.xticks([0.8, 0.9, 8], [0.8, 0.9, 8])
plt.show()
