# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

PATIENTS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19',
            '20', '21', '22', '23']
# 数据
index = 'FPR'
bar_width = 0.4
x = np.arange(len(PATIENTS))
x_name = list(pd.read_csv('../result/tfyou.csv')['Patient'][0:22])
you = pd.read_csv('../result/tfyou.csv')[0:22]
wu = pd.read_csv('../result/tfwu.csv')[0:22]

# 绘图 x 表示 从那里开始
plt.bar(x, you[index], bar_width, align='center')
plt.bar(x + bar_width, wu[index], bar_width, align="center")
plt.ylim([0, 0.25])

# 修改标题及x，y坐标轴字体及大小
plt.title(index, fontsize=40, fontproperties='Times New Roman')
# plt.xlabel('a', fontsize=20, fontweight='bold')
# plt.ylabel("数值", fontsize=15, fontweight='bold')

# # 修改坐标轴字体及大小
plt.yticks(fontproperties='Times New Roman', size=40)  # 设置大小及加粗
plt.xticks(ticks=x, labels=x_name, fontproperties='Times New Roman', size=30, rotation=45)
#
# # 设置标题
# plt.rcParams['font.sans-serif'] = ['KaiTi']  # 用来正常显示中文标签，如果想要用新罗马字体，改成 Times New Roman
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# plt.tight_layout()  # 解决绘图时上下标题重叠现象
#
# # 画线
# plt.vlines(starts2time, min(Mfcc1) - 10, max(Mfcc1) + 10, colors="black", linestyles="solid", lw=2)
# plt.vlines(ends2time, min(Mfcc1) - 10, max(Mfcc1) + 10, colors="black", linestyles="dashed", lw=2.5)
#
# # 添加图例
# plt.legend(['train acc', 'train loss'])  # 添加图例
# plt.legend(['train acc', 'train loss'], fontsize=12)  # 并且设置大小
#
# # 取消坐标轴刻度
# plt.xticks([])  # 去x坐标刻度
# plt.yticks([])  # 去y坐标刻度
# plt.axis('off')  # 去坐标轴
#
# # 取消savefig保存图片时的白色边框
# plt.savefig(pic_name, bbox_inches='tight', pad_inches=0.0)
#
# # 取消每一个的边框
# ax1 = plt.subplot(2, 3, 8)
# ax1.spines['right'].set_visible(False)  # 右边
# ax1.spines['top'].set_visible(False)  # 上边
# ax1.spines['left'].set_visible(False)  # 左边
# ax1.spines['bottom'].set_visible(False)  # 下边

plt.show()
