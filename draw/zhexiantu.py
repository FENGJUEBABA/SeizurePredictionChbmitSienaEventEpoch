# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd

plt.figure(figsize=(20, 10), dpi=100)
data = pd.read_excel('xiaorong.xlsx', index_col=0, header=0)
print(data)
print(type(data))
xlab = ['acc', 'spe', 'sen', 'auc', 'mcc']
dataslice = data.iloc[0:8, 0:5]  # 切片
st000 = dataslice.iloc[0, 0:5].tolist()
st001 = dataslice.iloc[1, 0:5].tolist()
st010 = dataslice.iloc[2, 0:5].tolist()
st100 = dataslice.iloc[3, 0:5].tolist()
st011 = dataslice.iloc[4, 0:5].tolist()
st110 = dataslice.iloc[5, 0:5].tolist()
st101 = dataslice.iloc[6, 0:5].tolist()
st111 = dataslice.iloc[7, 0:5].tolist()
plt.plot(xlab, st000, c='#828282', linestyle='-.', label="STB000", linewidth=4)
plt.plot(xlab, st001, c='#FF8C00', linestyle='-.', label="STB001", linewidth=4)
plt.plot(xlab, st010, c='#228B22', linestyle='-.', label="STB010", linewidth=4)
plt.plot(xlab, st100, c='#BDB76B', linestyle='-.', label="STB100", linewidth=4)
plt.plot(xlab, st011, c='#6495ED', linestyle='-.', label="STB011", linewidth=4)
plt.plot(xlab, st110, c='#9370DB', linestyle='-.', label="STB110", linewidth=4)
plt.plot(xlab, st101, c='#DAA520', linestyle='-.', label="STB101", linewidth=4)
plt.plot(xlab, st111, c='#CD5C5C', label="STB111", linewidth=4)
plt.scatter(xlab, st000, c='#828282', marker='s', linewidths=3)  # 加标记
plt.scatter(xlab, st001, c='#FF8C00', marker='s', linewidths=3)  # 加标记
plt.scatter(xlab, st010, c='#228B22', marker='s', linewidths=3)  # 加标记
plt.scatter(xlab, st100, c='#BDB76B', marker='s', linewidths=3)  # 加标记
plt.scatter(xlab, st011, c='#6495ED', marker='s', linewidths=3)  # 加标记
plt.scatter(xlab, st110, c='#9370DB', marker='s', linewidths=3)  # 加标记
plt.scatter(xlab, st101, c='#DAA520', marker='s', linewidths=3)  # 加标记
plt.scatter(xlab, st111, c='#CD5C5C', marker='s', linewidths=3)  # 加标记
plt.legend(loc='best')
plt.ylim([0.7, 1])
plt.grid(True, alpha=0.3)
plt.yticks(fontsize=30, family='Times New Roman')
plt.xticks(fontsize=30, family='Times New Roman')
plt.legend(fontsize=16)
# plt.xlabel("comparison of results", fontdict={'size': 16})
# plt.ylabel("", fontdict={'size': 16})

# plt.title("results of  Ablation experiments", fontdict={'size': 20})
plt.show()
