import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

you = pd.read_csv('../result/tfyou.csv')[0:22]
wu = pd.read_csv('../result/tfwu.csv')[0:22]
rawyou = pd.read_csv('../result/resulttraw128.csv')[0:22]
# Patient,Accuracy,Specificity,Sensitivity,AUC,MCC,F1,FPR
acc = rawyou['Accuracy']
spe = rawyou['Specificity']
sen = rawyou['Sensitivity']
auc = rawyou['AUC']
mcc = rawyou['MCC']
fpr = rawyou['FPR']
plt.grid(alpha=0.8, axis='y')
plt.boxplot(acc, patch_artist=False, showmeans=False, sym='+', positions=[1],
            # boxprops={'color': 'blue', 'facecolor': ''},
            boxprops={'color': 'blue', 'linewidth': 1.0}, flierprops={'markeredgecolor': 'red'},
            medianprops={'color': 'red'}, widths=0.9,
            meanprops={'marker': '', 'markerfacecolor': 'black'},
            whiskerprops={'linestyle': '--'})
plt.boxplot(spe, patch_artist=False, showmeans=False, sym='+', positions=[2],
            boxprops={'color': 'blue'}, flierprops={'markeredgecolor': 'red'},
            medianprops={'color': 'red'}, widths=0.9,
            meanprops={'marker': '', 'markerfacecolor': 'black'},
            whiskerprops={'linestyle': '--'})
plt.boxplot(sen, patch_artist=False, showmeans=False, sym='+', positions=[3],
            boxprops={'color': 'blue'}, flierprops={'markeredgecolor': 'red'},
            medianprops={'color': 'red'}, widths=0.9,
            meanprops={'marker': '', 'markerfacecolor': 'black'},
            whiskerprops={'linestyle': '--'})
plt.boxplot(auc, patch_artist=False, showmeans=False, sym='+', positions=[4],
            boxprops={'color': 'blue'}, flierprops={'markeredgecolor': 'red'},
            medianprops={'color': 'red'}, widths=0.9,
            meanprops={'marker': '', 'markerfacecolor': 'black'},
            whiskerprops={'linestyle': '--'})
plt.boxplot(mcc, patch_artist=False, showmeans=False, sym='+', positions=[5],
            boxprops={'color': 'blue'}, flierprops={'markeredgecolor': 'red'},
            medianprops={'color': 'red'}, widths=0.9,
            meanprops={'marker': '', 'markerfacecolor': 'black'},
            whiskerprops={'linestyle': '--'})
plt.boxplot(fpr, patch_artist=False, showmeans=False, sym='+', positions=[6],
            boxprops={'color': 'blue'}, flierprops={'markeredgecolor': 'red'},
            medianprops={'color': 'red'}, widths=0.9,
            meanprops={'marker': '', 'markerfacecolor': 'black'},
            whiskerprops={'linestyle': '--'})
plt.xticks([1, 2, 3, 4, 5, 6], ['Accuracy', 'Specificity', 'Sensitivity', 'AUC', 'MCC', 'FPR'],
           fontsize=20, family='Times New Roman', rotation=25)
plt.grid(alpha=0.8, axis='y')
plt.ylim([0, 1])
plt.yticks(fontsize=30, family='Times New Roman')
plt.xlabel('(c)', fontsize=20, family='Times New Roman')
# plt.ylabel('AUC', fontsize=18, family='Times New Roman')
plt.tight_layout()
plt.show()
