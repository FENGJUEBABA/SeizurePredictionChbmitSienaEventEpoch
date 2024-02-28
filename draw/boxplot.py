import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# acc = [0.7172,0.6940,0.7054,0.6987,0.6982,0.6883,0.6882,0.7145,0.68,0.7080,0.6996,0.7009]      #数据集
# plt.boxplot(acc)     #垂直显示箱线图
# plt.show()


# acc = pd.DataFrame({  # 用字典去建立数据表，第一列的列名a,列值是[8,2,3,4,5];第二列的列名是b，列值是 [5, 6, 7, 8]，以此类推
#     'SVM': [0.9573, 0.3649, 0.5967, 0.4927, 0.4515, 0.5970, 0.3707, 0.9430, 0.4753],
#     'NB': [0.5255, 0.3987, 0.5339, 0.4965, 0.4245, 0.5737, 0.3068, 0.7871, 0.4652],
#     'DT': [0.8165, 0.2495, 0.5791, 0.6726, 0.7445, 0.5013, 0.4984, 0.6287, 0.4984],
#     'our': [0.7172, 0.6940, 0.7054, 0.6987, 0.6982, 0.6883, 0.6882, 0.7145, 0.6800, 0.7080, 0.6996, 0.7009]
#     # 'our': [0.7172,0.6940,0.7054,0.6987,0.6982,0.6883,0.6882,0.7145,0.6800,0.7080,0.6996,0.7009]
# })
# acc.boxplot()
# plt.show()
# sen = pd.DataFrame({  # 用字典去建立数据表，第一列的列名a,列值是[8,2,3,4,5];第二列的列名是b，列值是 [5, 6, 7, 8]，以此类推
#     'SVM': [0.9950, 0.1880, 0.2130, 0.0627, 0.6742, 0.6566, 0, 0000, 0.9098, 0.0977],
#     'NB': [0.9975, 0.2719, 0.1992, 0.4486, 0.0414, 0.6579, 0.0714, 0.9461, 0.0000],
#     'DT': [8.0000, 0.4123, 0.2268, 0.3772, 0.7832, 0.5652, 0.8246, 0.2594, 0.0614],
#     'our': [0.7184, 0.6946, 0.7059, 0.7046, 0.6971, 0.6909, 0.7184, 0.7146, 0.6796, 0.7084, 0.6996, 0.7009]
#     # 'our': [0.7184,0.6946,0.7059,0.7046,0.6971,0.6909,0.7184,0.7146,0.6796,0.7084,0.6996,0.7009]
# })
#
# spe = pd.DataFrame({  # 用字典去建立数据表，第一列的列名a,列值是[8,2,3,4,5];第二列的列名是b，列值是 [5, 6, 7, 8]，以此类推
#     'SVM': [0.9193, 0.5420, 0.9824, 0.9355, 0.2290, 0.5367, 0.7409, 0.9762, 0.8523],
#     'NB': [0.0504, 0.5257, 0.8703, 0.5458, 0.8073, 0.4886, 0.5419, 0.6283, 0.9299],
#     'DT': [0.6318, 0.0866, 0.9332, 0.9768, 0.7059, 0.4367, 0.1727, 0.9975, 0.9349],
#     'our': [0.7159, 0.6935, 0.7049, 0.6925, 0.6992, 0.6857, 0.6579, 0.7143, 0.6805, 0.7077, 0.6996, 0.7009]
#     # 'our': [0.7159,0.6935,0.7049,0.6925,0.6992,0.6857,0.6579,0.7143,0.6805,0.7077,0.6996,0.7009]
# })
acc_cnn = [0.5021998742928976, 0.6175548589341693, 0.15829145728643215, 0.7972027972027972, 0.4996869129618034,
           0.6788413098236776, 0.1177207263619286, 0.897307451471509, 0.5497808390732624, 0.5012531328320802,
           0.6921151439299124, 0.5, 0.9128571428571428, 0.8335714285714285, 0.7271428571428571, 0.6914285714285714,
           0.8442857142857143, 0.535, 0.8221428571428572, 0.6885714285714286, 0.7335714285714285]
acc_chronoNet = [0.502232143, 0.495535714, 0.736607143, 0.828869048, 0.865327381, 0.770833333, 0.88764881, 0.761160714,
                 0.811011905, 0.494791667, 0.524088542, 0.755859375, 0.255208333, 0.863932292, 0.743489583, 0.533203125,
                 0.59765625, 0.875, 0.854166667, 0.96484375, 0.986979167]
acc_shb = [0.5007, 0.4986, 0.6464, 0.5007, 0.5129, 0.4950, 0.8264, 0.5007]
acc_svm = [0.9573, 0.3649, 0.5967, 0.4927, 0.4515, 0.5970, 0.3707, 0.9430, 0.4753]
acc_nb = [0.5255, 0.3987, 0.5339, 0.4965, 0.4245, 0.5737, 0.3068, 0.7871, 0.4652]
acc_dt = [0.8165, 0.2495, 0.5791, 0.6726, 0.7445, 0.5013, 0.4984, 0.6287, 0.4984]
acc_mv = [0.7172, 0.6940, 0.7054, 0.6987, 0.6982, 0.6883, 0.6882, 0.7145, 0.6800, 0.7080, 0.6996, 0.7009]
acc_our = [0.9222 + 0.0535, 0.8288 + 0.0199, 0.7039 - 0.0121, 0.8473 - 0.0364, 0.6828 + 0.1624, 0.9322 - 0.0079,
           0.6783 + 0.0470, 0.7731 + 0.0173, 0.7953 + 0.1174, 0.5335 + 0.2805, 0.4385 + 0.1549, 0.7580 - 0.1200,
           0.5597 + 0.1469, 0.8218 - 0.0193, 0.3189 - 0.0425, 0.9437 + 0.0219, 0.3714 + 0.0269, 0.8423 + 0.0175,
           0.6925 - 0.0313, 0.9418 + 0.0175]
acc_cnn = pd.DataFrame(acc_cnn)
acc_chronoNet = pd.DataFrame(acc_chronoNet)
acc_shb = pd.DataFrame(acc_shb)
acc_svm = pd.DataFrame(acc_svm)
acc_nb = pd.DataFrame(acc_nb)
acc_dt = pd.DataFrame(acc_dt)
acc_mv = pd.DataFrame(acc_mv)
acc_our = pd.DataFrame(acc_our)

# plt.figure(num=8)
# plt.boxplot((acc_cnn, acc_chronoNet, acc_shb, acc_svm, acc_nb, acc_dt, acc_mv, acc_our),
#             labels=('P-1D-CNN', 'ChornoNet', 'SHB', 'SVM', 'NB', 'DT', 'MV-TSK-FS', 'ours'), sym='+', autorange=True)
# plt.boxplot((acc_cnn, acc_chronoNet, acc_shb, acc_svm, acc_nb, acc_dt, acc_mv, acc_our),
#             labels=('P-1D-CNN', 'ChornoNet', 'SHB', 'SVM', 'NB', 'DT', 'MV-TSK-FS', 'ours'),
#             patch_artist=True,
#             boxprops={'color': 'black', 'facecolor': 'red'},
#             medianprops={'color': 'black'},
#             sym='+',
#             autorange=True)

plt.boxplot(acc_cnn, patch_artist=True, showmeans=True, sym='+', positions=[1],
            # boxprops={'color': 'blue', 'facecolor': ''},
            boxprops={'color': 'blue', 'facecolor': ''},
            medianprops={'color': 'red'}, widths=0.8,
            meanprops={'marker': '', 'markerfacecolor': 'black'})
plt.boxplot(acc_chronoNet, patch_artist=True, showmeans=True, sym='+', positions=[2],
            boxprops={'color': 'blue', 'facecolor': 'blue'},
            medianprops={'color': 'red'}, widths=0.8,
            meanprops={'marker': '', 'markerfacecolor': 'black'})
plt.boxplot(acc_shb, patch_artist=True, showmeans=True, sym='+', positions=[3],
            boxprops={'color': 'black', 'facecolor': 'blue'},
            medianprops={'color': 'red'}, widths=0.8,
            meanprops={'marker': '', 'markerfacecolor': 'black'})
plt.boxplot(acc_svm, patch_artist=True, showmeans=True, sym='+', positions=[4],
            boxprops={'color': 'black', 'facecolor': 'blue'},
            medianprops={'color': 'red'}, widths=0.8,
            meanprops={'marker': '', 'markerfacecolor': 'black'})
plt.boxplot(acc_nb, patch_artist=True, showmeans=True, sym='+', positions=[5],
            boxprops={'color': 'black', 'facecolor': 'blue'},
            medianprops={'color': 'red'}, widths=0.8,
            meanprops={'marker': '', 'markerfacecolor': 'black'})
plt.boxplot(acc_dt, patch_artist=True, showmeans=True, sym='+', positions=[6],
            boxprops={'color': 'blue', 'facecolor': 'blue'},
            medianprops={'color': 'red'}, widths=0.8,
            meanprops={'marker': '', 'markerfacecolor': 'black'})
plt.boxplot(acc_mv, patch_artist=True, showmeans=True, sym='+', positions=[7],
            boxprops={'color': 'black', 'facecolor': 'blue'},
            medianprops={'color': 'red'}, widths=0.8,
            meanprops={'marker': '', 'markerfacecolor': 'black'})
plt.boxplot(acc_our, patch_artist=True, showmeans=True, sym='+', positions=[8],
            boxprops={'color': 'black', 'facecolor': 'blue'},
            medianprops={'color': 'red'}, widths=0.8,
            meanprops={'marker': '', 'markerfacecolor': 'black'})
# no = [8, 2, 3, 4, 5, 6, 7, 8]
# lbs = ['P-1D-CNN', 'ChornoNet', 'SHB', 'SVM', 'NB', 'DT', 'MV-TSK-FS', 'OURS']
# plt.legend(lbs, bbox_to_anchor=(8, 0), loc='lower center', borderaxespad=0, ncol=2)
# plt.legend(lbs, loc='lower left', ncol=4, bbox_to_anchor=(8, 0), borderaxespad=0)
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], ['P-1D-CNN', 'ChornoNet', 'SHB', 'SVM', 'NB', 'DT', 'MV-TSK-FS', 'OURS'],
           fontsize=12, family='Times New Roman', rotation=20)
plt.grid(alpha=0.8, axis='y')
plt.ylim([0, 1])
plt.yticks(np.linspace(0, 1, 11), fontsize=12, family='Times New Roman')
plt.ylabel('Accuracy', fontsize=18, family='Times New Roman')
# plt.legend(020)
# plt.tight_layout()
plt.show()

sen_cnn = [0.0, 0.9108041005025, 0.27364438839848676, 0.686046511627907, 1.0, 0.35361216730038025,
           0.17418546365914786, 0.8583959899749374, 0.9536340852130326, 0.0012547051442910915, 0.8485607008760951, 0.0,
           0.8271428571428572, 0.6728571428571428, 0.4928571428571429, 0.99, 0.9185714285714286, 0.9771428571428571,
           0.6514285714285715, 0.41285714285714287, 0.6642857142857143]
sen_chronoNet = [1, 0, 0.777546662, 0.67195363, 0.819352102, 0.557630472, 0.984045358, 0.925533435, 0.923658423, 0,
                 0.050328146, 0.538350892, 0, 0.978232122, 0.620710108, 0.310190495, 0.578695941, 0.793532672, 1,
                 0.986465461, 0.974712225]
sen_shb = [0.0000, 0.0000, 0.9714, 0.0000, 1.0000, 0.0000, 0.9657, 0.0000]
sen_svm = [0.9950, 0.1880, 0.2130, 0.0627, 0.6742, 0.6566, 0, 0000, 0.9098, 0.0977]
sen_nb = [0.9975, 0.2719, 0.1992, 0.4486, 0.0414, 0.6579, 0.0714, 0.9461, 0.0000]
sen_dt = [1.0000, 0.4123, 0.2268, 0.3772, 0.7832, 0.5652, 0.8246, 0.2594, 0.0614]
sen_mv = [0.7184, 0.6946, 0.7059, 0.7046, 0.6971, 0.6909, 0.7184, 0.7146, 0.6796, 0.7084, 0.6996, 0.7009]
sen_our = [0.8445 + 0.1269, 0.6604 + 0.0386, 0.4550 - 0.0214, 0.9657 + 0.0043, 0.8487 - 0.0470, 0.9786 - 0.0057,
           0.9129 - 0.0599, 0.3980 + 0.1226, 0.6718 + 0.0200, 0.5889 + 0.2358, 0.8130 + 0.0978, 0.7166 + 0.2355,
           0.9729 + 0.0064, 0.6245 + 0.0774, 0.6468 - 0.0443, 0.6207 + 0.0849, 0.9862 - 0.0050, 0.7055 + 0.1278,
           0.4787 - 0.0462, 0.9712 - 0.0050]
sen_cnn = pd.DataFrame(sen_cnn)
sen_chronoNet = pd.DataFrame(sen_chronoNet)
sen_shb = pd.DataFrame(sen_shb)
sen_svm = pd.DataFrame(sen_svm)
sen_nb = pd.DataFrame(sen_nb)
sen_dt = pd.DataFrame(sen_dt)
sen_mv = pd.DataFrame(sen_mv)
sen_our = pd.DataFrame(sen_our)
plt.grid(alpha=0.8, axis='y')
plt.boxplot(sen_cnn, patch_artist=True, showmeans=True, sym='+', positions=[1],
            boxprops={'color': 'black', 'facecolor': '#FFDEAD'},
            medianprops={'color': 'black'}, widths=0.8,
            meanprops={'marker': 'D', 'markerfacecolor': 'black'})
plt.boxplot(sen_chronoNet, patch_artist=True, showmeans=True, sym='+', positions=[2],
            boxprops={'color': 'black', 'facecolor': '#BEBEBE'},
            medianprops={'color': 'black'}, widths=0.8,
            meanprops={'marker': 'D', 'markerfacecolor': 'black'})
plt.boxplot(sen_shb, patch_artist=True, showmeans=True, sym='+', positions=[3],
            boxprops={'color': 'black', 'facecolor': '#FF8247'},
            medianprops={'color': 'black'}, widths=0.8,
            meanprops={'marker': 'D', 'markerfacecolor': 'black'})
plt.boxplot(sen_svm, patch_artist=True, showmeans=True, sym='+', positions=[4],
            boxprops={'color': 'black', 'facecolor': '#7FFFD4'},
            medianprops={'color': 'black'}, widths=0.8,
            meanprops={'marker': 'D', 'markerfacecolor': 'black'})
plt.boxplot(sen_nb, patch_artist=True, showmeans=True, sym='+', positions=[5],
            boxprops={'color': 'black', 'facecolor': '#EEDD82'},
            medianprops={'color': 'black'}, widths=0.8,
            meanprops={'marker': 'D', 'markerfacecolor': 'black'})
plt.boxplot(sen_dt, patch_artist=True, showmeans=True, sym='+', positions=[6],
            boxprops={'color': 'black', 'facecolor': '#FF6A6A'},
            medianprops={'color': 'black'}, widths=0.8,
            meanprops={'marker': 'D', 'markerfacecolor': 'black'})
plt.boxplot(sen_mv, patch_artist=True, showmeans=True, sym='+', positions=[7],
            boxprops={'color': 'black', 'facecolor': '#EE82EE'},
            medianprops={'color': 'black'}, widths=0.8,
            meanprops={'marker': 'D', 'markerfacecolor': 'black'})
plt.boxplot(sen_our, patch_artist=True, showmeans=True, sym='+', positions=[8],
            boxprops={'color': 'black', 'facecolor': '#98FB98'},
            medianprops={'color': 'black'}, widths=0.8,
            meanprops={'marker': 'D', 'markerfacecolor': 'black'})

plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], ['P-1D-CNN', 'ChornoNet', 'SHB', 'SVM', 'NB', 'DT', 'MV-TSK-FS', 'OURS'],
           fontsize=12, family='Times New Roman', rotation=20)
plt.grid(alpha=0.8, axis='y')
plt.ylim([0, 1])
plt.yticks(np.linspace(0, 1, 11), fontsize=12, family='Times New Roman')
plt.ylabel('Sensitivity', fontsize=18, family='Times New Roman')
plt.tight_layout()
plt.show()

spe_cnn = [1.0, 0.32540675844806005, 0.04380475594493116, 0.904881101376721, 0, 1, 0.06132665832290363,
           0.9361702127659575, 0.14643304130162704, 1.0, 0.5356695869837297, 1, 0.9985714285714286, 0.9942857142857143,
           0.9614285714285714, 0.39285714285714285, 0.77, 0.09285714285714286, 0.9928571428571429, 0.9642857142857143,
           0.8028571428571428
           ]
spe_chronoNet = [0.001360544, 0.992778615, 0.688466165, 0.98119026, 0.908554394, 0.981774193, 0.794794075, 0.598992868,
                 0.700644437, 1, 0.991984892, 0.973527404, 0.521618798, 0.747512109, 0.863965786, 0.75509601,
                 0.61404162, 0.956872995, 0.709081374, 0.943006123, 0.998809524]
spe_shb = [1.0000, 0.9957, 0.3224, 1.0000, 0.0271, 0.9886, 0.6876, 1.0000]
spe_svm = [0.9193, 0.5420, 0.9824, 0.9355, 0.2290, 0.5367, 0.7409, 0.9762, 0.8523]
spe_nb = [0.0504, 0.5257, 0.8703, 0.5458, 0.8073, 0.4886, 0.5419, 0.6283, 0.9299]
spe_dt = [0.6318, 0.0866, 0.9332, 0.9768, 0.7059, 0.4367, 0.1727, 0.9975, 0.9349]
spe_mv = [0.7159, 0.6935, 0.7049, 0.6925, 0.6992, 0.6857, 0.6579, 0.7143, 0.6805, 0.7077, 0.6996, 0.7009]
spe_our = [1 - 0.02, 0.9971 - 0.0014, 0.9529 - 0.0029, 0.7289 - 0.0770, 0.8416 + 0.0257, 0.3792 + 0.3483,
           0.9514 + 0.0443, 0.9586 - 0.0186, 0.8744 + 0.0172, 1, 0.9675 - 0.25, 0.1625 + 0.0750, 0.5500 - 0.2425,
           0.4950 + 0.4162, 1, 0.5175 - 0.0300, 0.9012 + 0.0488, 0.2850 + 0.0787, 0.9787 - 0.0925, 0.9062 - 0.0162,
           0.9125 + 0.0398]

spe_cnn = pd.DataFrame(spe_cnn)
spe_chronoNet = pd.DataFrame(spe_chronoNet)
spe_shb = pd.DataFrame(spe_shb)
spe_svm = pd.DataFrame(spe_svm)
spe_nb = pd.DataFrame(spe_nb)
spe_dt = pd.DataFrame(spe_dt)
spe_mv = pd.DataFrame(spe_mv)
spa_our = pd.DataFrame(spe_our)
plt.grid(alpha=0.8, axis='y')
plt.boxplot(spe_cnn, patch_artist=True, showmeans=True, sym='+', positions=[1],
            boxprops={'color': 'black', 'facecolor': '#FFDEAD'},
            medianprops={'color': 'black'}, widths=0.8,
            meanprops={'marker': 'D', 'markerfacecolor': 'black'})
plt.boxplot(spe_chronoNet, patch_artist=True, showmeans=True, sym='+', positions=[2],
            boxprops={'color': 'black', 'facecolor': '#BEBEBE'},
            medianprops={'color': 'black'}, widths=0.8,
            meanprops={'marker': 'D', 'markerfacecolor': 'black'})
plt.boxplot(spe_shb, patch_artist=True, showmeans=True, sym='+', positions=[3],
            boxprops={'color': 'black', 'facecolor': '#FF8247'},
            medianprops={'color': 'black'}, widths=0.8,
            meanprops={'marker': 'D', 'markerfacecolor': 'black'})
plt.boxplot(spe_svm, patch_artist=True, showmeans=True, sym='+', positions=[4],
            boxprops={'color': 'black', 'facecolor': '#7FFFD4'},
            medianprops={'color': 'black'}, widths=0.8,
            meanprops={'marker': 'D', 'markerfacecolor': 'black'})
plt.boxplot(spe_nb, patch_artist=True, showmeans=True, sym='+', positions=[5],
            boxprops={'color': 'black', 'facecolor': '#EEDD82'},
            medianprops={'color': 'black'}, widths=0.8,
            meanprops={'marker': 'D', 'markerfacecolor': 'black'})
plt.boxplot(spe_dt, patch_artist=True, showmeans=True, sym='+', positions=[6],
            boxprops={'color': 'black', 'facecolor': '#FF6A6A'},
            medianprops={'color': 'black'}, widths=0.8,
            meanprops={'marker': 'D', 'markerfacecolor': 'black'})
plt.boxplot(spe_mv, patch_artist=True, showmeans=True, sym='+', positions=[7],
            boxprops={'color': 'black', 'facecolor': '#EE82EE'},
            medianprops={'color': 'black'}, widths=0.8,
            meanprops={'marker': 'D', 'markerfacecolor': 'black'})
plt.boxplot(spe_our, patch_artist=True, showmeans=True, sym='+', positions=[8],
            boxprops={'color': 'black', 'facecolor': '#98FB98'},
            medianprops={'color': 'black'}, widths=0.8,
            meanprops={'marker': 'D', 'markerfacecolor': 'black'})

plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], ['P-1D-CNN', 'ChornoNet', 'SHB', 'SVM', 'NB', 'DT', 'MV-TSK-FS', 'OURS'],
           fontsize=12, family='Times New Roman', rotation=20)
plt.grid(alpha=0.8, axis='y')
plt.ylim([0, 1])
plt.yticks(np.linspace(0, 1, 11), fontsize=12, family='Times New Roman')
plt.ylabel('Specificity', fontsize=18, family='Times New Roman')

plt.tight_layout()
plt.show()

auc_cnn = [0.9999905184510942, 0.9004801856592097, 0.040147915032504373, 0.9151992315976366, 0.3983315610678762,
           0.9971891353418643, 0.04338756779307468, 0.9706917481438264, 0.46421592153098645, 0.998159556409125,
           0.7274722314031463, 0.9980346938775511, 0.990108163265306, 0.884265306122449, 0.9695693877551019,
           0.938761224489796, 0.9602367346938776, 0.9922, 0.8513, 0.7260448979591836]
auc_chronoNet = [0.5, 0.483870968, 0.702272727, 0.736945813, 0.872922776, 0.806277056, 0.828125, 0.714705882, 0.78125,
                 0.5, 0.55, 0.75, 0.3125, 0.930555556, 0.727574751, 0.63546798, 0.597165992, 0.828431373, 0.857142857,
                 0.922422422, 0.985714286]
auc_dt = [0.827716995, 0.89722428, 0.747990302, 0.744704581, 0.779394448, 0.70821369, 0.410352878, 0.793739375,
          0.664297274, 0.782471627, 0.637680305, 0.500443489, 0.642317892, 0.728783003, 0.85, 0.195353057, 0.471946449,
          0.501279011, 0.526315789, 0.522581454, 0.788864348]
auc_mv = [0.7151, 0.6908, 0.6843, 0.7006, 0.6797, 0.7000, 0.7050, 0.7065, 0.6879, 0.6898, 0.7067, 0.6898, 0.7009,
          0.7015, 0.7030, 0.6995, 0.7017, 0.6962, 0.7014, 0.6984, 0.7069]
auc_shb = [0.8709964052287582, 0.8449, 0.8463, 0.8888, 0.8631, 0.8713]
auc_svm = [0.9819473020508396, 0.22464725175548658, 0.8749550197912919, 0.31195084485407065, 0.4262737569831964,
           0.5697027378573015, 0.009987421620383886, 0.9913598138023407, 0.37628959758595487]
auc_nb = [0.555634807, 0.750283164, 0.693939171, 0.879133427, 0.682404454, 0.890811002, 0.68829018, 0.629099243,
          0.643571722, 0.273644388, 0.45576221, 0.489226845, 0.418006306, 0.441185097, 0.780920656, 0.295369212,
          0.890358719, 0.489361702, 0.500626566, 0.630548246, 0.534205827]
auc_our = [0.9922 + 0.0023, 0.9846 - 0.0012, 0.8714 - 0.0131, 0.9652 + 0.0047, 0.9052 - 0.0087, 0.9582 + 0.0083,
           0.9728 + 0.0052, 0.8907 - 0.0491, 0.7447 + 0.0354, 0.9803 + 0.0174, 0.7079 + 0.1729, 0.3074 + 0.1171,
           0.8778 - 0.0642, 0.6264 + 0.0027, 0.9833 - 0.0383, 0.1959 - 0.0014, 0.9655 + 0.0141, 0.3350 + 0.0494,
           0.9731 - 0.0661, 0.8622 - 0.0356, 0.9766 + 0.0058]
auc_cnn = pd.DataFrame(auc_cnn)
auc_chronoNet = pd.DataFrame(auc_chronoNet)
auc_svm = pd.DataFrame(auc_svm)
auc_nb = pd.DataFrame(auc_nb)
auc_dt = pd.DataFrame(auc_dt)
auc_mv = pd.DataFrame(auc_mv)
auc_our = pd.DataFrame(auc_our)
plt.grid(alpha=0.8, axis='y')
plt.boxplot(auc_cnn, patch_artist=True, showmeans=True, sym='+', positions=[1],
            boxprops={'color': 'black', 'facecolor': '#FFDEAD'},
            medianprops={'color': 'black'}, widths=0.8,
            meanprops={'marker': 'D', 'markerfacecolor': 'black'})
plt.boxplot(auc_chronoNet, patch_artist=True, showmeans=True, sym='+', positions=[2],
            boxprops={'color': 'black', 'facecolor': '#BEBEBE'},
            medianprops={'color': 'black'}, widths=0.8,
            meanprops={'marker': 'D', 'markerfacecolor': 'black'})
plt.boxplot(auc_shb, patch_artist=True, showmeans=True, sym='+', positions=[3],
            boxprops={'color': 'black', 'facecolor': '#FF8247'},
            medianprops={'color': 'black'}, widths=0.8,
            meanprops={'marker': 'D', 'markerfacecolor': 'black'})
plt.boxplot(auc_svm, patch_artist=True, showmeans=True, sym='+', positions=[4],
            boxprops={'color': 'black', 'facecolor': '#7FFFD4'},
            medianprops={'color': 'black'}, widths=0.8,
            meanprops={'marker': 'D', 'markerfacecolor': 'black'})
plt.boxplot(auc_nb, patch_artist=True, showmeans=True, sym='+', positions=[5],
            boxprops={'color': 'black', 'facecolor': '#EEDD82'},
            medianprops={'color': 'black'}, widths=0.8,
            meanprops={'marker': 'D', 'markerfacecolor': 'black'})
plt.boxplot(auc_dt, patch_artist=True, showmeans=True, sym='+', positions=[6],
            boxprops={'color': 'black', 'facecolor': '#FF6A6A'},
            medianprops={'color': 'black'}, widths=0.8,
            meanprops={'marker': 'D', 'markerfacecolor': 'black'})
plt.boxplot(auc_mv, patch_artist=True, showmeans=True, sym='+', positions=[7],
            boxprops={'color': 'black', 'facecolor': '#EE82EE'},
            medianprops={'color': 'black'}, widths=0.8,
            meanprops={'marker': 'D', 'markerfacecolor': 'black'})
plt.boxplot(auc_our, patch_artist=True, showmeans=True, sym='+', positions=[8],
            boxprops={'color': 'black', 'facecolor': '#98FB98'},
            medianprops={'color': 'black'}, widths=0.8,
            meanprops={'marker': 'D', 'markerfacecolor': 'black'})
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], ['P-1D-CNN', 'ChornoNet', 'SHB', 'SVM', 'NB', 'DT', 'MV-TSK-FS', 'OURS'],
           fontsize=12, family='Times New Roman', rotation=20)
plt.grid(alpha=0.8, axis='y')
plt.ylim([0, 1])
plt.yticks(np.linspace(0, 1, 11), fontsize=12, family='Times New Roman')
plt.ylabel('AUC', fontsize=18, family='Times New Roman')
plt.tight_layout()
plt.show()
