import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 180  # 图形分辨率
pd.options.display.notebook_repr_html = False  # 表格显示
# 相关系数矩阵
a = np.loadtxt('global/a_entmax.txt')
a = a[:20][:]
# 热力图

sns.heatmap(a)
plt.show()
plt.savefig


b = np.loadtxt('global/a_softmax.txt')
b = b[:20][:]
# 热力图

sns.heatmap(b)
plt.show()

# c = np.loadtxt('global/g_entmax.txt')
# c = c[:20][:]
#
# sns.heatmap(c)
# plt.show()
#
#
# d = np.loadtxt('global/g_softmax.txt')
# d = d[:20][:]
#
# sns.heatmap(d)
# plt.show()
