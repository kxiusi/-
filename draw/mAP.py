import matplotlib.pyplot as plt
import numpy as np
YBow = [0.297,  0.489, 0.473, 0.475,0.296]
YTFIDF = [0.265, 0.485, 0.442, 0.470,0.293]
labels = ['MOOCCube_course','MIT', 'Caltech', 'CMU', 'Princeton']
bar_width = 0.35

# 绘图
plt.bar(np.arange(5), YBow, label='BoW', color='steelblue', alpha=0.7, width=bar_width)
plt.bar(np.arange(5) + bar_width, YTFIDF, label='TF-IDF', color='indianred', alpha=0.7, width=bar_width)

plt.xlabel('Datasets');
plt.ylabel('mAP');

plt.xticks(np.arange(5) + bar_width/2, labels);

for x, y in enumerate(YBow):
    plt.text(x, y + 100, '%s' % y, ha='center')

for x, y in enumerate(YTFIDF):
    plt.text(x + bar_width, y + 100, '%s' % y, ha='center')

plt.legend()
plt.savefig('mAP.pdf')

plt.show()
