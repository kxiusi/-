import matplotlib.pyplot as plt
import numpy as np

y1 = [47.59, 59.31]
y2 = [47.68, 59.44]
y3 = [47.75, 59.69]
y4 = [48.31, 59.88]
labels = ['P@10', 'P@20']
bar_width = 0.2

# 绘图
plt.bar(np.arange(2), y1, label='DIN4Rec w/o dep', color='#c82423', alpha=0.7, width=bar_width)
plt.bar(np.arange(2) + bar_width, y2, label='DIN4Rec-fus-emb', color='#f8ac8c', alpha=0.7, width=bar_width)
plt.bar(np.arange(2) + bar_width * 2, y3, label='DIN4Rec-fus-att', color='#9ac9db', alpha=0.7, width=bar_width)
plt.bar(np.arange(2) + bar_width * 3, y4, label='DIN4Rec', color='#2878b5', alpha=0.7, width=bar_width)

plt.xticks(np.arange(2) + bar_width * 1.5, labels)
plt.ylim(40, 62)
plt.ylabel("P(%)")

plt.legend(loc="upper left")
plt.savefig('fuse_P.pdf')

plt.show()

y1 = [24.77, 25.53]
y2 = [24.85, 25.67]
y3 = [25.03, 25.78]
y4 = [25.33, 25.81]
labels = ['MRR@10', 'MRR@20']
bar_width = 0.2

# 绘图
plt.bar(np.arange(2), y1, label='DIN4Rec w/o dep', color='#c82423', alpha=0.7, width=bar_width)
plt.bar(np.arange(2) + bar_width, y2, label='DIN4Rec-fus-emb', color='#f8ac8c', alpha=0.7, width=bar_width)
plt.bar(np.arange(2) + bar_width * 2, y3, label='DIN4Rec-fus-att', color='#9ac9db', alpha=0.7, width=bar_width)
plt.bar(np.arange(2) + bar_width * 3, y4, label='DIN4Rec', color='#2878b5', alpha=0.7, width=bar_width)

plt.xticks(np.arange(2) + bar_width * 1.5, labels)
plt.ylim((24, 26))
plt.ylabel("MRR(%)")

plt.legend()
plt.savefig('fuse_MRR.pdf')

plt.show()
