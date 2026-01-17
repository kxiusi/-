import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    # 直接返回sigmoid函数
    return np.exp(x) / (1. + np.exp(x))
def sparsemax(x):
    interval0 = [1 if (i <= -1) else 0 for i in x]
    interval1 = [1 if (i >-1 and i < 1) else 0 for i in x]
    interval2 = [1 if (i >= 1) else 0 for i in x]
    y = np.zeros([60,]) * interval0 + (0.5*x+0.5) * interval1 + 1 * interval2
    return y

# param:起点，终点，间距
x = np.arange(-3, 3, 0.1)
y1 = sigmoid(x)
plt.plot(x, y1,"--",label=r"$\alpha=1$(softmax)")
y2 = sparsemax(x)
plt.plot(x, y2,"--",label=r"$\alpha=2$(sparsemax)")
plt.xlabel('t')  # 设置x轴名称 x label
plt.legend()  # 自动检测要在图例中显示的元素，并且显示
plt.savefig("entmax.pdf")
plt.show()

