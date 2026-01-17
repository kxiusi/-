import matplotlib.pyplot as plt
import numpy as np
from brokenaxes import brokenaxes

x1 = [4, 8, 12]
y1 = [73.2230, 73.4407, 73.3189]
x2 = [4, 8, 12]
y2 = [59.6251, 59.88, 59.6083]

fig1 = plt.figure(figsize=(6, 3.5))
bax = brokenaxes(ylims=((59.5, 60), (73, 73.5)), hspace=.1, despine=False)
x = np.linspace(0, 1, 100)
bax.plot(x1, y1, label='MOOCCube_video', color="steelblue")
bax.plot(x2, y2, label='MOOCCube_course', color="darkorange")
bax.plot(x1, y1, 'o-', x2, y2, '+-')
bax.legend()
bax.set_xlabel('n_sample_num')
bax.set_ylabel('P@20(%)')
fig1.savefig("sample_num_P_20.pdf")

x1 = [4, 8, 12]
y1 = [43.5648, 43.5885, 43.4891]
x2 = [4, 8, 12]
y2 = [25.7348, 25.81, 25.7293]

fig1 = plt.figure(figsize=(6, 3.5))
bax = brokenaxes(ylims=((25.7, 25.9), (43.4, 43.7)), hspace=.1, despine=False)
x = np.linspace(0, 1, 100)
bax.plot(x1, y1, label='MOOCCube_video', color="steelblue")
bax.plot(x2, y2, label='MOOCCube_course', color="darkorange")
bax.plot(x1, y1, 'o-', x2, y2, '+-')
bax.legend()
bax.set_xlabel('n_sample_num')
bax.set_ylabel('MRR@20(%)')
fig1.savefig("sample_num_MRR_20.pdf")
fig1.show()

x1 = [4, 8, 12]
y1 = [66.10, 66.1648, 66.0812]
x2 = [4, 8, 12]
y2 = [47.5827, 47.7342, 47.6900]

fig1 = plt.figure(figsize=(6, 3.5))
bax = brokenaxes(ylims=((47.5, 47.8), (66, 66.2)), hspace=.1, despine=False)
x = np.linspace(0, 1, 100)
bax.plot(x1, y1, label='MOOCCube_video', color="steelblue")
bax.plot(x2, y2, label='MOOCCube_course', color="darkorange")
bax.plot(x1, y1, 'o-', x2, y2, '+-')
bax.legend()
bax.set_xlabel('n_sample_num')
bax.set_ylabel('P@10(%)')
fig1.savefig("sample_num_P_10.pdf")


x1 = [4, 8, 12]
y1 = [43.0628, 43.0792, 42.9812]
x2 = [4, 8, 12]
y2 = [24.9053, 24.9510, 24.9078]

fig1 = plt.figure(figsize=(6, 3.5))
bax = brokenaxes(ylims=((24.8, 25.0), (42.9, 43.2)), hspace=.1, despine=False)
x = np.linspace(0, 1, 100)
bax.plot(x1, y1, label='MOOCCube_video', color="steelblue")
bax.plot(x2, y2, label='MOOCCube_course', color="darkorange")
bax.plot(x1, y1, 'o-', x2, y2, '+-')
bax.legend()
bax.set_xlabel('n_sample_num')
bax.set_ylabel('MRR@10(%)')
fig1.savefig("sample_num_MRR_10.pdf")
fig1.show()
