import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12, 6))
xticks = ['s', 'sfd', 'asdf', 'asdf', 'asdf', 'adf', 'adsf', 'df', 'sfa']

plt.subplot(1, 2, 1)
plt.xticks(range(len(xticks)), xticks, rotation=17)
x = np.linspace(0, 8, 9)
y1 = x
y2 = 2 * x
plt.axhspan = 1
plt.axvspan = 0.01
plt.plot(x, y1, 'r-o', label='CS-SVM')
plt.plot(x, y2, 'g-s', label='CCS-SVM')
plt.legend()
plt.title(u'CCS-SVM和CS-SVM的G-mean比较')
plt.xlabel(u'数据集')
plt.ylabel(u'G-mean')

plt.subplot(1, 2, 2)
y1 = x
y2 = 2 * x
plt.axhspan = 1
plt.axvspan = 0.01
plt.plot(x, y1, 'r-o', label='CS-SVM')
plt.plot(x, y2, 'g-s', label='CCS-SVM')
plt.legend()
plt.title(u'CCS-SVM和CS-SVM的F1值比较')
plt.xlabel(u'数据集')
plt.ylabel(u'F1')
plt.show()
