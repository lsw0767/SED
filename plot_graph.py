from scipy import stats
import numpy as np
import math
import matplotlib.pyplot as plt

x_rage=np.arange(-13, 7, 0.01)
pdf = stats.norm.pdf(x_rage, 0, 2)
pdf = np.multiply(pdf, 1./max(pdf))
xlim = np.arange(0, 10, 0.005)

plt.figure(figsize=(3.5, 5))
plt.subplot(311)
plt.plot(xlim, pdf)

plt.subplot(312)
box = [1 if v>0.5 else 0 for v in pdf]
plt.plot(xlim, box, 'r')

plt.subplot(313)
plt.plot(xlim, box, 'r')
plt.plot(xlim, pdf)
plt.show()