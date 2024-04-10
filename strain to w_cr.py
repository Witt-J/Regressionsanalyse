import matplotlib.pyplot as plt
import numpy as np

def lorentz(x, gamma, s, alpha, w_cr):
    y_predicted = w_cr*gamma / (1+(x/s)**2)**alpha
    return y_predicted


x = np.linspace(-0.12, 0.12, 1000)
w_cr = np.linspace(0,500,20)
gamma = 24.6
s = 0.01872447
alpha = 1.25446139


strain_max = []
gradient = []

for i in range(len(w_cr)):
    y = lorentz(x, gamma, s, alpha, w_cr[i])
    strain_max.append(max(y))
    gradient.append(max(np.gradient(y,x))/1000)     # /1000 to be in microstrain per [mm]

plt.scatter(w_cr,strain_max)
plt.grid
plt.show()

plt.scatter(w_cr,gradient)
plt.grid
plt.show()

