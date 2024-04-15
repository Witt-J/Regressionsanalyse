import matplotlib.pyplot as plt
import numpy as np

def lorentz(x, gamma, s, alpha, w_cr):
    y_predicted = w_cr*gamma / (1+(x/s)**2)**alpha
    return y_predicted

def lorentz_lin(x, gamma, s, alpha, alpha_0, w_cr):
    y_predicted = w_cr*gamma / (1+(x/s)**2)**(alpha+w_cr*alpha_0)
    return y_predicted

x = np.linspace(-0.12, 0.12, 1000)
w_cr = np.linspace(0,500,20)
gamma = 24.6
s = 0.0184
alpha = 1.138902
alpha_0 = 0.0015

s_opt = 0.01872447
alpha_opt = 1.25446139

##### 1
# s = 0.017819
# alpha = 1.1603
# wcr = 56.88582611111112

#### 11
# s = 0.018969
# alpha = 1.29396
# wcr = 412.21442166666674

#interpolation
#s = 0.0184
#alpha = 1.138902
#alpha_0 = 0.00037616

a = 0.867
b = 1.012*10**(-3)

strain_max = []
gradient = []
gradient_regress = []

for i in range(len(w_cr)):
    y = lorentz_lin(x, gamma, s, alpha, alpha_0, w_cr[i])
    y_opt = lorentz(x, gamma, s_opt, alpha_opt, w_cr[i])
    strain_max.append(max(y))
    g = np.gradient(y,x)
    f = 1/(1+(x/s)**2)**alpha
    gradient.append(max(np.gradient(y,x))/1000)     # /1000 to be in microstrain per [mm]

    gradient_regress.append(a * w_cr[i] + b * w_cr[i] ** 2)

    #plt.plot(x,f)
    plt.plot(x,y,'blue')
    plt.plot(x,y_opt,'red')

plt.show()


plt.scatter(w_cr,strain_max)
plt.grid()
plt.show()

plt.plot(w_cr,gradient_regress)
plt.scatter(w_cr,gradient)
plt.grid()
plt.show()

