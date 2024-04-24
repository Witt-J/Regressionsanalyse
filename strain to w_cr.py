import matplotlib.pyplot as plt
import numpy as np

def lorentz(x, gamma, s, alpha, w_cr):
    y_predicted = w_cr*gamma / (1+(x/s)**2)**alpha
    return y_predicted

def lorentz_lin(x, gamma, s, alpha_0, alpha_1, w_cr):
    y_predicted = w_cr*gamma / (1+(x/s)**2)**(alpha_0+w_cr*alpha_1)
    return y_predicted

x = np.linspace(-0.12, 0.12, 1000)
w_cr = np.linspace(0,500,20)
gamma = 24.6

s = 0.018738
alpha_0 = 1.138902
alpha_1 = 0.0015

s_opt = 1.87382534e-02
alpha_0_opt = 1.19658912
alpha_1_opt = 0.000178386654

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
gradient_opt = []

for i in range(len(w_cr)):
    y = lorentz_lin(x, gamma, s, alpha_0, alpha_1, w_cr[i])
    y_opt = lorentz_lin(x, gamma, s_opt, alpha_0_opt, alpha_1_opt, w_cr[i])
    strain_max.append(max(y))
    g = np.gradient(y,x)
    #f = 1/(1+(x/s)**2)**alpha
    gradient.append(max(np.gradient(y,x))/1000)     # /1000 to be in microstrain per [mm]
    g_opt = np.gradient(y_opt,x)
    gradient_opt.append(max(np.gradient(y_opt,x))/1000)
    gradient_regress.append(a * w_cr[i] + b * w_cr[i] ** 2)

    #plt.plot(x,f)
    plt.plot(x,y,'blue')
    plt.plot(x,y_opt,'red')

plt.show()


plt.scatter(w_cr,strain_max)
plt.grid()
plt.show()

plt.plot(w_cr,gradient_regress,'k')
plt.plot(w_cr,gradient,linestyle="", marker ="o",color='b')
plt.plot(w_cr,gradient_opt,linestyle="", marker ="o",color='r')
plt.grid()
plt.show()

