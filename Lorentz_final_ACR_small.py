import numpy as np
import scipy.integrate
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import shapiro
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import sympy as sp


def r_squared(y_true, y_pred):
    # Berechnung des Residuals
    residuals = y_true - y_pred
    # Berechnung der Summe der quadrierten Residuen
    ss_res = np.sum(residuals ** 2)
    # Berechnung der Summe der quadrierten Abweichungen vom Mittelwert
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    # Berechnung des Bestimmtheitsgrads
    r2 = 1 - (ss_res / ss_tot)
    return r2

def lorentz(x, gamma, sigma0, sigma1, w_cr):
    y_predicted = w_cr*gamma / (1+(x/(sigma0+sigma1*w_cr))**2)           # virtuelle Daten mit freien Parametern

    return y_predicted

def lorentz_single(x, gamma, sigma, w_cr):
    y_predicted = w_cr*gamma / (1+(x/sigma)**2)           # virtuelle Daten mit freien Parametern

    return y_predicted

window = 3
weights = np.repeat(1.0, window)/window

###################################################
#                C3
###################################################
#w_cr_array_C3 = [20.32505944444444, 49.40898444444446, 82.72851166666668, 117.79031694444447, 152.61090000000004,
#                 186.28492277777787, 219.7565791468255, 254.01888764880965, 288.74052721428575, 323.19018488095253,
#                 358.5200473214287, 404.8731266666667]
w_cr_array_C3 = [20.32505944444444, 49.40898444444446, 82.72851166666668, 117.79031694444447, 152.61090000000004,
                 186.28492277777787]
#w_cr_array_C3 = np.array(w_cr_array_C3)

y_DFOS_C3 = []
for i in range(6):
    df = pd.read_csv(('Dehnungsverlauf/strainprofile_' + str((i+1)) + '_ACR1_0.65.csv'), sep='\t')
    y_DFOS_import = df.T.values[1]
    y_DFOS_import = [float(x) for x in y_DFOS_import]
    y_DFOS_import = np.array(y_DFOS_import)
    y_DFOS_C3.append(y_DFOS_import)


# Extract x values from dataframe
x_DFOS_C3 = df.T.values[0]
x_DFOS_C3 = np.array(x_DFOS_C3)

#masking
mask1 = x_DFOS_C3 > (-0.5213-0.08)
mask2 = x_DFOS_C3 < (-0.5213+0.08)

mask = mask1&mask2

x_DFOS_C3 = x_DFOS_C3[mask]


for i in range(6):
    y_DFOS_C3[i] = y_DFOS_C3[i][mask]

index_middle = np.argmax(y_DFOS_C3[4])
x_middle = x_DFOS_C3[index_middle]

x_DFOS_C3 = x_DFOS_C3-x_middle

delta_x = (abs(min(x_DFOS_C3)-max(x_DFOS_C3)))/2
x_start = -delta_x
x_end = delta_x

x_model_C3 = np.linspace(x_start, x_end, len(x_DFOS_C3))

List_linke_Wendepunkt_C3 = []
List_rechte_Wendepunkt_C3 = []
mittel_Wendepunkt_C3 = []
max_loc_list_C3 = []
max_list_C3 = []
max_grad_list_C3 = []

for i in range(6):
    max_loc_list_C3.append(x_model_C3[np.argmax(y_DFOS_C3[i])])
    max_list_C3.append((max(y_DFOS_C3[i])))
    average = np.convolve(y_DFOS_C3[i], weights, mode='valid')
    gradient = np.gradient(average,x_DFOS_C3[1]-x_DFOS_C3[0])/1000
    max_grad = max(gradient)
    max_grad_list_C3.append(max_grad)
    min_grad = min(gradient)
    max_index = np.argmax(gradient)
    min_index = np.argmin(gradient)
    List_linke_Wendepunkt_C3.append(abs(x_model_C3[max_index]))
    List_rechte_Wendepunkt_C3.append(x_model_C3[min_index])
    mittel_Wendepunkt_C3.append((abs(x_model_C3[max_index]) + x_model_C3[min_index]) / 2)
    #plt.plot(x_model_C3, gradient)
    #plt.show()

###################################################
#                C9
###################################################
#w_cr_array_C9 = [13.508229046546552, 33.35508384234234, 60.57087833333333, 91.36222694444447, 124.0084805555556,
#                 155.8088478194445, 186.80642809523803, 218.67845187499984, 249.28426033333318, 281.60293246428574,
#                 312.9477397738092, 311.6893757301583]
w_cr_array_C9 = [13.508229046546552, 33.35508384234234, 60.57087833333333, 91.36222694444447, 124.0084805555556,
                 155.8088478194445]
#w_cr_array_C9 = np.array(w_cr_array_C9)
y_DFOS_C9 = []
for i in range(6):
    df = pd.read_csv(('Dehnungsverlauf/strainprofile_' + str((i+1)) + '_ACR1_0.65.csv'), sep='\t')
    y_DFOS_import = df.T.values[1]
    y_DFOS_import = [float(x) for x in y_DFOS_import]
    y_DFOS_import = np.array(y_DFOS_import)
    y_DFOS_C9.append(y_DFOS_import)

# Extract x values from dataframe
x_DFOS_C9 = df.T.values[0]
x_DFOS_C9 = np.array(x_DFOS_C9)

#masking
mask1 = x_DFOS_C9 > (0.2515-0.05)
mask2 = x_DFOS_C9 < (0.2515+0.05)

mask = mask1&mask2

x_DFOS_C9 = x_DFOS_C9[mask]

for i in range(6):
    y_DFOS_C9[i] = y_DFOS_C9[i][mask]

index_middle = np.argmax(y_DFOS_C9[4])
x_middle = x_DFOS_C9[index_middle]

x_DFOS_C9 = x_DFOS_C9-x_middle

delta_x = (abs(min(x_DFOS_C9)-max(x_DFOS_C9)))/2
x_start = -delta_x
x_end = delta_x

x_model_C9 = np.linspace(x_start, x_end, len(x_DFOS_C9))

List_linke_Wendepunkt_C9 = []
List_rechte_Wendepunkt_C9 = []
mittel_Wendepunkt_C9 = []
max_loc_list_C9 = []
max_list_C9 = []
max_grad_list_C9 = []

for i in range(6):
    max_loc_list_C9.append(x_model_C9[np.argmax(y_DFOS_C9[i])])
    max_list_C9.append((max(y_DFOS_C9[i])))
    average = np.convolve(y_DFOS_C9[i], weights, mode='valid')
    gradient = np.gradient(average,x_DFOS_C9[1]-x_DFOS_C9[0])/1000
    max_grad = max(gradient)
    max_grad_list_C9.append(max_grad)
    min_grad = min(gradient)
    max_index = np.argmax(gradient)
    min_index = np.argmin(gradient)
    List_linke_Wendepunkt_C9.append(abs(x_model_C9[max_index]))
    List_rechte_Wendepunkt_C9.append(x_model_C9[min_index])
    mittel_Wendepunkt_C9.append((abs(x_model_C9[max_index]) + x_model_C9[min_index]) / 2)

    #plt.scatter(x_DFOS_C9,y_DFOS_C9[i])
    #plt.show()

###################################################
#                C12
###################################################
#w_cr_array_C12 = [26.63485680555558, 57.61857472222225, 93.37536, 131.65188833333337, 170.9102759722223,
#                  211.1452317361112, 250.20251930555568, 286.91174055555564, 323.16453363690516, 358.331415128969,
#                  391.60467962004043, 412.04237326984196]
w_cr_array_C12 = [26.63485680555558, 57.61857472222225, 93.37536, 131.65188833333337, 170.9102759722223,
                  211.1452317361112]
#w_cr_array_C9 = np.array(w_cr_array_C9)
y_DFOS_C12 = []
for i in range(6):
    df = pd.read_csv(('Dehnungsverlauf/strainprofile_' + str((i+1)) + '_ACR1_0.65.csv'), sep='\t')
    y_DFOS_import = df.T.values[1]
    y_DFOS_import = [float(x) for x in y_DFOS_import]
    y_DFOS_import = np.array(y_DFOS_import)
    y_DFOS_C12.append(y_DFOS_import)

# Extract x values from dataframe
x_DFOS_C12 = df.T.values[0]
x_DFOS_C12 = np.array(x_DFOS_C12)

#masking
mask1 = x_DFOS_C12 > (0.67275-0.08)
mask2 = x_DFOS_C12 < (0.67275+0.08)

mask = mask1&mask2

x_DFOS_C12 = x_DFOS_C12[mask]

for i in range(6):
    y_DFOS_C12[i] = y_DFOS_C12[i][mask]

index_middle = np.argmax(y_DFOS_C12[4])
x_middle = x_DFOS_C12[index_middle]

x_DFOS_C12 = x_DFOS_C12-x_middle

delta_x = (abs(min(x_DFOS_C12)-max(x_DFOS_C12)))/2
x_start = -delta_x
x_end = delta_x

x_model_C12 = np.linspace(x_start, x_end, len(x_DFOS_C12))

List_linke_Wendepunkt_C12 = []
List_rechte_Wendepunkt_C12 = []
mittel_Wendepunkt_C12 = []
max_loc_list_C12 = []
max_list_C12 = []
max_grad_list_C12 = []

for i in range(6):
    max_loc_list_C12.append(x_model_C12[np.argmax(y_DFOS_C12[i])])
    max_list_C12.append((max(y_DFOS_C12[i])))
    average = np.convolve(y_DFOS_C12[i], weights, mode='valid')
    gradient = np.gradient(average,x_DFOS_C12[1]-x_DFOS_C12[0])/1000
    max_grad = max(gradient)
    max_grad_list_C12.append(max_grad)
    min_grad = min(gradient)
    max_index = np.argmax(gradient)
    min_index = np.argmin(gradient)
    List_linke_Wendepunkt_C12.append(abs(x_model_C12[max_index]))
    List_rechte_Wendepunkt_C12.append(x_model_C12[min_index])
    mittel_Wendepunkt_C12.append((abs(x_model_C12[max_index]) + x_model_C12[min_index]) / 2)






def objective(params, w_cr, strain_max):
    gamma = params
    strain_predicted = w_cr*gamma #virtuelle Daten mit freien Parametern
    error = np.sum((strain_predicted - strain_max)**2)                                        #least square analyse
    return error

initial_params = [24.6]

# Optimierung
result = minimize(objective, initial_params, args=(w_cr_array_C3, max_list_C3))
print(result.x[0])
gamma_C3 = result.x[0]
result = minimize(objective, initial_params, args=(w_cr_array_C9, max_list_C9))
print(result.x[0])
gamma_C9 = result.x[0]
result = minimize(objective, initial_params, args=(w_cr_array_C12, max_list_C12))
print(result.x[0])
gamma_C12 = result.x[0]

def Regression(w_cr, gamma):
    strain_max = w_cr*gamma
    return strain_max

C3 = Regression(w_cr_array_C3,np.repeat(gamma_C3,len(w_cr_array_C3)))
C9 = Regression(w_cr_array_C9,np.repeat(gamma_C9,len(w_cr_array_C9)))
C12 = Regression(w_cr_array_C12,np.repeat(gamma_C12,len(w_cr_array_C12)))

plt.scatter(w_cr_array_C3,max_list_C3,label='Strain max C3')
plt.plot(w_cr_array_C3,C3,label='Reg C3')
plt.scatter(w_cr_array_C9,max_list_C9,label='Strain max C9')
plt.plot(w_cr_array_C9,C9,label='Reg C9')
plt.scatter(w_cr_array_C12,max_list_C12,label='Strain max C12')
plt.plot(w_cr_array_C12,C12,label='Reg C12')
plt.legend()
plt.show()

def objective_WP(params, w_cr, sigma_real):
    sigma0, sigma1 = params
    sigma_predicted = sigma0 + np.array(w_cr) * sigma1 #virtuelle Daten mit freien Parametern
    error = np.sum((np.array(sigma_real)*np.sqrt(3) - sigma_predicted)**2)                                        #least square analyse
    return error

def objective_WP_single(params, w_cr, sigma_real):
    sigma = params
    sigma_predicted = sigma  #virtuelle Daten mit freien Parametern
    error = np.sum((np.array(sigma_real)*np.sqrt(3) - sigma_predicted)**2)                                        #least square analyse
    return error

initial_params = [0.009, -0.0001]
initial_params_single = [1.6]


def Regression_sigma(w_cr, sigma0, sigma1):
    sig = sigma0 + np.array(w_cr) * sigma1
    return sig

result = minimize(objective_WP, initial_params, args=(w_cr_array_C3, mittel_Wendepunkt_C3))
sigma0_C3, sigma1_C3 = result.x
result = minimize(objective_WP_single, initial_params_single, args=(w_cr_array_C3, mittel_Wendepunkt_C3))
sigma_C3 = result.x
result = minimize(objective_WP, initial_params, args=(w_cr_array_C9, mittel_Wendepunkt_C9))
sigma0_C9, sigma1_C9 = result.x
result = minimize(objective_WP_single, initial_params_single, args=(w_cr_array_C9, mittel_Wendepunkt_C9))
sigma_C9 = result.x
result = minimize(objective_WP, initial_params, args=(w_cr_array_C12, mittel_Wendepunkt_C12))
sigma0_C12, sigma1_C12 = result.x
result = minimize(objective_WP_single, initial_params_single, args=(w_cr_array_C12, mittel_Wendepunkt_C12))
sigma_C12 = result.x



plt.scatter(w_cr_array_C3,mittel_Wendepunkt_C3,label='X WP C3')
plt.scatter(w_cr_array_C9,mittel_Wendepunkt_C9,label='X WP C9')
plt.scatter(w_cr_array_C12,mittel_Wendepunkt_C12,label='X WP C12')
plt.legend()
plt.show()


plt.scatter(w_cr_array_C3,np.array(mittel_Wendepunkt_C3)*np.sqrt(3),label='Sigma C3')
plt.plot(w_cr_array_C3, Regression_sigma(w_cr_array_C3,sigma0_C3,sigma1_C3))

plt.scatter(w_cr_array_C9,np.array(mittel_Wendepunkt_C9)*np.sqrt(3),label='Sigma C9')
plt.plot(w_cr_array_C9, Regression_sigma(w_cr_array_C9,sigma0_C9,sigma1_C9))

plt.scatter(w_cr_array_C12,np.array(mittel_Wendepunkt_C12)*np.sqrt(3),label='Sigma C12')
plt.plot(w_cr_array_C12, Regression_sigma(w_cr_array_C12,sigma0_C12,sigma1_C12))

plt.legend()
plt.show()

strain_max_list_all = max_list_C3 + max_list_C9 + max_list_C12
w_list_all = w_cr_array_C3 + w_cr_array_C9 + w_cr_array_C12
WP_all = mittel_Wendepunkt_C3+mittel_Wendepunkt_C9+mittel_Wendepunkt_C12

initial_params = [24.6]
result = result = minimize(objective, initial_params, args=(w_list_all, strain_max_list_all))

print('Gamma=',result.x[0])
g = result.x[0]

initial_params = [0.009, -0.0001]
result = minimize(objective_WP, initial_params, args=(w_list_all, WP_all))
sigma0, sigma1 = result.x
print('sigma0=',sigma0)
print('sigma1=',sigma1)


print('mit nur einem sigma')
initial_params = [0.009]
result = minimize(objective_WP_single,initial_params, args=(w_list_all, WP_all))
sigma = result.x
print('sigma=',sigma)


fig, axs = plt.subplots(3,3)

axs[0,0].plot(x_model_C3, lorentz(x_model_C3,g,sigma0,sigma1,w_cr_array_C3[1]),'k')
axs[0,0].plot(x_model_C3, lorentz_single(x_model_C3,g,sigma,w_cr_array_C3[1]),'r')
axs[0,0].scatter(x_DFOS_C3,y_DFOS_C3[1])

axs[0,1].plot(x_model_C9, lorentz(x_model_C9,g,sigma0,sigma1,w_cr_array_C9[1]),'k')
axs[0,1].plot(x_model_C9, lorentz_single(x_model_C9,g,sigma,w_cr_array_C9[1]),'r')
axs[0,1].scatter(x_DFOS_C9,y_DFOS_C9[1])

axs[0,2].plot(x_model_C12, lorentz(x_model_C12,g,sigma0,sigma1,w_cr_array_C12[1]),'k')
axs[0,2].plot(x_model_C12, lorentz_single(x_model_C12,g,sigma,w_cr_array_C12[1]),'r')
axs[0,2].scatter(x_DFOS_C3,y_DFOS_C12[1])

axs[1,0].plot(x_model_C3, lorentz(x_model_C3,g,sigma0,sigma1,w_cr_array_C3[3]),'k')
axs[1,0].plot(x_model_C3, lorentz_single(x_model_C3,g,sigma,w_cr_array_C3[3]),'r')
axs[1,0].scatter(x_DFOS_C3,y_DFOS_C3[3])

axs[1,1].plot(x_model_C9, lorentz(x_model_C9,g,sigma0,sigma1,w_cr_array_C9[3]),'k')
axs[1,1].plot(x_model_C9, lorentz_single(x_model_C9,g,sigma,w_cr_array_C9[3]),'r')
axs[1,1].scatter(x_DFOS_C9,y_DFOS_C9[3])

axs[1,2].plot(x_model_C12, lorentz(x_model_C12,g,sigma0,sigma1,w_cr_array_C12[3]),'k')
axs[1,2].plot(x_model_C12, lorentz_single(x_model_C12,g,sigma,w_cr_array_C12[3]),'r')
axs[1,2].scatter(x_DFOS_C12,y_DFOS_C12[3])

axs[2,0].plot(x_model_C3, lorentz(x_model_C3,g,sigma0,sigma1,w_cr_array_C12[5]),'k')
axs[2,0].plot(x_model_C3, lorentz_single(x_model_C3,g,sigma,w_cr_array_C3[5]),'r')
axs[2,0].scatter(x_DFOS_C3,y_DFOS_C3[5])

axs[2,1].plot(x_model_C9, lorentz(x_model_C9,g,sigma0,sigma1,w_cr_array_C9[5]),'k')
axs[2,1].plot(x_model_C9, lorentz_single(x_model_C9,g,sigma,w_cr_array_C9[5]),'r')
axs[2,1].scatter(x_DFOS_C9,y_DFOS_C9[5])

axs[2,2].plot(x_model_C12, lorentz(x_model_C12,g,sigma0,sigma1,w_cr_array_C12[5]),'k')
axs[2,2].plot(x_model_C12, lorentz_single(x_model_C12,g,sigma,w_cr_array_C12[5]),'r')
axs[2,2].scatter(x_DFOS_C12,y_DFOS_C12[5])

plt.show()


POS = -0.5213
w_cr_array_C3 = [20.32505944444444, 49.40898444444446, 82.72851166666668, 117.79031694444447, 152.61090000000004,
                 186.28492277777787, 219.7565791468255, 254.01888764880965, 288.74052721428575, 323.19018488095253,
                 358.5200473214287, 404.8731266666667]

POS = 0.2515
w_cr_array_C9 = [13.508229046546552, 33.35508384234234, 60.57087833333333, 91.36222694444447, 124.0084805555556,
                 155.8088478194445, 186.80642809523803, 218.67845187499984, 249.28426033333318, 281.60293246428574,
                 312.9477397738092, 311.6893757301583]

POS = 0.67275
w_cr_array_C12 = [26.63485680555558, 57.61857472222225, 93.37536, 131.65188833333337, 170.9102759722223,
                  211.1452317361112, 250.20251930555568, 286.91174055555564, 323.16453363690516, 358.331415128969,
                  391.60467962004043, 412.04237326984196]

