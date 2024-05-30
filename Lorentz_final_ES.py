import numpy as np
import scipy.integrate
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import shapiro
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import sympy as sp


# Name der CSV-Datei
#csv_file = 'cr5_ES_out_0.65'

# Load CSV file into a Pandas dataframe
#df = pd.read_csv('Dehnungsverlauf/strain_peak_for_ES_out_at_x_0.67_m.csv')
df = pd.read_csv('Dehnungsverlauf/strain_peak_for_ES_out_at_x_0.67_m.csv')

# Extract x values from dataframe
x_DFOS_C12 = df.T.values[0]
x_DFOS_C12 = np.array(x_DFOS_C12)

limit = 0.08
pos_cr = 0.6676

mask1 = x_DFOS_C12 > (pos_cr-limit)
mask2 = x_DFOS_C12 < (pos_cr+limit)

mask = mask1&mask2

x_DFOS_C12 = x_DFOS_C12[mask]
x_DFOS_C12 = x_DFOS_C12-pos_cr

delta_x = (abs(min(x_DFOS_C12)-max(x_DFOS_C12)))/2
x_start = -delta_x
x_end = delta_x

x_model_C12 = np.linspace(x_start, x_end, len(x_DFOS_C12))
x_model_C12_org = np.linspace(x_start, x_end, len(x_DFOS_C12))

#Liste der Rissbreiten einfügen und zu einem Array umwandeln


#CRACK C12
w_cr_array_C12 = [26.151312777777783, 56.88582611111112, 92.34463333333338, 130.0315105555556, 168.98190833333337,
              209.0284805555556, 248.09519944444452, 285.26361138888893, 322.0303594444445, 357.5927802777779,
              391.4908075000001, 412.21442166666674]



#w_cr_array_C12 = np.array(w_cr_array_C12)


y_DFOS = []
y_max = []

#einlesen der y_daten

for i in range(df.shape[1]-3):
    y_DFOS_temp = np.array(df.T.values[i+1])

    y_DFOS_temp = np.array(y_DFOS_temp)

    y_DFOS.append(y_DFOS_temp[mask])
    y_max.append(max(y_DFOS_temp))
    y_DFOS_temp = np.array(y_DFOS_temp)


#umformen in einen Array mit Shape von x_data und w_cr_array
np.stack(y_DFOS)
y_DFOS= np.array(y_DFOS)
y_max = np.array(y_max)


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



gamma = 24.6
sigma = 0.015762


window = 3
weights = np.repeat(1.0, window)/window
x_model_C12 = x_model_C12[1:-1]
#x_model_C12 = x_model_C12[0]

List_linke_Wendepunkt = []
List_rechte_Wendepunkt = []
mittel_Wendepunkt_C12 = []
max_loc_list = []
max_list_C12 = []
max_grad_list_C12 = []

for i in range(12):
    max_loc_list.append(x_model_C12[np.argmax(y_DFOS[i])])
    max_list_C12.append((max(y_DFOS[i])))
    average = np.convolve(y_DFOS[i], weights, mode='valid')
    gradient = np.gradient(average,x_DFOS_C12[1]-x_DFOS_C12[0])/1000
    max_grad = max(gradient)
    max_grad_list_C12.append(max_grad)
    min_grad = min(gradient)
    max_index = np.argmax(gradient)
    min_index = np.argmin(gradient)
    List_linke_Wendepunkt.append(abs(x_model_C12[max_index]))
    List_rechte_Wendepunkt.append(x_model_C12[min_index])
    mittel_Wendepunkt_C12.append((abs(x_model_C12[max_index]) + x_model_C12[min_index]) / 2)
    #plt.plot(x_model_C12, gradient)
    #plt.show()

#plt.scatter(w_l,max_list_C12)
#plt.show()

#plt.scatter(w_l,List_linke_Wendepunkt)
#plt.scatter(w_l,List_rechte_Wendepunkt)
#plt.plot(w_l, mittel_Wendepunkt_C12)
#plt.show()


def lorentz(x, gamma, sigma0, sigma1, w_cr):
    y_predicted = w_cr*gamma / (1+(x/(sigma0+sigma1*w_cr))**2)           # virtuelle Daten mit freien Parametern

    return y_predicted

def lorentz_single(x, gamma, sigma, w_cr):
    y_predicted = w_cr*gamma / (1+(x/sigma)**2)           # virtuelle Daten mit freien Parametern

    return y_predicted

###################################################
#                C3
###################################################
w_cr_array_C3 = [19.669451388888874, 48.233372777777745, 80.78568333333328, 115.46603972222213, 150.30286916666654,
                 184.53831499999987, 218.73831055555536, 253.81509111111086, 287.91341222222195, 322.43429111111084,
                 357.71768590277736, 402.8620627777773]
#w_cr_array_C3 = np.array(w_cr_array_C3)

y_DFOS_C3 = []
for i in range(12):
    df = pd.read_csv(('Dehnungsverlauf/strainprofile_' + str((i+1)) + '_ES_out_0.65.csv'), sep='\t')
    y_DFOS_C3.append(df.T.values[1])

# Extract x values from dataframe
x_DFOS_C3 = df.T.values[0]
x_DFOS_C3 = np.array(x_DFOS_C3)

#masking
mask1 = x_DFOS_C3 > (-0.514-0.08)
mask2 = x_DFOS_C3 < (-0.514+0.08)

mask = mask1&mask2

x_DFOS_C3 = x_DFOS_C3[mask]


for i in range(12):
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

for i in range(12):
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
w_cr_array_C9 = [12.746289345345351, 32.35402805555557, 58.232954722222225, 89.09878972222225, 121.80794166666666,
                 153.00820527777776, 184.5041462177177, 215.76272015765764, 247.2092651576577, 276.27144999999996,
                 305.6654791666669, 323.5282844444444]
#w_cr_array_C9 = np.array(w_cr_array_C9)
y_DFOS_C9 = []
for i in range(12):
    df = pd.read_csv(('Dehnungsverlauf/strainprofile_' + str((i+1)) + '_ES_out_0.65.csv'), sep='\t')
    y_DFOS_C9.append(df.T.values[1])

# Extract x values from dataframe
x_DFOS_C9 = df.T.values[0]
x_DFOS_C9 = np.array(x_DFOS_C9)

#masking
mask1 = x_DFOS_C9 > (0.25-0.08)
mask2 = x_DFOS_C9 < (0.25+0.08)

mask = mask1&mask2

x_DFOS_C9 = x_DFOS_C9[mask]

for i in range(12):
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

for i in range(12):
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
axs[0,2].scatter(x_DFOS_C12,y_DFOS[1])

axs[1,0].plot(x_model_C3, lorentz(x_model_C3,g,sigma0,sigma1,w_cr_array_C3[7]),'k')
axs[1,0].plot(x_model_C3, lorentz_single(x_model_C3,g,sigma,w_cr_array_C3[7]),'r')
axs[1,0].scatter(x_DFOS_C3,y_DFOS_C3[7])

axs[1,1].plot(x_model_C9, lorentz(x_model_C9,g,sigma0,sigma1,w_cr_array_C9[7]),'k')
axs[1,1].plot(x_model_C9, lorentz_single(x_model_C9,g,sigma,w_cr_array_C9[7]),'r')
axs[1,1].scatter(x_DFOS_C9,y_DFOS_C9[7])

axs[1,2].plot(x_model_C12, lorentz(x_model_C12,g,sigma0,sigma1,w_cr_array_C12[7]),'k')
axs[1,2].plot(x_model_C12, lorentz_single(x_model_C12,g,sigma,w_cr_array_C12[7]),'r')
axs[1,2].scatter(x_DFOS_C12,y_DFOS[7])

axs[2,0].plot(x_model_C3, lorentz(x_model_C3,g,sigma0,sigma1,w_cr_array_C12[11]),'k')
axs[2,0].plot(x_model_C3, lorentz_single(x_model_C3,g,sigma,w_cr_array_C3[11]),'r')
axs[2,0].scatter(x_DFOS_C3,y_DFOS_C3[11])

axs[2,1].plot(x_model_C9, lorentz(x_model_C9,g,sigma0,sigma1,w_cr_array_C9[11]),'k')
axs[2,1].plot(x_model_C9, lorentz_single(x_model_C9,g,sigma,w_cr_array_C9[11]),'r')
axs[2,1].scatter(x_DFOS_C9,y_DFOS_C9[11])

axs[2,2].plot(x_model_C12, lorentz(x_model_C12,g,sigma0,sigma1,w_cr_array_C12[11]),'k')
axs[2,2].plot(x_model_C12, lorentz_single(x_model_C12,g,sigma,w_cr_array_C12[11]),'r')
axs[2,2].scatter(x_DFOS_C12,y_DFOS[11])

plt.show()



"""
w_regresss = np.linspace(0,500,21)

gradient_opt = []
for i in range(len(w_regresss)):
    gradient_opt.append(max(np.gradient(lorentz(x_model_C3,g,sigma0,sigma1,w_regresss[i]),x_model_C3[1]-x_model_C3[0])/1000))

def regress_max(w_cr, alpha, beta):
    return alpha*w_cr+beta*w_cr**2

alpha = 0.867
beta = 1.012*10**(-3)

max_grad_list = max_grad_list_C3+max_grad_list_C9+max_grad_list_C12

plt.plot(w_regresss,regress_max(w_regresss,alpha,beta),label='Regression')
plt.scatter(w_list_all,max_grad_list,label='actual Cracks')
plt.scatter(w_regresss,gradient_opt,label='Lorentz')
plt.grid()
plt.legend()
plt.show()


l1 = lorentz(0.00676,24,sigma0,sigma1,400)
l2 = lorentz(0.00673,24,sigma0,sigma1,400)
x_model = np.linspace(-0.08, 0.08, 500)
l_400 = lorentz(x_model,24,sigma0,sigma1,400)
index = np.argmax(np.gradient(l_400))
x_loc = x_model_C12[index]
print(x_loc)
delta_x = 0.00676-0.00673


print((l2-l1)/(delta_x*1000))
gradient_400 = np.gradient(lorentz(x_model,24,sigma0,sigma1,400),edge_order=2)
print('last Line')



l_400_xC12 = lorentz(x_model_C12,24,sigma0,sigma1,400)


w_cr = 400
sigma = 0.01321

# Schritt 1: Definiere die Funktion
x = sp.symbols('x')
f = ((w_cr) * gamma)/(1+(x/sigma)**2)

# Schritt 2: Berechne die Steigung (Ableitung)
f_prime = sp.diff(f, x)

# Schritt 3: Plotte die Funktion und ihre Steigung
# Konvertiere die symbolische Funktion und Ableitung in NumPy-Funktionen
f_func = sp.lambdify(x, f, 'numpy')
f_prime_func = sp.lambdify(x, f_prime, 'numpy')

# Definiere den Definitionsbereich
x_values = np.linspace(-0.08, 0.08, 500)

# Berechne die Funktionswerte und die Ableitungswerte
y_values = f_func(x_values)
slope_values = f_prime_func(x_values)/1000

# Plotte die Funktion und ihre Steigung
plt.plot(x_values, slope_values, label='SciPy.dify')
#plt.plot(x_model,gradient_400, label='x=500pts')
#plt.plot(x_model_C12,np.gradient(l_400_xC12), label='x=250pts')
plt.plot(x_model_C12,np.gradient(l_400_xC12,x_model_C12[1]-x_model_C12[0])/1000, label='NP.Gradient')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient und Steigung')
plt.legend()
plt.grid(True)
plt.show()
#
plt.plot(x_values, y_values, label='Funktion:f(x)')
plt.plot(x_model,l_400, label='sig0 und sig1')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Lorentzfunktion')
plt.legend()
plt.grid(True)
plt.show()
"""










POS = -0.51
w_cr_array_C3 = [19.669451388888874, 48.233372777777745, 80.78568333333328, 115.46603972222213, 150.30286916666654,
                 184.53831499999987, 218.73831055555536, 253.81509111111086, 287.91341222222195, 322.43429111111084,
                 357.71768590277736, 402.8620627777773]

POS = 0.25
w_cr_array_C9 = [12.746289345345351, 32.35402805555557, 58.232954722222225, 89.09878972222225, 121.80794166666666,
                 153.00820527777776, 184.5041462177177, 215.76272015765764, 247.2092651576577, 276.27144999999996,
                 305.6654791666669, 323.5282844444444]

POS = 0.67
w_cr_array_C12 = [26.151312777777783, 56.88582611111112, 92.34463333333338, 130.0315105555556, 168.98190833333337,
              209.0284805555556, 248.09519944444452, 285.26361138888893, 322.0303594444445, 357.5927802777779,
              391.4908075000001, 412.21442166666674]

