import numpy as np
import scipy.integrate
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import shapiro
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Name der CSV-Datei
#csv_file = 'cr5_ES_out_0.65'

# Load CSV file into a Pandas dataframe
#df = pd.read_csv('Dehnungsverlauf/strain_peak_for_ES_out_at_x_0.67_m.csv')
df = pd.read_csv('Dehnungsverlauf/strain_peak_for_ES_out_at_x_0.67_m.csv')

# Extract x values from dataframe
x_DFOS = df.T.values[0]
x_DFOS = np.array(x_DFOS)

limit = 0.08
pos_cr = 0.6676

mask1 = x_DFOS > (pos_cr-limit)
mask2 = x_DFOS < (pos_cr+limit)

mask = mask1&mask2

x_DFOS = x_DFOS[mask]
x_DFOS = x_DFOS-pos_cr

delta_x = (abs(min(x_DFOS)-max(x_DFOS)))/2
x_start = -delta_x
x_end = delta_x

x_model = np.linspace(x_start, x_end, len(x_DFOS))

#Liste der Rissbreiten einfügen und zu einem Array umwandeln


#CRACK C12
w_cr_array = [26.151312777777783, 56.88582611111112, 92.34463333333338, 130.0315105555556, 168.98190833333337,
              209.0284805555556, 248.09519944444452, 285.26361138888893, 322.0303594444445, 357.5927802777779,
              391.4908075000001, 412.21442166666674]



w_cr_array = np.array(w_cr_array)

#mit Meshgrid 2D Arrays erzeugen, welche die gleiche Dimension (shape) haben -> damit diese mit dem y_daten überlager werden können
x_model, w_cr_array = np.meshgrid(x_model, w_cr_array)

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

w_cr_list = []

for i in range(12):
    integral = np.trapz(y_DFOS[i],x=x_model[i])
    w_cr_list.append(integral)


w_cr_list = np.array(w_cr_list)

#mit Meshgrid 2D Arrays erzeugen, welche die gleiche Dimension (shape) haben -> damit diese mit dem y_daten überlager werden können
x_Dfos2, w_cr_list = np.meshgrid(x_DFOS, w_cr_list)

# Definition der zu optimierenden Funktion
def objective(params, x, y, w_cr):
    gamma, sigma = params
    y_predicted = w_cr*gamma / (1+(x/sigma)**2)  #virtuelle Daten mit freien Parametern
    #for i in range(df.shape[1]-3):
        #y_predicted[i] = y_predicted[i]-y_predicted[i,0]
        #y_predicted[i] = (b * w_cr[i] + c) * y_predicted[i]
    error = np.sum((y_predicted - y)**2)                                        #least square analyse
    return error

# Definition der Funktion, um später virtuelle y_daten erzeugen zu können
def lorentz(x, gamma, sigma, w_cr):
    y_predicted = w_cr*gamma / (1+(x/sigma)**2)           # virtuelle Daten mit freien Parametern

    return y_predicted

initial_params = [23, 0.016]

# Optimierung
result = minimize(objective, initial_params, args=(x_model, y_DFOS, w_cr_array))

# Ausgabe der optimierten Parameter
optimized_params = result.x
print("Optimierte Parameter:", optimized_params)
gamma, sigma = optimized_params

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

#erstellen der y Werte mit der Laplace funktion und den optimierten parametern
y_lp = lorentz(x_model, gamma, sigma, w_cr_array)
for i in range(12):
    # PLOT
    #plt.scatter(x_model[i], y_DFOS[i])
    #plt.plot(x_model[i], y_lp[i], color='k')
    #plt.show()

    r2 = r_squared(y_DFOS[i], y_lp[i])
    print(f"R-Squared: {r2}")

list = []
w_cr = []

for i in range(12):
    result = minimize(objective, initial_params, args=(x_model[i], y_DFOS[i], w_cr_array[i]))

    # Ausgabe der optimierten Parameter
    optimized_params = result.x
    print("Optimierte Parameter:", optimized_params)
    #plt.scatter(x_model[i],y_DFOS[i])
    #plt.show()

    #plt.scatter(w_cr_array[i][i],result.x[0])
    list.append(result.x[1])
    w_cr.append(w_cr_array[i][i])
#plt.show()
#plt.plot(w_cr,list)
#plt.show()


gamma = 24.6
sigma = (1/gamma)*0.41
def lorentz_opt(x, gamma, sigma, w_cr, alpha, beta):
    y_predicted = w_cr*gamma / (1+(x/(sigma*(1+alpha*w_cr+beta*w_cr**2)))**2)           # virtuelle Daten mit freien Parametern

    return y_predicted


x_model, list = np.meshgrid(x_DFOS, list)
def objective_quad(params, w_cr, sigma, sigma_list):
    alpha, beta = params
    sigma_predicted = sigma * (1+alpha*w_cr+beta*w_cr**2) #virtuelle Daten mit freien Parametern
    error = np.sum((sigma_predicted - sigma_list)**2)                                        #least square analyse
    return error

initial_params = [-0.005, -0.005]
result = minimize(objective_quad, initial_params, args=(w_cr_list,sigma,list))

alpha, beta = result.x
print(alpha)
print(beta)

for i in range(12):
    y_lor = lorentz_opt(x_model[i],gamma, sigma, w_cr_list[i], alpha, beta)

    print(sigma*(1+alpha*w_cr_list[i][i]+beta*w_cr_list[i][i]**2))
    plt.scatter(x_model[i],y_DFOS[i])
    plt.plot(x_model[i],y_lor,'k')
    plt.show()
