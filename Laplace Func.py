import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import shapiro
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Name der CSV-Datei
csv_file = 'cr5_ES_out_0.65'

# Load CSV file into a Pandas dataframe
df = pd.read_csv('cr5_ES_out_0.65.csv')

# Extract x and y values from dataframe
x_DFOS_half = df['x_axis']

# x spiegel und zusammenfügen
x_DFOS_mirr = x_DFOS_half[::-1]
x_DFOS = np.append(x_DFOS_half, x_DFOS_mirr * -1)

#eng Verteilte Daten erstellen um späten Plots auszugeben
x_model = np.linspace(min(x_DFOS), max(x_DFOS), 1000)

#Liste der Rissbreiten einfügen und zu einem Array umwandeln
w_cr_array = [13.4019275, 32.5236925, 58.6297725, 89.3305725, 121.855565, 152.95553, 185.6038275, 216.53853, 248.827085, 280.6140025, 312.583635, 330.2024375, 343.39955, 368.0849575]
w_cr_array = np.array(w_cr_array)

#mit Meshgrid 2D Arrays erzeugen, welche die gleiche Dimension (shape) haben -> damit diese mit dem y_daten überlager werden können
x_DFOS, w_cr_array = np.meshgrid(x_DFOS, w_cr_array)
y_DFOS = []

#einlesen der y_daten
for i in range(df.shape[1]-1):
    y_DFOS_half = df['strain_u'+str(i+1)]
    y_DFOS_mirr = y_DFOS_half[::-1]
    y_temp = np.append(y_DFOS_half, y_DFOS_mirr)
    y_DFOS.append(y_temp)

#umformen in einen Array mit Shape von x_data und w_cr_array
y_DFOS= np.array(y_DFOS)

# Definition der zu optimierenden Funktion
def objective(params, x, y, w_cr):
    sig, mu, a, b, c = params
    y_predicted = ((w_cr))*((1/(2*sig))*a*np.exp(-(np.abs(x-mu)*b)/sig)) + c    #virtuelle Daten mit freien Parametern
    error = np.sum((y_predicted - y)**2)                                        #least square analyse
    return error

# Definition der Funktion, um später virtuelle y_daten erzeugen zu können
def laplace_f(x, sig, mu, a, b, c, w_cr):
    return ((w_cr))*((1/(2*sig))*a*np.exp(-(np.abs(x-mu)*b)/sig)) + c

def laplace_f_individual(x, sig, mu, a, b, c):
    return ((1/(2*sig))*a*np.exp(-(np.abs(x-mu)*b)/sig)) + c
# Anfangswerte für die Optimierung
initial_params = [0.00455, 0, 3.1, 0.3, 1]

# Optimierung
result = minimize(objective, initial_params, args=(x_DFOS, y_DFOS, w_cr_array), method='L-BFGS-B')

# Ausgabe der optimierten Parameter
optimized_params = result.x
print("Optimierte Parameter:", optimized_params)
sig, mu, a, b, c = optimized_params

# Funktion für den gleitenden Durchschnitt
def moving_average1(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def moving_average2(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Fenstergröße für den gleitenden Durchschnitt
window_size = 20

#erstellen der y Werte mit der Laplace funktion und den optimierten parametern
for i in range(14):
    y_lp = laplace_f(x_model, sig, mu, a, b, c, w_cr_array[i,i])
    index_max, value_max = max(enumerate(y_DFOS[i]), key=lambda x: x[1])
    for k in range(len(y_lp)):
        if y_lp[k]>value_max:
            y_lp[k] = value_max
    ma_results1 = moving_average1(y_lp, window_size)
    ma_results2 = moving_average2(y_lp, window_size)

    popt, pcov = curve_fit(laplace_f_individual, x_DFOS[i], y_DFOS[i], p0=[0.00455, 0, 3.1, 0.3, 1])
    sig_compare, mu_compare, a_compare, b_compare, c_compare = popt
    y_individual = laplace_f_individual(x_model, sig_compare, mu_compare, a_compare, b_compare, c_compare)

    # PLOT
    plt.scatter(x_DFOS[i], y_DFOS[i])
    #plt.plot(x_model, y_lp, color='k')
    plt.plot(x_model, ma_results1, color='k')

    #reducing the Array x_model to have the same lenght as the ma_res2
    length_res = len(ma_results2)
    length_x_model = len(x_model)
    rounded_up = ((length_x_model-length_res)/2).__ceil__()
    rounded_down = ((length_x_model-length_res)/2).__floor__()

    #plt.plot(x_model[rounded_up:len(x_model)-rounded_down], ma_results2, color='k')
    plt.plot(x_model, y_individual, color='red')
    plt.show()


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
print("Laplace Func without post processing")
for i in range(14):
    r2 = r_squared(y_DFOS[i], laplace_f(x_DFOS[i], sig, mu, a, b, c, w_cr_array[i,i]))
    print(f"R-Squared: {r2}")

print("-----")
print("Laplace function with proper peak ")
for i in range(14):
    y_lp = laplace_f(x_DFOS[i], sig, mu, a, b, c, w_cr_array[i, i])
    index_max, value_max = max(enumerate(y_DFOS[i]), key=lambda x: x[1])
    for k in range(len(y_lp)):
        if y_lp[k] > value_max:
            y_lp[k] = value_max

    # Fenstergröße für den gleitenden Durchschnitt
    window_size = 4

    #moving average mit gleicher Länge des Arrays
    ma_results1 = moving_average1(y_lp, window_size)

    r2 = r_squared(y_DFOS[i], ma_results1)
    print(f"R-Squared: {r2}")

