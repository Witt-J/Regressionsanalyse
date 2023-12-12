import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import shapiro
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Name der CSV-Datei
csv_file = 'strainprofile_-_ES_out_0.65'

# Load CSV file into a Pandas dataframe
df = pd.read_csv('Dehnungsverlauf/strainprofile_1_ES_out_0.65.csv', sep='\t')

# Extract x values from dataframe
x_DFOS = df.T.values[0]

#RISS 1
#index_start = 1880
#index_end = 2171

#RISS 2
#index_start = 3972
#index_end = 4264

#RISS 3
index_start = 4285
index_end = 4510

x_DFOS = np.array(x_DFOS)
x_DFOS = x_DFOS[index_start:index_end]

delta_x = (abs(min(x_DFOS)+max(x_DFOS)))/2
x_start = -delta_x
x_end = delta_x

x_model = np.linspace(x_start, x_end, (index_end-index_start))

#Liste der Rissbreiten einfügen und zu einem Array umwandeln

#Riss 1 bei -0.72m nur 1 bis 11!!!
#w_cr_array = [25.817609999999995, 62.73302027777779, 103.47890944444444, 145.7053686111111, 186.29669500000003, 226.75738944444447, 269.18947972222225, 312.55836444444446, 357.41525805555557, 403.28825333333333, 460.0975383333333]
#Riss 2 bei 0.62m
#w_cr_array = [26.151312777777783, 56.88582611111112, 92.34463333333338, 130.0315105555556, 168.98190833333337, 209.0284805555556, 248.09519944444452, 285.26361138888893, 322.0303594444445, 357.5927802777779, 391.4908075000001, 412.21442166666674, 429.1949597361114, 460.94397132539683]
#Riss 3 bei 0.86m
#w_cr_array = [15.175723333333327, 35.66578888888887, 63.07872638888885, 94.73801638888885, 128.80793611111108, 164.02423194444435, 198.59538472222215, 232.53970638888882, 266.78888888888883, 300.4199358333332, 332.0820208333332, 351.71996527777765, 368.59756555555543, 388.8326902777777]
w_cr_array = [15.175723333333327, 35.66578888888887, 63.07872638888885, 94.73801638888885, 128.80793611111108, 164.02423194444435, 198.59538472222215, 232.53970638888882, 266.78888888888883, 300.4199358333332, 332.0820208333332]

w_cr_array = np.array(w_cr_array)

#mit Meshgrid 2D Arrays erzeugen, welche die gleiche Dimension (shape) haben -> damit diese mit dem y_daten überlager werden können
x_model, w_cr_array = np.meshgrid(x_model, w_cr_array)
y_DFOS = []

#einlesen der y_daten
for i in range(11):
    df = pd.read_csv('Dehnungsverlauf/strainprofile_'+str(i+1)+'_ES_out_0.65.csv', sep='\t')

    # Extract x values from dataframe
    y_DFOS_temp = np.array(df.T.values[1])
    y_DFOS_temp = y_DFOS_temp[index_start:index_end]
    y_DFOS.append(y_DFOS_temp)

#umformen in einen Array mit Shape von x_data und w_cr_array
np.stack(y_DFOS)
y_DFOS= np.array(y_DFOS)

# Definition der zu optimierenden Funktion
def objective(params, x, y, w_cr):
    sig, a, b, c = params
    y_predicted = ((w_cr))*((1/(2*sig))*a*np.exp(-(np.abs(x)*b)/sig)) + c    #virtuelle Daten mit freien Parametern
    error = np.sum((y_predicted - y)**2)                                        #least square analyse
    return error

# Definition der Funktion, um später virtuelle y_daten erzeugen zu können
def laplace_f(x, sig, a, b, c, w_cr):
    return ((w_cr))*((1/(2*sig))*a*np.exp(-(np.abs(x)*b)/sig)) + c

def laplace_f_individual(x, sig, a, b, c):
    return ((1/(2*sig))*a*np.exp(-(np.abs(x)*b)/sig)) + c
# Anfangswerte für die Optimierung
initial_params = [0.00455, 3.1, 0.3, 1]

# Optimierung
result = minimize(objective, initial_params, args=(x_model, y_DFOS, w_cr_array), method='L-BFGS-B')

# Ausgabe der optimierten Parameter
optimized_params = result.x
print("Optimierte Parameter:", optimized_params)
sig, a, b, c = optimized_params

# Funktion für den gleitenden Durchschnitt
def moving_average1(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def moving_average2(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Fenstergröße für den gleitenden Durchschnitt
window_size = 10

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
for i in range(11):
    y_lp = laplace_f(x_model[i], sig, a, b, c, w_cr_array[i,i])
    index_max, value_max = max(enumerate(y_DFOS[i]), key=lambda x: x[1])
    for k in range(len(y_lp)):
        if y_lp[k]>value_max:
            y_lp[k] = value_max
    ma_results1 = moving_average1(y_lp, window_size)
    ma_results2 = moving_average2(y_lp, window_size)

    #popt, pcov = curve_fit(laplace_f_individual, x_model[i], y_DFOS[i], p0=[0.00455, 3.1, 0.3, 1])
    #sig_compare, a_compare, b_compare, c_compare = popt

    r2 = r_squared(y_DFOS[i], ma_results1)
    print(f"R-Squared: {r2}")

    # PLOT
    plt.scatter(x_model[i], y_DFOS[i])
    plt.plot(x_model[i], y_lp, color='k')
    plt.plot(x_model[i], ma_results1, color='r')

    #reducing the Array x_model to have the same lenght as the ma_res2
    length_res = len(ma_results2)
    length_x_model = len(x_model)
    rounded_up = ((length_x_model-length_res)/2).__ceil__()
    rounded_down = ((length_x_model-length_res)/2).__floor__()

    #plt.plot(x_model[rounded_up:len(x_model)-rounded_down], ma_results2, color='k')
    #plt.plot(x_model, y_individual, color='red')
    plt.show()



print("Laplace Func without post processing")
for i in range(11):
    r2 = r_squared(y_DFOS[i], laplace_f(x_model[i], sig, a, b, c, w_cr_array[i,i]))
    print(f"R-Squared: {r2}")

"""
print("-----")
print("Laplace function with proper peak ")

for i in range(11):
    y_lp = laplace_f(x_model[i], sig, a, b, c, w_cr_array[i, i])
    index_max, value_max = max(enumerate(y_DFOS[i]), key=lambda x: x[1])
    for k in range(len(y_lp)):
        if y_lp[k] > value_max:
            y_lp[k] = value_max

    # Fenstergröße für den gleitenden Durchschnitt
    window_size = 10

    #moving average mit gleicher Länge des Arrays
    ma_results1 = moving_average1(y_lp, window_size)

    r2 = r_squared(y_DFOS[i], ma_results1)
    print(f"R-Squared: {r2}")
"""
