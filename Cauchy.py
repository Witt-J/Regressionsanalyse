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
#df = pd.read_csv('Dehnungsverlauf/strain_peak_for_ES_out_at_x_0.67_m.csv')
df = pd.read_csv('Dehnungsverlauf/strain_peak_for_ES_out_at_x_0.67_m.csv')

# Extract x values from dataframe
x_DFOS = df.T.values[0]
x_DFOS = np.array(x_DFOS)


delta_x = (abs(min(x_DFOS)-max(x_DFOS)))/2
x_start = -delta_x
x_end = delta_x

x_model = np.linspace(x_start, x_end, len(x_DFOS))

#Liste der Rissbreiten einfügen und zu einem Array umwandeln
'''
w_cr_array = [26.151312777777783, 56.88582611111112, 92.34463333333338, 130.0315105555556, 168.98190833333337,
              209.0284805555556, 248.09519944444452, 285.26361138888893, 322.0303594444445, 357.5927802777779,
              391.4908075000001, 412.21442166666674, 429.1949597361114, 460.94397132539683]
              '''
w_cr_array = [26.151312777777783, 56.88582611111112, 92.34463333333338, 130.0315105555556, 168.98190833333337,
              209.0284805555556, 248.09519944444452, 285.26361138888893, 322.0303594444445, 357.5927802777779,
              391.4908075000001, 412.21442166666674]
w_cr_strain = [26.151312777777783, 56.88582611111112, 92.34463333333338, 130.0315105555556, 168.98190833333337,
              209.0284805555556, 248.09519944444452, 285.26361138888893, 322.0303594444445, 357.5927802777779,
              391.4908075000001, 412.21442166666674]


w_cr_array = np.array(w_cr_array)

#mit Meshgrid 2D Arrays erzeugen, welche die gleiche Dimension (shape) haben -> damit diese mit dem y_daten überlager werden können
x_model, w_cr_array = np.meshgrid(x_model, w_cr_array)
y_DFOS = []
y_max = []

#einlesen der y_daten
'''
for i in range(df.shape[1]-1):
    y_DFOS_half = df['strain_u'+str(i+1)]
    y_DFOS_mirr = y_DFOS_half[::-1]
    y_temp = np.append(y_DFOS_half, y_DFOS_mirr)
    y_DFOS.append(y_temp)
'''
for i in range(df.shape[1]-3):
    y_DFOS_temp = np.array(df.T.values[i+1])
    y_first = y_DFOS_temp[0]
    y_last = y_DFOS_temp[0]
    avg_y = (y_first+y_last)/2
    y_DFOS_temp = np.array(y_DFOS_temp)
    y_DFOS_temp = y_DFOS_temp-avg_y

    y_DFOS.append(y_DFOS_temp)
    y_max.append(max(y_DFOS_temp))

#umformen in einen Array mit Shape von x_data und w_cr_array
np.stack(y_DFOS)
y_DFOS= np.array(y_DFOS)
y_max = np.array(y_max)
# Definition der zu optimierenden Funktion
# Definition der zu optimierenden Funktion
def objective(params, x, y, w_cr):
    gamma, a = params
    y_predicted = w_cr*(gamma / (np.pi * a * (gamma**2 + x**2)) )  #virtuelle Daten mit freien Parametern
    error = np.sum((y_predicted - y)**2)                                        #least square analyse
    return error

# Definition der Funktion, um später virtuelle y_daten erzeugen zu können
def cauchy(x, gamma, a, w_cr):
    return w_cr*(gamma / (np.pi * a * (gamma**2 + x**2)))  #virtuelle Daten mit freien Parametern


initial_params = [ 0.016, 0.8]


# Optimierung
result = minimize(objective, initial_params, args=(x_model, y_DFOS, w_cr_array))

# Ausgabe der optimierten Parameter
optimized_params = result.x
print("Optimierte Parameter:", optimized_params)
gamma, a = optimized_params


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
for i in range(12):
    y_lp = cauchy(x_model[i], gamma, a, w_cr_array[i,i])

    # PLOT
    plt.scatter(x_model[i], y_DFOS[i])
    plt.plot(x_model[i], y_lp, color='k')
    plt.show()

    r2 = r_squared(y_DFOS[i], cauchy(x_model[i], gamma, a, w_cr_array[i, i]))
    print(f"R-Squared: {r2}")

plt.scatter(w_cr_strain,y_max)
plt.show()
