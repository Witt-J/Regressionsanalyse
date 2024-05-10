import numpy as np
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
#df = pd.read_csv('Dehnungsverlauf/strain_peak_for_ES_out_at_x_0.67_m.csv')


y_DFOS = []
for i in range(12):
    df = pd.read_csv(('Dehnungsverlauf/strainprofile_' + str((i+1)) + '_ES_out_0.65.csv'), sep='\t')
    y_DFOS.append(df.T.values[1])


# Extract x values from dataframe
x_DFOS = df.T.values[0]
x_DFOS = np.array(x_DFOS)

#masking
mask1 = x_DFOS > (-0.514-0.07)
mask2 = x_DFOS < (-0.514+0.07)

mask = mask1&mask2

x_DFOS = x_DFOS[mask]
x_DFOS = x_DFOS+0.514

for i in range(12):
    y_DFOS[i] = y_DFOS[i][mask]


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
'''
#CRACK C12
w_cr_array = [26.151312777777783, 56.88582611111112, 92.34463333333338, 130.0315105555556, 168.98190833333337,
              209.0284805555556, 248.09519944444452, 285.26361138888893, 322.0303594444445, 357.5927802777779,
              391.4908075000001, 412.21442166666674]
 '''
#CRACK C3
w_cr_array = [19.922467499999, 24.4241725000004, 80.9428099999999, 115.362032499999, 150.148829999999, 184.150102499999,
              218.735855, 253.7521025, 287.824875,322.2732825, 357.12274, 401.6919725]

w_cr_array = np.array(w_cr_array)

gamma = 24.6
#mit Meshgrid 2D Arrays erzeugen, welche die gleiche Dimension (shape) haben -> damit diese mit dem y_daten überlager werden können
x_model, w_cr_array = np.meshgrid(x_model, w_cr_array)

#y_DFOS = []
#y_max = []

#einlesen der y_daten
'''
for i in range(df.shape[1]-1):
    y_DFOS_half = df['strain_u'+str(i+1)]
    y_DFOS_mirr = y_DFOS_half[::-1]
    y_temp = np.append(y_DFOS_half, y_DFOS_mirr)
    y_DFOS.append(y_temp)

for i in range(df.shape[1]-3):
    y_DFOS_temp = np.array(df.T.values[i+1])

    y_DFOS_temp = np.array(y_DFOS_temp)

    y_DFOS.append(y_DFOS_temp)
    y_max.append(max(y_DFOS_temp))
 '''
#umformen in einen Array mit Shape von x_data und w_cr_array
np.stack(y_DFOS)
y_DFOS= np.array(y_DFOS)
#y_max = np.array(y_max)
# Definition der zu optimierenden Funktion
# Definition der zu optimierenden Funktion
def objective(params, x, y, w_cr, gamma):
    s, alpha = params
    y_predicted = w_cr*gamma / (1+(x/s)**2)**alpha  #virtuelle Daten mit freien Parametern
    #for i in range(df.shape[1]-3):
        #y_predicted[i] = y_predicted[i]-y_predicted[i,0]
        #y_predicted[i] = (b * w_cr[i] + c) * y_predicted[i]
    error = np.sum((y_predicted - y)**2)                                        #least square analyse
    return error

# Definition der Funktion, um später virtuelle y_daten erzeugen zu können
def lorentz(x, gamma, s, alpha, w_cr):
    y_predicted = w_cr*gamma / (1+(x/s)**2)**alpha           # virtuelle Daten mit freien Parametern
    #for i in range(df.shape[1] - 3):
    #    y_predicted[i] = y_predicted[i] - y_predicted[i, 0]
    #    y_predicted[i] = (b * w_cr[i] + c) * y_predicted[i]
    return y_predicted

initial_params = [ 0.018, 1.25]

# Optimierung
result = minimize(objective, initial_params, args=(x_model, y_DFOS, w_cr_array, gamma))

res1 = minimize(objective, initial_params, args=(x_model[1], y_DFOS[1], w_cr_array[1], gamma))
res2 = minimize(objective, initial_params, args=(x_model[11], y_DFOS[11], w_cr_array[11], gamma))

# Ausgabe der optimierten Parameter
optimized_params = result.x
print("Optimierte Parameter:", optimized_params)
s, alpha = optimized_params


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
y_lp = lorentz(x_model, gamma, s, alpha, w_cr_array)
for i in range(12):
    # PLOT
    plt.scatter(x_model[i], y_DFOS[i])
    plt.plot(x_model[i], y_lp[i], color='k')
    plt.show()

    r2 = r_squared(y_DFOS[i], y_lp[i])
    print(f"R-Squared: {r2}")


