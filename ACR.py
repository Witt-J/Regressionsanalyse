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
df = pd.read_csv('Dehnungsverlauf/strainprofile_1_ACR1_0.65.csv', sep='\t')

# Extract x values from dataframe
x_DFOS = df.T.values[0]
x_DFOS = np.array(x_DFOS)

pos_cr = 0.6727
limit = 0.12

#masking
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
"""
w_cr_array = [26.63485680555558, 57.61857472222225, 93.37536, 131.65188833333337, 170.9102759722223,
              211.1452317361112, 250.20251930555568, 286.91174055555564, 323.16453363690516, 358.331415128969, 391.60467962004043]
"""
w_cr_array = [26.63485680555558, 57.61857472222225, 93.37536, 131.65188833333337]



w_cr_array = np.array(w_cr_array)

gamma = 31.2
#mit Meshgrid 2D Arrays erzeugen, welche die gleiche Dimension (shape) haben -> damit diese mit dem y_daten überlager werden können
x_model, w_cr_array = np.meshgrid(x_model, w_cr_array)

y_DFOS = []
y_max = []

#einlesen der y_daten
for i in range(4):
    df = pd.read_csv('Dehnungsverlauf/strainprofile_'+str(i+1)+'_ACR1_0.65.csv', sep='\t')

    # Extract x values from dataframe
    y_DFOS_temp = np.array(df.T.values[1])
    y_DFOS_temp = list(map(float, y_DFOS_temp))
    y_DFOS_temp = np.array(y_DFOS_temp)
    y_DFOS.append(y_DFOS_temp[mask])


#umformen in einen Array mit Shape von x_data und w_cr_array
np.stack(y_DFOS)
y_DFOS= np.array(y_DFOS)
y_max = np.array(y_max)
# Definition der zu optimierenden Funktion
# Definition der zu optimierenden Funktion

def objective(params, x, y, w_cr, gamma):
    s, alpha_0, alpha_1 = params
    y_predicted = w_cr*gamma / (1+(x/s)**2)**(alpha_0 + w_cr * alpha_1)  #virtuelle Daten mit freien Parametern
    #for i in range(df.shape[1]-3):
        #y_predicted[i] = y_predicted[i]-y_predicted[i,0]
        #y_predicted[i] = (b * w_cr[i] + c) * y_predicted[i]
    error = np.sum((y_predicted - y)**2)                                        #least square analyse
    return error

# Definition der Funktion, um später virtuelle y_daten erzeugen zu können
def lorentz(x, gamma, s, alpha_0, alpha_1, w_cr):
    y_predicted = w_cr*gamma / (1+(x/s)**2)**(alpha_0 + w_cr * alpha_1)           # virtuelle Daten mit freien Parametern
    #for i in range(df.shape[1] - 3):
    #    y_predicted[i] = y_predicted[i] - y_predicted[i, 0]
    #    y_predicted[i] = (b * w_cr[i] + c) * y_predicted[i]
    return y_predicted

initial_params = [ 0.018, 1.25, 0.003]

# Optimierung
result = minimize(objective, initial_params, args=(x_model, y_DFOS, w_cr_array, gamma))

res1 = minimize(objective, initial_params, args=(x_model[3], y_DFOS[3], w_cr_array[3], gamma))
#res2 = minimize(objective, initial_params, args=(x_model[10], y_DFOS[10], w_cr_array[10], gamma))


# Ausgabe der optimierten Parameter
optimized_params = result.x
print("Optimierte Parameter:", optimized_params)
s, alpha_0, alpha_1 = optimized_params


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
y_lp = lorentz(x_model, gamma, s, alpha_0, alpha_1, w_cr_array)
for i in range(10):
    # PLOT
    plt.scatter(x_model[i], y_DFOS[i])
    plt.plot(x_model[i], y_lp[i], color='k')
    plt.show()

    r2 = r_squared(y_DFOS[i], y_lp[i])
    print(f"R-Squared: {r2}")


