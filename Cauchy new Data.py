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
#index_start = 4022
#index_end = 4190

#RISS 3
index_start = 4270
index_end = 4535

x_DFOS = np.array(x_DFOS)
x_DFOS = x_DFOS[index_start:index_end]

delta_x = (abs(min(x_DFOS)+max(x_DFOS)))/2
x_start = -delta_x/2
x_end = delta_x/2

x_model = np.linspace(x_start, x_end, (index_end-index_start))

#Liste der Rissbreiten einfügen und zu einem Array umwandeln

#Riss 1 bei -0.72m nur 1 bis 11!!!
#w_cr_array = [25.817609999999995, 62.73302027777779, 103.47890944444444, 145.7053686111111, 186.29669500000003, 226.75738944444447, 269.18947972222225, 312.55836444444446, 357.41525805555557, 403.28825333333333, 460.0975383333333]
#Riss 2 bei 0.62m
#w_cr_array = [26.151312777777783, 56.88582611111112, 92.34463333333338, 130.0315105555556, 168.98190833333337, 209.0284805555556, 248.09519944444452, 285.26361138888893, 322.0303594444445, 357.5927802777779, 391.4908075000001]
#Riss 3 bei 0.86m
w_cr_array = [15.175723333333327, 35.66578888888887, 63.07872638888885, 94.73801638888885, 128.80793611111108, 164.02423194444435, 198.59538472222215, 232.53970638888882, 266.78888888888883, 300.4199358333332, 332.0820208333332]


w_cr_array = np.array(w_cr_array)

#mit Meshgrid 2D Arrays erzeugen, welche die gleiche Dimension (shape) haben -> damit diese mit dem y_daten überlager werden können
x_model, w_cr_array = np.meshgrid(x_model, w_cr_array)
y_DFOS = []

#einlesen der y_daten (14 for Crack 2 and 3, 11 for Crack 1)
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
    gamma, a, b, c = params
    y_predicted = w_cr*(gamma / (np.pi * a * (gamma**2 + b*x**2)) )+c  #virtuelle Daten mit freien Parametern
    error = np.sum((y_predicted - y)**2)                                        #least square analyse
    return error

# Definition der Funktion, um später virtuelle y_daten erzeugen zu können
def cauchy(x, gamma, a, b, c, w_cr):
    return w_cr*(gamma / (np.pi * a * (gamma**2 + b*x**2)) +c)  #virtuelle Daten mit freien Parametern


# Anfangswerte für die Optimierung
initial_params = [ 0.14, 0.12, 1, 0]

# Optimierung
result = minimize(objective, initial_params, args=(x_model, y_DFOS, w_cr_array), method='L-BFGS-B')

# Ausgabe der optimierten Parameter
optimized_params = result.x
print("Optimierte Parameter:", optimized_params)
gamma, a, b, c = optimized_params

gamma_list = []
a_list = []
b_list = []
c_list = []

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
r_list = []
#erstellen der y Werte mit der Laplace funktion und den optimierten parametern (14 for Crack 2 , 11 for Crack 1, 3)
for i in range(11):
    y_lp = cauchy(x_model[i], gamma, a, b, c, w_cr_array[i,i])


    # PLOT
    plt.scatter(x_model[i], y_DFOS[i])
    plt.plot(x_model[i], y_lp, color='k')
    #plt.show()

    r2 = r_squared(y_DFOS[i], cauchy(x_model[i], gamma, a, b, c, w_cr_array[i, i]))
    r_list.append(r2)
    print(f"R-Squared: {r2}")

print('-----')
average_list = sum(r_list)/len(r_list)
print(average_list)
"""
plt.plot(gamma_list)
plt.show()
plt.plot(a_list)
plt.show()
plt.plot(b_list)
plt.show()
plt.plot(c_list)
plt.show()
"""
