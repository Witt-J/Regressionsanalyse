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
#df = pd.read_csv('Dehnungsverlauf/strain_peak_for_ES_out_at_x_0.67_m.csv')


# Extract x values from dataframe
x_DFOS = df.T.values[0]

print('Welcher Riss soll betrachtet werden 1-5?')
riss_nr = input()
riss_nr = int(riss_nr)

if riss_nr==1:
    #RISS 1
    index_start = 1016
    index_end = 1324
elif riss_nr==2:
    #RISS 2
    index_start = 1880
    index_end = 2171
elif riss_nr==3:
    #RISS 3
    index_start = 3956
    index_end = 4265
elif riss_nr==4:
    #RISS 4
    index_start = 4404
    index_end = 4525
elif riss_nr==5:
    #RISS 5
    index_start = 4987
    index_end = 5295
else:
    index_start = 0
    index_end = len(x_DFOS)-1

x_DFOS = np.array(x_DFOS)
x_DFOS = x_DFOS[index_start:index_end]

if riss_nr==4:
    delta_x = (abs(min(x_DFOS) - max(x_DFOS)))
    x_start = 0
    x_end = delta_x
    x_DFOS = np.linspace(x_start, x_end, (index_end - index_start))

    x_DFOS_half = x_DFOS
    x_DFOS_mirr = x_DFOS_half[::-1]
    x_DFOS = np.append(x_DFOS_mirr * -1, x_DFOS_half)

delta_x = (abs(min(x_DFOS)-max(x_DFOS)))/2
x_start = -delta_x
x_end = delta_x

if riss_nr!=4:
    x_model = np.linspace(x_start, x_end, (index_end-index_start))
else:
    x_model = x_DFOS
#Liste der Rissbreiten einfügen und zu einem Array umwandeln

if riss_nr==1:
    # Riss 1 bei -1.24m
     w_cr_array = [6.45345638888889, 17.004621111111106, 30.809519722222223, 48.46142888888889, 68.32797833333335, 88.4466988888889, 110.62932111111111, 132.7303502777778, 156.13874611111112, 180.50695722222224, 206.1053186111111]
elif riss_nr==2:
    # Riss 2 bei -0.72m
     w_cr_array = [25.817609999999995, 62.73302027777779, 103.47890944444444, 145.7053686111111, 186.29669500000003, 226.75738944444447, 269.18947972222225, 312.55836444444446, 357.41525805555557, 403.28825333333333, 460.0975383333333]
elif riss_nr==3:
    #Riss 3 bei 0.62m
    w_cr_array = [26.151312777777783, 56.88582611111112, 92.34463333333338, 130.0315105555556, 168.98190833333337, 209.0284805555556, 248.09519944444452, 285.26361138888893, 322.0303594444445, 357.5927802777779, 391.4908075000001]
elif riss_nr==4:
    #Riss 4 bei 0.86m
    w_cr_array = [15.175723333333327, 35.66578888888887, 63.07872638888885, 94.73801638888885, 128.80793611111108, 164.02423194444435, 198.59538472222215, 232.53970638888882, 266.78888888888883, 300.4199358333332, 332.0820208333332]
elif riss_nr==5:
    #Riss 5 bei 1.34m
    w_cr_array = [11.23331805555556, 14.37383638888889, 25.528461111111103, 37.86212444444444, 50.84252333333333, 64.41449805555555, 78.98109194444446, 93.72599166666666, 108.68798055555554, 123.13810166666667, 137.46988666666664]



w_cr_array = np.array(w_cr_array)

#mit Meshgrid 2D Arrays erzeugen, welche die gleiche Dimension (shape) haben -> damit diese mit dem y_daten überlager werden können
x_model, w_cr_array = np.meshgrid(x_model, w_cr_array)
y_DFOS = []

#einlesen der y_daten (14 for Crack 2 and 3, 11 for Crack 1)
for i in range(11):
    df = pd.read_csv('Dehnungsverlauf/strainprofile_'+str(i+1)+'_ES_out_0.65.csv', sep='\t')

    if riss_nr != 4:
        # Extract x values from dataframe
        y_DFOS_temp = np.array(df.T.values[1])
        y_DFOS_temp = y_DFOS_temp[index_start:index_end]
        y_DFOS.append(y_DFOS_temp)

    if riss_nr == 4:
        y_DFOS_half = np.array(df.T.values[1])
        y_DFOS_half = y_DFOS_half[index_start:index_end]
        y_DFOS_mirr = y_DFOS_half[::-1]
        y_DFOS_temp = np.append(y_DFOS_mirr, y_DFOS_half)
        y_DFOS.append(y_DFOS_temp)

#umformen in einen Array mit Shape von x_data und w_cr_array
np.stack(y_DFOS)
y_DFOS= np.array(y_DFOS)

# Definition der zu optimierenden Funktion
def objective(params, x, y, w_cr):
    gamma, a, b, c = params
    y_predicted = w_cr*c*(b*gamma / (np.pi * a * (gamma**2 + x**2))) #virtuelle Daten mit freien Parametern
    #error = np.sum((y_predicted - y)**2)                                        #least square analyse

    n_steps = (y_predicted.shape)[0]
    lenght = (y_predicted.shape)[1]
    third = (lenght/3).__floor__()

    error1 = np.sum((y_predicted[0:n_steps,0:third] - y[0:n_steps,0:third]) **2)
    error2 = np.sum((y_predicted[0:n_steps,third:lenght-third] - y[0:n_steps,third:lenght-third])**4)                                        #least square analyse
    error3 = np.sum((y_predicted[0:n_steps,lenght-third:lenght] - y[0:n_steps,lenght-third:lenght]) **2)

    error = error1+error2+error3
    return error

# Definition der Funktion, um später virtuelle y_daten erzeugen zu können
def cauchy(x, gamma, a, b, c, w_cr):
    return w_cr*c*(b*gamma / (np.pi * a * (gamma**2 + x**2)))  #virtuelle Daten mit freien Parametern


# Anfangswerte für die Optimierung
initial_params = [ 0.017, 0.82, 1, 1]

# Optimierung
result = minimize(objective, initial_params, args=(x_model, y_DFOS, w_cr_array), method='L-BFGS-B')

# Ausgabe der optimierten Parameter
optimized_params = result.x
print("Optimierte Parameter:", optimized_params)
gamma, a, b, c= optimized_params

gamma_list = []
a_list = []


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
print('polt? y/n')
q_plot = input()
if q_plot=='y':
    bool_plot = True
elif q_plot=='n':
    bool_plot = False

for i in range(11):
    y_lp = cauchy(x_model[i], gamma, a, b, c, w_cr_array[i,i])
    #y_lp_compare = cauchy(x_model[i], 0.2313576, 0.06110288, w_cr_array[i, i])

    # PLOT
    if bool_plot:
        plt.scatter(x_model[i], y_DFOS[i])
        plt.plot(x_model[i], y_lp, color='k')
        #plt.plot(x_model[i], y_lp_compare, color='r')
        plt.show()

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
