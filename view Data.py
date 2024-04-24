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

x_DFOS = np.array(x_DFOS)

w_cr_array = [15.175723333333327, 35.66578888888887, 63.07872638888885, 94.73801638888885, 128.80793611111108, 164.02423194444435, 198.59538472222215, 232.53970638888882, 266.78888888888883, 300.4199358333332, 332.0820208333332]

w_cr_array = np.array(w_cr_array)

#mit Meshgrid 2D Arrays erzeugen, welche die gleiche Dimension (shape) haben -> damit diese mit dem y_daten überlager werden können
x_DFOS, w_cr_array = np.meshgrid(x_DFOS, w_cr_array)
y_DFOS = []

#einlesen der y_daten (14 for Crack 2 and 3, 11 for Crack 1)
for i in range(11):
    df = pd.read_csv('Dehnungsverlauf/strainprofile_'+str(i+1)+'_ES_out_0.65.csv', sep='\t')

    # Extract x values from dataframe
    y_DFOS_temp = np.array(df.T.values[1])
    #y_DFOS_temp = list(map(float, y_DFOS_temp))
    #y_DFOS_temp = np.array(y_DFOS_temp)
    y_DFOS.append(y_DFOS_temp)

#umformen in einen Array mit Shape von x_data und w_cr_array
np.stack(y_DFOS)
y_DFOS= np.array(y_DFOS)

plt.plot(x_DFOS[4],y_DFOS[4])
#plt.show()

#Position Cracks
pos_cr = [-0.255, -0.156, -0.028, 0.033, 0.25] #cm
x0 = pos_cr
w_cr = [123.9, 20.4, 112.6, 81.6, 121.8] #microm

x_prediction = np.linspace(-0.38, 0.38, 7600)
x_start = x_prediction[0]

gamma = 1.65128426e-02
a = 9.16119657e-01
b = 1.48343394e-04
c = 1.11589189e+00

def cauchy(x, gamma, a, b, c, x0, w_cr, pos_cr, limit):
    y_predicted = w_cr * (gamma / (np.pi * a * (gamma ** 2 + (x-x0) ** 2)))  # virtuelle Daten mit freien Parametern
    x_list = x.tolist()

    gesuchter_wert = (pos_cr-limit)
    nächstes_element = min(x_list, key=lambda x: abs(x - gesuchter_wert))

    index_inf_l = x_list.index(nächstes_element)

    y_predicted = y_predicted - y_predicted[index_inf_l]
    y_predicted = (b * w_cr + c) * y_predicted
    return y_predicted



def lorentz(x, gamma, s, alpha, x0, w_cr, pos_cr, limit):
    y_predicted = w_cr*gamma / (1+((x-x0)/s)**2)**alpha  # virtuelle Daten mit freien Parametern
    x_list = x.tolist()

    gesuchter_wert = (pos_cr - limit)
    nächstes_element = min(x_list, key=lambda x: abs(x - gesuchter_wert))

    index_inf_l = x_list.index(nächstes_element)

    y_predicted = y_predicted - y_predicted[index_inf_l]
    #y_predicted = (b * w_cr + c) * y_predicted
    return y_predicted

def lorentz_lin(x, gamma, s, alpha_0, alpha_1, x0, w_cr, pos_cr, limit):
    y_predicted = w_cr*gamma / (1+((x-x0)/s)**2)**(alpha_0+w_cr*alpha_1)  # virtuelle Daten mit freien Parametern
    x_list = x.tolist()

    gesuchter_wert = (pos_cr - limit)
    nächstes_element = min(x_list, key=lambda x: abs(x - gesuchter_wert))

    index_inf_l = x_list.index(nächstes_element)

    y_predicted = y_predicted - y_predicted[index_inf_l]
    #y_predicted = (b * w_cr + c) * y_predicted
    return y_predicted

y = [0] * 7600
y_prediction = np.array(y)
data = [x_prediction,y_prediction]
#limit for Einflusslänge of crack (influence lenght?)
limit = 0.12

"""
for i in range(len(pos_cr)):
    y_single_crack = cauchy(x_prediction, gamma, a, b, c, x0[i], w_cr[i], pos_cr[i], limit)

    mask1 = x_prediction > (pos_cr[i]+limit)
    mask2 = x_prediction < (pos_cr[i]-limit)
    y_single_crack[mask1] = 0
    y_single_crack[mask2] = 0

    plt.plot(x_prediction,y_single_crack,c='m')
    y_prediction = y_prediction+y_single_crack

plt.plot(x_prediction,y_prediction,c='r')
plt.show()
"""
gamma = 24.6
#C12
#s = 0.01872447
#alpha = 1.25446139

#C3
#s = 0.01881
#alpha = 1.25823

#from Gradient
#s = 0.016
#alpha = 1.6

#interpolation
s = 0.0184
alpha_0 = 1.138902
alpha_1 = 0.0015

for i in range(len(pos_cr)):
    y_single_crack = lorentz_lin(x_prediction, gamma, s, alpha_0, alpha_1, x0[i], w_cr[i], pos_cr[i], limit)

    mask1 = x_prediction > (pos_cr[i]+limit)
    mask2 = x_prediction < (pos_cr[i]-limit)
    y_single_crack[mask1] = 0
    y_single_crack[mask2] = 0

    plt.plot(x_prediction,y_single_crack,c='m')
    y_prediction = y_prediction+y_single_crack

#plt.scatter(x_DFOS[4],y_DFOS[4])
plt.plot(x_prediction,y_prediction,c='r')
plt.show()



