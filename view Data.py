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
    y_DFOS.append(y_DFOS_temp)

#umformen in einen Array mit Shape von x_data und w_cr_array
np.stack(y_DFOS)
y_DFOS= np.array(y_DFOS)

plt.scatter(x_DFOS[4],y_DFOS[4])
#plt.show()

#Position Cracks
pos_cr = [-0.255, -0.156, -0.028, 0.033, 0.25] #cm
x0 = pos_cr
w_cr = [123.9, 20.4, 112.6, 81.6, 121.8] #microm

x_prediction = np.linspace(-0.36, 0.36, 7200)

gamma = 0.01621281
a = 0.80975164

def cauchy(x, gamma, a, x0, w_cr):
    return w_cr*(gamma / (np.pi * a * (gamma**2 + (x-x0)**2)) )  #virtuelle Daten mit freien Parametern

y = [0] * 7200
y_prediction = np.array(y)
data = [x_prediction,y_prediction]

'''
for i in range(len(pos_cr)):
    x_single_crack = np.linspace(-0.1+pos_cr[i],0.1+pos_cr[i],2000)
    y_single_crack = cauchy(x_single_crack,gamma,a,w_cr[i])
    data_single_crack = [x_single_crack,y_single_crack]
    for j,k in data_single_crack.items():
        if k in data:
            data_single_crack[k] += k
'''
def moving_average1(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')


# Fenstergröße für den gleitenden Durchschnitt
window_size = 30


for i in range(len(pos_cr)):
    y_single_crack = cauchy(x_prediction, gamma, a, x0[i], w_cr[i])

    limit = 0.15

    mask1 = x_prediction > (pos_cr[i]+limit)
    mask2 = x_prediction < (pos_cr[i]-limit)
    y_single_crack[mask1] = 0
    y_single_crack[mask2] = 0

    y_single_crack = moving_average1(y_single_crack,window_size)

    plt.plot(x_prediction,y_single_crack,c='m')
    y_prediction = y_prediction+y_single_crack

plt.plot(x_prediction,y_prediction,c='r')
plt.show()
print('last statement')



