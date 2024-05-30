import numpy as np
import scipy.integrate
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import shapiro
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import sympy as sp



# Load CSV file into a Pandas dataframe
#df = pd.read_csv('Dehnungsverlauf/strain_peak_for_ES_out_at_x_0.67_m.csv')
df = pd.read_csv('Dehnungsverlauf/strain_peak_for_ES_out_at_x_0.67_m.csv')

# Extract x values from dataframe
x_DFOS_C12 = df.T.values[0]
x_DFOS_C12 = np.array(x_DFOS_C12)

# limit beschreibt den Abstand nach rechts und links von Rissmitte pos_cr, limit in [m]
limit = 0.08
pos_cr = 0.6676

#erstellt Masken, wodruch der Array gekürzt werden kann, da die Rohdaten alle Risse beinhalten
mask1 = x_DFOS_C12 > (pos_cr-limit)
mask2 = x_DFOS_C12 < (pos_cr+limit)

#kombination beider Masken
mask = mask1&mask2

#Überlagerung/ Abschneiden der x Daten, damit nur die x Daten rund um die Rissspitze vorhanden sind
x_DFOS_C12 = x_DFOS_C12[mask]
x_DFOS_C12 = x_DFOS_C12-pos_cr

#ein Delta x bestimmen, was dem Unterschied zwischen den größten und kleinsten x Wert repräsentiert
delta_x = (abs(min(x_DFOS_C12)-max(x_DFOS_C12)))/2
x_start = -delta_x
x_end = delta_x

#mit linspace daten Punkte im gleichmäßigen Abstand
x_model_C12 = np.linspace(x_start, x_end, len(x_DFOS_C12))

#Liste der Rissbreiten einfügen
#CRACK C12
w_cr_array_C12 = [26.151312777777783, 56.88582611111112, 92.34463333333338, 130.0315105555556, 168.98190833333337,
              209.0284805555556, 248.09519944444452, 285.26361138888893, 322.0303594444445, 357.5927802777779,
              391.4908075000001, 412.21442166666674]

#Liste für das Einlesen der y_daten erstellen
y_DFOS = []
y_max = []

#einlesen der y_daten
for i in range(df.shape[1]-3):
    y_DFOS_temp = np.array(df.T.values[i+1])

    y_DFOS_temp = np.array(y_DFOS_temp)

    y_DFOS.append(y_DFOS_temp[mask])
    #maximum der y_daten
    y_max.append(max(y_DFOS_temp))
    y_DFOS_temp = np.array(y_DFOS_temp)


#umformen in einen Array mit Shape von x_data und w_cr_array
np.stack(y_DFOS)
y_DFOS= np.array(y_DFOS)
y_max = np.array(y_max)

#R_quadrat definition um ein Fehlermaß bestimmen zu können
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

#fenster definieren, wodurch ein laufender Mittelwert bestimmt wert, die Wichtung ist hier jeweils 1/3 bei einer Fenstergröße von 3
window = 3
weights = np.repeat(1.0, window)/window
x_model_C12 = x_model_C12[1:-1]
#x_model_C12 = x_model_C12[0]

List_linke_Wendepunkt = []
List_rechte_Wendepunkt = []
mittel_Wendepunkt_C12 = []
max_loc_list = []
max_list_C12 = []
max_grad_list_C12 = []

for i in range(12):
    max_loc_list.append(x_model_C12[np.argmax(y_DFOS[i])])
    max_list_C12.append((max(y_DFOS[i])))
    average = np.convolve(y_DFOS[i], weights, mode='valid')
    gradient = np.gradient(average,x_DFOS_C12[1]-x_DFOS_C12[0])/1000
    max_grad = max(gradient)
    max_grad_list_C12.append(max_grad)
    min_grad = min(gradient)
    max_index = np.argmax(gradient)
    min_index = np.argmin(gradient)
    List_linke_Wendepunkt.append(abs(x_model_C12[max_index]))
    List_rechte_Wendepunkt.append(x_model_C12[min_index])
    mittel_Wendepunkt_C12.append((abs(x_model_C12[max_index]) + x_model_C12[min_index]) / 2)



#Definition der Lorentzfunktion mit sigma0 und sigma1, um eine lineare Abhängigkeit von w_cr zu definieren
def lorentz(x, gamma, sigma0, sigma1, w_cr):
    y_predicted = w_cr*gamma / (1+(x/(sigma0+sigma1*w_cr))**2)           # virtuelle Daten mit freien Parametern

    return y_predicted

#lorentz_single ist die definition mit nur 2 freien Parametern: gamma und sigma
def lorentz_single(x, gamma, sigma, w_cr):
    y_predicted = w_cr*gamma / (1+(x/sigma)**2)           # virtuelle Daten mit freien Parametern

    return y_predicted

###################################################
#                C3
###################################################
#Liste der Rissbreiten einfügen FOSANALYSIS
w_cr_array_C3 = [19.669451388888874, 48.233372777777745, 80.78568333333328, 115.46603972222213, 150.30286916666654,
                 184.53831499999987, 218.73831055555536, 253.81509111111086, 287.91341222222195, 322.43429111111084,
                 357.71768590277736, 402.8620627777773]
#w_cr_array_C3 = np.array(w_cr_array_C3)
#Liste für das Einlesen der y_daten erstellen
y_DFOS_C3 = []
for i in range(12):
    df = pd.read_csv(('Dehnungsverlauf/strainprofile_' + str((i+1)) + '_ES_out_0.65.csv'), sep='\t')
    y_DFOS_C3.append(df.T.values[1])

# Extract x values from dataframe
x_DFOS_C3 = df.T.values[0]
x_DFOS_C3 = np.array(x_DFOS_C3)

#masking
mask1 = x_DFOS_C3 > (-0.514-0.08)
mask2 = x_DFOS_C3 < (-0.514+0.08)

#combining the mask
mask = mask1&mask2

#Maske anwenden auf die x_werte
x_DFOS_C3 = x_DFOS_C3[mask]

#maske anwenden auf die y_werte
for i in range(12):
    y_DFOS_C3[i] = y_DFOS_C3[i][mask]

#Rissmitte über den größten y_wert bestimmen und dann den dazugehörigen Index aus der x_daten wählen
index_middle = np.argmax(y_DFOS_C3[4])
x_middle = x_DFOS_C3[index_middle]
#den Ausschnitt der x daten mit dem x_wert vom strain peak subtrahieren
x_DFOS_C3 = x_DFOS_C3-x_middle

delta_x = (abs(min(x_DFOS_C3)-max(x_DFOS_C3)))/2
x_start = -delta_x
x_end = delta_x

#mit linspace in gleichen Abständen x_werte zwischen min und max erstellen
x_model_C3 = np.linspace(x_start, x_end, len(x_DFOS_C3))

#Listen erstellen, welche in der Schleife befüllt werden können(für jeden Belastungsschritt)
List_linke_Wendepunkt_C3 = []
List_rechte_Wendepunkt_C3 = []
mittel_Wendepunkt_C3 = []
#ort des Maximums
max_loc_list_C3 = []
# größe des Maximums
max_list_C3 = []
#größe des größten Gradienten
max_grad_list_C3 = []

#schleife um jeden Belastungspunkt einzeln auszuwerten
for i in range(12):
    max_loc_list_C3.append(x_model_C3[np.argmax(y_DFOS_C3[i])])
    max_list_C3.append((max(y_DFOS_C3[i])))
    #mittelwert mit dem bestimmten Fenster über die Y_daten laufen lassen
    average = np.convolve(y_DFOS_C3[i], weights, mode='valid')
    #Gradienten bestimmen (Array)
    gradient = np.gradient(average,x_DFOS_C3[1]-x_DFOS_C3[0])/1000
    #größten Wert des Gradienten Array bestimmen
    max_grad = max(gradient)
    max_grad_list_C3.append(max_grad)
    min_grad = min(gradient)
    max_index = np.argmax(gradient)
    min_index = np.argmin(gradient)
    List_linke_Wendepunkt_C3.append(abs(x_model_C3[max_index]))
    List_rechte_Wendepunkt_C3.append(x_model_C3[min_index])
    mittel_Wendepunkt_C3.append((abs(x_model_C3[max_index]) + x_model_C3[min_index]) / 2)
    #plt.plot(x_model_C3, gradient)
    #plt.show()

###################################################
#                C9
###################################################
w_cr_array_C9 = [12.746289345345351, 32.35402805555557, 58.232954722222225, 89.09878972222225, 121.80794166666666,
                 153.00820527777776, 184.5041462177177, 215.76272015765764, 247.2092651576577, 276.27144999999996,
                 305.6654791666669, 323.5282844444444]
#w_cr_array_C9 = np.array(w_cr_array_C9)
y_DFOS_C9 = []
for i in range(12):
    df = pd.read_csv(('Dehnungsverlauf/strainprofile_' + str((i+1)) + '_ES_out_0.65.csv'), sep='\t')
    y_DFOS_C9.append(df.T.values[1])

# Extract x values from dataframe
x_DFOS_C9 = df.T.values[0]
x_DFOS_C9 = np.array(x_DFOS_C9)

#masking
mask1 = x_DFOS_C9 > (0.25-0.08)
mask2 = x_DFOS_C9 < (0.25+0.08)

mask = mask1&mask2

x_DFOS_C9 = x_DFOS_C9[mask]

for i in range(12):
    y_DFOS_C9[i] = y_DFOS_C9[i][mask]

index_middle = np.argmax(y_DFOS_C9[4])
x_middle = x_DFOS_C9[index_middle]

x_DFOS_C9 = x_DFOS_C9-x_middle

delta_x = (abs(min(x_DFOS_C9)-max(x_DFOS_C9)))/2
x_start = -delta_x
x_end = delta_x

x_model_C9 = np.linspace(x_start, x_end, len(x_DFOS_C9))

List_linke_Wendepunkt_C9 = []
List_rechte_Wendepunkt_C9 = []
mittel_Wendepunkt_C9 = []
max_loc_list_C9 = []
max_list_C9 = []
max_grad_list_C9 = []

for i in range(12):
    max_loc_list_C9.append(x_model_C9[np.argmax(y_DFOS_C9[i])])
    max_list_C9.append((max(y_DFOS_C9[i])))
    average = np.convolve(y_DFOS_C9[i], weights, mode='valid')
    gradient = np.gradient(average,x_DFOS_C9[1]-x_DFOS_C9[0])/1000
    max_grad = max(gradient)
    max_grad_list_C9.append(max_grad)
    min_grad = min(gradient)
    max_index = np.argmax(gradient)
    min_index = np.argmin(gradient)
    List_linke_Wendepunkt_C9.append(abs(x_model_C9[max_index]))
    List_rechte_Wendepunkt_C9.append(x_model_C9[min_index])
    mittel_Wendepunkt_C9.append((abs(x_model_C9[max_index]) + x_model_C9[min_index]) / 2)

    #plt.scatter(x_DFOS_C9,y_DFOS_C9[i])
    #plt.show()

#################################################################
#OPTIMIERUNG
#################################################################

#Zu minimierendes Ziel für die Bestimmung von gamma
def objective(params, w_cr, strain_max):
    gamma = params
    strain_predicted = w_cr*gamma #virtuelle Daten mit freien Parametern
    error = np.sum((strain_predicted - strain_max)**2)                                        #least square analyse
    return error

initial_params = [24.6]

# Optimierung
result = minimize(objective, initial_params, args=(w_cr_array_C3, max_list_C3))
print(result.x[0])
gamma_C3 = result.x[0]
result = minimize(objective, initial_params, args=(w_cr_array_C9, max_list_C9))
print(result.x[0])
gamma_C9 = result.x[0]
result = minimize(objective, initial_params, args=(w_cr_array_C12, max_list_C12))
print(result.x[0])
gamma_C12 = result.x[0]

#Funktion um strain_max zu bestimmen
def Regression(w_cr, gamma):
    strain_max = w_cr*gamma
    return strain_max

C3 = Regression(w_cr_array_C3,np.repeat(gamma_C3,len(w_cr_array_C3)))
C9 = Regression(w_cr_array_C9,np.repeat(gamma_C9,len(w_cr_array_C9)))
C12 = Regression(w_cr_array_C12,np.repeat(gamma_C12,len(w_cr_array_C12)))

plt.scatter(w_cr_array_C3,max_list_C3,label='Strain max C3')
plt.plot(w_cr_array_C3,C3,label='Reg C3')
plt.scatter(w_cr_array_C9,max_list_C9,label='Strain max C9')
plt.plot(w_cr_array_C9,C9,label='Reg C9')
plt.scatter(w_cr_array_C12,max_list_C12,label='Strain max C12')
plt.plot(w_cr_array_C12,C12,label='Reg C12')
plt.legend()
plt.show()

#Zu Minimierendes Ziel für die Bestimmung von Sigma, der Zusammenhang von Sigma und dem Wendepunkt ist im Latex Dokument erläutert
# x_WP = sigma/WURZEL(3)  bzw WURZEL(3)*x_WP = sigma
def objective_WP(params, w_cr, sigma_real):
    sigma0, sigma1 = params
    sigma_predicted = sigma0 + np.array(w_cr) * sigma1 #virtuelle Daten mit freien Parametern
    error = np.sum((np.array(sigma_real)*np.sqrt(3) - sigma_predicted)**2)                                        #least square analyse
    return error

#Zu Minimierendes Ziel für die Bestimmung von Sigma, der Zusammenhang von Sigma und dem Wendepunkt ist im Latex Dokument erläutert
# x_WP = sigma/WURZEL(3)  bzw WURZEL(3)*x_WP = sigma
def objective_WP_single(params, w_cr, sigma_real):
    sigma = params
    sigma_predicted = sigma  #virtuelle Daten mit freien Parametern
    error = np.sum((np.array(sigma_real)*np.sqrt(3) - sigma_predicted)**2)                                        #least square analyse
    return error

#bestimmung der Anfangsparameter
initial_params = [0.009, -0.0001]
initial_params_single = [1.6]

#lin Reg um sigma zu beschreiben, wenn dieser Parameter von w_cr abhängig ist
def Regression_sigma(w_cr, sigma0, sigma1):
    sig = sigma0 + np.array(w_cr) * sigma1
    return sig

#bestimmung der Werte für die Individuellen Risse
result = minimize(objective_WP, initial_params, args=(w_cr_array_C3, mittel_Wendepunkt_C3))
sigma0_C3, sigma1_C3 = result.x
result = minimize(objective_WP_single, initial_params_single, args=(w_cr_array_C3, mittel_Wendepunkt_C3))
sigma_C3 = result.x
result = minimize(objective_WP, initial_params, args=(w_cr_array_C9, mittel_Wendepunkt_C9))
sigma0_C9, sigma1_C9 = result.x
result = minimize(objective_WP_single, initial_params_single, args=(w_cr_array_C9, mittel_Wendepunkt_C9))
sigma_C9 = result.x
result = minimize(objective_WP, initial_params, args=(w_cr_array_C12, mittel_Wendepunkt_C12))
sigma0_C12, sigma1_C12 = result.x
result = minimize(objective_WP_single, initial_params_single, args=(w_cr_array_C12, mittel_Wendepunkt_C12))
sigma_C12 = result.x



plt.scatter(w_cr_array_C3,mittel_Wendepunkt_C3,label='X WP C3')
plt.scatter(w_cr_array_C9,mittel_Wendepunkt_C9,label='X WP C9')
plt.scatter(w_cr_array_C12,mittel_Wendepunkt_C12,label='X WP C12')
plt.legend()
plt.show()

#Darstellung der WP*WURZEL(3) zu den optimierten sigma Parametern
plt.scatter(w_cr_array_C3,np.array(mittel_Wendepunkt_C3)*np.sqrt(3),label='Sigma C3')
plt.plot(w_cr_array_C3, Regression_sigma(w_cr_array_C3,sigma0_C3,sigma1_C3))

plt.scatter(w_cr_array_C9,np.array(mittel_Wendepunkt_C9)*np.sqrt(3),label='Sigma C9')
plt.plot(w_cr_array_C9, Regression_sigma(w_cr_array_C9,sigma0_C9,sigma1_C9))

plt.scatter(w_cr_array_C12,np.array(mittel_Wendepunkt_C12)*np.sqrt(3),label='Sigma C12')
plt.plot(w_cr_array_C12, Regression_sigma(w_cr_array_C12,sigma0_C12,sigma1_C12))

plt.legend()
plt.show()

#Kombination aller Listen damit eine Optimierung global erfolgen kann
strain_max_list_all = max_list_C3 + max_list_C9 + max_list_C12
w_list_all = w_cr_array_C3 + w_cr_array_C9 + w_cr_array_C12
WP_all = mittel_Wendepunkt_C3+mittel_Wendepunkt_C9+mittel_Wendepunkt_C12

#initialer Parameter für gamma
initial_params = [24.6]
result = result = minimize(objective, initial_params, args=(w_list_all, strain_max_list_all))

print('Gamma=',result.x[0])
g = result.x[0]

#initiale Parameter für sigma0 und sigma1
initial_params = [0.009, -0.0001]
result = minimize(objective_WP, initial_params, args=(w_list_all, WP_all))
sigma0, sigma1 = result.x
print('sigma0=',sigma0)
print('sigma1=',sigma1)

#initioale parameter für nur ein sigma
print('mit nur einem sigma')
initial_params = [0.009]
result = minimize(objective_WP_single,initial_params, args=(w_list_all, WP_all))
sigma = result.x
print('sigma=',sigma)

#Multiplot um die verschiedenen Risse zu verschiedenen Verschiebungszeitpunkten darzustellen
#schwarze Funktion ist mit sigma0 und sigma1
#rote Funktion hat nur die beiden Parameter gamma und sigma
fig, axs = plt.subplots(3,3)

axs[0,0].plot(x_model_C3, lorentz(x_model_C3,g,sigma0,sigma1,w_cr_array_C3[1]),'k')
axs[0,0].plot(x_model_C3, lorentz_single(x_model_C3,g,sigma,w_cr_array_C3[1]),'r')
axs[0,0].scatter(x_DFOS_C3,y_DFOS_C3[1])

axs[0,1].plot(x_model_C9, lorentz(x_model_C9,g,sigma0,sigma1,w_cr_array_C9[1]),'k')
axs[0,1].plot(x_model_C9, lorentz_single(x_model_C9,g,sigma,w_cr_array_C9[1]),'r')
axs[0,1].scatter(x_DFOS_C9,y_DFOS_C9[1])

axs[0,2].plot(x_model_C12, lorentz(x_model_C12,g,sigma0,sigma1,w_cr_array_C12[1]),'k')
axs[0,2].plot(x_model_C12, lorentz_single(x_model_C12,g,sigma,w_cr_array_C12[1]),'r')
axs[0,2].scatter(x_DFOS_C12,y_DFOS[1])

axs[1,0].plot(x_model_C3, lorentz(x_model_C3,g,sigma0,sigma1,w_cr_array_C3[7]),'k')
axs[1,0].plot(x_model_C3, lorentz_single(x_model_C3,g,sigma,w_cr_array_C3[7]),'r')
axs[1,0].scatter(x_DFOS_C3,y_DFOS_C3[7])

axs[1,1].plot(x_model_C9, lorentz(x_model_C9,g,sigma0,sigma1,w_cr_array_C9[7]),'k')
axs[1,1].plot(x_model_C9, lorentz_single(x_model_C9,g,sigma,w_cr_array_C9[7]),'r')
axs[1,1].scatter(x_DFOS_C9,y_DFOS_C9[7])

axs[1,2].plot(x_model_C12, lorentz(x_model_C12,g,sigma0,sigma1,w_cr_array_C12[7]),'k')
axs[1,2].plot(x_model_C12, lorentz_single(x_model_C12,g,sigma,w_cr_array_C12[7]),'r')
axs[1,2].scatter(x_DFOS_C12,y_DFOS[7])

axs[2,0].plot(x_model_C3, lorentz(x_model_C3,g,sigma0,sigma1,w_cr_array_C12[11]),'k')
axs[2,0].plot(x_model_C3, lorentz_single(x_model_C3,g,sigma,w_cr_array_C3[11]),'r')
axs[2,0].scatter(x_DFOS_C3,y_DFOS_C3[11])

axs[2,1].plot(x_model_C9, lorentz(x_model_C9,g,sigma0,sigma1,w_cr_array_C9[11]),'k')
axs[2,1].plot(x_model_C9, lorentz_single(x_model_C9,g,sigma,w_cr_array_C9[11]),'r')
axs[2,1].scatter(x_DFOS_C9,y_DFOS_C9[11])

axs[2,2].plot(x_model_C12, lorentz(x_model_C12,g,sigma0,sigma1,w_cr_array_C12[11]),'k')
axs[2,2].plot(x_model_C12, lorentz_single(x_model_C12,g,sigma,w_cr_array_C12[11]),'r')
axs[2,2].scatter(x_DFOS_C12,y_DFOS[11])

plt.show()


POS = -0.51
w_cr_array_C3 = [19.669451388888874, 48.233372777777745, 80.78568333333328, 115.46603972222213, 150.30286916666654,
                 184.53831499999987, 218.73831055555536, 253.81509111111086, 287.91341222222195, 322.43429111111084,
                 357.71768590277736, 402.8620627777773]

POS = 0.25
w_cr_array_C9 = [12.746289345345351, 32.35402805555557, 58.232954722222225, 89.09878972222225, 121.80794166666666,
                 153.00820527777776, 184.5041462177177, 215.76272015765764, 247.2092651576577, 276.27144999999996,
                 305.6654791666669, 323.5282844444444]

POS = 0.67
w_cr_array_C12 = [26.151312777777783, 56.88582611111112, 92.34463333333338, 130.0315105555556, 168.98190833333337,
              209.0284805555556, 248.09519944444452, 285.26361138888893, 322.0303594444445, 357.5927802777779,
              391.4908075000001, 412.21442166666674]

