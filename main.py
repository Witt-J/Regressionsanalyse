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
y_DFOS_half = df['strain_u2']

# x und y spiegeln
x_DFOS_mirr = x_DFOS_half[::-1]
y_DFOS_mirr = y_DFOS_half[::-1]

x_DFOS = np.append(x_DFOS_half, x_DFOS_mirr * -1)
y_DFOS = np.append(y_DFOS_half, y_DFOS_mirr)

# print(x_DFOS)
# print(y_DFOS)

#plt.plot(x_DFOS, y_DFOS, 'go--', label="original")

#plt.show()


# Approximation der Dehnungsspitze mit Gauß-Funktion
# Beispiel siehe: https://github.com/lukepolson/youtube_channel/blob/main/Python%20Tutorial%20Series/curvefit1.ipynb
def gauss_f(x, y, A, mu, sig):
    return y + A * np.exp(-(x - mu) ** 2 / sig ** 2) #todo z does not make any difference here when using curve fit

#todo Double gauss with addition
def gauss_double_f(x, y, A1, mu1, sig1, A2, mu2, sig2):
    return (y + (A1 * np.exp(-(x - mu1) ** 2 / sig1 ** 2)) + (A2 * np.exp(-(x - mu2) ** 2 / sig2 ** 2)))

def gauss_poly2_f(x, y, A, mu, sig, p1, p2, p3):
    return (y + A * np.exp(-(x - mu) ** 2 / sig ** 2)) + (p1+p2*x+p3*x**2)

def exp_f(x, a, b, c, d):
    return np.exp(a*x+b)*c + d

def gauss_poly4_f(x, y, A, mu, sig, a4, b4, c4):
    return (y + A * np.exp(-(x - mu) ** 2 / sig ** 2)) + (a4 + b4 * x ** 2 + c4 * x**4)

def gauss_multi_poly4_f(x, y, A, mu, sig, am4, bm4, cm4):
    return (y + A * np.exp(-(x - mu) ** 2 / sig ** 2)) * (am4 + bm4 * x ** 2 + cm4 * x**4)

def laplace_f(x, sig, mu, a, b, c):
    return ((1/(2*sig))*a*np.exp(-(np.abs(x-mu)*b)/sig)) + c

def cauchy_pdf(x, gamma, a, b, c):
    return 1.0 / (np.pi * gamma * a * (1 + (b*(x) / gamma)**2)) +c

#todo look into spline fitting


# x_data = x_stuff[(x_stuff>725) & (x_stuff<850)] # die Regressionsfunktion kann auf einen bestimmten Bereich angewendet werden
# y_data = y_stuff[(x_stuff>725) & (x_stuff<850)]

popt, pcov = curve_fit(gauss_f, x_DFOS, y_DFOS, p0=[100, 9400, 0, 0.015])

#double gauss
popt2, pcov2 = curve_fit(gauss_double_f, x_DFOS, y_DFOS, p0=[1, 10, 0, 0.015, 10, 0, 1])

#gauss poly2
popt3, pcov3 = curve_fit(gauss_poly2_f, x_DFOS, y_DFOS, p0=[1, 10, 0, 0.015, 1 ,1 ,1])

#gauss poly4
popt4, pcov4 = curve_fit(gauss_poly4_f, x_DFOS, y_DFOS, p0=[1, 10, 0, 0.015, 1 ,1 ,1])

#gauss multi poly4
popt5, pcov5 = curve_fit(gauss_multi_poly4_f, x_DFOS, y_DFOS, p0=[1, 10, 0, 0.015, 1 ,1 ,1])

#laplace
popt6, pcov6 = curve_fit(laplace_f, x_DFOS, y_DFOS, p0=[7.466458104581132e-05, 0, 0.13664100265080145, 0.003925559390623247, -3.6492848974456256])

popt7, pcov7 = curve_fit(cauchy_pdf, x_DFOS, y_DFOS, p0=[ 7.5e-03, 0.13664100265080145, 0.003925559390623247, -3.6492848974456256])

#find data maximum and its corresponding index
#slice array to split it at its maximum in order to properly fit and exp function
max_index, max_value = max(enumerate(y_DFOS), key=lambda x: x[1])
y_data_exp = y_DFOS[1:max_index+1]
x_data_exp = x_DFOS[1:max_index+1]

#fitting optimazation for exp function
expopt, expcov = curve_fit(exp_f, x_data_exp, y_data_exp, p0=[1, 1, 1, 1])
#gathering the optimized parameters
a_opt, b_opt, c_opt, d_opt = expopt
#filling the exp function with values using opt parameters
x_data_exp_fine = np.linspace(min(x_data_exp), max(x_data_exp),100)
y_data_exp_fine = exp_f(x_data_exp_fine, a_opt, b_opt, c_opt, d_opt)
#for i in range(len(y_data_exp_fine)):
#    if y_data_exp_fine[i]>max_value:
#        y_data_exp_fine[i]=max_value
#PLOT

#print(popt)  # popt contains the values for A, mu and sig

y_opt, A_opt, mu_opt, sig_opt = popt
y_double_opt, A1_opt, mu1_opt, sig1_opt, A2_opt, mu2_opt, sig2_opt =popt2
y_gausspoly2, A_gausspoly2, mu_gausspoly2, sig_gausspoly2, p1_gausspoly2, p2_gausspoly2, p3_gausspoly2 = popt3
y_gausspoly4, A_gausspoly4, mu_gausspoly4, sig_gausspoly4, p1_gausspoly4, p2_gausspoly4, p3_gausspoly4 = popt4
y_gmp4, A_gmp4, mu_gmp4, sig_gmp4, am4, bm4, cm4 = popt5
sig_lp, mu_lp, a_lp, b_lp, c_lp = popt6
gamma_c, a_c, b_c, c_c = popt7

###TRYOUT ZONE MAGENTA
#A_opt_2 = 1.1 * A_opt
#sig_opt_2 = 1.1 * sig_opt
###TRYOUT ZONE

x_model = np.linspace(min(x_DFOS), max(x_DFOS), 1000)
y_model = gauss_f(x_model, y_opt, A_opt, mu_opt, sig_opt)
#y_model_2 = gauss_f(x_model, y_opt, A_opt_2, mu_opt, sig_opt_2)
y_model_double = gauss_double_f(x_model, y_double_opt, A1_opt, mu1_opt, sig1_opt, A2_opt, mu2_opt, sig2_opt)
y_model_gausspoly2 = gauss_poly2_f(x_model,y_gausspoly2,A_gausspoly2,mu_gausspoly2,sig_gausspoly2, p1_gausspoly2, p2_gausspoly2, p3_gausspoly2)
y_m_gp4 = gauss_poly4_f(x_model,y_gausspoly4,A_gausspoly4,mu_gausspoly4,sig_gausspoly4, p1_gausspoly4, p2_gausspoly4, p3_gausspoly4)
y_m_gmp4 = gauss_multi_poly4_f(x_model,y_gmp4, A_gmp4, mu_gmp4, sig_gmp4, am4, bm4, cm4)
y_cauchy = cauchy_pdf(x_model, gamma_c, a_c, b_c, c_c)

plt.scatter(x_DFOS, y_DFOS)
plt.plot(x_model,y_model , color='grey')
plt.plot(x_model,y_cauchy , color='red')
plt.show()


# Extract x and y values from dataframe
w_cr = [13.4019275, 32.5236925, 58.6297725, 89.3305725, 121.855565, 152.95553, 185.6038275, 216.53853, 248.827085, 280.6140025, 312.583635, 330.2024375, 343.39955, 368.0849575]

a_analyse = []
b_analyse = []
c_analyse = []
sig_analyse = []
y_lp = []

def function1(x, y, i):
    return w_cr[i]

for i in range(14):
    y_DFOS_half = df['strain_u'+str(i+1)]

    # x und y spiegeln
    y_DFOS_mirr = y_DFOS_half[::-1]

    y_DFOS = []
    y_DFOS = np.append(y_DFOS_half, y_DFOS_mirr)

    popt, pcov = curve_fit(laplace_f, x_DFOS, y_DFOS, p0=[0.00455, 0, 3.1, 0.3, 1])
    sig, mu, a, b, c = popt
    a_analyse.append(a)
    b_analyse.append(b)
    c_analyse.append(c)
    sig_analyse.append(sig)
    y_lp.append(laplace_f(x_model, sig, mu, a, b, c))

    z_real = function1(x_DFOS, y_DFOS, i)
    z_virtual = function1(x_model, y_lp, i)

    #PLOT
    #plt.scatter(x_DFOS, y_DFOS)
    #plt.plot(x_model,y_lp[i], color='k')
    #plt.show()




print(a_analyse)
print(b_analyse)
print(c_analyse)
print(sig_analyse)
print(w_cr)

plt.scatter(w_cr, a_analyse)
plt.show()
plt.scatter(w_cr, b_analyse)
plt.show()
plt.scatter(w_cr, c_analyse)
plt.show()
plt.scatter(w_cr, sig_analyse)
plt.show()

#define the exp function

"""
#find the max gradient of data
y_m_lp = laplace_f(x_model, sig_lp, mu_lp, a_lp, b_lp, c_lp)
g = np.gradient(y_DFOS)
index_g, value_g = max(enumerate(g), key=lambda x: x[1])
index_max, value_max = max(enumerate(y_DFOS), key=lambda x: x[1])
x_g = x_DFOS[index_g]
x_max = x_DFOS[index_max]

#seperate the data
x_exp1 = x_DFOS[0:index_g+1]
y_exp1 = y_DFOS[0:index_g+1]
x_exp2 = x_DFOS[index_g:index_max+1]
y_exp2 = y_DFOS[index_g:index_max+1]

def exp_f_1(x, a, b, c):
    return (x-x_g)*np.exp(a*x+b)*c + y_exp1[index_g]

def exp_f_2(x, a, b, c):
    return (x-x_g)*np.exp(a*x+b)*c + y_exp1[index_g]

def exp_f_11(x, a, b, c, d):
    return np.exp(a*x+b)*c + d

def exp_f_22(x, a, b, c, d):
    return np.exp(a*x+b)*c + d

#optimzizing
expopt1, expcov1 = curve_fit(exp_f_1, x_exp1, y_exp1, p0=[1, 1, 1])
expopt2, expcov2 = curve_fit(exp_f_2, x_exp2, y_exp2, p0=[1, 1, 1])

expopt11, expcov11 = curve_fit(exp_f_11, x_exp1, y_exp1, p0=[1, 1, 1, 1])
expopt22, expcov22 = curve_fit(exp_f_22, x_exp2, y_exp2, p0=[1, 1, 1, 1])

#parameter of omptimization to variables
a1, b1, c1 = expopt1
a2, b2, c2= expopt2

a11, b11, c11, d11 = expopt11
a22, b22, c22, d22 = expopt22

#create the data
x_virtual1 = np.linspace(min(x_DFOS), x_g, 1000)
x_virtual2 = np.linspace(x_g, x_max, 1000)
y_virtual1 = exp_f_1(x_virtual1, a1, b1, c1)
y_virtual2 = exp_f_2(x_virtual2, a2, b2, c2)

y_virtual11 = exp_f_11(x_virtual1, a11, b11, c11, d11)
y_virtual22 = exp_f_22(x_virtual2, a22, b22, c22, d22)


#are the max gradient points on the same x_value
gradient = []
for i in range(14):
    y_DFOS_half = df['strain_u'+str(i+1)]
    g = np.gradient(y_DFOS_half)
    index_g, value_g = max(enumerate(g), key=lambda x: x[1])
    x_g = x_DFOS[index_g]
    gradient.append(x_g)

print(gradient)
"""

"""
#####using minimize instead of curve_fit in order to get constraints
#find the max gradient of data
y_m_lp = laplace_f(x_model, sig_lp, mu_lp, a_lp, b_lp, c_lp)
g = np.gradient(y_DFOS)
index_g, value_g = max(enumerate(g), key=lambda x: x[1])
index_max, value_max = max(enumerate(y_DFOS), key=lambda x: x[1])
x_g = x_DFOS[index_g]
y_g = y_DFOS[index_g]
x_max = x_DFOS[index_max]

#seperate the data
x_exp1 = x_DFOS[0:index_g+1]
y_exp1 = y_DFOS[0:index_g+1]
x_exp2 = x_DFOS[index_g:index_max+1]
y_exp2 = y_DFOS[index_g:index_max+1]

# Definition der Funktionen mit diskreten Randbedingungen
def function1(params1, x, y):
    a, b, c, d = params1
    predictions = np.exp(a*x+b)*c + d  #definition of the function
    error = np.sum((predictions - y)**2)
    return error

#Exponential Funktion
def exponential1(x, params1):
    a, b, c, d = params1
    return np.exp(a*x+b)*c + d

# Ableitung der Exponentialfunktion
def exponential_derivative1(x, params1):
    a, b, c, d = params1
    return a * c * np.exp(a * x +b)

def function2(params2, x, y):
    a, b, c, d = params2
    predictions = np.exp(a*x+b)*c + d  #definition of the function
    error = np.sum((predictions - y)**2)
    return error

#Exponential Funktion
def exponential2(x, params2):
    a, b, c, d = params2
    return np.exp(a*x+b)*c + d

# Ableitung der Exponentialfunktion
def exponential_derivative2(x, params2):
    a, b, c, d = params2
    return a * c * np.exp(a * x +b)

# Anfangswerte für die Optimierung
initial_params1 = [60, 8, 0.7, 40]
initial_params2 = [-90, 1, -15, value_max]

# Festlegen der Randbedingungen
constraint1 = {
    'type': 'eq',
    'fun': lambda params1: exponential_derivative1(x_g, params1) - value_g, #Anstieg WP
    'fun': lambda params1: exponential1(x_g, params1) - y_g,                #Position WP
    'fun': lambda params1: exponential1(x_exp1[0], params1) - y_exp1[0]     #erster Wert y übereinstimmung
}
constraint2 = {
    'type': 'eq',
    'fun': lambda params2: exponential_derivative2(x_g, params2) - value_g,
    'fun': lambda params2: exponential2(x_g, params2) - y_g,
    'fun': lambda params2: exponential1(x_max, params2) - value_max,
    'fun': lambda params2: exponential_derivative2(x_max, params2) - 10,
}


# Optimierung
result1 = minimize(function1, initial_params1, args=(x_exp1, y_exp1), constraints=constraint1)
result2 = minimize(function2, initial_params2, args=(x_exp2, y_exp2), constraints=constraint2)


# Ausgabe der Ergebnisse
optimized_params1 = result1.x
optimized_params2 = result2.x


# Verwenden Sie die optimierten Parameter, um die Funktion zu erstellen
def optimized_function1(x):
    return optimized_params1[2] * np.exp(optimized_params1[0] * x+optimized_params1[1]) + optimized_params1[3]

def optimized_function2(x):
    return optimized_params2[2] * np.exp(optimized_params2[0] * x+optimized_params2[1]) + optimized_params2[3]

# Beispiel: Vorhersage für neue Daten
predictions1 = optimized_function1(x_exp1)
predictions2 = optimized_function2(x_exp2)
"""



#####PLOT#####

#plt.scatter(x_DFOS, y_DFOS)


#plt.plot(x_model, y_model, color='r')
#plt.plot(x_model, y_model_2, color='m')
#plt.plot(x_data_exp_fine,y_data_exp_fine, color='k')
#plt.plot(x_model,y_model_double, color='darkred')
#plt.plot(x_model,y_model_gausspoly2, color='dodgerblue')
#plt.plot(x_model,y_m_gp4  , color='gold')
#plt.plot(x_model,y_m_lp, color='darkred')
#plt.plot(x_virtual1,y_virtual1, color='gold')
#plt.plot(x_virtual2,y_virtual2, color='gold')
#plt.plot(x_virtual1,y_virtual11, color='r')
#plt.plot(x_virtual2,y_virtual22, color='r')
#plt.plot(x_exp1,predictions1, color='m')
#plt.plot(x_exp2,predictions2, color='m')

#plt.plot(x_model,y_m_gmp4 , color='lime')


#plt.show()
"""
# Shapiro-Wilk-Test --> sind die Daten normalverteilt?
stat, p_value = shapiro(y_DFOS)

# Interpretation des Ergebnisses
print(f"Statistik: {stat}, p-Wert: {p_value}")
if p_value > 0.05:
    print("Die Daten könnten normalverteilt sein.")
else:
    print("Die Daten sind wahrscheinlich nicht normalverteilt.")
"""
# ggf. ist die t-Verteilung besser geeignet? https://www.geo.fu-berlin.de/en/v/soga-py/Basics-of-statistics/Continous-Random-Variables/Students-t-Distribution/Students-t-Distribution-in-Python/index.html


# create polynomial fit
"""
for i in range(1, 14):
    coeffs = np.polyfit(x_DFOS, y_DFOS, i)  # calculate fit
    yyr = np.polyval(coeffs, x_DFOS)  # evaluate the function

    plt.plot(x_DFOS, y_DFOS, 'go--', label="original")
    # plt.plot(x_cr5_dict[14], yy_noisy,'k.', label="noisy data")
    plt.plot(x_DFOS, yyr, 'r-', label="regression_{}".format(i))

    #plt.legend()
    # plt.savefig("regression.png")
    #plt.show()
#plt.show()
"""
# print(coeffs)


# coeffs = np.polyfit(x_DFOS, y_DFOS, 13)  # calculate fit
# yyr    = np.polyval(coeffs, x_DFOS)  # evaluate the function

# plt.plot(x_DFOS, y_DFOS, 'go--', label="original")
# #plt.plot(x_cr5_dict[14], yy_noisy,'k.', label="noisy data")
# plt.plot(x_DFOS, yyr,'r-', label="regression")

'''
plt.legend()
plt.savefig("regression.png")
plt.show()
print(coeffs)
'''

# #Approximation der Dehnungsspitze mit Gauß-Funktion
# def gaussian(x, mu, sig):
#     return (
#         1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
#     )

# x_gauß = np.arange(-0.1, 0.1, 0.00065)

# mu = 0
# sig = 0.015

# y_gauß = 350*gaussian(x_gauß, mu, sig)

# #Plot
# plt.plot(x_DFOS, y_DFOS, label="strain")
# plt.plot(x_gauß, y_gauß, label = "Gauß") # x_gauß, mu, sig
# plt.legend()
# plt.show()

# #Dehnungsdifferenzen
# #y_DFOS_gradient = np.gradient(x_DFOS, y_DFOS)
# y_DFOS_diff = np.diff(y_DFOS)
# x_DFOS_diff = [(x_DFOS[i] + x_DFOS[i+1]) / 2 for i in range(len(x_DFOS)-1)]
# y_gauß_diff = np.gradient(y_gauß)
# x_gauß_diff = [(x_gauß[i] + x_gauß[i+1]) / 2 for i in range(len(x_gauß)-1)]


# plt.plot(x_DFOS_diff, y_DFOS_diff, label = "DFOSdiff")
# plt.plot(x_gauß, y_gauß_diff, label = "Gaußdiff")

# plt.legend()
# plt.show()

# print("x_measurement: ", x_DFOS)
# print("y_measurement: ", y_DFOS)

# for mu, sig in [(-1, 1), (0, 2), (2, 3)]:
#   mp.plot(x_gauß, gaussian(x_gauß, mu, sig))


# # Define some test data which is close to Gaussian
# data = numpy.random.normal(size=10000)

# hist, bin_edges = numpy.histogram(data, density=True)
# bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

# # Define model function to be used to fit to the data above:
# def gauss(x, *p):
# ,,A, mu, sigma = p
# ,,return A*numpy.exp(-(x-mu)**2/(2.*sigma**2))

# # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
# p0 = [1., 0., 1.]

# coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)

# # Get the fitted curve
# hist_fit = gauss(bin_centres, *coeff)

# plt.plot(bin_centres, hist, label='Test data')
# plt.plot(bin_centres, hist_fit, label='Fitted data')

# # Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
# print('Fitted mean = ', coeff[1])
# print('Fitted standard deviation = ', coeff[2])

# plt.show()


