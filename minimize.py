import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Beispiel Daten
x_data = np.linspace(-5, 5, 100)
y_data = np.linspace(-5, 5, 100)
x_data, y_data = np.meshgrid(x_data, y_data)
z_data_true = np.sin(np.sqrt(x_data**2 + y_data**2))

# Hinzufügen von Rauschen zu den Daten
noise = 0.1 * np.random.normal(size=z_data_true.shape)
z_data_noisy = z_data_true + noise

# Definition der zu optimierenden Funktion
def objective(params, x, y, z_true):
    a, b, c = params
    z_predicted = a * np.sin(b * np.sqrt(x**2 + y**2)) + c
    error = np.sum((z_predicted - z_true)**2)
    return error

# Anfangswerte für die Optimierung
initial_params = [1, 1, 0]

# Optimierung
result = minimize(objective, initial_params, args=(x_data, y_data, z_data_noisy), method='L-BFGS-B')

# Ausgabe der optimierten Parameter
optimized_params = result.x
print("Optimierte Parameter:", optimized_params)

# Verwenden Sie die optimierten Parameter, um die Funktion zu erstellen
def optimized_function(x, y):
    a, b, c = optimized_params
    return a * np.sin(b * np.sqrt(x**2 + y**2)) + c

# Beispiel: Vorhersage für neue Daten
new_x_data = np.linspace(-5, 5, 100)
new_y_data = np.linspace(-5, 5, 100)
new_x_data, new_y_data = np.meshgrid(new_x_data, new_y_data)
predictions = optimized_function(new_x_data, new_y_data)

# Plot der Daten und der optimierten Funktion
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_data, y_data, z_data_noisy, label='Noisy Data')
ax.plot_surface(new_x_data, new_y_data, predictions, alpha=0.5, label='Optimized Function', cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
plt.show()
