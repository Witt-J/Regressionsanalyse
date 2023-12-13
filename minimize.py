import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Generate example data
x_data = np.linspace(0, 1, 100)
y_data_true = np.exp(-(x_data - 0.5)**2 / (2 * 0.1**2))+ np.random.normal(0, 0.05, 100) # True Gaussian curve

# Objective function to be minimized (sum of squared differences)
def objective_function(params, x, y_target):
    mu = params[0]
    y_predicted = np.exp(-(x - mu)**2 / (2 * 0.1**2))  # Gaussian curve with fixed amplitude and sigma
    return np.sum((y_predicted - y_target)**2)

# Constraint: value at x=0.5 should be 1
def constraint_function(params, x, y_target):
    mu = params[0]
    y_predicted = np.exp(-(x - mu)**2 / (2 * 0.1**2))  # Gaussian curve with fixed amplitude and sigma
    return y_predicted[x == 0.5] - 1

# Initial guess for the optimization
initial_guess = [0.4]

# Define the constraint
constraint = {'type': 'eq', 'fun': constraint_function, 'args': (x_data, y_data_true)}

# Perform the optimization
result = minimize(objective_function, initial_guess, args=(x_data, y_data_true), constraints=constraint)

# Display the results
optimized_mu = result.x[0]
print("Optimized mu:", optimized_mu)

# Evaluate the optimized Gaussian curve
y_data_optimized = np.exp(-(x_data - optimized_mu)**2 / (2 * 0.1**2))

# Plotting
plt.plot(x_data, y_data_true, label="True Gaussian Curve")
plt.plot(x_data, y_data_optimized, label="Optimized Gaussian Curve", linestyle='dashed', color='red')
plt.scatter([0.5], [1], color='green', marker='o', label="Constraint Point (x=0.5, y=1)")
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Fitting a Gaussian Curve with Constraint")
plt.show()
