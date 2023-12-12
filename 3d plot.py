import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the 2D functions
def function1(x, y):
    return np.sin(np.sqrt(x**2 + y**2))

def function2(x, y):
    return np.cos(np.sqrt(x**2 + y**2))

def function3(x, y):
    return np.exp(-(x**2 + y**2)/10)

# Create a grid of points
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)

# Evaluate the functions at each point
z1 = function1(x, y)
z2 = function2(x, y)
z3 = function3(x, y)

# Plot the functions in 3D with different constant z values
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, z1, cmap='viridis', alpha=0.7, label='Function 1', zdir='z', offset=0)
ax.plot_surface(x, y, z2, cmap='plasma', alpha=0.7, label='Function 2', zdir='z', offset=5)
ax.plot_surface(x, y, z3, cmap='cividis', alpha=0.7, label='Function 3', zdir='z', offset=10)

# Customize the plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Show the plot
plt.show()
