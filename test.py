import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
F0 = 10.0   # arbitrary units 
m = 1.0     # mass
g = 9.81    # gravity
x0 = 1.0    # arbitrary distance
omega = 2.0 # frequency
phi = 0.5   # phase shift
h = 10.0    # height

# Time span
t_span = (0, 10)  # time from 0 to 10 seconds
t_eval = np.linspace(*t_span, 500)  # 500 points for evaluation

# Define the system of ODEs
def equations(t, y):
    z, v = y  # y[0] is z (position), y[1] is v (velocity)
    dzdt = v
    dvdt = (F0 / (h - z)) - (F0 / z) - g + (x0 * omega**2 * phi * h * np.sin(omega * t))
    return [dzdt, dvdt]

# Initial conditions: [z(0), v(0)] where z(0) is initial position, v(0) is initial velocity
z0 = 1.0  # starting position
v0 = 0.0  # starting velocity
initial_conditions = [z0, v0]

# Solve the ODE system
solution = solve_ivp(equations, t_span, initial_conditions, t_eval=t_eval, method='RK45')

# Extract results
z = solution.y[0]  # position over time
v = solution.y[1]  # velocity over time
t = solution.t     # time points

a = (F0 / (h - z)) - (F0 / z) - g + (x0 * omega**2 * phi * h * np.sin(omega * t))

# Plot results
plt.figure(figsize=(12, 6))
 
# Compute acceleration over time using the same equations

# Plot results
plt.figure(figsize=(12, 9))  # Adjust figure size to fit three plots

# Plot position
plt.subplot(3, 1, 1)
plt.plot(t, z, label='Position (z)')
plt.title('Position, Velocity, and Acceleration of the Magnet Over Time')
plt.ylabel('Position (z)')
plt.grid(True)

# Plot velocity
plt.subplot(3, 1, 2)
plt.plot(t, v, label='Velocity (v)', color='r')
plt.ylabel('Velocity (v)')
plt.grid(True)

# Plot acceleration
# plt.subplot(3, 1, 3)
# plt.plot(t, a, label='Acceleration (a)', color='g')
# plt.xlabel('Time (t)')
# plt.ylabel('Acceleration (a)')
# plt.grid(True)

plt.tight_layout()
plt.savefig('/home/yerlan/projects/Magnet/graph_z_v_a.png')
