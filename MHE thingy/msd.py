import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters for the mass-spring-damper system
m = 1.0   # Mass (kg)
k = 10.0  # Spring constant (N/m)
c = 1.0   # Damping coefficient (N*s/m)

# Define the system of ODEs
def mass_spring_damper(t, y):
    x, v = y  # y[0] is position, y[1] is velocity
    dxdt = v
    dvdt = -(c/m)*v - (k/m)*x
    return [dxdt, dvdt]

# Initial conditions
x0 = 1.0  # Initial displacement (m)
v0 = 0.0  # Initial velocity (m/s)

# Time span for the simulation
t_span = (0, 10)            # Time range from 0 to 10 seconds
t_eval = np.linspace(0, 10, 1000)  # Time points where solution is evaluated

# Solve the ODE
sol = solve_ivp(mass_spring_damper, t_span, [x0, v0], t_eval=t_eval)

# Plot the trajectory

plt.rcParams.update({
    'font.size': 14,            # General font size
    'axes.titlesize': 14,       # Title font size
    'axes.labelsize': 14,       # Axes label font size
    'legend.fontsize': 13,      # Legend font size
    'xtick.labelsize': 14,      # X-tick label font size
    'ytick.labelsize': 14       # Y-tick label font size
})
plt.plot(sol.t, sol.y[0], label='Displacement (m)')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.title('Mass-Spring-Damper System')
plt.legend()
plt.grid()
plt.show()
