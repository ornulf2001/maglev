import do_mpc
from casadi import *
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt  # Fix: matplotlib import corrected

####################### Model Setup #############################

model_type = "continuous"
model = do_mpc.model.Model(model_type)

# Define model parameters (for spring-mass-damper)
d = 0.1  # Damping coefficient
m = 1    # Mass
k = 2    # Spring constant

# Define state variables
theta = model.set_variable("_x", "theta")
omega = model.set_variable("_x", "omega")

# Define input
u = model.set_variable("_u", "u")
u_value = 0

# Define dynamic equations (spring-mass-damper, right side of equation using Newton's laws)
model.set_rhs('theta', omega)
model.set_rhs("omega", -k/m*theta - d/m*omega + u/m)


# Define measurement model
model.set_meas("theta_meas", theta)
model.set_meas("omega_meas", omega) 
model.setup()

######################## MHE Setup ##############################
mhe = do_mpc.estimator.MHE(model)

N = 1000
dt = 0.1

mhe.settings.n_horizon=N
mhe.settings.t_step=dt
mhe.settings.check_for_mandatory_settings()

#mhe.set_param(n_horizon=N)
#mhe.set_param(t_step=dt)
#mhe.set_param(store_full_solution=True)  # Store full solution history

# Set weight matrices
P_x = np.eye(2)
P_v = np.eye(2)
P_p = np.eye(0)

mhe.set_default_objective(P_x, P_v, P_p)


###################### Measurement Setup ################################
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'measured_rotation.csv')
data = pd.read_csv(filename)

time = data.iloc[:, 0].values  # Time steps
theta_meas = data.iloc[:, 1].values  # Measured rotation angle from offline dataset

# Define the measurement function
def measurement_function(x):
    return  x # Measurement is just the rotation angle (first state)

# Set the measurement function
mhe.set_y_fun(measurement_function)

# Now setup the MHE (important to do before the loop)
mhe.setup()

# Pass measurements into the MHE estimator
for t, theta in zip(time, theta_meas):
    y_meas = np.array([theta])
    mhe.make_step(y_meas)  # Feed measurements into MHE

########################## Run Simulation  ###########################
simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step=dt)
simulator.setup()

steps = N - 1

# Simulate the system over time
for i in range(steps):
    u0 = np.array([u_value]).reshape(1, 1)
    simulator.make_step(u0)




######################### Plotting the estimates #######################
theta_estimates = []
omega_estimates = []

for i in range(steps):
    estimate = mhe.data.get_estimate()  # Get the current state estimate
    theta_estimates.append(estimate[0])  # First state: theta (rotation angle)
    omega_estimates.append(estimate[1])  # Second state: omega (angular velocity)

plt.figure(figsize=(10, 5))

# Plot estimated rotation angle (theta)
plt.subplot(2, 1, 1)
plt.plot(time, theta_estimates, label='Estimated Theta (Rotation Angle)')
plt.xlabel('Time [s]')
plt.ylabel('Theta [rad]')
plt.legend()

# Plot estimated angular velocity (omega)
plt.subplot(2, 1, 2)
plt.plot(time, omega_estimates, label='Estimated Omega (Angular Velocity)', color='orange')
plt.xlabel('Time [s]')
plt.ylabel('Omega [rad/s]')
plt.legend()

plt.tight_layout()
plt.show()
