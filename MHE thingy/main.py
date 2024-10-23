# Written by Ørnulf Damsgaard, NTNU - ITK


#Changelog:
#18:22 - 21.10.2024 - This still doesnt work. I have no idea how the measurement function is supposed to be implemented.
#                     It gets treaded as an int for some reason, which causes problems. idk what data type its supposed to be

#16:20 - 22.10.2024 - Still doesnt work, but i have learned some new stuff using another example. Trying to implement
import do_mpc
from casadi import *
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt  


####################### Model Setup #############################

model_type = "continuous"
model = do_mpc.model.Model(model_type)

# Define model parameters (for spring-mass-damper)


# Define state variables
theta = model.set_variable("_x", "theta")
omega = model.set_variable("_x", "omega")
k_param = model.set_variable("parameter", "k_param")
m_param = model.set_variable("parameter", "m_param")
d_param = model.set_variable("parameter", "d_param")

# Define input
u = model.set_variable("_u", "u")
u_value = 0

# Define dynamic equations (spring-mass-damper, right side of equation using Newton's laws)
model.set_rhs('theta', omega, process_noise=True)
model.set_rhs("omega", -k_param/m_param*theta - d_param/m_param*omega + u/m_param, process_noise=True)


# Define measurement model
model.set_meas("theta_meas", theta, meas_noise=True)
model.setup()

######################## MHE Setup ##############################
mhe = do_mpc.estimator.MHE(model,["k_param","m_param","d_param"])

N = 2
dt = 0.4

mhe.settings.n_horizon=N
mhe.settings.t_step=dt
mhe.settings.meas_from_data=True
mhe.settings.check_for_mandatory_settings()

#mhe.set_param(n_horizon=N)
#mhe.set_param(t_step=dt)
#mhe.set_param(store_full_solution=True)  # Store full solution history

# Set weight matrices
P_x = 0.3*np.eye(2)
P_p = 0.3*np.eye(3)
P_v=np.diag(np.array([0.2]))
P_w=np.diag(np.array([0.2,0.2]))

mhe.set_default_objective(P_x,P_v,P_p,P_w)
mhe.setup()


###################### Measurement Setup ################################
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'measured_rotation.csv')
data = pd.read_csv(filename)

time = data.iloc[:, 0].values  # Time steps
theta_meas = data.iloc[:, 1].values  # Measured rotation angle from offline dataset

# Define the measurement function
#def measurement_function(x):
  #  return  x # Measurement is just the rotation angle (first state)

# Set the measurement function
#mhe.set_y_fun(measurement_function)

# Now setup the MHE (important to do before the loop)



########################## Run Simulation  ###########################
simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step=dt)
p_template_sim = simulator.get_p_template()
def p_fun_sim(t_now):
    p_template_sim['k_param'] = 15
    p_template_sim['m_param'] = 0.3
    p_template_sim['d_param'] = 1
    return p_template_sim
simulator.set_p_fun(p_fun_sim)
simulator.setup()

#Initial guess
x0=np.array([[0.1],[0.1]])
x0_mhe=x0*(1+0.2*np.random.randn(1,1))

simulator.x0=x0
mhe.x0 = x0_mhe
mhe.p_est0["k_param"] = 13
mhe.p_est0["m_param"] = 0.25
mhe.p_est0["d_param"] = 0.9
mhe.set_initial_guess()

# Pass measurements into the MHE estimator
for t, theta in zip(time, theta_meas):
    y_meas = np.array([[theta]])
    x0 = mhe.make_step(y_meas)  # Feed measurements into MHE
#steps = N - 1

# Simulate the system over time
#for i in range(steps):
#    u0 = np.array([u_value]).reshape(1, 1)
#    simulator.make_step(u0)






mhe_graphics = do_mpc.graphics.Graphics(mhe.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)

fig, ax = plt.subplots(3, sharex=True, figsize=(8,4))
fig.align_ylabels()

fig_p, ax_p = plt.subplots(3, figsize=(8,2))

sim_graphics.add_line(var_type='_x', var_name='theta', axis=ax[0], label='Simulated theta')
mhe_graphics.add_line(var_type='_x', var_name='theta', axis=ax[0], label='MHE estimated theta')

sim_graphics.add_line(var_type='_x', var_name='omega', axis=ax[1], label='Simulated omega')
mhe_graphics.add_line(var_type='_x', var_name='omega', axis=ax[1], label='MHE estimated omega')

sim_graphics.add_line(var_type='_u', var_name='u', axis=ax[2], label='Simulated u')
mhe_graphics.add_line(var_type='_u', var_name='u', axis=ax[2], label='MHE estimated u')

# Parameter plot (alpha)
sim_graphics.add_line(var_type='_p', var_name='k_param', axis=ax_p[0], label='Simulated k')
mhe_graphics.add_line(var_type='_p', var_name='k_param', axis=ax_p[0], label='MHE estimated k')
sim_graphics.add_line(var_type='_p', var_name='m_param', axis=ax_p[1], label='Simulated m')
mhe_graphics.add_line(var_type='_p', var_name='m_param', axis=ax_p[1], label='MHE estimated m')
sim_graphics.add_line(var_type='_p', var_name='d_param', axis=ax_p[2], label='Simulated d')
mhe_graphics.add_line(var_type='_p', var_name='d_param', axis=ax_p[2], label='MHE estimated d')



ax[0].set_ylabel('theta')
ax[1].set_ylabel("omega")
ax[2].set_ylabel("motor [?]")
ax[2].set_xlabel('time [s]')

for line_i in sim_graphics.result_lines.full:
    line_i.set_alpha(0.4)
    line_i.set_linewidth(6)

lines_labels = [ax[0].get_legend_handles_labels(), ax[1].get_legend_handles_labels(),ax[2].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

# Adjust the legend for both plots
combined_labels = ['Simulated', 'MHE Estimated']
fig.legend(lines[:2], combined_labels, loc='upper center', ncol=2)
if len(simulator.data['_x']) > 0 and len(mhe.data['_x']) > 0:
    sim_graphics.plot_results()
    mhe_graphics.plot_results()
else:
    print("No data available to plot.")
# Reset the limits on all axes in graphic to show the data.
mhe_graphics.reset_axes()

# Mark the time after a full horizon is available to the MHE.
ax[0].axvline(1)
ax[1].axvline(1)

# Show the figure:
plt.show(block=True)










































def plot():
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
