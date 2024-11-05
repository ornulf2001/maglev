# Written by Ørnulf Damsgaard, NTNU - ITK


#Changelog:
#18:22 - 21.10.2024 - This still doesnt work. I have no idea how the measurement function is supposed to be implemented.
#                     It gets treaded as an int for some reason, which causes problems. idk what data type its supposed to be

#16:20 - 22.10.2024 - Still doesnt work, but i have learned some new stuff using another example. Trying to implement
#19:10 - 24.10.2024 - Something is working here now. Weird behaviour from omega when i bound the parameters. Unbounded 
#                     parameters give ok omega estimate but sinusoidal parameters (also negative)
#17:51 - 05.11.2024 - I tried disabling the estimation of msd parameters, and instead setting them to fixed values. Added offset for theta to 
#                     match oscillations better with measurements. Also disabled some simulation stuff that isnt needed, and disabled plots 
#                     for parameters. Also cleaned up comments and unused code.

import do_mpc
from casadi import *
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt  


####################### Model Setup #############################
model_type = "continuous"
model = do_mpc.model.Model(model_type)

# Define state variables
theta = model.set_variable("_x", "theta")
omega = model.set_variable("_x", "omega")

# Param values that work good: [k,m,d] = [1.4, 1, 0.005]
k_param = 1.4
m_param = 1
d_param = 0.005
offset=54 # theta offset to match oscillations to data

# Define dynamic equations (mass-spring-damper, right side of equation using Newton's laws)
model.set_rhs('theta', omega, process_noise=True)
model.set_rhs("omega", -k_param/m_param*(theta-offset) - d_param/m_param*omega, process_noise=True)


# Define measurement model, measure only position
model.set_meas("theta_meas", theta, meas_noise=True)
model.setup()



######################## MHE Setup ##############################
mhe = do_mpc.estimator.MHE(model)   #,["k_param","m_param","d_param"])
N = 15
dt = 0.1

#MHE settings
mhe.settings.n_horizon=N
mhe.settings.t_step=dt
mhe.settings.meas_from_data=True
mhe.settings.check_for_mandatory_settings()

# Set weight matrices
P_x = 20*np.eye(2)
P_p = None # No parameter estimation so no need for weights on them
P_v=np.array([[10]])
P_w=np.diag(np.array([1,10]))
mhe.set_default_objective(P_x,P_v,P_p,P_w)

# Below are weights that have worked ok before (Px, Pp, Pv, Pw)
#   [PARAMS ESTIMATED] These weights give ok theta and kinda ok omega, but messy parameters when unbounded: (10, 5, 5, [10,50])
#   [PARAMS ESTIMATED] With bounded parameters, these values give ok theta but spikey omega, also unrealistic periodic parameter spikes: (5, 5, 1, [10,50])
#   [PARAMS FIXED] These look good, maybe need more fine tuning of k and d to match amplitude and decay: (20, None, 10, [1,10])

# MHE Bounds
mhe.bounds["lower","_x","omega"]=-200
mhe.bounds["upper","_x","omega"]=200
mhe.setup()


###################### Measurement Setup ################################
#Load data set
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'measured_rotation.csv') 
data = pd.read_csv(filename)

n_data = len(data)
used_data_ratio = 0.3 # Ratio of total dataset used. Can use less data for debugging to reduce computation time
time = data.iloc[:int(n_data*used_data_ratio), 0].values  # Time steps
theta_meas = data.iloc[:int(n_data*used_data_ratio), 1].values  # Measured rotation angle from offline dataset

########################## Run MHE  ###########################
# #Initial guess
x0=np.array([[70],[-2.4]])
x0_mhe=x0*(1+0.02*np.random.randn(1,1))
mhe.x0 = x0_mhe
mhe.set_initial_guess()

# Pass measurements into the MHE estimator
y_meas_list = [] # for plotting gt
for theta in theta_meas:
    y_meas = np.array([[theta]])
    y_meas_list.append(theta)
    mhe.make_step(y_meas)  # Feed measurements into MHE

#Save the MHE data to a CSV file, Data columns: (0,1,y) = ("Theta est.", "Omega est.", "Thea meas.")
data_x = pd.DataFrame(mhe.data["_x"]) 
data_x["y"] = y_meas_list
data_x.to_csv("mhe_states.csv", index=False)




#################### Plotting ############################
mhe_graphics = do_mpc.graphics.Graphics(mhe.data)
fig, ax = plt.subplots(2, sharex=True, figsize=(8,4))
fig.align_ylabels()

#Plotting theta and omega vs time
ax[0].plot(mhe.data["_time"], y_meas_list, label='gt', linestyle='--', color='green')
mhe_graphics.add_line(var_type='_x', var_name='theta', axis=ax[0], label='MHE estimated theta', color="red")
mhe_graphics.add_line(var_type='_x', var_name='omega', axis=ax[1], label='MHE estimated omega',color="orange")

# Adding axis labels
ax[0].set_ylabel("theta")
ax[1].set_ylabel("omega")
ax[1].set_xlabel('time [s]')


# Adding legends
lines_labels = [ax[0].get_legend_handles_labels(), ax[1].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
combined_labels = ['gt', 'MHE Est. Theta', "MHE Est. Omega"]
fig.legend(lines[:3], combined_labels, loc='upper center', ncol=2)

# idk what this does
mhe_graphics.reset_axes()

# Mark the time after a full horizon is available to the MHE.
ax[0].axvline(mhe.settings.t_step * mhe.settings.n_horizon)
ax[1].axvline(mhe.settings.t_step * mhe.settings.n_horizon)

# Show the figure:
plt.show()
