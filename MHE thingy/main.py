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
#16.22 - 04.12.2024 - Added code for performance indicators MSE and MAE, as well as timers to calculate step runtimes and total runtimes. 
#                     Also increased plot font sizes etc. 
import do_mpc
from casadi import *
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt 
import seaborn as sns 
import time
start_time=time.time()


####################### Model Setup #############################
model_type = "continuous"
model = do_mpc.model.Model(model_type)

# Define state variables
theta = model.set_variable("_x", "theta")
omega = model.set_variable("_x", "omega")

# Param values that work good: [k,m,d] = [1.4, 1, 0.005]
k_param = 1.4
I_param = 1
d_param = 0.005
offset=54 # theta offset to match oscillations to data

# Define dynamic equations (mass-spring-damper, right side of equation using Newton's laws)
model.set_rhs('theta', omega, process_noise=True)
model.set_rhs("omega", -k_param/I_param*(theta-offset) - d_param/I_param*omega, process_noise=True)


# Define measurement model, measure only position
model.set_meas("theta_meas", theta, meas_noise=True)
model.setup()



######################## MHE Setup ##############################
mhe = do_mpc.estimator.MHE(model)   #,["k_param","I_param","d_param"])
N = 1
dt = 0.033

#MHE settings
mhe.settings.n_horizon=N
mhe.settings.t_step=dt
mhe.settings.meas_from_data=True
mhe.settings.check_for_mandatory_settings()

# Set weight matrices   
Test=8
P_x = 10*np.eye(2)  
P_p = None # No parameter estimation so no need for weights on them
P_v=np.array([[300]])
P_w=np.diag(np.array([3,3]))
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
used_data_ratio = 0.2 # Ratio of total dataset used. Can use less data for debugging to reduce computation time
theta_meas = data.iloc[:int(n_data*used_data_ratio), 1].values  # Measured rotation angle from offline dataset'
########################## Run MHE  ###########################
# #Initial guess
x0=np.array([[70],[-2.4]])
x0_mhe=x0*(1+0.02*np.random.randn(1,1))
mhe.x0 = x0_mhe
mhe.set_initial_guess()

step_runtimes=[]
y_meas_list = [] # for plotting the measurements
# Pass measurements into the MHE estimator
for theta in theta_meas:
    step_start=time.time()
    y_meas = np.array([[theta]])
    y_meas_list.append(theta)
    mhe.make_step(y_meas)  # Feed measurements into MHE
    step_runtime=time.time()-step_start
    step_runtimes.append(step_runtime)

#Save the MHE data to a CSV file, Data columns: (0,1,y) = ("Theta est.", "Omega est.", "Thea meas.")
data_x = pd.DataFrame(mhe.data["_x"]) 
data_x["y"] = y_meas_list
data_x.to_csv(f"MHE tests V2/Test {Test}/mhe_states.csv", index=False)

total_runtime=time.time() - start_time

################### Performance #########################

theta_est= mhe.data["_x"][:,0]

errors = y_meas_list-theta_est
MSE = np.mean(errors**2)
MAE = np.mean(np.abs(errors))


#################### Plotting ############################

plt.rcParams.update({
    'font.size': 14,            # General font size
    'axes.titlesize': 14,       # Title font size
    'axes.labelsize': 14,       # Axes label font size
    'legend.fontsize': 13,      # Legend font size
    'xtick.labelsize': 14,      # X-tick label font size
    'ytick.labelsize': 14       # Y-tick label font size
})


mhe_graphics = do_mpc.graphics.Graphics(mhe.data)
fig, ax = plt.subplots(2, sharex=True, figsize=(8,4))
fig.align_ylabels()

#Plotting theta and omega vs time
ax[0].plot(mhe.data["_time"], y_meas_list, label='Measurement', linestyle='--', color='green')
mhe_graphics.add_line(var_type='_x', var_name='theta', axis=ax[0], label='MHE estimated theta', color="red")
mhe_graphics.add_line(var_type='_x', var_name='omega', axis=ax[1], label='MHE estimated omega',color="orange")

# Adding axis labels
ax[0].set_ylabel("Theta")
ax[1].set_ylabel("Omega")
ax[1].set_xlabel('Time [s]')


# Adding legends
lines_labels = [ax[0].get_legend_handles_labels(), ax[1].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
combined_labels = ['Measurement', 'MHE Est. Theta', "MHE Est. Omega"]
fig.legend(lines[:3], combined_labels, loc='upper center', ncol=2)

# idk what this does
mhe_graphics.reset_axes()

# Mark the time after a full horizon is available to the MHE.
ax[0].axvline(mhe.settings.t_step * mhe.settings.n_horizon)
ax[1].axvline(mhe.settings.t_step * mhe.settings.n_horizon)

#Plotting performance indicators
plt.figure()
sns.histplot(errors, kde=True, bins=20, edgecolor="black")
plt.ylabel("Frequency")
plt.xlabel("Error")
plt.title("Error distribution")
textstr = f"MSE: {MSE:.3f}\nMAE: {MAE:.3f}"
props = dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.5)  # Add padding
plt.text(0.95, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right', bbox=props)

plt.figure()
plt.plot(np.array(step_runtimes)*1000, label="Step Runtime", color="royalblue")
plt.xlabel("Time step")
plt.ylabel("Runtime [ms]")
plt.title("MHE Step Runtimes")
plt.legend(loc="upper right")
textstr = f"Total Runtime: {total_runtime:.3f} [s]"
props = dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.5)  # Add padding
plt.text(0.985, 0.88, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right', bbox=props)

# Show the figures:
plt.show()
