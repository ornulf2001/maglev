import do_mpc
from casadi import *
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt  

model_type = "continuous"
model = do_mpc.model.Model(model_type)

x=model.set_variable("_x", "x")
motor=model.set_variable("_u","motor")
alpha=model.set_variable("parameter","alpha")

x_meas = model.set_meas("x_meas",x, meas_noise=True)      # y=x
motor_meas = model.set_meas("motor_meas", motor, meas_noise=True)

model.set_rhs("x",alpha*x+motor, process_noise=True)                                       # \dot{x} = \alpha*x
model.setup()

mhe = do_mpc.estimator.MHE(model, ['alpha'])

setup_mhe={
    "t_step": 0.1,
    "n_horizon": 15,
    "store_full_solution":True,
    "meas_from_data":True
}
mhe.set_param(**setup_mhe)

P_x=np.eye(1)
P_p=0.01*np.eye(1)
P_v=np.diag(np.array([1,1]))
P_w=np.diag(np.array([1]))
mhe.set_default_objective(P_x,P_v,P_p,P_w)
mhe.setup()

simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step = 0.1)
p_template_sim = simulator.get_p_template()
def p_fun_sim(t_now):
    p_template_sim['alpha'] = -2.25e-2
    return p_template_sim
simulator.set_p_fun(p_fun_sim)
simulator.setup()

x0=np.array([1])
x0_mhe=x0*(1+0.2*np.random.randn(1,1))

simulator.x0=x0
mhe.x0 = x0_mhe
mhe.p_est0 = -0.02
mhe.set_initial_guess()


def random_u(u0):
    # Hold the current value with 80% chance or switch to new random value.
    u_next = (0.5-np.random.rand(1,1))*np.pi # New candidate value.
    switch = np.random.rand() >= 0.8 # switching? 0 or 1.
    u0 = (1-switch)*u0 + switch*u_next # Old or new value.
    return u0

np.random.seed(999) #make it repeatable

u0 = np.zeros((1,1))
for i in range(50):
    u0 = random_u(u0) # Control input
    v0 = 0.1*np.random.randn(model.n_v,1) # measurement noise
    y0 = simulator.make_step(u0, v0=v0)
    x0 = mhe.make_step(y0) # MHE estimation step



mhe_graphics = do_mpc.graphics.Graphics(mhe.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)

fig, ax = plt.subplots(2, sharex=True, figsize=(8,4))
fig.align_ylabels()

fig_p, ax_p = plt.subplots(1, figsize=(8,2))

sim_graphics.add_line(var_type='_x', var_name='x', axis=ax[0], label='Simulated x')
mhe_graphics.add_line(var_type='_x', var_name='x', axis=ax[0], label='MHE estimated x')

sim_graphics.add_line(var_type='_u', var_name='motor', axis=ax[1], label='Simulated motor')
mhe_graphics.add_line(var_type='_u', var_name='motor', axis=ax[1], label='MHE estimated motor')

# Parameter plot (alpha)
sim_graphics.add_line(var_type='_p', var_name='alpha', axis=ax_p, label='Simulated alpha')
mhe_graphics.add_line(var_type='_p', var_name='alpha', axis=ax_p, label='MHE estimated alpha')

ax[0].set_ylabel('pos [m]')
ax[1].set_ylabel("motor [?]")
ax[1].set_xlabel('time [s]')

sim_graphics.result_lines['_x', 'x', 0]
for line_i in sim_graphics.result_lines.full:
    line_i.set_alpha(0.4)
    line_i.set_linewidth(6)

lines_labels = [ax[0].get_legend_handles_labels(), ax[1].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

# Adjust the legend for both plots
combined_labels = ['Simulated', 'MHE Estimated']
fig.legend(lines[:2], combined_labels, loc='upper center', ncol=2)
sim_graphics.plot_results()
mhe_graphics.plot_results()
# Reset the limits on all axes in graphic to show the data.
mhe_graphics.reset_axes()

# Mark the time after a full horizon is available to the MHE.
ax[0].axvline(setup_mhe['n_horizon']*setup_mhe["t_step"])
ax[1].axvline(setup_mhe['n_horizon']*setup_mhe["t_step"])

# Show the figure:
plt.show(block=True)
    