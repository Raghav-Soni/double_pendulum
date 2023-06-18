import os
from datetime import datetime
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.sac_lqr.sac_lqr import SACController
from double_pendulum.utils.plotting import plot_timeseries

# model parameters
design = "design_C.0"
model = "model_3.0"
robot = "acrobot"
friction_compensation = True


if robot == "pendubot":
    torque_limit = [6.0, 0.0]
    active_act = 0
elif robot == "acrobot":
    torque_limit = [0.0, 6.0]
    active_act = 1

model_par_path = "../../data/system_identification/identified_parameters/"+design+"/"+model+"/model_parameters.yml"
mpar = model_parameters(filepath=model_par_path)

mpar_con = model_parameters(filepath=model_par_path)
mpar_con.set_motor_inertia(0.)
if friction_compensation:
    mpar_con.set_damping([0., 0.])
    mpar_con.set_cfric([0., 0.])
mpar_con.set_torque_limit(torque_limit)

# simulation parameters
dt = 0.005
t_final = 10.0
integrator = "runge_kutta"
goal = [np.pi, 0., 0., 0.]

# noise
process_noise_sigmas = [0.0, 0.0, 0.0, 0.0]
meas_noise_sigmas = [0.0, 0.0, 0.05, 0.05]
delay_mode = "posvel"
delay = 0.01
u_noise_sigmas = [0.0, 0.0]
u_responsiveness = 1.0
perturbation_times = []
perturbation_taus = []

# filter args
meas_noise_vfilter = "none"
meas_noise_cut = 0.1
filter_kwargs = {"lowpass_alpha": [1., 1., 0.2, 0.2],
                 "kalman_xlin": goal,
                 "kalman_ulin": [0., 0.],
                 "kalman_process_noise_sigmas": process_noise_sigmas,
                 "kalman_meas_noise_sigmas": meas_noise_sigmas,
                 "ukalman_integrator": integrator,
                 "ukalman_process_noise_sigmas": process_noise_sigmas,
                 "ukalman_meas_noise_sigmas": meas_noise_sigmas}

if robot == "acrobot":
    x0 = [0.0, 0.0, 0.0, 0.0]
    roa_in = [170*np.pi/180, 10*np.pi/180, 5, 5]
    roa_out = [170*np.pi/180, 10*np.pi/180, 1, 1]
    # x0 = [np.pi+0.05, -0.2, 0.0, 0.0]
    # x0 = [np.pi+0.1, -0.4, 0.0, 0.0]

    Q = np.diag([0.64, 0.99, 0.78, 0.64])
    R = np.eye(2)*0.27

elif robot == "pendubot":
    x0 = [0.0, 0.0, 0.0, 0.0]
    roa_in = [160*np.pi/180, 15*np.pi/180, 0.5, 1.0]
    roa_out = [160*np.pi/180, 20*np.pi/180, 1, 1]
    # x0 = [np.pi-0.2, 0.3, 0., 0.]
    #x0 = [np.pi-0.05, 0.1, 0.0, 0.0]

    Q = np.diag([0.0125, 6.5, 6.88, 9.36])
    R = np.diag([2.4, 2.4])

timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", design, model, robot, "lqr", timestamp)
os.makedirs(save_dir)

plant = SymbolicDoublePendulum(model_pars=mpar)

sim = Simulator(plant=plant)
sim.set_process_noise(process_noise_sigmas=process_noise_sigmas)
sim.set_measurement_parameters(meas_noise_sigmas=meas_noise_sigmas,
                               delay=delay,
                               delay_mode=delay_mode)
sim.set_motor_parameters(u_noise_sigmas=u_noise_sigmas,
                         u_responsiveness=u_responsiveness)

dir_path = "../.."  #Enter the directory path to double_pendulum repo

controller = SACController(model_pars=mpar_con)
controller.set_goal(goal)
controller.update_dir_path(dir_path)  # Updating the directory for weights
controller.set_roa(roa_in, roa_out)
controller.set_cost_matrices(Q=Q, R=R)
controller.set_parameters(failure_value=0.0,
                          cost_to_go_cut=1000)
controller.set_filter_args_(filt=meas_noise_vfilter, x0=goal, dt=dt, plant=plant,
                           simulator=sim, velocity_cut=meas_noise_cut,
                           filter_kwargs=filter_kwargs)


if friction_compensation:
    controller.set_friction_compensation_(damping=mpar.b, coulomb_fric=mpar.cf)
controller.init()

T, X, U = sim.simulate_and_animate(t0=0.0, x0=x0,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator=integrator,
                                   save_video=False,
                                   video_name=os.path.join(save_dir, "simulation"))


plot_timeseries(T, X, U, None,
                plot_energy=False,
                X_filt=controller.x_filt_hist,
                X_meas=sim.meas_x_values,
                U_con=controller.u_hist[1:],
                U_friccomp=controller.u_fric_hist,
                pos_y_lines=[0.0, np.pi],
                tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
                save_to=os.path.join(save_dir, "timeseries"))