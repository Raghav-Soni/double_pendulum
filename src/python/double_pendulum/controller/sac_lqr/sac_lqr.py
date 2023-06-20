import os
import yaml
import numpy as np

from stable_baselines3 import SAC
from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.controller.lqr.lqr_controller import LQRController


class SACController(AbstractController):
    def __init__(self,
                 mass=[0.5, 0.6],
                 length=[0.3, 0.2],
                 com=[0.3, 0.2],
                 damping=[0.1, 0.1],
                 coulomb_fric=[0.0, 0.0],
                 gravity=9.81,
                 inertia=[None, None],
                 torque_limit=[0.0, 1.0],
                 model_pars=None):

        super().__init__()

        self.mass = mass
        self.length = length
        self.com = com
        self.damping = damping
        self.cfric = coulomb_fric
        self.gravity = gravity
        self.inertia = inertia
        self.torque_limit = torque_limit

        if model_pars is not None:
            self.mass = model_pars.m
            self.length = model_pars.l
            self.com = model_pars.r
            self.damping = model_pars.b
            self.cfric = model_pars.cf
            self.gravity = model_pars.g
            self.inertia = model_pars.I
            # self.Ir = model_pars.Ir
            # self.gr = model_pars.gr
            self.torque_limit = model_pars.tl
        
        self.lqr = LQRController(model_pars=model_pars)
        
        if(self.torque_limit[1] == 0.0):
            self.robot = "pendubot"
            self.active_act = 0
        elif(self.torque_limit[0] == 0.0):
            self.robot = "acrobot"
            self.active_act = 1

        self.dir_path = None

        
        

    def init_(self):
        self.lqr.init()
        # dir_path = os.path.dirname(os.path.realpath(__file__)) 
        if(self.robot == "pendubot"): 
            dir_path = self.dir_path + "/src/python/double_pendulum/controller/sac_lqr/src/weights/pendubot.zip"
        if(self.robot == "acrobot"):
            dir_path = self.dir_path + "/src/python/double_pendulum/controller/sac_lqr/src/weights/acrobot.zip"

        self.model = SAC.load(dir_path)

        self.obs_buffer = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])

        self.inside_flag = False

        self.max_vel = 30
        self.max_tq = 6



    def get_control_output_(self, x, t=None):
        """
        The function to compute the control input for the double pendulum's
        actuator(s).

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        t : float, optional
            time, unit=[s]
            (Default value=None)

        Returns
        -------
        array_like
            shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
        """
        self.obs_buffer[0] = self.obs_buffer[1]
        self.obs_buffer[1] = self.obs_buffer[2]


        a1_cos = np.cos(x[0])
        a1_abs = np.arccos(a1_cos)
        a2_cos = np.cos(x[1])
        a2_abs = np.arccos(a2_cos)

        if(self.inside_flag == True):
            if(a1_abs < self.roa_out[0] or a2_abs > self.roa_out[1]):
                self.inside_flag = False
        else:
            if(a1_abs > self.roa_in[0] and a2_abs < self.roa_in[1] and abs(x[2]) < self.roa_in[2] and abs(x[3]) < self.roa_in[3]):
                self.inside_flag = True

        if(self.inside_flag == False):

            a1_r = x[0]%(2*np.pi)
            if(a1_r > np.pi):
                a1_r = a1_r - 2*np.pi
            a2_r = x[1]%(2*np.pi)
            if(a2_r > np.pi):
                a2_r = a2_r - 2*np.pi
            self.obs_buffer[2][0] = a1_r/np.pi
            self.obs_buffer[2][1] = a2_r/np.pi
            self.obs_buffer[2][2] = x[2]/self.max_vel
            self.obs_buffer[2][3] = x[3]/self.max_vel

            obs = np.concatenate((self.obs_buffer[0], self.obs_buffer[1], self.obs_buffer[2]))
            action, _states = self.model.predict(obs, deterministic = True)

            action[0] = self.max_tq*action[0]


            u = [0.0, 0.0]
            u[self.active_act] = action[0]

            u[0] = np.clip(u[0], -self.torque_limit[0], self.torque_limit[0])
            u[1] = np.clip(u[1], -self.torque_limit[1], self.torque_limit[1])
        else:
            u = self.lqr.get_control_output(x, t)
        return u

    def set_goal(self, x=[np.pi, 0., 0., 0.]):
        self.lqr.set_goal(x)
    
    def set_parameters(self, failure_value=np.nan,
                       cost_to_go_cut=15.):
        
        self.lqr.set_parameters(failure_value=0.0, cost_to_go_cut=1000)
    
    def set_cost_matrices(self, Q, R):
        self.lqr.set_cost_matrices(Q=Q, R=R)

    def set_filter_args_(self, filt, x0, dt, plant,
                        simulator, velocity_cut,
                        filter_kwargs):
        self.lqr.set_filter_args(filt=filt, x0=x0, dt=dt, plant=plant,
                           simulator=simulator, velocity_cut=velocity_cut,
                           filter_kwargs=filter_kwargs)
    
    def set_friction_compensation_(self, damping, coulomb_fric):
        self.lqr.set_friction_compensation(damping=damping, coulomb_fric=coulomb_fric)
    
    def set_roa(self, roa_in, roa_out):
        self.roa_in = roa_in
        self.roa_out = roa_out

    def update_dir_path(self, dir_path):
        self.dir_path = dir_path