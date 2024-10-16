# %%
import numpy as np
from multiwfs.dynamics import StateSpaceDynamics, StateSpaceObservation, simulate
from multiwfs.controller import Openloop, Integrator, LQG, MPC

# np.random.seed(1)

dynamics = StateSpaceDynamics(
    np.array([[0.995, 0.0], [0.0, 0.0]]), # state to state
    np.array([[0.0], [1.0]]), # state to input
    np.array([[1e-2, 0.0], [0.0, 0.0]]), # state covariance
)

openloop = Openloop(p=dynamics.input_size)
integrator = Integrator(s=1, p=1, gain=0.3, leak=0.999)

observation_one = StateSpaceObservation(
    np.array([[1.0, -1.0]]), # state to measure
    np.array([[0.0]]), # input to measure
    np.array([[1e-2]]) # measure covariance
)

observation_two = StateSpaceObservation(
    np.array([[1.0, -1.0], [1.0, -1.0]]), # state to measure
    np.array([[0.0], [0.0]]), # input to measure
    np.array([[1e-2, 0], [0, 1e-2]]) # measure covariance
)

for observation in [observation_one]:
    mpc = MPC(dynamics, observation)
    lqg = LQG(dynamics, observation)
    controllers = [openloop, integrator, lqg, mpc]
    sim_res = simulate(dynamics, observation, controllers, nsteps=1000);
    # %%
