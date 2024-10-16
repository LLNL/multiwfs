# %%
import numpy as np
from multiwfs.utils import rms, genpsd
from multiwfs.dare import solve_dare
from multiwfs.dynamics import StateSpaceDynamics, StateSpaceObservation, simulate
from multiwfs.controller import Openloop, Integrator, LQG, MPC
from matplotlib import pyplot as plt

np.random.seed(1)

W = np.zeros((6,6))
W[0,0] = 1e-2

dynamics = StateSpaceDynamics(
    np.array([
        [0.995, 0.0, 0.0, 0.0, 0.0, 0.0], 
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0153856, 0.0307712, 0.0153856, 1.63484, -0.696743, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]), # state to state
    np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [1.0]]), # state to input
    W # state covariance
)

observation_slow = StateSpaceObservation(
    np.array([[0.0153856, 0.0307712, 0.0153856, 1.63484, -0.696743, -1.0]]), # state to measure
    np.array([[0.0]]), # input to measure
    np.array([[1e-2]]) # measure covariance
)

observation_fast = StateSpaceObservation(
    np.array([[1.0, 0.0, 0.0, 0.0, 0.0, -1.0]]), # state to measure
    np.array([[0.0]]), # input to measure
    np.array([[1e-2]]) # measure covariance
)

openloop = Openloop(p=dynamics.input_size)
integrator = Integrator(s=1, p=1, gain=0.3, leak=0.999)

# %%
obs_to_minimize = np.array([[1, 0, 0, 0, 0, -1]])
Q = obs_to_minimize.T @ obs_to_minimize
lqg_zoh = LQG(dynamics, observation_slow, Q=Q, name="LQG ZOH")
lqg_fast = LQG(dynamics, observation_fast, Q=Q, name="LQG fast")
lqg_tenth = LQG(dynamics, observation_fast, Q=Q, name="LQG slow", observe_every=10)
sim = simulate(dynamics, observation, [openloop, lqg_fast, lqg_tenth, lqg_zoh]);
# %%
