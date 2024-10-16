# %%
import numpy as np
from matplotlib import pyplot as plt
from multiwfs.utils import rms
from multiwfs.dynamics import StateSpaceDynamics, StateSpaceObservation, simulate
from multiwfs.controller import Openloop, Integrator, LQG

dynamics = StateSpaceDynamics(
    np.array([[0.995]]), # state to state
    np.array([[0.0]]), # state to input
    np.array([[1e-1]]), # state covariance
)

x = np.zeros(dynamics.state_size)
N = 100
states = np.zeros((N, dynamics.state_size))
for i in range(N):
    x = dynamics.A @ x + dynamics.process_dist.rvs()
    states[i,:] = x
    
observations_one = StateSpaceObservation(
    np.array([[1.0]]), # state to measure
    np.array([[1.0]]), # input to measure
    np.array([[1e-2]]) # measure covariance
)
noises_one = observations_one.measure_dist.rvs(N)
observations_two = StateSpaceObservation(
    np.array([[1.0], [1.0]]), # state to measure
    np.array([[1.0], [1.0]]), # input to measure
    np.array([[1e-2, 0], [0, 1e-2]]) # measure covariance
)
noises_two = observations_two.measure_dist.rvs(N)
noises_two[:,0] = noises_one

fig, axs = plt.subplots(1, 2, figsize=(10,5))
for (observation, noise, ax) in zip([observations_one, observations_two], [noises_one, noises_two], axs):
    lqg = LQG(dynamics, observation)
    estims = np.zeros(N)
    resids = np.zeros(N)
    for i in range(N):
        x = states[i]
        y = observation.C @ x + noise[i]
        lqg.predict()
        lqg.update(y)
        states[i] = x[0]
        estims[i] = lqg.x[0]
        resids[i] = x[0] - lqg.x[0]
        
    ax.plot(states, label="Truth")
    ax.plot(estims, label="Kalman filter estimates")
    ax.plot(resids, label="Residual")
    ax.set_title(f"WFSs = {observation.C.shape[0]}, KF resid = {rms(resids):.3f}")
    ax.legend()
# %%
