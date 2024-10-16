# %%
import numpy as np
from multiwfs.utils import rms, genpsd
from multiwfs.dare import solve_dare
from multiwfs.dynamics import StateSpaceDynamics, StateSpaceObservation, simulate
from multiwfs.controller import Openloop, Integrator, LQG, MPC
from matplotlib import pyplot as plt

np.random.seed(1)

dynamics = StateSpaceDynamics(
    np.array([[0.995, 0.0, 0.0], [0.0, 0.995, 0.0], [0.0, 0.0, 0.0]]), # state to state
    np.array([[0.0], [0.0], [1.0]]), # state to input
    np.array([[1e-2, 0.0, 0.0], [0.0, 1e-2, 0.0], [0.0, 0.0, 0.0]]), # state covariance
)

observation = StateSpaceObservation(
    np.array([[1.0, 0.0, -1.0], [1.0, 1.0, -1.0]]), # state to measure
    np.array([[0.0], [0.0]]), # input to measure
    np.array([[1e-2, 0], [0, 1e-2]]) # measure covariance
)

openloop = Openloop(p=dynamics.input_size)
integrator = Integrator(s=1, p=1, gain=0.3, leak=0.999)
# %%
# put this into the LQG class
lqg = LQG(dynamics, observation)
# Cost matrix to minimize error on the science WFS without caring about the auxiliary WFS
Ccost = np.copy(lqg.C)
Ccost[1] = 0.0
lqg.Q = Ccost.T @ Ccost
u_lim = 1.0
lqg.Pcon = solve_dare(dynamics.A, dynamics.B, lqg.Q, lqg.R, S=lqg.S)
lqg.L = -np.linalg.pinv(lqg.R + dynamics.B.T @ lqg.Pcon @ dynamics.B) @ (lqg.S.T + dynamics.B.T @ lqg.Pcon @ dynamics.A)
mpc = MPC(dynamics, observation, Q=lqg.Q, R=lqg.R, u_lim=u_lim, y_limidx=[1], y_limval=[1.0 - 1e-3])
sim = simulate(dynamics, observation, [openloop, integrator, lqg, mpc], nsteps=10000, u_lim=u_lim);

# %%
lqg_sci, lqg_aux = sim["LQG"]["noiseless_measurements"].T
mpc_sci, mpc_aux = sim["MPC"]["noiseless_measurements"].T
# %%
rms(lqg_sci) # error on the science WFS
# %%
rms(lqg_aux)
# error on the auxiliary WFS
# %%
rms(mpc_sci) # error on the science WFS
# %%
rms(mpc_aux)
# %%
plt.loglog(*genpsd(lqg_sci, dt=1e-3), label="LQG: science WFS", color="r")
plt.loglog(*genpsd(mpc_sci, dt=1e-3), label="MPC: science WFS", color="b")
plt.loglog(*genpsd(lqg_aux, dt=1e-3), label="LQG: auxiliary WFS", ls="--", color="r")
plt.loglog(*genpsd(mpc_aux, dt=1e-3), label="MPC: auxiliary WFS", ls="--", color="b")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.legend()
# %%
