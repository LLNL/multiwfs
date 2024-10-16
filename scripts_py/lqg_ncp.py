# %%
import numpy as np
from multiwfs.utils import rms, genpsd, pretty_print
from multiwfs.dynamics import StateSpaceDynamics, StateSpaceObservation
from multiwfs.controller import Openloop, Integrator, LQG
from matplotlib import pyplot as plt
from scipy.linalg import block_diag

np.random.seed(1)

f_loop = 200
# written 2021-11-27
def vib_coeffs(f, k, fs):
    w = 2 * np.pi * f
    a1 = 2 * np.exp(-k * w / fs) * np.cos(w * np.sqrt(1 - k**2) / fs)
    a2 = -np.exp(-2 * k * w / fs)
    return a1, a2

vibration_ncp_freq = 1.0
a1vib, a2vib = vib_coeffs(vibration_ncp_freq, 0.0, 200.0)

W = np.zeros((8,8))
W[0,0] = 1e-2
W[1,1] = 1e-6 # maybe later I'll generate these timeseries without this

A = block_diag(
    np.array([[0.995]]),
    np.array([[a1vib, a2vib], [1, 0]]),
    np.array([[0.0, 0.0, 0.0, 0.0, 0.0], 
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0307712, 0.0153856, 1.63484, -0.696743, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]])
)
A[3,0] = 1.0
A[5,0] = 0.0153856

B = np.zeros(8).reshape((8, 1))
B[7] = 1.0

dynamics = StateSpaceDynamics(A, B, W)

observation = StateSpaceObservation(
    np.array([
        [0.0153856, 0.0, 0.0, 0.0307712, 0.0153856, 1.63484, -0.696743, -1.0],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        ]),
    np.zeros((2, 1)), # input to measure
    np.array([[1e-2, 0], [0, 1e-2]]) # measure covariance
)

openloop = Openloop(p=dynamics.input_size)
Ccost = np.copy(observation.C)
Ccost[1,:] = 0
Q = Ccost.T @ Ccost
lqg = LQG(dynamics, observation, Q=Q)
# %%
n = 50_000
openloop_common_path = np.zeros(n)
openloop_common_path[0] = np.random.normal(0, 1e-2)
for i in range(1, n):
    openloop_common_path[i] = openloop_common_path[i-1] + np.random.normal(0, 1e-2)
    
vibration_ncp = np.sin(2 * np.pi * np.arange(n) * vibration_ncp_freq / f_loop)
vibration_ncp *= 0.1 * rms(openloop_common_path) / rms(vibration_ncp)

plt.loglog(*genpsd(openloop_common_path, 1/200), label="Common path")
plt.loglog(*genpsd(openloop_common_path + vibration_ncp, 1/200), label="Non common path")
# %%
measurements = np.vstack((openloop_common_path, openloop_common_path + vibration_ncp)).T
control_commands = np.zeros(n)
for (i, m) in enumerate(measurements):
    lqg.observe_law(m)
    control_commands[i] = lqg.control_law()[0]
    
plt.loglog(*genpsd(openloop_common_path, 1/200), label="OL common path")
plt.loglog(*genpsd(openloop_common_path + vibration_ncp, 1/200), label="OL non common path")
plt.loglog(*genpsd(openloop_common_path - control_commands, 1/200), label="Science closed-loop")
plt.legend()
# %%
# This doesn't seem to be going great, let's ask Lisa how this setup should work
