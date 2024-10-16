# %%
# Attempt at implementing implicit and then explicit MPC for a sine-wave process.

# main implementations of everything will be the Julia, I'll make a data format for explicit MPC solutions that can be read by Julia either in simulation or on SEAL

import numpy as np
from matplotlib import pyplot as plt
import cvxpy as cp
from scipy.stats import multivariate_normal as mvn
from multiwfs.controller import MPC
from multiwfs.dynamics import StateSpaceDynamics, StateSpaceObservation

# Set up the random process.
f_over_f_loop = 0.01
A = np.array([[2 * np.cos(2 * np.pi * f_over_f_loop), -1], [1, 0]])
C = np.array([1, 0])
W = np.array([[1, 0], [0, 0]])
process_noise = mvn(np.zeros(2), W, allow_singular=True)
