"""
Implementation of a *generic* Linear-Quadratic-Gaussian observer (Kalman filter) and controller (LQR)
"""

import warnings
from copy import copy
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as mvn
from tqdm import tqdm, trange

from .utils import rms, genpsd

class StateSpaceDynamics:
    """
    A state-space dynamics model, for making simulation setups.
    LQG uses this to make its internal model.
    
    x - the state
    A - the time-evolution matrix
    B - the input-to-state matrix
    C - the state-to-measurement matrix
    D - the input-to-measurement matrix
    W - the process noise matrix (covariance around Ax)
    V - the measurement noise matrix (covariance around Cx)
    """
    def __init__(self, A, B, W):
        self.A, self.B, self.W = A, B, W
        n = A.shape[0]
        warnings_on = False
        # for the obsrank/conrank bit, look back at sealrtc
        self.recompute()

    def recompute(self):
        s = self.A.shape[0]
        assert self.A.shape == (s, s), "A must be a square matrix."
        p = self.B.shape[1]
        assert len(self.B.shape) == 2, "got wrong number of dimensions in B."
        assert self.B.shape == (s, p), f"B must have dimension matching A: got {self.B.shape[0]} whereas {s} was expected in dimension 0, or got wrong number of dimensions."
        self.process_dist = mvn(cov=self.W, allow_singular=True)

    @property
    def state_size(self):
        return self.A.shape[0]

    @property
    def input_size(self):
        return self.B.shape[1]

    def __repr__(self):
        return f"State space dynamics model with state size {self.state_size} and input size {self.input_size}."
    
class StateSpaceObservation:
    """
    A state-space observation model. Only split off from dynamics so that we can get the same random seed runs when we change the observation model.
    """
    def __init__(self, C, D, V):
        self.C, self.D, self.V = C, D, V
        self.recompute()
        
    def recompute(self):
        m = self.C.shape[0]
        assert len(self.C.shape) == 2, "got wrong number of dimensions in C."
        # some of these checks are useless now
        assert self.C.shape == (m, self.state_size), f"C must have dimension matching A: got {self.C.shape[1]} whereas {self.state_size} was expected in dimension 1."
        assert len(self.D.shape) == 2, "got wrong number of dimensions in D."
        assert self.D.shape == (m, self.input_size), f"D must have dimensions matching B and C: got {self.D.shape} whereas {(m, self.input_size)} was expected."
        self.measure_dist = mvn(cov=self.V, allow_singular=True)
        
    @property
    def state_size(self):
        return self.C.shape[1]

    @property
    def input_size(self):
        return self.D.shape[1]
        
    @property
    def measure_size(self):
        return self.C.shape[0]
        
    def __repr__(self):
        return f"State space observation model with state size {self.state_size}, input size {self.input_size} and measurement size {self.measure_size}."    

def simulate(dynamics, observation, controllers, nsteps=10000, plot=True, u_lim=np.inf):
    states_one = np.zeros((nsteps, dynamics.state_size))
    states_one[0] = dynamics.process_dist.rvs()
    sim = {
        c.name : {
            "states": copy(states_one),
            "inputs": np.zeros((nsteps, dynamics.input_size)),
            "measurements": np.zeros((nsteps, observation.measure_size)),
            "noiseless_measurements": np.zeros((nsteps, observation.measure_size)),
        }
        for c in controllers
    }

    for j in trange(nsteps):
        process_noise, measure_noise = dynamics.process_dist.rvs(), observation.measure_dist.rvs()
        for c in controllers:
            sim_c = sim[c.name]
            s, i, m, nm = sim_c["states"], sim_c["inputs"], sim_c["measurements"], sim_c["noiseless_measurements"]
            nm[j-1] = observation.C @ s[j-1] + observation.D @ i[j-1]
            m[j-1] = nm[j-1] + measure_noise
            i[j] = np.maximum(-u_lim, np.minimum(c(m[j-1], j), u_lim))
            s[j] = dynamics.A @ s[j-1] + dynamics.B @ i[j] + process_noise
        
    for c in controllers:
        c.reset()

    if plot:
        nsteps_plot = min(1000, nsteps)
        times = np.arange(nsteps_plot)
        _, axs = plt.subplots(1, 2, figsize=(10,6))
        plt.suptitle("Simulated control results")
        meastoplot = lambda meas: np.convolve(np.linalg.norm(meas, axis=1)[:nsteps_plot], np.ones(10) / 10, 'same')
        for c in controllers:
            measurements = sim[c.name]["noiseless_measurements"]
            rmsval = rms(measurements, axis=(0,1))
            axs[0].plot(times, meastoplot(measurements), label=f"{c.name}, rms = {rmsval:.3f}")
            measurement_energy = np.sqrt(
                np.mean(measurements ** 2, axis=1)
            )
            freqs, psd = genpsd(measurement_energy, dt=1/1000) # change this later
            axs[1].loglog(freqs, psd, label=f"{c.name} PSD")

        axs[0].set_title("Control residuals")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Simulated time-averaged RMS error")
        axs[0].legend()
        axs[1].set_title("Residual PSD")
        axs[1].set_xlabel("Frequency (Hz)")
        axs[1].set_ylabel("Simulated power")
        axs[1].legend()

    return sim

# look back on sealrtc for the delay version
