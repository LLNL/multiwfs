"""
Various control strategies.
"""

from abc import ABC
from functools import partial
import numpy as np
import cvxpy as cp

from .dare import solve_dare

class Controller(ABC):
    def reset(self):
        pass

    def __call__(self, measurement, iteration):
        if hasattr(self, "observe_every") and iteration % self.observe_every == 0:
            self.observe_law(measurement)
        return self.control_law()

    def observe_law(self, measurement):
        pass

    def control_law(self):
        pass

class Openloop(Controller):
    def __init__(self, p=2):
        self.name = "openloop"
        self.u = np.zeros((p,))
    
    def control_law(self):
        return self.u

class Integrator(Controller):
    def __init__(self, s, p, gain=0.5, leak=0.999):
        self.s = s
        self.p = p
        self.gain = gain
        self.leak = leak
        self.u = np.zeros((p,))
        self.state = np.zeros((s,))
        self.name = "integrator"

    def reset(self):
        self.u = np.zeros((self.p,))

    def observe_law(self, measurement):
        self.state = measurement[:self.p]
        
    def control_law(self):
        self.u = self.gain * self.state + self.leak * self.u
        return self.u

class LQG(Controller):
    def __init__(self, dyn, obs, name="LQG", Q=None, R=None, observe_every=1):
        self.name = name
        self.A, self.B, self.C, self.D = dyn.A, dyn.B, obs.C, obs.D
        if Q is None:
            Q = obs.C.T @ obs.C
        if R is None:
            R = obs.D.T @ obs.D
        self.Q, self.R = Q, R
        S = self.S = obs.C.T @ obs.D
        self.x = np.zeros((dyn.state_size,))
        self.u = np.zeros((dyn.input_size,))
        self.Pobs = solve_dare(dyn.A.T, obs.C.T, dyn.W, obs.V)
        self.Pcon = solve_dare(dyn.A, dyn.B, Q, R, S=S)
        self.K = self.Pobs @ obs.C.T @ np.linalg.pinv(obs.C @ self.Pobs @ obs.C.T + obs.V)
        self.L = -np.linalg.pinv(R + dyn.B.T @ self.Pcon @ dyn.B) @ (S.T + dyn.B.T @ self.Pcon @ dyn.A)
        self.observe_every = observe_every
        
    def measure(self):
        return self.C @ self.x + self.D @ self.u
        
    def predict(self):
        self.x = self.A @ self.x + self.B @ self.u

    def update(self, y):
        self.x = self.x + self.K @ (y - self.measure())
        
    def control_law(self):
        self.u = self.L @ self.x
        return self.u

    def observe_law(self, measurement):
        self.predict()
        self.update(measurement)
        
class MPC(Controller):
    def __init__(self, dyn, obs, name="MPC", horizon=1, Q=None, R=None, S=None, u_lim=np.inf, y_limidx=[], y_limval=[]):
        self.name = name
        self.horizon = horizon
        self.u_lim = u_lim
        self.A, self.B, self.C, self.D = dyn.A, dyn.B, obs.C, obs.D
        if Q is None:
            self.Q = obs.C.T @ obs.C
        else:
            self.Q = Q
        if R is None:
            self.R = obs.D.T @ obs.D
        else:
            self.R = R
        self.S = obs.C.T @ obs.D
        self.x_opt = cp.Variable((dyn.state_size, horizon))
        self.u_opt = cp.Variable((dyn.input_size, horizon))
        self.Pobs = solve_dare(dyn.A.T, obs.C.T, dyn.W, obs.V)
        self.K = self.Pobs @ obs.C.T @ np.linalg.pinv(obs.C @ self.Pobs @ obs.C.T + obs.V)
        cost = 0
        constr = []
        self.x_curr = cp.Parameter((dyn.state_size,))
        self.x_curr.value = np.zeros((dyn.state_size,))
        xlast = self.x_curr
        for t in range(self.horizon):
            xtp1 = self.x_opt[:,t]
            ut = self.u_opt[:,t]
            cost += cp.quad_form(xtp1, self.Q) + cp.quad_form(ut, self.R)
            constr += [xtp1 == self.A @ xlast + self.B @ ut]
            xtp2 = self.A @ xtp1 + self.B @ ut
            obs_lim1 = cp.abs(self.C @ xtp1)
            for (idx, val) in zip(y_limidx, y_limval):
                constr += [obs_lim1[idx] <= val]
            xlast = xtp1
        self.problem = cp.Problem(cp.Minimize(cost), constr)
        self.problem.solve()
        
    def measure(self):
        return self.C @ self.x + self.D @ self.u
        
    def predict(self):
        self.x_curr.value = self.A @ self.x + self.B @ self.u

    def update(self, y):
        self.x_curr.value = self.x + self.K @ (y - self.measure())
        
    @property
    def x(self):
        return self.x_curr.value
        
    @property
    def u(self):
        # return np.zeros(self.B.shape[1])
        u = self.u_opt.value
        if u is not None:
            self.u_prev = u[:,0]
        return self.u_prev
        
    def control_law(self):
        # at the current time, you're constrained on state = the KF-recovered state
        # due to the separation principle, this is the best we can do
        # after that, we assume x[next] = Ax[curr] + Bu[curr]
        # LQG doesn't look at W or V anyway so that's fine
        # and we try and minimize x^T Qx + x^T Su + u^T Ru
        self.problem.solve()
        try:
            assert np.abs((self.C @ (self.A @ self.x + self.B @ self.u))[1]) <= 1.0
        except AssertionError:
            print(self.problem.status)
            print("curr state", self.x)
            print("expected state", self.A @ self.x + self.B @ self.u)
            print("expected obs", self.C @ (self.A @ self.x + self.B @ self.u))
        return np.maximum(-self.u_lim, np.minimum(self.u, self.u_lim))

    def observe_law(self, measurement):
        self.predict()
        self.update(measurement)
        
        