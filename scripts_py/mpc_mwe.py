# %%
import numpy as np
import cvxpy as cp
from multiwfs.dare import solve_dare
from multiwfs.utils import rms

A = np.array([[0.995, 0.0], [0.0, 0.0]]) # state to state
B = np.array([[0.0], [1.0]]) # input to state
C = np.array([[1.0, -1.0]]) # state to measurement
Q = C.T @ C
R = np.array([[0.0]])
state_size = A.shape[0]
input_size = B.shape[1]
measurement_size = C.shape[0]

# assume perfect state visibility, so the unconstrained MPC problem is the LQR problem
# for real cases you should solve this over a horizon, I'll just set this to 1 because I have a simple 1D state

horizon = 1
x_opt = cp.Variable((state_size, horizon))
u_opt = cp.Variable((input_size, horizon))
cost = 0
constr = []
x_curr = cp.Parameter((state_size,))
x_curr.value = np.zeros((state_size,))
xlast = x_curr
xtp1 = x_opt[:,t]
ut = u_opt[:,t]
cost += cp.quad_form(xtp1, Q) + cp.quad_form(ut, R)
constr += [xtp1 == A @ xlast + B @ ut]
xlast = xtp1
    
problem = cp.Problem(cp.Minimize(cost), constr)
Pcon = solve_dare(A, B, Q, R)
L = -np.linalg.pinv(R + B.T @ Pcon @ B) @ (B.T @ Pcon @ A)

# %%
# check that you can solve the LQR problem
nsteps = 5000
np.random.seed(1)
x = np.zeros((state_size,))
u = np.zeros((input_size,))
yhistory = np.zeros((measurement_size, nsteps))
for do_control in [False, True]:
    for i in range(nsteps):
        x = A@x + B@u
        x[0] += np.random.normal(0.0, 1e-2) # driving noise
        if do_control:
            u = L@x
        yhistory[:,i] = C@x
        
    print(f"RMS with do_control {str(do_control).ljust(5)} = {rms(yhistory[0,:]):.3f}")
# %%
# now, replace the LQR gain with the MPC solver
nsteps = 5000
np.random.seed(1)
x = np.zeros((state_size,))
u = np.zeros((input_size,))
yhistory = np.zeros((measurement_size, nsteps))
for do_control in [False, True]:
    for i in range(nsteps):
        x = A@x + B@u
        x[0] += np.random.normal(0.0, 1e-2) # driving noise
        if do_control:
            x_curr.value = x
            problem.solve()
            u[0] = u_opt.value[0,0]
        yhistory[:,i] = C@x
        
    print(f"RMS with do_control {str(do_control).ljust(5)} = {rms(yhistory[0,:]):.3f}")
# %%
# now let's add in MPC constraints
constr += [cp.abs(xtp1[0]) <= 1.0]
problem = cp.Problem(cp.Minimize(cost), constr)
nsteps = 5000
np.random.seed(1)
x = np.zeros((state_size,))
u = np.zeros((input_size,))
yhistory = np.zeros((measurement_size, nsteps))
for do_control in [False, True]:
    for i in range(nsteps):
        x = A@x + B@u
        x[0] += np.random.normal(0.0, 1e-2) # driving noise
        if do_control:
            x_curr.value = x
            problem.solve()
            u[0] = u_opt.value[0,0]
        yhistory[:,i] = C@x
        
    print(f"RMS with do_control {str(do_control).ljust(5)} = {rms(yhistory[0,:]):.3f}")
# %%
