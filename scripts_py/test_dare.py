# %%
from multiwfs.dare import solve_dare
import numpy as np
# %%
A = np.array([[1.99483, -0.994986], [1.0, 0.0]])
C = np.array([[1.0], [0.0]])
W = np.eye(2)
V = np.eye(1)
# %%
solve_dare(A.T, C, W, V)
# %%
