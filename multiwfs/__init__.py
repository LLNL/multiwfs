from matplotlib import pyplot as plt

plt.rc('font', family='serif',size=12)
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif" : "cmr10",
    "axes.formatter.use_mathtext" : True
})

from .dynamics import StateSpaceDynamics