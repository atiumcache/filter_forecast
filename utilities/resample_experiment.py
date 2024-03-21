import numpy as np
from scipy.integrate import solve_ivp


def growthEQ(t, y):
    r = 0.1
    k = 10000
    return r*y*(1 - (y/k))

time = 50
sol = solve_ivp(fun=growthEQ, t_span=[0, time], y0=[2, 5,], t_eval=range(time))

