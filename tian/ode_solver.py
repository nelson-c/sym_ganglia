from __future__ import print_function, division
import numpy as np

# CALL tspn.py
def update(func, y_init, t_len, cc, params, tstep):
    num_vars = len(y_init)
    y = np.zeros((t_len, num_vars))
    y[0] = y_init

    # Exponential Euler
    for i in range(0, t_len):
        if i <= t_len - 2:
            y[i+1] = func(y[i], cc[i], params, tstep)

    return y