from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import ode_solver
import tspn

# constants - current clamp
cc_amplitude = -120                     # pA; current step amplitude
start_cc = 5000                         # ms; start time of current step
end_cc = 8000                           # ms; end time of current step

# constants - time
step_size = 0.25                        # ms
t_total = end_cc + 1200                 # ms; total simulation time
t_len = int(t_total / step_size)
t_template = np.arange(0, t_total, step_size)

# initialization
y0 = []
y0.append(-65)                  # initial membrane potential; mV
y0.append(0.001)                # intracellular [Ca2+]; mM
y0.append(0.0000422117)         # m
y0.append(0.9917)               # h
y0.append(0.00264776)           # n
y0.append(0.5873)               # mA
y0.append(0.1269)               # hA
y0.append(0.0517)               # mh
y0.append(0.000025)             # mM
y0.append(7.6e-5)               # mCaL
y0.append(0.94)                 # hCaL
y0.append(0.4)                  # s
y0.append(0.000025)             # mKCa
y0.append(0)                    # INa
y0.append(0)                    # IK
y0.append(0)                    # ICaL
y0.append(0)                    # IM
y0.append(0)                    # IKCa
y0.append(0)                    # IA
y0.append(0)                    # Ih
y0.append(0)                    # Ileak
y0.append(0)                    # mh_inf

# parameters
G = [0, 400, 300, 5, 10, 10, 1, 0.4, 0.5, 100,]  # idx, GNa, GK, GCaL, GM, GKCa, GA, Gh, Gleak, Capacitance

# compute current clamp (cc) template
cc_template = np.zeros((t_len, 1))

for i in range(0, t_len):
    if int(start_cc / step_size) <= i <= int(end_cc / step_size):
        cc_template[i] = cc_amplitude
    else:
        cc_template[i] = 0

# CALL ode_solver.py
y = ode_solver.update(tspn.step, y0, t_len, cc_template, G, step_size)
sio.savemat('y.mat', {'output':y})

# calculate ISI
spkCount = 0
spkIdx = np.zeros(100, dtype=np.int)

slope = np.sign(np.diff(y[:,0]))
for i in range(20001, 32000):
    if slope[i-1] == 1 and slope[i] == -1 and y[i,0] > -20:  # spike detection criteria
        spkCount += 1
        spkIdx[spkCount] = i

spkIdx = spkIdx[spkIdx > 0]                                  # truncate unfilled slots
isi = np.diff(t_template[spkIdx]) / 1e3
isi_inv = [1 / ind for ind in isi]

np.set_printoptions(threshold=np.nan)
print("# of ISI is %d" % len(isi))