from brian2 import *
from scipy import stats
from matplotlib.pyplot import *

defaultclock.dt = 0.25*ms

morpho = Cylinder(length=10*cm, diameter=2*238*um, n=1000, type='axon')

# constants - current clamp
cc_amplitude = -120                     # pA; current step amplitude
start_cc = 5000                         # ms; start time of current step
end_cc = 8000                           # ms; end time of current step

step_size = 0.25                        # ms
t_total = end_cc + 1200                 # ms; total simulation time
t_len = int(t_total / step_size)
t_template = np.arange(0, t_total, step_size)

init_mV = -65*mV                  # initial membrane potential; mV
intra_Ca = 0.001*mM                # intracellular [Ca2+]; mM

m = 0.0000422117       # m
h = 0.9917               # h
n = 0.00264776           # n
mA = 0.5873               # mA
hA = 0.1269               # hA
mh = 0.0517               # mh
mM = 0.000025             # mM
mCaL = 7.6e-5               # mCaL
hCaL = 0.94                 # hCaL
s = 0.4                  # s
mKCa = 0.000025             # mKCa
I_Na = 0                    # INa
I_K = 0                    # IK
I_CaL = 0                    # ICaL
I_M = 0                    # IM
I_KCa = 0                    # IKCa
I_A = 0                    # IA
I_h = 0                    # Ih
I_leak = 0                    # Ileak
mh_inf = 0                    # mh_inf

E_Na = 60  # mV; reverse potential of INa
E_K = -90
E_h = -31.6
E_leak = -55
E_syn = 0
E_Ca = 120

idx = 0

G_Na = 400
G_K = 300
G_CaL = 5
G_M = 10
G_KCa = 10
G_A = 1
G_h = 0.4
G_leak = 0.5
G_syn = 0

Capacitance = 100

f = 0.01  # percent of free to bound Ca2+
alpha = 0.002  # uM/pA; convertion factor from current to concentration
kCaS = 0.024  # /ms; Ca2+ removal rate, kCaS is proportional to  1/tau_removal; 0.008 - 0.025

SCa = 1  # uM; half-saturation of [Ca2+]; 25uM in Ermentrount book, 0.2uM in Kurian et al. 2011
tauKCa_0 = 50  # ms
tau_hA_scale = 100  # scaling factor for tau_hA


eqs = '''
'''

