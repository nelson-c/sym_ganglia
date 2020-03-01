from __future__ import print_function, division
from numpy import exp
from numpy import power
import numpy as np
import matplotlib.pyplot as plt


def tSPN(gmax: [], iclamp: [], y0: [], gsyn):
    gmax = np.array(gmax)
    iclamp = np.array(iclamp)
    y0 = np.array(y0)
    gsyn = np.array(gsyn)

    if y0.size != 22:
        y0 = [- 65, 0.001, 4.22117 * 10 ** (-5), 0.9917, 0.00264776, 0.5873, 0.1269, 0.0517, 0.5, 2.5 * 10 ** (-5),
              7.6 * 10 ** (-5), 0.94, 0.4,
              2.5 * 10 ** (-5), 0, 0, 0, 0, 0, 0, 0, 0]

    if gsyn.size != 0:
        if len(gsyn) < len(iclamp):
            diff = len(iclamp) - len(gsyn)
            gsyn = np.append(gsyn, [0] * diff, axis=0)
        elif len(gsyn) > len(iclamp):
            gsyn = gsyn[:len(iclamp)]
    else:
        gsyn = np.zeros_like(iclamp)

    if len(gmax) == 9:
        gmax = np.append(gmax, [0])

    iclamp_lin = np.reshape(iclamp, (len(iclamp), -1))

    I = np.zeros([len(y0), len(iclamp_lin)])

    for index in range(0, len(iclamp_lin)):
        dy = tSPN_step(y0, iclamp_lin[index], gmax, .1, gsyn[index])
        I[:, index] = dy.T

    # V = reshape(I(1,:), size(iclamp));
    # I = reshape(I, length(dy), [], size(iclamp, 2));
    # I = permute(I, [2 3 1]);

    print(I)
    print(len(iclamp))
    V = I[1, :len(iclamp)]
    print(V)
    return (V, I)



def tSPN_step(dydt, iclamp, gmax, dt, Gsyn):

    ## initialization
    V = dydt[0]             # Somatic membrane voltage (mV)
    CaS = dydt[1]           # somatic [Ca2+]
    m = dydt[2]             # Na activation
    h = dydt[3]             # Na inactivation
    n = dydt[4]             # K activation
    mA = dydt[5]            # A activation
    hA = dydt[6]            # A inactivation
    mh = dydt[7]            # h activation
    mh_inf = dydt[8]        # h steady-state activation
    mM = dydt[9]            # M activation
    mCaL = dydt[10]          # CaL activation
    hCaL = dydt[11]         # CaL inactivation
    s = dydt[12]            # Na slow inactivation
    mKCa = dydt[13]         # KCa activation


    GNa = gmax[0]           # nS; maximum conductance of INa
    GK = gmax[1]
    GCaL = gmax[2]
    GM = gmax[3]
    GKCa = gmax[4]
    GA = gmax[5]
    Gh = gmax[6]
    Gleak = gmax[7]
    C = gmax[8]
    Ginjury = gmax[9]       # added to simulate impalement injury

    E_Na = 60                # mV; reverse potential of INa
    E_K = -90
    E_h = -31.6
    E_leak = -55
    E_syn = 0
    E_Ca = 120
    E_injury = -15          # reversal potential for injury induced leak = solution of GHK if all permeabilities are equal

    f = 0.01                   # percent of free to bound Ca2+
    alpha = 0.002              # uM/pA; convertion factor from current to concentration
    kCaS = 0.024               # /ms; Ca2+ removal rate, kCaS is proportional to  1/tau_removal; 0.008 - 0.025
    # A = 1.26e-5              % cm^2; cell surface area; radius is 10um
    # Ca_out = 2               % mM; extracellular Ca2+ concentration

    SCa = 1                    # uM; half-saturation of [Ca2+]; 25uM in Ermentrount book, 0.2uM in Kurian et al. 2011
    tauKCa_0 = 50              # ms
    tauh_0_act = 1             # 10, 1
    tauh_0_inact = 1           # 10, 5         % ms; vary from 50ms to 1000ms
    tau_mA_scale = 1
    tau_hA_scale = 10         # scaling factor for tau_hA

    ## update dydt
    # Sodium current (pA), Wheeler & Horn 2004 or Yamada et al., 1989
    alpha_m = 0.36 * (V + 33) / (1 - exp(-(V + 33) / 3))
    beta_m = - 0.4 * (V + 42) / (1 - exp((V + 42) / 20))
    m_inf = alpha_m / (alpha_m + beta_m)
    tau_m = 2 / (alpha_m + beta_m)
    m_next = m_inf + (m - m_inf) * exp(-dt / tau_m) if dt < tau_m else m_inf

    alpha_h = - 0.1 * (V + 55) / (1 - exp((V + 55) / 6))
    beta_h = 4.5 / (1 + exp(-V / 10))
    h_inf = alpha_h / (alpha_h + beta_h)
    tau_h = 2 / (alpha_h + beta_h)
    h_next = h_inf + (h - h_inf) * exp(-dt / tau_h) if dt < tau_h else h_inf

    alpha_s = 0.0077 / (1 + exp((V - 18) / 9))  # Miles et al., 2005
    beta_s = 0.0077 / (1 + exp((18 - V) / 9))
    tau_s = 129.2
    s_inf = alpha_s / (alpha_s + beta_s)
    s_next = s_inf + (s - s_inf) * exp(-dt / tau_s) if dt < tau_s else s_inf

    gNa = GNa * power(m_next, 2) * h_next
    I_Na = gNa * (V - E_Na)

    # Potassium current (pA), Wheeler & Horn 2004 or Yamada et al., 1989
    alpha_n_20 = 0.0047 * (V - 8) / (1 - exp(-(V - 8) / 12))
    beta_n_20 = exp(-(V + 127) / 30)
    n_inf = alpha_n_20 / (alpha_n_20 + beta_n_20)
    alpha_n = 0.0047 * (V + 12) / (1 - exp(-(V + 12) / 12))
    beta_n = exp(-(V + 147) / 30)
    tau_n = 1 / (alpha_n + beta_n)
    n_next = n_inf + (n - n_inf) * exp(-dt / tau_n) if dt < tau_n else n_inf

    gK = GK * power(n_next, 4)
    I_K = gK * (V - E_K)

    # Calcium current (pA), L-type, Bhalla & Bower, 1993
    alpha_mCaL = 7.5 / (1 + exp((13 - V) / 7))
    beta_mCaL = 1.65 / (1 + exp((V - 14) / 4))
    mCaL_inf = alpha_mCaL / (alpha_mCaL + beta_mCaL)
    tau_mCaL = 1 / (alpha_mCaL + beta_mCaL)
    mCaL_next = mCaL_inf + (mCaL - mCaL_inf) * exp(-dt / tau_mCaL) if dt < tau_mCaL else mCaL_inf

    alpha_hCaL = 0.0068 / (1 + exp((V + 30) / 12))
    beta_hCaL = 0.06 / (1 + exp(-V / 11))
    hCaL_inf = alpha_hCaL / (alpha_hCaL + beta_hCaL)
    tau_hCaL = 1 / (alpha_hCaL + beta_hCaL)
    hCaL_next = hCaL_inf + (hCaL - hCaL_inf) * exp(-dt / tau_hCaL) if dt < tau_hCaL else hCaL_inf

    gCaL = GCaL * mCaL_next * hCaL_next
    I_CaL = gCaL * (V - E_Ca)

    # M current (pA), Wheeler & Horn, 2004
    mM_inf = 1 / (1 + exp(-(V + 35) / 10))
    tau_mM = 2000 / (3.3 * (exp((V + 35) / 40) + exp(-(V + 35) / 20)))
    mM_next = mM_inf + (mM - mM_inf) * exp(-dt / tau_mM) if dt < tau_mM else mM_inf
    gM = GM * power(mM_next, 2)
    I_M = gM * (V - E_K)

    # Somatic KCa current (pA), Ermentrout & Terman 2010
    mKCa_inf = CaS ** 2 / (CaS ** 2 + SCa ** 2)
    tau_mKCa = tauKCa_0 / (1 + (CaS / SCa) ** 2)
    mKCa_next = mKCa_inf + (mKCa - mKCa_inf) * exp(-dt / tau_mKCa) if dt < tau_mKCa else mKCa_inf
    gKCa = GKCa * power(mKCa_next, 1)
    I_KCa = gKCa * (V - E_K)

    # A-type potassium current (pA), Rush & Rinzel, 1995
    mA_inf = (0.0761 * exp((V + 94.22) / 31.84) / (1 + exp((V + 1.17) / 28.93))) ** (1/3)
    tau_mA = 0.3632 + 1.158 / (1 + exp((V + 55.96) / 20.12))
    mA_next = mA_inf + (mA - mA_inf) * exp(-dt / tau_mA) if dt < tau_mA else mA_inf

    hA_inf = (1 / (1 + exp(0.069 * (V + 53.3)))) ** 4
    tau_hA = (0.124 + 2.678 / (1 + exp((V + 50) / 16.027))) * tau_hA_scale
    hA_next = hA_inf + (hA - hA_inf) * exp(-dt / tau_hA) if dt < tau_hA else hA_inf

    gA = GA * power(mA_next, 3) * hA_next
    I_A = gA * (V - E_K)

    # Ih (pA), Based on Kullmann et al., 2016
    mh_inf_next = 1 / (1 + exp((V + 87.6) / 11.7))
    tau_mh_activ = tauh_0_act * 53.5 + 67.7 * exp(-(V + 120) / 22.4)
    tau_mh_deactiv = tauh_0_inact * 40.9 - 0.45 * V
    tau_mh = tau_mh_activ if mh_inf_next > mh_inf else tau_mh_deactiv
    mh_next = mh + (dt * (mh_inf_next-mh)/tau_mh) if dt<tau_mh else mh_inf_next
    gh = Gh * mh_next
    I_h = gh * (V - E_h)

    # Leak current (pA)
    I_leak = Gleak * (V - E_leak)

    I_injury = Ginjury * (V - E_injury)

    I_leak = I_leak + I_injury

    # Synaptic current (pA)
    I_syn = Gsyn * (V - E_syn)

    # Somatic calcium concentration (uM), Kurian et al., 2011 & Methods of Neuronal Modeling, p. 490. 12.24
    CaS_next = CaS * exp(-f * kCaS * dt) - alpha / kCaS * I_CaL * (1 - exp(-f * kCaS * dt))

    ## update voltage
    g_inf = gNa + gCaL + gK + gA + gM + gKCa + gh + Gleak + Gsyn + Ginjury
    V_inf = (int(iclamp) + gNa * E_Na + gCaL * E_Ca + (gK + gA + gM + gKCa) * E_K + gh * E_h + Gleak * E_leak + Ginjury * E_injury + Gsyn * E_syn) / g_inf
    tau_tspn = C / g_inf
    V_next = V_inf + (V - V_inf) * exp(-dt / tau_tspn)

    dy = [V_next, CaS_next, m_next, h_next, n_next, mA_next, hA_next, mh_next, mM_next, mCaL_next, hCaL_next, s_next, mKCa_next, I_Na, I_K,
          I_CaL, I_M, I_KCa, I_A, I_h, I_leak, mh_inf_next]

    dy = np.array(dy)

    return dy


'''
%   generates a synaptic conductance trace

%   event_time: vector of times in ms at which a synaptic event occurs
%   event_scale: vector of corresponding amplitude of synaptic event in nanosiemens

%   gsyn: synaptic conductance in nS
'''
def tSPN_gsyn(event_time: [], event_scale: []):

    dt = 0.1
    event_index = [round(t/dt) for t in event_time]
    tau_rise = 1
    tau_decay = 15
    t = np.arange(0, 200+dt, dt)
    gsyn = np.zeros(max(event_index)+len(t))
    e_tau_decay = [exp(-i/tau_decay) for i in t]
    e_tau_rise = [exp(-i/tau_rise) for i in t]
    gEPSC = [a - b for a, b in zip(e_tau_decay, e_tau_rise)]
    gEPSC = [item/max(gEPSC) for item in gEPSC]

    temp = 0
    for index in range(len(event_time)):
        gEPSC_temp = [item*event_scale[index] for item in gEPSC]
        gsyn[event_index[index]:len(t)+event_index[index]] += gEPSC_temp
    print(gsyn)
    return gsyn


'''
%   This code calculates the amount of current required to hold a model cell
%   at a given holding voltage. 
%
%   gmax: vector of parameters for a specific model cell in the following
%   order: [GNa GK GCaL GM GKCa GA GH GLeak Cm GImp] (GImp is optional, default is 0)
%   Conductance is measured in nS, capacitance is measured in pF. 
%
%   vhold: target holding voltage in mV
%
%   bounds: (optional) 2-element vector which sets the upper and lower bounds 
%   for the injected current in pA. default is [-100 100];
%
%   ihold: current required to hold the given model cell at the specified
%   holding voltage.
%
%   y0: value of all parameters at the end of the simulation. Can be used as
%   the initial value for subsequent simulations. Useful for reducing
%   initialization time. 
'''
def tSPN_ihold(gmax: [],vhold,bounds):
    return

if __name__ == "__main__":
    # gmax = [1] * 9
    # iclamp = [1]*10
    # y0 = []
    # gsyn = []
    # tSPN(gmax, iclamp, y0, gsyn)
    plt.plot(tSPN_gsyn([1, 2, 3], [1, 2, 3]))
    plt.show()
