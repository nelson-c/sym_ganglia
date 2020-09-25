from __future__ import print_function, division
from numpy import exp
from numpy import power
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
import math
import time

'''
% This code runs the model cell simulation.

% gmax: vector of parameters for a specific model cell in the following
%  order: [GNa GK GCaL GM GKCa GA GH GLeak Cm GImp] (GImp is optional, default is 0)
%  Conductance is measured in nS, capacitance is measured in pF. 
%
% iclamp: Vector or 2D matrix of injected current in pA. iclamp is linearized 
%  and outputs are reshaped to match iclamp dimensions. Time step is 0.1ms.
%
% y0: (optional) vector of initial values for the simulation in the
%  following order
%     V
%     [Ca2+]
%     m
%     h
%     n
%     mA
%     hA
%     mh
%     mh_inf 
%     mM
%     mCaL
%     hCaL
%     s
%     mKCa
%     INa
%     IK
%     ICaL
%     IM
%     IKCa
%     IA
%     Ih
%     Ileak
%
% gsyn: (optional) vector of synaptic conductance in nS. 

% V: voltage trace of cellular response to injected current.
%
% I: matrix of additional parameters as they change over time. Same order
%  as y0.
'''


def tSPN(gmax: [], iclamp: [], y0: [], gsyn):
    gmax = np.array(gmax, dtype=np.float)
    iclamp = np.array(iclamp, dtype=np.float)
    y0 = np.array(y0, dtype=np.float)
    gsyn = np.array(gsyn, dtype=np.float)

    if y0.size != 22:
        y0 = np.array(
            [-65, 0.001, 4.22117 * 10 ** (-5), 0.9917, 0.00264776, 0.5873, 0.1269, 0.0517, 0.5, 2.5 * 10 ** (-5),
             7.6 * 10 ** (-5), 0.94, 0.4, 2.5 * 10 ** (-5), 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float)

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

    iclamp_lin = iclamp.flatten('F')
    gsyn = gsyn.flatten('F')
    I = np.zeros([len(y0), len(iclamp_lin)])
    dy = tSPN_step(y0, iclamp_lin[0], gmax, .1, gsyn[0])

    for index in range(0, len(iclamp_lin)):
        dy = tSPN_step(dy, iclamp_lin[index], gmax, .1, gsyn[index])
        I[:, index] = dy.T

    V = I[0, :]
    I = np.reshape(I, [len(dy), len(iclamp), len(iclamp[0])], order='F')
    I = np.transpose(I, (1, 2, 0))
    return V, I


'''
Called by tSPN
'''


def tSPN_step(dydt, iclamp, gmax, dt, Gsyn):
    V = dydt[0]  # Somatic membrane voltage (mV)
    CaS = dydt[1]  # somatic [Ca2+]
    m = dydt[2]  # Na activation
    h = dydt[3]  # Na inactivation
    n = dydt[4]  # K activation
    mA = dydt[5]  # A activation
    hA = dydt[6]  # A inactivation
    mh = dydt[7]  # h activation
    mh_inf = dydt[8]  # h steady-state activation
    mM = dydt[9]  # M activation
    mCaL = dydt[10]  # CaL activation
    hCaL = dydt[11]  # CaL inactivation
    s = dydt[12]  # Na slow inactivation
    mKCa = dydt[13]  # KCa activation

    GNa = gmax[0]  # nS; maximum conductance of INa
    GK = gmax[1]
    GCaL = gmax[2]
    GM = gmax[3]
    GKCa = gmax[4]
    GA = gmax[5]
    Gh = gmax[6]
    Gleak = gmax[7]
    C = gmax[8]
    Ginjury = gmax[9]  # added to simulate impalement injury

    E_Na = 60  # mV; reverse potential of INa
    E_K = -90
    E_h = -31.6
    E_leak = -55
    E_syn = 0
    E_Ca = 120
    E_injury = -15  # reversal potential for injury induced leak = solution of GHK if all permeabilities are equal

    f = 0.01  # percent of free to bound Ca2+
    alpha = 0.002  # uM/pA; convertion factor from current to concentration
    kCaS = 0.024  # /ms; Ca2+ removal rate, kCaS is proportional to  1/tau_removal; 0.008 - 0.025
    # A = 1.26e-5           # cm^2; cell surface area; radius is 10um
    # Ca_out = 2            # mM; extracellular Ca2+ concentration

    SCa = 1  # uM; half-saturation of [Ca2+]; 25uM in Ermentrount book, 0.2uM in Kurian et al. 2011
    tauKCa_0 = 50  # ms
    tauh_0_act = 1  # 10, 1
    tauh_0_inact = 1  # 10, 5         % ms; vary from 50ms to 1000ms
    tau_mA_scale = 1
    tau_hA_scale = 10  # scaling factor for tau_hA

    alpha_m = 0.36 * (V + 33) / (1 - exp(-(V + 33) / 3))
    beta_m = - 0.4 * (V + 42) / (1 - exp((V + 42) / 20))
    m_inf = alpha_m / (alpha_m + beta_m)
    tau_m = 2 / (alpha_m + beta_m)
    if dt < tau_m:
        m_next = m_inf + (m - m_inf) * exp(-dt / tau_m)
    else:
        m_next = m_inf

    alpha_h = - 0.1 * (V + 55) / (1 - exp((V + 55) / 6))
    beta_h = 4.5 / (1 + exp(-V / 10))
    h_inf = alpha_h / (alpha_h + beta_h)
    tau_h = 2 / (alpha_h + beta_h)
    if dt < tau_h:
        h_next = h_inf + (h - h_inf) * exp(-dt / tau_h)
    else:
        h_next = h_inf

    alpha_s = 0.0077 / (1 + exp((V - 18) / 9))
    beta_s = 0.0077 / (1 + exp((18 - V) / 9))
    tau_s = 129.2
    s_inf = alpha_s / (alpha_s + beta_s)
    if dt < tau_s:
        s_next = s_inf + (s - s_inf) * exp(-dt / tau_s)
    else:
        s_next = s_inf

    gNa = GNa * power(m_next, 2) * h_next
    I_Na = gNa * (V - E_Na)

    alpha_n_20 = 0.0047 * (V - 8) / (1 - exp(-(V - 8) / 12))
    beta_n_20 = exp(-(V + 127) / 30)
    n_inf = alpha_n_20 / (alpha_n_20 + beta_n_20)
    alpha_n = 0.0047 * (V + 12) / (1 - exp(-(V + 12) / 12))
    beta_n = exp(-(V + 147) / 30)
    tau_n = 1 / (alpha_n + beta_n)
    if dt < tau_n:
        n_next = n_inf + (n - n_inf) * exp(-dt / tau_n)
    else:
        n_next = n_inf

    gK = GK * power(n_next, 4)
    I_K = gK * (V - E_K)

    alpha_mCaL = 7.5 / (1 + exp((13 - V) / 7))
    beta_mCaL = 1.65 / (1 + exp((V - 14) / 4))
    mCaL_inf = alpha_mCaL / (alpha_mCaL + beta_mCaL)
    tau_mCaL = 1 / (alpha_mCaL + beta_mCaL)
    if dt < tau_mCaL:
        mCaL_next = mCaL_inf + (mCaL - mCaL_inf) * exp(-dt / tau_mCaL)
    else:
        mCaL_next = mCaL_inf

    alpha_hCaL = 0.0068 / (1 + exp((V + 30) / 12))
    beta_hCaL = 0.06 / (1 + exp(-V / 11))
    hCaL_inf = alpha_hCaL / (alpha_hCaL + beta_hCaL)
    tau_hCaL = 1 / (alpha_hCaL + beta_hCaL)
    if dt < tau_hCaL:
        hCaL_next = hCaL_inf + (hCaL - hCaL_inf) * exp(-dt / tau_hCaL)
    else:
        hCaL_next = hCaL_inf

    gCaL = GCaL * mCaL_next * hCaL_next
    I_CaL = gCaL * (V - E_Ca)

    mM_inf = 1 / (1 + exp(-(V + 35) / 10))
    tau_mM = 2000 / (3.3 * (exp((V + 35) / 40) + exp(-(V + 35) / 20)))
    if dt < tau_mM:
        mM_next = mM_inf + (mM - mM_inf) * exp(-dt / tau_mM)
    else:
        mM_next = mM_inf

    gM = GM * power(mM_next, 2)
    I_M = gM * (V - E_K)

    mKCa_inf = CaS ** 2 / (CaS ** 2 + SCa ** 2)
    tau_mKCa = tauKCa_0 / (1 + (CaS / SCa) ** 2)
    if dt < tau_mKCa:
        mKCa_next = mKCa_inf + (mKCa - mKCa_inf) * exp(-dt / tau_mKCa)
    else:
        mKCa_next = mKCa_inf

    gKCa = GKCa * power(mKCa_next, 1)
    I_KCa = gKCa * (V - E_K)

    mA_inf = (0.0761 * exp((V + 94.22) / 31.84) / (1 + exp((V + 1.17) / 28.93))) ** (1 / 3)
    tau_mA = (0.3632 + 1.158 / (1 + exp((V + 55.96) / 20.12))) * tau_mA_scale
    if dt < tau_mA:
        mA_next = mA_inf + (mA - mA_inf) * exp(-dt / tau_mA)
    else:
        mA_next = mA_inf

    hA_inf = (1 / (1 + exp(0.069 * (V + 53.3)))) ** 4
    tau_hA = (0.124 + 2.678 / (1 + exp((V + 50) / 16.027))) * tau_hA_scale
    if dt < tau_hA:
        hA_next = hA_inf + (hA - hA_inf) * exp(-dt / tau_hA)
    else:
        hA_next = hA_inf

    gA = GA * power(mA_next, 3) * hA_next
    I_A = gA * (V - E_K)

    mh_inf_next = 1 / (1 + exp((V + 87.6) / 11.7))
    if mh_inf_next > mh_inf:
        tau_mh = tauh_0_act * (53.5 + 67.7 * exp((V + 120) / 22.4))
    else:
        tau_mh = tauh_0_inact * (40.9 - 0.45 * V)

    if dt < tau_mh:
        mh_next = mh + (dt * (mh_inf_next - mh) / tau_mh)
    else:
        mh_next = mh_inf_next

    gh = Gh * mh_next
    I_h = gh * mh * (V - E_h)
    I_leak = Gleak * (V - E_leak)
    I_injury = Ginjury * (V - E_injury)
    I_leak = I_leak + I_injury
    I_syn = Gsyn * (V - E_syn)

    CaS_next = CaS * exp(-f * kCaS * dt) - alpha / kCaS * I_CaL * (1 - exp(-f * kCaS * dt))

    g_inf = gNa + gCaL + gK + gA + gM + gKCa + gh + Gleak + Gsyn + Ginjury
    V_inf = (iclamp + gNa * E_Na + gCaL * E_Ca + (
            gK + gA + gM + gKCa) * E_K + gh * E_h + Gleak * E_leak + Ginjury * E_injury + Gsyn * E_syn) / g_inf
    tau_tspn = C / g_inf
    V_next = V_inf + (V - V_inf) * exp(-dt / tau_tspn)

    dy = [V_next, CaS_next, m_next, h_next, n_next, mA_next, hA_next, mh_next, mh_inf_next, mM_next, mCaL_next,
          hCaL_next, s_next, mKCa_next, I_Na, I_K, I_CaL, I_M, I_KCa, I_A, I_h, I_leak]

    return np.array(dy)


'''
%   generates a synaptic conductance trace

%   event_time: vector of times in ms at which a synaptic event occurs
%   event_scale: vector of corresponding amplitude of synaptic event in nanosiemens

%   gsyn: synaptic conductance in nS
'''


def tSPN_gsyn(event_time: [], event_scale: []):
    dt = 0.1
    event_index = [round(t / dt) for t in event_time]
    tau_rise = 1
    tau_decay = 15
    t = np.arange(0, 200 + dt, dt)
    gsyn = np.zeros(int(max(event_index)) + len(t))
    e_tau_decay = [exp(-i / tau_decay) for i in t]
    e_tau_rise = [exp(-i / tau_rise) for i in t]
    gEPSC = [a - b for a, b in zip(e_tau_decay, e_tau_rise)]
    gEPSC = [item / max(gEPSC) for item in gEPSC]

    temp = 0
    for index in range(len(event_time)):
        gEPSC_temp = [item * event_scale[index] for item in gEPSC]
        gsyn[int(event_index[index]):len(t) + int(event_index[index])] += gEPSC_temp
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

The code is kinda slow compared to matlab version
y0 is different
'''


def tSPN_ihold(gmax: [], vhold, bounds: []):
    gmax = np.array(gmax)
    bounds = np.array(bounds)

    if bounds.size is 2:
        i_lb = min(bounds)
        i_ub = max(bounds)
    else:
        i_lb = -100
        i_ub = 100

    n_reps = round(math.log(i_ub - i_lb, 2))
    y0 = []

    for index in range(n_reps):
        ihold = (i_ub + i_lb) / 2
        iclamp = np.array([[ihold] * 10000], dtype='float')
        V, I = tSPN(gmax, iclamp, y0, [])

        is_firing = max(V) > 0

        if is_firing or V[-1] > vhold:
            i_ub = ihold
        else:
            i_lb = ihold
            y0 = I[-1, 0, :]
    return ihold, y0


'''
% This code calculates the input conductance of a model cell. 

% gmax: vector of parameters for a specific model cell in the following
%  order: [GNa GK GCaL GM GKCa GA GH GLeak Cm GImp] (GImp is optional, default is 0)
%  Conductance is measured in nS, capacitance is measured in pF. 
%
% ihold: current required to hold the given model cell at the specified
%  holding voltage. May be obtained from tSPN_ihold.
%
% y0: value of all parameters at the end of the simulation. Can be used as
%  the initial value for subsequent simulations. Useful for reducing
%  initialization time. May be obtained from tSPN_ihold.

% gin: calculated input conductance of the model cell in nS. 
'''


def tSPN_gin(gmax: [], ihold, y0):
    itest = -5
    iclamp = np.array([ihold] * 100000, dtype='float')
    iclamp[50000:-1] = ihold + itest

    V, I = tSPN(gmax, [iclamp], y0, [])
    del_V = min(V[50000:-1]) - V[49999]

    gin = itest / del_V
    return gin


def tSPN_synThresh():
    return .4344, -50, -40, [300, 2000, 0, 0, 0, 0, .4344, 100]


'''
%this code calculates the rheobase current magnitude

% gmax: vector of parameters for a specific model cell in the following
%  order: [GNa GK GCaL GM GKCa GA GH GLeak Cm GImp] (GImp is optional, default is 0)
%  Conductance is measured in nS, capacitance is measured in pF. 
%
% ihold: current required to hold the given model cell at the specified
%  holding voltage. May be obtained from tSPN_ihold.
%
% y0: value of all parameters at the end of the simulation. Can be used as
%  the initial value for subsequent simulations. Useful for reducing
%  initialization time. May be obtained from tSPN_ihold.

% irheo: minimal current required to produce a single action potential, i.e. rheobase (pA).
'''


def tSPN_rheo(gmax, ihold, y0):
    i_ub = 1000.
    i_lb = 0.

    sth = 0
    tol = .1

    n_iters = round(math.log((i_ub - i_lb) / tol, 2))

    for index in range(n_iters):
        iclamp = np.array([ihold] * 5000, dtype='float')
        iclamp[100:-1] = ihold + (i_ub + i_lb) / 2
        V, I = tSPN(gmax, [iclamp], y0, [])

        isfiring = max(V) > sth

        if isfiring:
            irheo = i_ub
            i_ub = float((i_ub + i_lb) / 2)
        else:
            i_lb = float((i_ub + i_lb) / 2)
    return irheo


def tSPN_network():
    # freqs = 10. ^ linspace(-2, 2, 100);
    freqs = np.array([.1, 1, 10, 100])
    synaptic_gain = np.array([0] * len(freqs))

    # Define an MxN matrix. M = postganglionic neurons. N = preganglionic neurons.

    for index in range(len(freqs)):
        # 1 postganglionic, 5 preganglionics
        net = np.array([[10, 3, 3, 3, 3]])

        postN = net.shape[0]
        preN = net.shape[1]

        # For each preganglionic, generate a vector of spike times pulled from a
        # given distribution.

        dt = .1  # .1ms sampling interval
        HR = 10  # HR = 10Hz
        # f = .1 #fR = 1Hz
        f = freqs[index]

        n_samples = 1000 / HR / dt  # samples per cardiac cycle

        t = np.arange(n_samples) * dt  # ms

        # pdf_type = 'tri'
        # pdf_type = 'uni'
        pdf_type = 'duty'  # duty cycle
        # pdf_type = 'sin'
        # pdf_type = 'wrap'  #??

        if pdf_type is 'tri':
            pdf = t
        if pdf_type is 'uni':
            pdf = np.ones(int(n_samples))
        if pdf_type is 'duty':
            dc = 0.2
            pdf = np.zeros(int(n_samples))
            pdf[:int(dc * len(pdf))] = 1
        if pdf_type is 'sin':
            pdf = -(np.cos(2 * np.pi * HR * t / 1000)) + 1

        ## pdf starts at 0, matlab starts at non-0
        pdf = pdf / sum(pdf) * f / HR
        n_cycles = 1000

        cont_pdf = np.matlib.repmat(pdf, 1, n_cycles)
        cont_t = np.matlib.repmat(t, 1, n_cycles)

        pre_event_time = [None] * preN
        for i in range(preN):
            mcs = np.random.random_sample(cont_pdf.size)
            pre_event_time[i] = np.asarray(np.where(mcs < cont_pdf[0])) * dt

        # For each postganglionic, generate a waveform of synaptic conductance based
        # on preganglionic firing and network connectivity matrix, run the simulation
        # with given synaptic waveform and detect spikes

        gmax = np.array([400, 2000, 1.2, 10, 10, 10, 1, 1, 100])  # for now, assume uniform population
        iclamp = np.array([0] * 10000, dtype='float')

        y0 = None
        V, I = tSPN(gmax, [iclamp], y0, [])
        y0 = I[-1, -1, :]
        iclamp = np.array([0] * n_cycles * int(n_samples), dtype='float')

        post_event_time = [None] * postN
        for i in range(postN):
            gsyn_tot = np.array([0.] * (len(iclamp) + 2001))  # 2001?
            syn_weights = net[i, :]
            for j in range(preN):
                if syn_weights[j] > 0:
                    event_time = pre_event_time[j].ravel()
                    event_scale = np.array([syn_weights[j]] * len(event_time))
                    gsyn = tSPN_gsyn(event_time, event_scale)
                    gsyn_tot[: len(gsyn)] += gsyn
            V, I = tSPN(gmax, [iclamp], y0, [gsyn_tot])
            post_event_time[i] = np.asarray(np.where(np.diff(V > 0) == 1)) * dt

            plt.subplot(211)
            plt.plot(V)
            plt.subplot(212)
            plt.plot(gsyn_tot)
        plt.show()
        events = pre_event_time[0]
        plt.plot(np.remainder(events, 1000 / HR), np.zeros(len(events)), 'b*')
        events = post_event_time[0]
        plt.plot(np.remainder(events, 1000 / HR), np.zeros(len(events)) + 1, 'r*')
        plt.show()

        sV = V + 100
        plt.plot(sV * np.sin(cont_t * 2 * np.pi / 100)[0], sV * np.cos(cont_t * 2 * np.pi / 100)[0])
        plt.plot(pdf * max(sV) / max(pdf) * np.sin(t * 2 * np.pi / 100),
                 pdf * max(sV) / max(pdf) * np.cos(t * 2 * np.pi / 100), 'r')
        plt.show()

        f1 = len(pre_event_time[0][0]) / (n_cycles / HR)
        n = 0
        for i in range(len(pre_event_time)):
            n += len(pre_event_time[i])

        fm = float(n / len(pre_event_time) / (n_cycles / HR))
        fp = float(len(post_event_time[0][0]) / (n_cycles / HR))

        synaptic_gain[index] = float(fp / f1)
        print(fp, f1)

    plt.semilogx(freqs, synaptic_gain)
    plt.show()
    plt.plot(freqs, synaptic_gain * freqs)
    plt.show()


def tSPN_synapticGain():
    freqs = np.array([.1, 1, 10, 100])
    ###Synaptic Gain, CVC vs MVC
    ###Calculate the synaptic gain for a cutaneous vasoconstrictor (uniform
    ###probability distribution, setup.pdf_type = 'uni') versus a muscle
    ###vasoconstrictor (sinusoidally distributed pdf, setup.pdf_type = 'sin').
    ###Calculate the synaptic gain at frequencies ranging from 10^-2 (.01H) to

    setup = defaultSetup()
    for i in range(len(freqs)):
        setup.freq = freqs[i]
        out = tSPN_net(setup)


class defaultSetup:
    def __init__(self):
        self.duration = 100  # 100s simulation
        self.net = np.array([10, 3, 3, 3, 3])  # 1 primary input, 4 secondary inputs
        self.freq = 1  # target firing rate
        self.HR = 10  # heart rate
        self.dt = .1  # sampling interval
        self.pdf_type = 'uni'  # uniform firing probability
        self.gmax = np.array([400, 2000, 1.2, 10, 10, 10, 1, 1, 100])  # default gmax
        self.thresh = 0
        self.ihold = 0
        self.tol = .01


def tSPN_net(setup):
    setup.n_cycles = setup.duration * setup.HR
    setup.n_samples = 1000 / setup.HR / setup.dt
    setup.iclamp = np.array([setup.ihold] * setup.n_cycles * setup.n_samples, dtype='float')

    event_time = spikeTrain(setup)
    setup.pre_event_time = event_time

    gsyn_tot = calcGsyn(setup)
    setup.gsyn_tot = gsyn_tot

    V_tot = runSim(setup)
    setup.V_tot = V_tot

    event_time = spikeDetect(setup)
    setup.post_event_time = event_time


def spikeTrain(setup):
    pdf_type = setup.pdf_type
    freq = setup.freq
    HR = setup.HR
    n_cycles = setup.n_cycles
    net = setup.net
    dt = setup.dt
    n_samples = setup.n_samples

    t = np.arange(n_samples) * dt

    # pdf_type = 'tri'
    # pdf_type = 'uni'
    pdf_type = 'duty'  # duty cycle
    # pdf_type = 'sin'
    # pdf_type = 'wrap'  #??

    if pdf_type is 'tri':
        pdf = t
    if pdf_type is 'uni':
        pdf = np.ones(int(n_samples))
    if pdf_type is 'duty':
        dc = 0.2
        pdf = np.zeros(int(n_samples))
        pdf[:int(dc * len(pdf))] = 1
    if pdf_type is 'sin':
        pdf = -(np.cos(2 * np.pi * HR * t / 1000)) + 1


if __name__ == "__main__":
    gmax = np.array([400, 3000, 1.2, 40, 60, 80, 1, 2, 100])
    ihold, y0 = tSPN_ihold(gmax, -70, np.array([-100,100]))
    iclamp = np.array([ihold] * 50000, dtype='float')
    iclamp[10000:40000]+=50
    # y0 = []
    gsyn = []
    V, I = tSPN(gmax, [iclamp], y0, gsyn)
    plt.plot(V)
    plt.show()
    # print(I)
    # plt.plot(tSPN_gsyn([1, 2, 3], [1, 2, 3]))
    # plt.show()
    # ihold, y0 = tSPN_ihold(gmax, -20., [])
    # print(tSPN_gin(gmax, -30., []))
    # print(tSPN_rheo(gmax, -30, []))
    # tSPN_network()
    # tSPN_synapticGain()
