function [V, I] = tSPN(gmax,iclamp,y0,gsyn)

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

if nargin==2 || isempty(y0)
    y0 = [-65 .001 0.0000422117 0.9917 0.00264776 0.5873 0.1269 0.0517 0.5 0.000025... %initial value
    7.6e-5 0.94 0.4 0.000025 0 0 0 0 0 0 0 0];
end

if nargin==4
    if length(gsyn)<numel(iclamp)
        gsyn(numel(iclamp))=0;
    elseif length(gsyn)>numel(iclamp)
        gsyn((numel(iclamp)+1):end)=[];
    end
    gsyn = reshape(gsyn,size(iclamp));
else
    gsyn=zeros(size(iclamp));
end

if length(gmax)==9
    gmax(10)=0;
end

iclamp_lin = iclamp(:);

I = zeros(length(y0), length(iclamp_lin));
dy = tSPN_step(y0,iclamp_lin(1),gmax,.1,gsyn(1));
for ind = 1:length(iclamp_lin)
    dy = tSPN_step(dy, iclamp_lin(ind), gmax, .1, gsyn(ind)); 
    I(:,ind) = dy';'
end

V=reshape(I(1,:),size(iclamp));
I = reshape(I,length(dy),[],size(iclamp,2));
I = permute(I,[2 3 1]);

function dy = tSPN_step(dydt, iclamp, gmax, dt, Gsyn)
%% initialization
    V = dydt(1);  % Somatic membrane voltage (mV)
    CaS = dydt(2);  % somatic (Ca2+)
    m = dydt(3);  % Na activation
    h = dydt(4);  % Na inactivation
    n = dydt(5);  % K activation
    mA = dydt(6);  % A activation
    hA = dydt(7);  % A inactivation
    mh = dydt(8);  % h activation
    mh_inf = dydt(9); % h steady-state activation
    mM = dydt(10);  % M activation
    mCaL = dydt(11);  % CaL activation
    hCaL = dydt(12);  % CaL6 inactivation
    s = dydt(13);  % Na slow inactivation
    mKCa = dydt(14);  % KCa activation

    GNa = gmax(1);  % nS; maximum conductance of INa
    GK = gmax(2);
    GCaL = gmax(3);
    GM = gmax(4);
    GKCa = gmax(5);
    GA = gmax(6);
    Gh = gmax(7);
    Gleak = gmax(8);
    C = gmax(9);
    Ginjury = gmax(10); %added to simulate impalement injury

    E_Na = 60;                  % mV; reverse potential of INa
    E_K = -90; 
    E_h = -31.6;
    E_leak = -55;
    E_syn = 0;
    E_Ca = 120;
    E_injury = -15;  %reversal potential for injury induced leak = solution of GHK if all permeabilities are equal

    f = 0.01;                   % percent of free to bound Ca2+
    alpha = 0.002;              % uM/pA; convertion factor from current to concentration
    kCaS = 0.024;               % /ms; Ca2+ removal rate, kCaS is proportional to  1/tau_removal; 0.008 - 0.025
    % A = 1.26e-5              % cm^2; cell surface area; radius is 10um
    % Ca_out = 2               % mM; extracellular Ca2+ concentration

    SCa = 1;                   % uM; half-saturation of (Ca2+); 25uM in Ermentrount book, 0.2uM in Kurian et al. 2011
    tauKCa_0 = 50;             % ms
    tauh_0_act = 1; %10, 1
    tauh_0_inact = 1; %10, 5         % ms; vary from 50ms to 1000ms
    tau_mA_scale = 1;
    tau_hA_scale = 10;          % scaling factor for tau_hA (default 10)

    %% update dydt
    % Sodium current (pA), Wheeler & Horn 2004 or Yamada et al., 1989
    alpha_m = 0.36 * (V + 33) / (1 - exp(-(V + 33) / 3));
    beta_m = - 0.4 * (V + 42) / (1 - exp((V + 42) / 20));
    m_inf = alpha_m / (alpha_m + beta_m);
    tau_m = 2 / (alpha_m + beta_m);
    if dt < tau_m
        m_next = m_inf + (m - m_inf) * exp(-dt / tau_m); 
    else
        m_next = m_inf;
    end

    alpha_h = - 0.1 * (V + 55) / (1 - exp((V + 55) / 6));
    beta_h = 4.5 / (1 + exp(-V / 10));
    h_inf = alpha_h / (alpha_h + beta_h);
    tau_h = 2 / (alpha_h + beta_h);
    if dt < tau_h
        h_next = h_inf + (h - h_inf) * exp(-dt / tau_h);
    else
        h_next = h_inf;
    end

    alpha_s = 0.0077 / (1 + exp((V - 18) / 9));  % Miles et al., 2005
    beta_s = 0.0077 / (1 + exp((18 - V) / 9));
    tau_s = 129.2;
    s_inf = alpha_s / (alpha_s + beta_s);
    if dt < tau_s 
        s_next = s_inf + (s - s_inf) * exp(-dt / tau_s); 
    else
        s_next = s_inf;
    end

    gNa = GNa * power(m_next, 2) * h_next;
    I_Na = gNa * (V - E_Na);

    % Potassium current (pA), Wheeler & Horn 2004 or Yamada et al., 1989
    alpha_n_20 = 0.0047 * (V - 8) / (1 - exp(-(V - 8) / 12));
    beta_n_20 = exp(-(V + 127) / 30);
    n_inf = alpha_n_20 / (alpha_n_20 + beta_n_20);
    alpha_n = 0.0047 * (V + 12) / (1 - exp(-(V + 12) / 12));
    beta_n = exp(-(V + 147) / 30);
    tau_n = 1 / (alpha_n + beta_n);
    if dt < tau_n 
        n_next = n_inf + (n - n_inf) * exp(-dt / tau_n) ;
    else
        n_next = n_inf;
    end

    gK = GK * power(n_next, 4);
    I_K = gK * (V - E_K);

    % Calcium current (pA), L-type, Bhalla & Bower, 1993
    alpha_mCaL = 7.5 / (1 + exp((13 - V) / 7));
    beta_mCaL = 1.65 / (1 + exp((V - 14) / 4));
    mCaL_inf = alpha_mCaL / (alpha_mCaL + beta_mCaL);
    tau_mCaL = 1 / (alpha_mCaL + beta_mCaL);
    if dt < tau_mCaL 
        mCaL_next = mCaL_inf + (mCaL - mCaL_inf) * exp(-dt / tau_mCaL) ;
    else
        mCaL_next = mCaL_inf;
    end

    alpha_hCaL = 0.0068 / (1 + exp((V + 30) / 12));
    beta_hCaL = 0.06 / (1 + exp(-V / 11));
    hCaL_inf = alpha_hCaL / (alpha_hCaL + beta_hCaL);
    tau_hCaL = 1 / (alpha_hCaL + beta_hCaL);
    if dt < tau_hCaL
        hCaL_next = hCaL_inf + (hCaL - hCaL_inf) * exp(-dt / tau_hCaL);  
    else
        hCaL_next = hCaL_inf;
    end

    gCaL = GCaL * mCaL_next * hCaL_next;
    I_CaL = gCaL * (V - E_Ca);

    % M current (pA), Wheeler & Horn, 2004
    mM_inf = 1 / (1 + exp(-(V + 35) / 10));
    tau_mM = 2000 / (3.3 * (exp((V + 35) / 40) + exp(-(V + 35) / 20)));
    if dt < tau_mM
        mM_next = mM_inf + (mM - mM_inf) * exp(-dt / tau_mM);
    else
        mM_next = mM_inf;
    end
    gM = GM * power(mM_next, 2);
    I_M = gM * (V - E_K);

    % Somatic KCa current (pA), Ermentrout & Terman 2010
    mKCa_inf = CaS ^ 2 / (CaS ^ 2 + SCa ^ 2);
    tau_mKCa = tauKCa_0 / (1 + (CaS / SCa) ^ 2);
    if dt < tau_mKCa 
        mKCa_next = mKCa_inf + (mKCa - mKCa_inf) * exp(-dt / tau_mKCa); 
    else
        mKCa_next = mKCa_inf;
    end
    gKCa = GKCa * power(mKCa_next, 1);
    I_KCa = gKCa * (V - E_K);

    % A-type potassium current (pA), from Rush and Rinzel, 1995
    mA_inf = (0.0761 * exp((V + 94.22) / 31.84) / (1 + exp((V + 1.17) / 28.93))) ^ (1/3);
    tau_mA = (0.3632 + 1.158 / (1 + exp((V + 55.96) / 20.12))) * tau_mA_scale;
    if dt < tau_mA 
        mA_next = mA_inf + (mA - mA_inf) * exp(-dt / tau_mA); 
    else
        mA_next = mA_inf;
    end
    hA_inf = (1 / (1 + exp(0.069 * (V + 53.3)))) ^ 4;
    tau_hA = (0.124 + 2.678 / (1 + exp((V + 50) / 16.027))) * tau_hA_scale;
    if dt < tau_hA 
        hA_next = hA_inf + (hA - hA_inf) * exp(-dt / tau_hA); 
    else
        hA_next = hA_inf;
    end
    gA = GA * power(mA_next, 3) * hA_next;
    I_A = gA * (V - E_K);

    % Ih (pA), Kullman et. al. 2016
    mh_inf_next = 1 / (1 + exp((V + 87.6) / 11.7));
    if mh_inf_next > mh_inf
        tau_mh = tauh_0_act * (53.5 + 67.7 * exp((V + 120) / 22.4)); %I think the - sign before 22.4 was a typo
    else
        tau_mh = tauh_0_inact * (40.9 - 0.45 * V);
    end
    
    if dt < tau_mh
        mh_next = mh + (dt * (mh_inf_next-mh)/tau_mh);
    else
        mh_next = mh_inf_next;
    end   
        
    gh = Gh * mh_next;
    I_h = gh * mh * (V - E_h);

    % Leak current (pA)
    I_leak = Gleak * (V - E_leak);
    
    I_injury = Ginjury * (V - E_injury); %added to simulate injury
    
    I_leak = I_leak + I_injury;

    % Synaptic current (pA)
    I_syn = Gsyn * (V - E_syn);

    % Somatic calcium concentration (uM), Kurian et al., 2011 & Methods of Neuronal Modeling, p. 490. 12.24
    CaS_next = CaS * exp(-f * kCaS * dt) - alpha / kCaS * I_CaL * (1 - exp(-f * kCaS * dt));

    %% update voltage
    g_inf = gNa + gCaL + gK + gA + gM + gKCa + gh + Gleak + Gsyn + Ginjury;
    V_inf = (iclamp + gNa * E_Na + gCaL * E_Ca + (gK + gA + gM + gKCa) * E_K + gh * E_h + Gleak * E_leak + Ginjury * E_injury + Gsyn * E_syn) / g_inf;
    tau_tspn = C / g_inf;
    V_next = V_inf + (V - V_inf) * exp(-dt / tau_tspn);

    dy = [V_next, CaS_next, m_next, h_next, n_next, mA_next, hA_next, mh_next, mh_inf_next, mM_next, mCaL_next, hCaL_next, s_next, mKCa_next, I_Na, I_K,...
          I_CaL, I_M, I_KCa, I_A, I_h, I_leak];