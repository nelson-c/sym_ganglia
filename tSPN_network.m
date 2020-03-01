%% Set up simulation

% freqs = 10.^linspace(-2,2,100);
freqs = [.1 1 10 100];


%Define an MxN matrix. M = #postganglionic neurons. N = #preganglionic
%neurons.
for trial_ind = 1:length(freqs)

    net = [ %1 postganglionic, 5 preganglionics
    10 3 3 3 3
    ];

    [postN preN] = size(net);

%For each preganglionic, generate a vector of spike times pulled from a
%given distribution.

    dt = .1; %.1ms sampling interval
    HR = 10; %HR = 10Hz
%     f = .1; %fR = 1Hz
    f = freqs(trial_ind);

    n_samples = 1000/HR/dt; % #samples per cardiac cycle

    t = (1:n_samples)*dt; %ms

    pdf_type = 'uni';
    switch pdf_type
        case 'tri'
            pdf=t;
        case 'uni'
            pdf=ones(size(t));
        case 'duty'
            dc = 0.2;
            pdf = zeros(size(t));
            pdf(1:dc*length(pdf))=1;
        case 'sin'
            pdf = -cos(2*pi*HR*t/1000)+1;
        case 'wrap' %wrapped normal
    end
    pdf = pdf/sum(pdf)*f/HR; %normalize pdf to desired overall firing rate
    n_cycles = 1000; % #cardiac cycles

    cont_pdf = repmat(pdf,1,n_cycles);
    cont_t = repmat(t,1,n_cycles);
    
    pre_event_time = cell(1,preN);
    for ind = 1:preN
        mcs = rand(size(cont_pdf)); %monte-carlo simulation
        pre_event_time{ind} = find(mcs<cont_pdf)*dt; %ms
    end

%For each postganglionic, generate a waveform of synaptic conductance based
%on preganglionic firing and network connectivity matrix, run the simulation
%with given synaptic waveform and detect spikes 

    gmax = [400 2000 1.2 10 10 10 1 1 100]; %for now, assume uniform population

    iclamp = zeros(1,10000);
    [V I] = tSPN(gmax,iclamp);
    y0 = I(1,end,:);
    iclamp = zeros(1,n_cycles*n_samples);
    
    post_event_time = cell(1,postN);
    for ind = 1:postN
        gsyn_tot = zeros(1,length(iclamp)+2001);
        syn_weights = net(ind,:);
        for jnd = 1:preN
            if syn_weights(jnd)>0
                event_time = pre_event_time{jnd};
                event_scale = zeros(size(event_time))+syn_weights(jnd); %for now, amplitudes are constant. Eventually should be poisson 
                gsyn = tSPN_gsyn(event_time,event_scale);
                gsyn_tot(1:length(gsyn)) = gsyn_tot(1:length(gsyn))+gsyn;            
            end
        end
        V = tSPN(gmax,iclamp,y0,gsyn_tot);
        post_event_time{ind}=find(diff(V>0)==1)*dt;
        
        figure(1),
        subplot(2,1,1),plot(V),hold on
        subplot(2,1,2),plot(gsyn_tot),hold on
    end

    figure(2),hold on
    events = pre_event_time{1};
    plot(mod(events,1000/HR),zeros(size(events)),'b*')

    events = post_event_time{1};
    plot(mod(events,1000/HR),zeros(size(events))+1,'r*')

    figure(3)
    sV = V+100;

    clf
    plot(sV.*sin(cont_t*2*pi/100),sV.*cos(cont_t*2*pi/100))
    hold on, 
    plot(pdf*max(sV)/max(pdf).*sin(t*2*pi/100),pdf*max(sV)/max(pdf).*cos(t*2*pi/100),'r')
    events = post_event_time{1};
    % plot(,'r*')
    axis equal

% synaptic gain
    f1 = length(pre_event_time{1})/(n_cycles/HR); %firing rate of primary

    n = 0;
    for ind = 1:length(pre_event_time)
        n = n + length(pre_event_time{ind});
    end
    fm = n/length(pre_event_time)/(n_cycles/HR); %mean firing rate of all pre

    fp = length(post_event_time{1})/(n_cycles/HR);
    
    synaptic_gain(trial_ind) = fp/f1;
end

figure(4),semilogx(freqs,synaptic_gain)
figure(5),plot(freqs,synaptic_gain.*freqs)

%%
network_matrix = [
    3 0 0 0 0
    0 1 0 0 0
    0 0 1 0 0
    0 0 0 1 0
    0 0 0 0 1
    3 1 1 1 1
];

[post_N,pre_N] = size(network_matrix);

n_cycles = 10;
cycle_dur = 1000;
pre_st = cycle_dur*(1:n_cycles); %1Hz

pre_fire = rand(pre_N,n_cycles)<.2;

gmax = [400 2000 1.2 10 10 10 1 1 100];
iclamp = zeros(1,10000);
[V I] = tSPN(gmax,iclamp);
y0 = I(1,end,:);
iclamp = zeros(1,n_cycles*cycle_dur*10);

clf
for ind = 1:post_N
    syn_weights = network_matrix(ind,:);
    event_time = [];
    event_scale = [];
    for jnd = 1:pre_N
        event_time = [event_time pre_st(pre_fire(jnd,:))];
        event_scale = [event_scale ones(1,sum(pre_fire(jnd,:)))*syn_weights(jnd)];
    end
    
    
    gsyn = tSPN_gsyn(event_time, event_scale);
    [V I] = tSPN(gmax,iclamp,y0,gsyn);

    plot(V+ind*20),hold on
end


%% Probability distribution functions
dt = .0001; %.1ms
HR = 10; %HR = 10Hz
f = 1; %fR = 10Hz

n_samples = 1/HR/dt;  
T = 1/HR;

t = (1:n_samples)*dt; %seconds

pdf_type = 'duty';
switch pdf_type
    case 'tri'
        pdf=t;
    case 'uni'
        pdf=ones(size(t));
    case 'duty'
        dc = 0.2;
        pdf = zeros(size(t));
        pdf(1:dc*length(pdf))=1;
    case 'sin'
        pdf = -cos(2*pi*HR*t)+1;
end

pdf = pdf/sum(pdf)*f/HR; %normalize pdf to desired overall firing rate

% plot(t,pdf),hold on, plot(t,pdf2,'r')

n_cycles = 10000;

cont_pdf = repmat(pdf,1,n_cycles);
cont_t = repmat(t,1,n_cycles);
mcs = rand(size(cont_pdf)); %monte-carlo simulation

spike_train = mcs<cont_pdf; 
% clf
% plot(cont_pdf),hold on,plot(mcs<cont_pdf,'r') 
clf
plot(cont_t,spike_train), hold on, plot(t,pdf/max(pdf),'r')

sum(spike_train)/length(spike_train)/dt
