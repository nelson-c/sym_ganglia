%% Plot example traces
%Run a simulation for a preganglionic network that fires at 0.1, 1, 10, and
%100 Hz and plot the voltage and synaptic traces. 
freqs = [.1 1 10 100];
setup = defaultSetup;

for ind = 1:length(freqs)
    setup.freq = freqs(ind);
    out = tSPN_net(setup);
    figure(1)
    plot(out.V_tot), hold on
    figure(2)
    plot(out.gsyn_tot), hold on
end

%% Synaptic Gain, CVC vs MVC
%Calculate the synaptic gain for a cutaneous vasoconstrictor (uniform
%probability distribution, setup.pdf_type = 'uni') versus a muscle
%vasoconstrictor (sinusoidally distributed pdf, setup.pdf_type = 'sin').
%Calculate the synaptic gain at frequencies ranging from 10^-2 (.01H) to
%10^2 (100Hz) with values logarithmically distributed.
st = tic;
n_events = 1000;
freqs = 10.^linspace(-2,2,200);
% freqs = [.01 .1 100];
setup = defaultSetup;
setup.duration = 1000;

for jnd = 1:2
    if jnd==1
        setup.pdf_type = 'uni';
    elseif jnd==2
        setup.pdf_type = 'sin';
    end

    n1 = zeros(size(freqs));
    np = zeros(size(setup.net,1),length(freqs));
    dur = zeros(size(freqs));
    
    for ind = 1:length(freqs)
        setup.freq = freqs(ind);
        tic
        while n1(ind)<n_events
        %keep running the simulation until there are enough primary presynaptic
        %events to make an appropriate calculation
    %         out = tSPN_net(setup);
            out = tSPN_sparseNet(setup);
            n1(ind)   = n1(ind)  + out.n_pre(1);
            np(:,ind) = np(:,ind)  + out.n_post';
            dur(ind)  = dur(ind) + out.duration;
            toc
        end
        disp(['completed FR = ' num2str(freqs(ind))])
    end
    
    figure(1), hold on
    plot(freqs,np'./repmat(dur',1,size(np,1)));

    figure(2), hold on
    synGainN = np./repmat(n1,size(np,1),1);
    semilogx(freqs,synGainN);
    
end
toc(st)    


%% How does N impact firing rate (takes about 6 hours to run)
%Calculate the postganglionic firing rate and synaptic gain as a function
%of the number of presynaptic inputs. 

st = tic;
n_events = 1000; %need ~1000 events to make appropriate calculation

%Calculate the synatpic gain at frequencies ranging from 10^-2 (.01H) to
%10^2 (100Hz) with values logarithmically distributed.
freqs = 10.^linspace(-2,2,200);

% freqs = .1;
% dur = n_events./freqs;
setup = defaultSetup;

%The setup.net variable dictates the network connections. The first row
%indicates a single primary input (gsyn = 10) and no secondary inputs. The
%second row keeps the same primary input, but adds a single secondary input
%(gsyn = 3). Each row adds another secondary input until the last row,
%which is one primary input and 9 secondary inputs. 
setup.net = [10 0 0 0 0 0 0 0 0 0
             10 3 0 0 0 0 0 0 0 0
             10 3 3 0 0 0 0 0 0 0
             10 3 3 3 0 0 0 0 0 0 
             10 3 3 3 3 0 0 0 0 0
             10 3 3 3 3 3 0 0 0 0
             10 3 3 3 3 3 3 0 0 0
             10 3 3 3 3 3 3 3 0 0
             10 3 3 3 3 3 3 3 3 0
             10 3 3 3 3 3 3 3 3 3];
setup.duration = 1000;
n1 = zeros(size(freqs));
np = zeros(size(setup.net,1),length(freqs));
dur = zeros(size(freqs));

for ind = 1:length(freqs)
    setup.freq = freqs(ind);
    tic
    while n1(ind)<n_events 
        %keep running the simulation until there are enough primary presynaptic
        %events to make an appropriate calculation
%         out = tSPN_net(setup);
        out = tSPN_sparseNet(setup);
        n1(ind)   = n1(ind)  + out.n_pre(1);
        np(:,ind) = np(:,ind)  + out.n_post';
        dur(ind)  = dur(ind) + out.duration;
        toc
    end
    disp(['completed FR = ' num2str(freqs(ind))])
end

figure(1)
plot(freqs,np'./repmat(dur',1,size(np,1)));

figure(2)
% synGainN = fp./repmat(f1,size(fp(1),1));
synGainN = np./repmat(n1,size(np,1),1);
semilogx(freqs,synGainN);
toc(st)


%% How does impalement conductance impact synaptic integration?
%Calculate the postganglionic firing rate and synaptic gain as a result
%of impalement conductance. 

st = tic;
%first need to set holding current to compare at the same RMP
setup = defaultSetup;
V = tSPN(setup.gmax,zeros(1,10000));
vhold = V(end);

n_events = 1000;
freqs = 10.^linspace(-2,2,200);
% freqs = [0.01 0.1 1 10 100];
% freqs = 0.01;
setup.duration = 1000;

%set gimp at 0nS (whole cell simulation) or 7nS (microelectrode simulation)
for gimp = [0 7]
    setup.gmax(10) = gimp;
    if gimp == 7
        [ihold y0] = tSPN_ihold(setup.gmax,vhold,[-1000 1000]);
        setup.ihold = ihold;
    end
        
    n1 = zeros(size(freqs));
    np = zeros(size(setup.net,1),length(freqs));
    dur = zeros(size(freqs));
    
    for ind = 1:length(freqs)
        setup.freq = freqs(ind);
        tic
        while n1(ind)<n_events
            out = tSPN_sparseNet(setup);
            n1(ind)   = n1(ind)  + out.n_pre(1);
            np(:,ind) = np(:,ind)  + out.n_post';
            dur(ind)  = dur(ind) + out.duration;
            toc
        end
        disp(['completed FR = ' num2str(freqs(ind))])
    end
    

    figure(1), hold on
    plot(freqs,np'./repmat(dur',1,size(np,1)));

    figure(2), hold on
    synGainN = np./repmat(n1,size(np,1),1);
    semilogx(freqs,synGainN);

end
toc(st)

%% How does amplitude of secondary synapses impact synaptic gain? (18hrs)
%Calculate the postganglionic firing rate and synaptic gain as a function
%of secondary synaptic event amplitude. Note that each secondary event has 
%identical amplitude in these simulations. In reality, there should be an
%amplitude distribution but I haven't implemented that yet. 

st = tic;
n_events = 1000; %need ~1000 events to make appropriate calculation
freqs = 10.^linspace(-2,2,200);
% freqs = [.01 .1 1 10 100];
% freqs = .1;
% dur = n_events./freqs;
setup = defaultSetup;

%this setup.net variable shows a single primary input and four secondary
%inputs. The primary input always has an amplitude of 10nS, while the
%secondary inputs vary from 1 to 9nS. 
setup.net = [10 0 0 0 0 
             10 1 1 1 1
             10 2 2 2 2
             10 3 3 3 3
             10 4 4 4 4
             10 5 5 5 5
             10 6 6 6 6
             10 7 7 7 7
             10 8 8 8 8
             10 9 9 9 9];
setup.duration = 1000;
n1 = zeros(size(freqs));
np = zeros(size(setup.net,1),length(freqs));
dur = zeros(size(freqs));

for ind = 1:length(freqs)
    setup.freq = freqs(ind);
    tic
    while n1(ind)<n_events
%         out = tSPN_net(setup);
        out = tSPN_sparseNet(setup);
        n1(ind)   = n1(ind)  + out.n_pre(1);
        np(:,ind) = np(:,ind)  + out.n_post';
        dur(ind)  = dur(ind) + out.duration;
        toc
    end
    disp(['completed FR = ' num2str(freqs(ind))])
end

figure(1)
plot(freqs,np'./repmat(dur',1,size(np,1)));

figure(2)
% synGainN = fp./repmat(f1,size(fp(1),1));
synGainN = np./repmat(n1,size(np,1),1);
semilogx(freqs,synGainN);
toc(st)

%% How do membrane currents affect threshold synaptic events?
%Na K CaL M KCa A H Leak Cm
%calculate synaptic threshold amplitude as a result of changing individual
%parameters

holdRMP = true;

mult = linspace(0,2,100); %change conductance/capacitance from 0 to twice default value

setup = defaultSetup;
V = tSPN(setup.gmax,zeros(1,10000));
vhold = V(end);

for jnd = 1:length(setup.gmax)
    setup = defaultSetup;
    cur = setup.gmax(jnd)*mult;

    for ind = 1:length(cur)
        setup.gmax(jnd) = cur(ind);
        if holdRMP
            [ihold, y0] = tSPN_ihold(setup.gmax,vhold,[-100 100]);
            setup.ihold = ihold;
        end
        amp(ind,jnd) = gsynThresh(setup);
    end
    figure(jnd)
    plot(mult,amp(:,jnd))
    axis([0 2 0 6])
end

%% Change currents to half and double (48hrs to run)
%Calculate the postganglionic firing rate and synaptic gain as a function
%of changing individual parameters (Na, K, Cm, etc.). In each case, I take
%the default value of one parameter and double it, or halve it, and see
%how it affects the postganglionic firing rate to the same presynatpic
%input. 

%Many of the effects seen here are likely due to a change in input
%resistance, membrane time constant, or resting membrane potential. It
%would be important to control for these in the next round of sinulations.
%I never got around to implementing that. 

%Na K CaL M KCa A H Leak Cm


st = tic;
n_events = 1000; %need ~1000 events to make appropriate calculation
freqs = 10.^linspace(-2,2,200);
% freqs = [.01 .1 1 10 100];
% freqs = [10 100];
% dur = n_events./freqs;
setup = defaultSetup;
setup.duration = 1000;


Gmax = setup.gmax;
for ind = 1:9
    gmax = setup.gmax;
    gmax(ind)=setup.gmax(ind)*2;
    Gmax = [Gmax; gmax];
    
    gmax = setup.gmax;
    gmax(ind)=setup.gmax(ind)*0.5;
    Gmax = [Gmax; gmax];
    
end

n1 = zeros(size(Gmax,1),length(freqs));
np = zeros(size(Gmax,1),length(freqs));
dur = zeros(size(Gmax,1),length(freqs));

for jnd = 1:size(Gmax,1)
    setup.gmax = Gmax(jnd,:);
    for ind = 1:length(freqs)
        setup.freq = freqs(ind);
        tic
        while n1(jnd,ind)<n_events
    %         out = tSPN_net(setup);
            out = tSPN_sparseNet(setup);
            n1(jnd,ind)   = n1(jnd,ind)  + out.n_pre(1);
            np(jnd,ind) = np(jnd,ind)  + out.n_post';
            dur(jnd,ind)  = dur(jnd,ind) + out.duration;
            toc
        end
        disp(['completed FR = ' num2str(freqs(ind))])
    end
    
end

figure(1)
for ind = 1:9
    subplot(3,3,ind)
    plot(freqs,np(1,:)./dur(1,:),'k'),hold on
    plot(freqs,np(2*ind,:)./dur(2*ind,:),'b')
    plot(freqs,np(2*ind+1,:)./dur(2*ind+1,:),'r')
end

figure(2)
for ind = 1:9
    subplot(3,3,ind)
    synGainN = np(1,:)./n1(1,:);
    plot(freqs,synGainN,'k'),hold on
    synGainN = np(2*ind,:)./n1(2*ind,:);
    plot(freqs,synGainN,'b')
    synGainN = np(2*ind+1,:)./n1(2*ind+1,:);
    plot(freqs,synGainN,'r')
end


% figure(2)
% % synGainN = fp./repmat(f1,size(fp(1),1));
% 
% semilogx(freqs,synGainN);

save('fp01-100varG.mat','freqs','n1','np','dur')

toc(st)


%% plot amplitude vs difference between events
%If two synaptic events of equal amplitude are a given distance apart, what
%amplitude would they each need to have in order to recruit an action
%potential in the postganglionic cell?

setup = defaultSetup;

ampTh = gsynThresh(setup);
amp = linspace(ampTh/2, ampTh, 100);

delay = zeros(size(amp));
for ind = 1:length(amp)
    delay(ind) = calcDelay(setup,amp(ind));
end

plot(delay(1:end-1),amp(1:end-1))




%% FUNCTIONS
function setup = tSPN_sparseNet(setup)
%Runs the network simulation. This function runs a modified simulation that
%saves time by not running the simulation unless there are multiple events
%within a given temporal window. Runs faster than tSPN_net, but doesn't
%return as many variables. 

    setup.n_cycles = setup.duration*setup.HR; % # of cardiac cycles
    setup.n_samples = 1000/setup.HR/setup.dt; % #samples per cardiac cycle
    setup.iclamp = zeros(1,setup.n_cycles*setup.n_samples)+setup.ihold; %
    
%For each preganglionic, generate a vector of spike times pulled from a
%given distribution.

    event_time = spikeTrain(setup);
    setup.pre_event_time = event_time;

%determine which of the preganglionic inputs have the possibility of
%colliding

    setup.window = 1000; %1 second window for possible summation
  
    event_vec = [];
    pre_vec = [];

    gmax = setup.gmax;
    iclamp = zeros(1,10000)+setup.ihold;
    [V, I] = tSPN(gmax,iclamp);
    y0 = I(1,end,:);

    for ind = 1:length(event_time)
        event_vec = [event_vec event_time{ind}];
        pre_vec = [pre_vec repmat(ind,1,length(event_time{ind}))];
        [event_vec index] = sort(event_vec);
        pre_vec = pre_vec(index);
    end

    %calculate the threshold synaptic conductance
    ampTh = gsynThresh(setup);
    
    n_post = zeros(1,size(setup.net,1));
    for ind = 1:size(setup.net,1)
        amp_vec = setup.net(ind,pre_vec);
        nevent_vec = event_vec(amp_vec~=0);
        npre_vec = pre_vec(amp_vec~=0);
        amp_vec(amp_vec==0)=[];

        if ~isempty(nevent_vec)
            %split preganglionic spike train into groups separated by more
            %than 1s of inactivity. 
            dvec = diff(nevent_vec)>1000;
            grp = cumsum([1 diff(nevent_vec)>1000]);

            for jnd = 1:grp(end)
                if sum(grp==jnd)==1 
                %if there's only one event within the window, determine if
                %it's above or below threshold. If it's suprathreshold,
                %increase the number of postganglionic spikes by 1. 
                    if amp_vec(grp==jnd)>ampTh
                        n_post(ind)=n_post(ind)+1;
                    end
                else
                %if there are multiple events within the window, run a
                %short simulation that includes only the events within the
                %window. 
                    time = nevent_vec(grp==jnd);
                    event_time = time-time(1)+10;
                    event_scale = amp_vec(grp==jnd);
                    gsyn = tSPN_gsyn(event_time,event_scale);
                    setup.V_tot = tSPN(gmax,zeros(size(gsyn))+setup.ihold,y0,gsyn);
                    spike_time = spikeDetect(setup);
                    n_post(ind) = n_post(ind)+length(spike_time{1});
%                     cla,plot(setup.V_tot),hold on,plot(gsyn),title(length(spike_time{1})),pause(.5),drawnow
                end
            end

        end
    end

    setup.n_pre = length(setup.pre_event_time{1});
    setup.n_post = n_post;
end

function setup = tSPN_net(setup)
%Runs the network simulation. This function runs the full simulation and
%returns all relevant variables. Takes more time to run than
%tSPN_sparseNet.

    setup.n_cycles = setup.duration*setup.HR; % # of cardiac cycles
    setup.n_samples = 1000/setup.HR/setup.dt; % #samples per cardiac cycle
    setup.iclamp = zeros(1,setup.n_cycles*setup.n_samples)+setup.ihold; %
    
%For each preganglionic, generate a vector of spike times pulled from a
%given distribution.
       
    event_time = spikeTrain(setup);
    setup.pre_event_time = event_time;
    
%For each postganglionic, generate a waveform of synaptic conductance based
%on preganglionic firing and network connectivity matrix, 

    gsyn_tot = calcGsyn(setup);
    setup.gsyn_tot = gsyn_tot;

%run the simulation with given synaptic waveform 

    V_tot = runSim(setup);
    setup.V_tot = V_tot;
      
%detect spikes in voltage traces    
    
    event_time = spikeDetect(setup);
    setup.post_event_time = event_time;
   
%calculate firing rate

    [n_pre, n_post] = calcFR(setup);
    setup.n_pre     = n_pre; %number of presynaptic events
    setup.n_post    = n_post; %number of postsynaptic events
    setup.preFR     = n_pre./setup.duration; %presynaptic firing rate
    setup.postFR    = n_post./setup.duration; %postsynaptic firing rate

end

function setup = defaultSetup
%defines a default network setup. All simulations modify this setup
%variable to test how individual variables affect synaptic gain and
%postganglionic firing rate. 
    setup.duration = 100;       %100s simulation
    setup.net = [10 3 3 3 3];   %1 primary input, 4 secondary inputs
    setup.freq = 1;             %target firing rate
    setup.HR = 10;              %heart rate
    setup.dt = .1;              %sampling interval
    setup.pdf_type = 'uni';     %uniform firing probability
    setup.gmax = [400 2000 1.2 10 10 10 1 1 100];   %default gmax
    setup.thresh = 0;
    setup.ihold = 0;
    setup.tol = .01;
end

function delay = calcDelay(setup, amp)
%calculates the maximum inter-event time (delay) between successive events,
%each of the same amplitude(amp), that lead to an action potential. In 
%other words, how close to 2 synaptic events of the same size have to be in order to
%recruit a spike?
    gmax = setup.gmax;
    iclamp = zeros(1,10000)+setup.ihold;
    [V, I] = tSPN(gmax,iclamp);
    y0 = I(1,end,:);
    
    d_ub = 1000;
    d_lb = 0;
    
    n_iters = ceil(log2((d_ub-d_lb)/.1));
    
    for ind = 1:n_iters
        delay = mean([d_ub d_lb]);
        gsyn = tSPN_gsyn(10 + [0 delay],amp+[0 0]);
        iclamp = zeros(size(gsyn));
        V = tSPN(gmax,iclamp,y0,gsyn);
%         cla,plot(V),hold on,plot(gsyn),pause(1)
        if any(V>0)
            d_lb = delay;
        else
            d_ub = delay;
        end
    end
end

function amp = gsynThresh(setup)
%calculates the threshold synaptic weight i.e. the minimal synaptic
%conductance that leads to an acton potential
    gmax = setup.gmax;
    iclamp = zeros(1,10000)+setup.ihold;
    [V, I] = tSPN(gmax,iclamp);
    y0 = I(1,end,:);
    
    a_ub = 100;
    a_lb = 0;
    tol = setup.tol;
    n_iters = ceil(log2((a_ub-a_lb)/tol));
    
    for ind = 1:n_iters
        amp = (a_ub+a_lb)/2;
        gsyn = tSPN_gsyn(10,amp);
        V = tSPN(gmax,zeros(size(gsyn))+setup.ihold,y0,gsyn);
        if any(V>0)
            a_ub = amp;
        else
            a_lb = amp;
        end
    end

end

function event_time = spikeTrain(setup)
%Generates a spike train of presynaptic events with the specified
%characteristics. 
    pdf_type    = setup.pdf_type;
    freq        = setup.freq;
    HR          = setup.HR;
    n_cycles	= setup.n_cycles;
    net         = setup.net;
    dt          = setup.dt; 
    n_samples   = setup.n_samples;

    t = (1:n_samples)*dt; %ms

    %Generate a probability density function with the desired shape:
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
    
    pdf = pdf/sum(pdf)*freq/HR; %normalize pdf to desired overall firing rate
    
    cont_pdf = repmat(pdf,1,n_cycles); %repeat this pdf to the desired simulation duration
    
    preN = size(net,2);
    event_time = cell(1,preN);
    for ind = 1:preN
        mcs = rand(size(cont_pdf)); %monte-carlo simulation, rejection sampling
        event_time{ind} = find(mcs<cont_pdf)*dt; %ms
    end
end

function gsyn_tot = calcGsyn(setup)
%Calculate the synaptic conductance trace for each preganglionic neuron and
%sum them all together. 

    iclamp = setup.iclamp;
    net = setup.net;
    [postN preN] = size(net);
    pre_event_time = setup.pre_event_time;
    
    gsyn_tot = zeros(postN,length(iclamp));
    for ind = 1:postN
        syn_weights = net(ind,:);
        for jnd = 1:preN
            if syn_weights(jnd)>0
                event_time = pre_event_time{jnd};
                event_scale = zeros(size(event_time))+syn_weights(jnd); %for now, amplitudes are constant. Eventually should be a distribution 
                gsyn = tSPN_gsyn(event_time,event_scale);
                    gsyn(length(iclamp)+1)=0;
                    gsyn=gsyn(1:length(iclamp));
                gsyn_tot(ind,1:length(gsyn)) = gsyn_tot(ind,1:length(gsyn))+gsyn;            
            end
        end
    end
end

function V_tot = runSim(setup)
%Run the simulation with given intrinsic and synaptic conductances
    gsyn_tot    = setup.gsyn_tot;
    gmax        = setup.gmax;
    
    iclamp = zeros(1,10000)+setup.ihold;
    [V, I] = tSPN(gmax,iclamp);
    y0 = I(1,end,:);
    
    iclamp = setup.iclamp;
    V_tot = zeros(size(gsyn_tot));
    for ind = 1:size(gsyn_tot,1)
        V = tSPN(gmax,iclamp,y0,gsyn_tot(ind,:));
        V_tot(ind,:) = V;
    end

end
    
function event_time = spikeDetect(setup)
%detect spikes in the resultant trace using a threshold method
    V_tot   = setup.V_tot;
    thresh  = setup.thresh;
    dt      = setup.dt;
    
    event_time = cell(1,size(V_tot,1));
    for ind = 1:size(V_tot,1)
        event_time{ind}=find(diff(V_tot(ind,:)>thresh)==1)*dt;      
    end

end

function [n_pre, n_post] = calcFR(setup)
%Calculate the number of spikes in the pre- and postganglionic neurons
    pre_event_time  = setup.pre_event_time;
    post_event_time = setup.post_event_time;
    n_cycles        = setup.n_cycles;
    HR              = setup.HR;
    
    n_pre = zeros(1,length(pre_event_time));
    for ind = 1:length(pre_event_time)
        n_pre(ind) = length(pre_event_time{ind});
    end
        
    n_post = zeros(1,length(post_event_time));
    for ind = 1:length(post_event_time)
        n_post(ind) = length(post_event_time{ind});
    end
    
end