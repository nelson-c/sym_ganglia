function gsyn = tSPN_gsyn(event_time,event_scale)
%generates a synaptic conductance trace

%event_time: vector of times in ms at which a synaptic event occurs
%event_scale: vector of corresponding amplitude of synaptic event in nanosiemens

%gsyn: synaptic conductance in nS

dt = .1;

event_ind = round(event_time/dt);

tau_rise = 1;
tau_decay = 15;

t = 0:dt:200;

gsyn = zeros(1,max(event_ind)+length(t));

gEPSC = exp(-t/tau_decay)-exp(-t/tau_rise);
gEPSC = gEPSC/max(gEPSC);

for ind = 1:length(event_time)
    gsyn(event_ind(ind)+(1:length(t))) = gsyn(event_ind(ind)+(1:length(t))) + gEPSC*event_scale(ind);
end
