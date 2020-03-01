function [ihold, y0] = tSPN_ihold(gmax,vhold,bounds)

% This code calculates the amount of current required to hold a model cell
% at a given holding voltage. 
%
% gmax: vector of parameters for a specific model cell in the following
%  order: [GNa GK GCaL GM GKCa GA GH GLeak Cm GImp] (GImp is optional, default is 0)
%  Conductance is measured in nS, capacitance is measured in pF. 
%
% vhold: target holding voltage in mV
%
% bounds: (optional) 2-element vector which sets the upper and lower bounds 
% for the injected current in pA. default is [-100 100];
%
% ihold: current required to hold the given model cell at the specified
%  holding voltage.
%
% y0: value of all parameters at the end of the simulation. Can be used as
%  the initial value for subsequent simulations. Useful for reducing
%  initialization time. 

if nargin==3
    i_lb = min(bounds);
    i_ub = max(bounds);
else
    i_lb = -100;
    i_ub = 100;
end

n_reps = ceil(log2(i_ub - i_lb));

iclamp = zeros(10000,1);

ihold = nan;
y0 = [];

for ind = 1:n_reps;
    ihold = (i_ub + i_lb) / 2;
    iclamp(:) = ihold;
    [V I] = tSPN(gmax,iclamp,y0);
    
    is_firing = max(V)>0;
    
    if is_firing || V(end)>vhold
        i_ub = ihold;
    else
        i_lb = ihold;
        y0 = I(end,1,:);
    end
end
