function irheo = tSPN_rheo(gmax,ihold,y0)
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
i_ub = 1000;
i_lb = 0;

sth = 0; %spike threshold
tol = .1;

n_iters = ceil(log2((i_ub-i_lb)/tol));

for ind = 1:n_iters
    iclamp = zeros(1,5000)+ihold;
    iclamp(100:end) = ihold + mean([i_ub i_lb]);
    V = tSPN(gmax,iclamp,y0);

    isfiring = max(V)>sth;
    
    if isfiring
        irheo = i_ub;
        i_ub = mean([i_ub i_lb]);
    else
        i_lb = mean([i_ub i_lb]);
    end
end
