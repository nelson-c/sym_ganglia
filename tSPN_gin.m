function gin = tSPN_gin(gmax,ihold,y0)
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

itest = -5;

iclamp = zeros(1,100000)+ihold;
iclamp(50000:end)=ihold+itest;

V = tSPN(gmax,iclamp,y0);
% figure(4)
% plot(V),hold on
del_V = min(V(50000:end))-V(49999);

gin = itest/del_V;

