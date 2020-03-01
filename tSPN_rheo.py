# Generated with SMOP  0.41
from libsmop import *
# tSPN_rheo.m

    
@function
def tSPN_rheo(gmax=None,ihold=None,y0=None,*args,**kwargs):
    varargin = tSPN_rheo.varargin
    nargin = tSPN_rheo.nargin

    #this code calculates the rheobase current magnitude
    
    # gmax: vector of parameters for a specific model cell in the following
#  order: [GNa GK GCaL GM GKCa GA GH GLeak Cm GImp] (GImp is optional, default is 0)
#  Conductance is measured in nS, capacitance is measured in pF.
    
    # ihold: current required to hold the given model cell at the specified
#  holding voltage. May be obtained from tSPN_ihold.
    
    # y0: value of all parameters at the end of the simulation. Can be used as
#  the initial value for subsequent simulations. Useful for reducing
#  initialization time. May be obtained from tSPN_ihold.
    
    # irheo: minimal current required to produce a single action potential, i.e. rheobase (pA).
    i_ub=1000
# tSPN_rheo.m:16
    i_lb=0
# tSPN_rheo.m:17
    sth=0
# tSPN_rheo.m:19
    
    tol=0.1
# tSPN_rheo.m:20
    n_iters=ceil(log2((i_ub - i_lb) / tol))
# tSPN_rheo.m:22
    for ind in arange(1,n_iters).reshape(-1):
        iclamp=zeros(1,5000) + ihold
# tSPN_rheo.m:25
        iclamp[arange(100,end())]=ihold + mean(concat([i_ub,i_lb]))
# tSPN_rheo.m:26
        V=tSPN(gmax,iclamp,y0)
# tSPN_rheo.m:27
        isfiring=max(V) > sth
# tSPN_rheo.m:29
        if isfiring:
            irheo=copy(i_ub)
# tSPN_rheo.m:32
            i_ub=mean(concat([i_ub,i_lb]))
# tSPN_rheo.m:33
        else:
            i_lb=mean(concat([i_ub,i_lb]))
# tSPN_rheo.m:35
    