# Generated with SMOP  0.41
from libsmop import *
# tSPN_gin.m

    
@function
def tSPN_gin(gmax=None,ihold=None,y0=None,*args,**kwargs):
    varargin = tSPN_gin.varargin
    nargin = tSPN_gin.nargin

    # This code calculates the input conductance of a model cell.
    
    # gmax: vector of parameters for a specific model cell in the following
#  order: [GNa GK GCaL GM GKCa GA GH GLeak Cm GImp] (GImp is optional, default is 0)
#  Conductance is measured in nS, capacitance is measured in pF.
    
    # ihold: current required to hold the given model cell at the specified
#  holding voltage. May be obtained from tSPN_ihold.
    
    # y0: value of all parameters at the end of the simulation. Can be used as
#  the initial value for subsequent simulations. Useful for reducing
#  initialization time. May be obtained from tSPN_ihold.
    
    # gin: calculated input conductance of the model cell in nS.
    
    itest=- 5
# tSPN_gin.m:17
    iclamp=zeros(1,100000) + ihold
# tSPN_gin.m:19
    iclamp[arange(50000,end())]=ihold + itest
# tSPN_gin.m:20
    V=tSPN(gmax,iclamp,y0)
# tSPN_gin.m:22
    # figure(4)
# plot(V),hold on
    del_V=min(V(arange(50000,end()))) - V(49999)
# tSPN_gin.m:25
    gin=itest / del_V
# tSPN_gin.m:27