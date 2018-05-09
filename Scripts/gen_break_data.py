# -*- coding: utf-8 -*-
"""
Updated on May 8, 2018
Author: Daniel Garrett (dg622@cornell.edu)
"""

import EtaBase
import os
import numpy as np
from funcs import *

transit_data = ['Natalie9p1','Mulders2015','Burke_m','Burke_g']
directory = os.path.dirname(os.path.dirname(__file__))
paths = [os.path.join(directory,'Data',name,name+'.json') for name in transit_data]
etas = [EtaBase.EtaBase(path) for path in paths]

M, au, al, Ru, Rl, T, mu, s = grab_all_data(etas)
args = (au,al,Ru,Rl,T,mu,s)

#ndim, walkers = 7, 500 # break
ndim, walkers = 9, 500 # breakt
#ndim, walkers = 11, 500 # breakt2
#ndim, walkers = 13, 500 # breakt3

lnprob = 'breakt'
#p0 = (0.406,-0.455,1.265,1.048,-0.633,-0.949,2.956) # BRMAll
#p0 = (0.400,-0.453,1.267,1.047,-0.634,-0.951,2.967) # BRMFGK
#p0 = (0.019,-0.641,1.096,1.005,-0.164,-0.876,2.759,-3.829,-2.625) # BRMtAll
p0 = (0.019,-0.641,1.096,1.005,-0.164,-0.876,2.759,-3.829,-2.625) # BRMtFGK
#p0 = (-0.044,-0.556,1.096,1.005,-0.140,-0.902,2.735,-4.115,3.850,-2.489,-2.596) # BRMt2All
#p0 = (-0.044,-0.556,1.096,1.005,-0.140,-0.902,2.735,-4.115,3.850,-2.489,-2.596) # BRMt2FGK
#p0 = (0.00696,-0.520,1.099,1.004,-0.153,-0.891,2.743,-3.300,0.724,-20.20,-1.286,-5.442,-43.32) # BRMt3All
#p0 = (0.02,-0.5,1.2,1.,-0.5,-1.,3.7,1.,-7.8,-81.,0.05,-6.8,-43.) # BRMt3FGK

nburn = 1000
nsteps = 1000

burnin, samples = gen_MCMC(lnprob,args,p0,ndim,walkers,nburn,nsteps)

#labels = ['ln C0','ln C1','alpha0','alpha1','beta0','beta1','Rb'] # break
labels = ['ln C0','ln C1','alpha0','alpha1','beta0','beta1','Rb','gamma0','gamma1'] # breakt
#labels = ['ln C0','ln C1','alpha0','alpha1','beta0','beta1','Rb','gamma0','mu0',\
#          'gamma1','mu1'] # breakt2
#labels = ['ln C0','ln C1','alpha0','alpha1','beta0','beta1','Rb','gamma0','mu0',\
#          'nu0','gamma1','mu1','nu1'] # breakt3

print_results(samples,labels,lnprob,args)

# only save these once
bpath = os.path.join(directory,'MCMC Data FGK',lnprob+'_burnin.npy')
np.save(bpath,burnin)
spath = os.path.join(directory,'MCMC Data FGK',lnprob+'_samples.npy')
np.save(spath,samples)