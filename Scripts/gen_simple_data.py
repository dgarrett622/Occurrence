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
ndim, walkers = 3, 500 # simple
#ndim, walkers = 4, 500 # simplet
#ndim, walkers = 5, 500 # simplet2
#ndim, walkers = 6, 500 # simplet3
lnprob = 'simple'
#p0 = (0.467,1.244,-1.238) # SMAll
#p0 = (0.461,1.246,-1.235) # SMFGK
p0 = (2.481,1.256,-0.616) # SMM
#p0 = (0.382,1.169,-1.215,-2.961) # SMtAll
#p0 = (0.381,1.174,-1.212,-2.843) # SMtFGK
#p0 = (0.340,1.171,-1.211,-3.116,2.857) # SMt2All
#p0 = (0.402,1.173,-1.214,-2.762,-1.432) # SMt2FGK
#p0 = (0.386,1.169,-1.195,-1.942,-1.643,-33.43) # SMt3All
#p0 = (0.149,1.192,-1.377,-1.742,33.13,486.0) # SMt3FGK

nburn = 1000 
nsteps = 1000

burnin, samples = gen_MCMC(lnprob,args,p0,ndim,walkers,nburn,nsteps)

labels = ['ln C','alpha','beta'] # simple
#labels = ['ln C','alpha','beta','gamma'] # simplet
#labels = ['ln C','alpha','beta','gamma','mu'] # simplet2
#labels = ['ln C','alpha','beta','gamma','mu','nu'] # simplet3

print_results(samples,labels,lnprob,args)

## only save these once
#bpath = os.path.join(directory,'MCMC Data M',lnprob+'_burnin.npy')
#spath = os.path.join(directory,'MCMC Data M',lnprob+'_samples.npy')
#np.save(bpath,burnin)
#np.save(spath,samples)