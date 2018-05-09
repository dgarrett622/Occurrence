# -*- coding: utf-8 -*-
"""
Updated on May 8, 2018
Author: Daniel Garrett (dg622@cornell.edu)
"""

import os
from funcs import *
import numpy as np
import EtaBase
import matplotlib.pyplot as plt

models = ['break','breakt','breakt2','breakt3']
labels = [['ln C0','ln C1','alpha0','alpha1','beta0','beta1','Rb'],\
          ['ln C0','ln C1','alpha0','alpha1','beta0','beta1','Rb','gamma0','gamma1'],\
          ['ln C0','ln C1','alpha0','alpha1','beta0','beta1','Rb','gamma0',\
           'mu0','gamma1','mu1'],\
          ['ln C0','ln C1','alpha0','alpha1','beta0','beta1','Rb','gamma0',\
           'mu0','nu0','gamma1','mu1','nu1']]

transit_data = ['Natalie9p1','Mulders2015','Burke_m','Burke_g']
directory = os.path.dirname(os.path.dirname(__file__))
paths = [os.path.join(directory,'Data',name,name+'.json') for name in transit_data]
etas = [EtaBase.EtaBase(path) for path in paths]

M, au, al, Ru, Rl, T, mu, s = grab_all_data(etas)
args = (au,al,Ru,Rl,T,mu,s)

for i in xrange(len(models)):
    print('----------------------------------------------------------')
    print('Fit data for {} model'.format(models[i]))
    path = os.path.join(directory,'MCMC Data All',models[i]+'_samples.npy')
    samples = np.load(path)
    print_results(samples,labels[i],models[i],args)