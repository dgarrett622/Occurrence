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

models = ['simple','simplet','simplet2','simplet3']
labels = [['ln C','alpha','beta'],['ln C','alpha','beta','gamma'],\
          ['ln C','alpha','beta','gamma','mu'],\
          ['ln C','alpha','beta','gamma','mu','nu']]

transit_data = ['Natalie9p1','Mulders2015','Burke_m','Burke_g']
directory = os.path.dirname(os.path.dirname(__file__))
paths = [os.path.join(directory,'Data',name,name+'.json') for name in transit_data]
etas = [EtaBase.EtaBase(path) for path in paths]

M, au, al, Ru, Rl, T, mu, s = grab_all_data(etas)
args = (au,al,Ru,Rl,T,mu,s)

for i in xrange(len(models)):
    print('----------------------------------------------------------')
    print('Fit data for {} model'.format(models[i]))
    path = os.path.join(directory,'MCMC Data FGK',models[i]+'_samples.npy')
    samples = np.load(path)
    print_results(samples,labels[i],models[i],args)