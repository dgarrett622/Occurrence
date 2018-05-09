# -*- coding: utf-8 -*-
"""
Updated on May 8, 2018
Author: Daniel Garrett (dg622@cornell.edu)
"""

import os
from funcs import *
import numpy as np
import EtaBase
import scipy.optimize as optimize
import matplotlib.pyplot as plt

data_dir = os.path.dirname(__file__)

transit_data = ['Natalie9p1','Mulders2015','Burke_m','Burke_g']

directory = os.path.dirname(os.path.dirname(__file__))
paths = [os.path.join(directory,'Data',name,name+'.json') for name in transit_data]
etas = [EtaBase.EtaBase(path) for path in paths]

M, au, al, Ru, Rl, T, mu, s = grab_all_data(etas)
args = (au,al,Ru,Rl,T,mu,s)

def min_func(x,au=au,al=al,Ru=Ru,Rl=Rl,T=T,mu=mu,s=s):
    """Chi-square function
    
    Args:
        x (ndarray):
            Model Parameters
        args (tuple):
            Data values needed
    
    Returns:
        f (float):
            Chi-square value
    """
    model_args = tuple(x)
#    model = simple_model(au,al,Ru,Rl,T,*model_args)
    model = breakt3_model(au,al,Ru,Rl,T,*model_args)
    chi2 = ((mu-model)/s)**2
    f = np.sum(chi2)
    return f

#x0 = [2.472,1.258,-0.617] # simple models
x0 = [-0.930,-0.223,1.063,1.007,-0.266,-1.019,2.606,-57.84,124.3,1716.,-0.235,-8.060,-80.94] # break models
res = optimize.minimize(min_func,x0,args=args)
print(res.x)
print('chi-square: {}'.format(res.fun))