# -*- coding: utf-8 -*-
"""
Updated on May 8, 2018
Author: Daniel Garrett (dg622@cornell.edu)
"""

import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt

# plot settings
plt.rc('text',usetex=True)
plt.rc('font',weight='bold',family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath',r'\renewcommand{\seriesdefault}{\bfdefault}']

def simple_model(au,al,Ru,Rl,T,lnC=np.log(0.5),alpha=-0.61,beta=-1.16):
    """
    Model for occurrence rates with no break radius and no dependence on
    stellar effective temperature
    
    Args:
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Stellar effective temperatures normalized by T_eff_sun
        lnC (float):
            Occurrence rate density for Earth at T_eff_sun
        alpha (float):
            Power law index for semi-major axis
        beta (float):
            Power law index for planetary radius
    
    Returns:
        eta (ndarray):
            Integrated occurrence values for the semi-major axis and radius bins
    
    """
    
    eta = np.exp(lnC)/(alpha*beta)*(au**alpha-al**alpha)*(Ru**beta-Rl**beta)

    return eta

def lnlike_simple(theta,au,al,Ru,Rl,T,x,s):
    """
    Returns the sum of the log-likelihood of the simple model with no 
    dependence on effective temperature
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC, alpha, beta
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        x (ndarray):
            Occurrence rate from data
        s (ndarray):
            Occurrence rate standard deviation from data
    
    Returns:
        f (float):
            Value of the sum of the log-likelihood
    """
    
    model = simple_model(au,al,Ru,Rl,T,*theta)

    f = -0.5*(np.sum(np.log(2.*np.pi*s**2)+(x-model)**2/s**2))
    
    return f

def lnprior_simple(theta):
    """
    Returns the log-prior for the simple model with no dependence on 
    stellar effective temperature
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC, alpha, beta
    
    Returns:
        f (float):
            Value of the log-prior
    """
    # upper and lower bounds on hyperparameters
    lnCu, lnCl = 10.,-5.
    alphau, alphal = 2., -2.
    betau, betal = 2., -2.
    
    # get parameter values for this set
    lnC, alpha, beta = theta
    # check if parameters are within the bounds
    lnC_check = lnCl < lnC < lnCu
    alpha_check = alphal < alpha < alphau
    beta_check = betal < beta < betau
        
    if lnC_check and alpha_check and beta_check:
        f = -np.log(lnCu-lnCl) - np.log(alphau-alphal) - np.log(betau-betal)
    else:
        f = -np.inf
    
    return f

def lnprob_simple(theta,au,al,Ru,Rl,T,x,s):
    """
    Returns the sum of the log-likelihood and log-prior (distribution to sample)
    for the model with no dependence on stellar effective temperature
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC, alpha, beta
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        x (ndarray):
            Occurrence rate from data
        s (ndarray):
            Occurrence rate standard deviation from data
    
    Returns:
        f (float):
            Value of the desired sample distribution
    """
    
    lp = lnprior_simple(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_simple(theta,au,al,Ru,Rl,T,x,s)

def simplet_model(au,al,Ru,Rl,T,lnC=np.log(0.5),alpha=-0.61,beta=-1.16,gamma=0.5):
    """
    Model for occurrence rates with no break radius using linear function on T_eff
    
    Args:
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        lnC (float):
            Occurrence rate density for Earth at T_eff_sun
        alpha (float):
            Power law index for semi-major axis
        beta (float):
            Power law index for planetary radius
        gamma (float):
            Linear coefficient in occurrence dependence on T_eff
    
    Returns:
        eta (ndarray):
            Integrated occurrence values for the semi-major axis and radius bins
    
    """
    x = [gamma]
    f = t_dep(x,T)
    eta = np.exp(lnC)/(alpha*beta)*(au**alpha-al**alpha)*(Ru**beta-Rl**beta)*f

    return eta

def lnlike_simplet(theta,au,al,Ru,Rl,T,x,s):
    """
    Returns the sum of the log-likelihood (Gaussian, independent error bars)
    for simple model with linear dependence on T_eff
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC, alpha, beta, gamma
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        x (ndarray):
            Occurrence rate from data
        s (ndarray):
            Occurrence rate standard deviation from data
    
    Returns:
        f (float):
            Value of the sum of the log-likelihood
    """
    
    model = simplet_model(au,al,Ru,Rl,T,*theta)

    f = -0.5*(np.sum(np.log(2.*np.pi)+2.*np.log(s)+(x-model)**2/s**2))
    
    return f

def lnprior_simplet(theta):
    """
    Returns the log-prior for simple model with linear dependence on T_eff
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC, alpha, beta, gamma
    
    Returns:
        f (float):
            Value of the log-prior
    """
    # upper and lower bounds on hyperparameters
    lnCu, lnCl = 10.,-5.
    alphau, alphal = 2., -2.
    betau, betal = 2., -2.
    gammau, gammal = 100., -100.
    
    # get parameter values for this set
    lnC, alpha, beta, gamma = theta
    # check if parameters are within the bounds
    lnC_check = lnCl < lnC < lnCu
    alpha_check = alphal < alpha < alphau
    beta_check = betal < beta < betau
    gamma_check = gammal < gamma < gammau
    
    if lnC_check and alpha_check and beta_check and gamma_check:
        f = -np.log(lnCu-lnCl) - np.log(alphau-alphal) - np.log(betau-betal) -\
            np.log(gammau-gammal)
    else:
        f = -np.inf
    
    return f

def lnprob_simplet(theta,au,al,Ru,Rl,T,x,s):
    """
    Returns the sum of the log-likelihood and log-prior (distribution to sample)
    for the simple model with linear dependence on T_eff
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC, alpha, beta, gamma
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        x (ndarray):
            Occurrence rate from data
        s (ndarray):
            Occurrence rate standard deviation from data
    
    Returns:
        f (float):
            Value of the desired sample distribution
    """
    
    lp = lnprior_simplet(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_simplet(theta,au,al,Ru,Rl,T,x,s)

def simplet2_model(au,al,Ru,Rl,T,lnC=np.log(0.5),alpha=-0.61,beta=-1.16,gamma=0.5,mu=0.25):
    """
    Model for occurrence rates with no break radius quadratic dependence on T_eff
    
    Args:
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        lnC (float):
            Occurrence rate density for Earth at T_eff_sun
        alpha (float):
            Power law index for semi-major axis
        beta (float):
            Power law index for planetary radius
        gamma (float):
            Linear coefficient in occurrence dependence on T_eff
        mu (float):
            Quadratic coefficient in occurrence dependence on T_eff
    
    Returns:
        eta (ndarray):
            Integrated occurrence values for the semi-major axis and radius bins
    
    """
    
    eta = np.exp(lnC)/(alpha*beta)*(au**alpha-al**alpha)*(Ru**beta-Rl**beta)*\
            (1.+gamma*T+mu*T**2)

    return eta

def lnlike_simplet2(theta,au,al,Ru,Rl,T,x,s):
    """
    Returns the sum of the log-likelihood (Gaussian, independent error bars)
    for the simple model with quadratic dependence on T_eff
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC, alpha, beta, gamma, mu
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        x (ndarray):
            Occurrence rate from data
        s (ndarray):
            Occurrence rate standard deviation from data
    
    Returns:
        f (float):
            Value of the sum of the log-likelihood
    """
    
    model = simplet2_model(au,al,Ru,Rl,T,*theta)

    f = -0.5*(np.sum(np.log(2.*np.pi)+2.*np.log(s)+(x-model)**2/s**2))
    
    return f

def lnprior_simplet2(theta):
    """
    Returns the log-prior for the simple model with quadratic dependence
    on T_eff
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC, alpha, beta, gamma, mu
    
    Returns:
        f (float):
            Value of the log-prior
    """
    # upper and lower bounds on hyperparameters
    lnCu, lnCl = 10.,-5.
    alphau, alphal = 2., -2.
    betau, betal = 2., -2.
    gammau, gammal = 100., -100.
    muu, mul = 500., -500.
    
    # get parameter values for this set
    lnC, alpha, beta, gamma, mu = theta
    # check if parameters are within the bounds
    lnC_check = lnCl < lnC < lnCu
    alpha_check = alphal < alpha < alphau
    beta_check = betal < beta < betau
    gamma_check = gammal < gamma < gammau
    mu_check = mul < mu < muu
    
    if lnC_check and alpha_check and beta_check and gamma_check and mu_check:
        f = -np.log(lnCu-lnCl) - np.log(alphau-alphal) - np.log(betau-betal) -\
            np.log(gammau-gammal) - np.log(muu-mul)
    else:
        f = -np.inf
    
    return f

def lnprob_simplet2(theta,au,al,Ru,Rl,T,x,s):
    """
    Returns the sum of the log-likelihood and log-prior (distribution to sample)
    for the simple model with quadratic dependence on T_eff
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC, alpha, beta, gamma, mu
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        x (ndarray):
            Occurrence rate from data
        s (ndarray):
            Occurrence rate standard deviation from data
    
    Returns:
        f (float):
            Value of the desired sample distribution
    """
    
    lp = lnprior_simplet2(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_simplet2(theta,au,al,Ru,Rl,T,x,s)

def simplet3_model(au,al,Ru,Rl,T,lnC=np.log(0.5),alpha=-0.61,beta=-1.16,gamma=0.5,mu=0.25,nu=0.25):
    """
    Model for occurrence rates with no break radius cubic dependence on T_eff
    
    Args:
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        lnC (float):
            Occurrence rate density for Earth at T_eff_sun
        alpha (float):
            Power law index for semi-major axis
        beta (float):
            Power law index for planetary radius
        gamma (float):
            Linear coefficient in occurrence dependence on T_eff
        mu (float):
            Quadratic coefficient in occurrence dependence on T_eff
        nu (float):
            Cubic coefficient in occurrence dependence on T_eff
    
    Returns:
        eta (ndarray):
            Integrated occurrence values for the semi-major axis and radius bins
    
    """
    
    eta = np.exp(lnC)/(alpha*beta)*(au**alpha-al**alpha)*(Ru**beta-Rl**beta)*\
            (1. + gamma*T + mu*T**2 + nu*T**3)

    return eta

def lnlike_simplet3(theta,au,al,Ru,Rl,T,x,s):
    """
    Returns the sum of the log-likelihood (Gaussian, independent error bars)
    for simple model with cubic dependence on T_eff
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC, alpha, beta, gamma, mu, nu
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        x (ndarray):
            Occurrence rate from data
        s (ndarray):
            Occurrence rate standard deviation from data
    
    Returns:
        f (float):
            Value of the sum of the log-likelihood
    """
    
    model = simplet3_model(au,al,Ru,Rl,T,*theta)

    f = -0.5*(np.sum(np.log(2.*np.pi)+2.*np.log(s)+(x-model)**2/s**2))
    
    return f

def lnprior_simplet3(theta):
    """
    Returns the log-prior for the simple model with cubic dependence on T_eff
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC, alpha, beta, gamma, mu, nu
    
    Returns:
        f (float):
            Value of the log-prior
    """
    # upper and lower bounds on hyperparameters
    lnCu, lnCl = 10.,-5.
    alphau, alphal = 2., -2.
    betau, betal = 2., -2.
    gammau, gammal = 100., -100.
    muu, mul = 500., -500.
    nuu, nul = 5000., -5000.
    
    # get parameter values for this set
    lnC, alpha, beta, gamma, mu, nu = theta
    # check if parameters are within the bounds
    lnC_check = lnCl < lnC < lnCu
    alpha_check = alphal < alpha < alphau
    beta_check = betal < beta < betau
    gamma_check = gammal < gamma < gammau
    mu_check = mul < mu < muu
    nu_check = nul < nu < nuu
    
    if lnC_check and alpha_check and beta_check and gamma_check and mu_check \
        and nu_check:
        f = -np.log(lnCu-lnCl) - np.log(alphau-alphal) - np.log(betau-betal) -\
            np.log(gammau-gammal) - np.log(muu-mul) - np.log(nuu-nul)
    else:
        f = -np.inf
    
    return f

def lnprob_simplet3(theta,au,al,Ru,Rl,T,x,s):
    """
    Returns the sum of the log-likelihood and log-prior (distribution to sample)
    for the simple model with cubic dependence on T_eff
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC, alpha, beta, gamma, mu, nu
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        x (ndarray):
            Occurrence rate from data
        s (ndarray):
            Occurrence rate standard deviation from data
    
    Returns:
        f (float):
            Value of the desired sample distribution
    """
    
    lp = lnprior_simplet3(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_simplet3(theta,au,al,Ru,Rl,T,x,s)

def break_model(au,al,Ru,Rl,T,lnC0=np.log(0.5),lnC1=np.log(0.6),alpha0=-0.61,\
                 alpha1=-0.5,beta0=-1.16,beta1=-1.0,Rb=3.4):
    """
    Model for occurrence rates with one break radius no T_eff dependence
    
    Args:
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        lnC0 (float):
            Occurrence rate density for Earth at T_eff_sun for Rp < Rb
        lnC1 (float):
            Occurrence rate density for Rp >= Rb
        alpha0 (float):
            Power law index for semi-major axis for Rp < Rb
        alpha1 (float):
            Power law index for semi-major axis for Rp >= Rb
        beta0 (float):
            Power law index for planetary radius for Rp < Rb
        beta1 (float):
            Power law index for planetary radius for Rp >= Rb
        Rb (float):
            Break radius
    
    Returns:
        eta (ndarray):
            Integrated occurrence values for the semi-major axis and radius bins
    """
    # initialize output
    eta = np.inf*np.ones(au.shape)
    # data where Ru <= Rb
    below = Ru <= Rb
    eta[below] = np.exp(lnC0)/(alpha0*beta0)*\
                (au[below]**alpha0-al[below]**alpha0)*(Ru[below]**beta0-Rl[below]**beta0)
    # data where Rl <= Rb and Ru >= Rb
    sandwich = (Rl <= Rb)&(Ru >= Rb)
    eta[sandwich] = np.exp(lnC0)/(alpha0*beta0)*\
                (au[sandwich]**alpha0-al[sandwich]**alpha0)*(Rb**beta0-Rl[sandwich]**beta0)
    eta[sandwich]+= np.exp(lnC1)/(alpha1*beta1)*\
                (au[sandwich]**alpha1-al[sandwich]**alpha1)*(Ru[sandwich]**beta1-Rb**beta1)
    # data where Rl and Ru >= Rb
    above = Rl >= Rb
    eta[above] = np.exp(lnC1)/(alpha1*beta1)*\
                (au[above]**alpha1-al[above]**alpha1)*(Ru[above]**beta1-Rl[above]**beta1)
    
    return eta

def lnlike_break(theta,au,al,Ru,Rl,T,x,s):
    """
    Returns the sum of the log-likelihood (Gaussian, independent error bars)
    for the model with one break radius and no dependence on T_eff
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC0, lnC1, alpha0, alpha1, beta0, 
            beta1, Rb
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        x (ndarray):
            Occurrence rate from data
        s (ndarray):
            Occurrence rate standard deviation from data
    
    Returns:
        f (float):
            Value of the sum of the log-likelihood
    """
    # get model values
    model = break_model(au,al,Ru,Rl,T,*theta)

    f = -0.5*(np.sum(np.log(2.*np.pi)+2.*np.log(s)+(x-model)**2/s**2))
    
    return f

def lnprior_break(theta):
    """
    Returns the log-prior for the model with one break radius and no dependence
    on T_eff
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC0, lnC1, alpha0, alpha1, beta0,
            beta1, Rb
    
    Returns:
        f (float):
            Value of the log-prior
    """
    # upper and lower bounds on hyperparameters
    lnC0u, lnC0l = 10., -5.
    lnC1u, lnC1l = 10., -5.
    alpha0u, alpha0l = 2., -2.
    alpha1u, alpha1l = 2., -2.
    beta0u, beta0l = 2., -2.
    beta1u, beta1l = 2., -2.
    Rbl, Rbu = 0.44, 26.
    
    # get parameter values for this set
    lnC0, lnC1, alpha0, alpha1, beta0, beta1, Rb = theta
    # check if parameters are within the bounds
    lnC0_check = lnC0l < lnC0 < lnC0u
    lnC1_check = lnC1l < lnC1 < lnC1u
    alpha0_check = alpha0l < alpha0 < alpha0u
    alpha1_check = alpha1l < alpha1 < alpha1u
    beta0_check = beta0l < beta0 < beta0u
    beta1_check = beta1l < beta1 < beta1u
    Rb_check = Rbl < Rb < Rbu
    
    all_checks = lnC0_check and lnC1_check and alpha0_check and alpha1_check \
                and beta0_check and beta1_check and Rb_check 
    if all_checks:
        f = -np.log(lnC0u-lnC0l) - np.log(lnC1u-lnC1l) - np.log(alpha0u-alpha0l) \
            -np.log(alpha1u-alpha1l) - np.log(beta0u-beta0l) - np.log(beta1u-beta1l) \
            -np.log(Rbu-Rbl)
    else:
        f = -np.inf
    
    return f

def lnprob_break(theta,au,al,Ru,Rl,T,x,s):
    """
    Returns the sum of the log-likelihood and log-prior (distribution to sample)
    for the model with one break radius and no dependence on T_eff
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC0, lnC1, alpha0, alpha1, beta0,
            beta1
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        x (ndarray):
            Occurrence rate from data
        s (ndarray):
            Occurrence rate standard deviation from data
    
    Returns:
        f (float):
            Value of the desired sample distribution
    """
    
    lp = lnprior_break(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_break(theta,au,al,Ru,Rl,T,x,s)

def breakt_model(au,al,Ru,Rl,T,lnC0=np.log(0.5),lnC1=np.log(0.6),alpha0=-0.61,\
                 alpha1=-0.5,beta0=-1.16,beta1=-1.0,Rb=3.4,gamma0=0.5,\
                 gamma1=0.5):
    """
    Model for occurrence rates with one break radius linear dependent T_eff
    
    Args:
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        lnC0 (float):
            Occurrence rate density for Earth at T_eff_sun for Rp < Rb
        lnC1 (float):
            Occurrence rate density for Rp >= Rb
        alpha0 (float):
            Power law index for semi-major axis for Rp < Rb
        alpha1 (float):
            Power law index for semi-major axis for Rp >= Rb
        beta0 (float):
            Power law index for planetary radius for Rp < Rb
        beta1 (float):
            Power law index for planetary radius for Rp >= Rb
        Rb (float):
            Break radius
        gamma0 (float):
            Linear coefficient in occurrence dependence on T_eff for Rp < Rb
        gamma1 (float):
            Linear coefficient in occurrence dependence on T_eff for Rp >= Rb
    
    Returns:
        eta (ndarray):
            Integrated occurrence values for the semi-major axis and radius bins
    """
    # initialize output
    eta = np.inf*np.ones(au.shape)
    # T_eff quadratic dependence
    f0 = t_dep([gamma0],T)
    f1 = t_dep([gamma1],T)

    # data where Ru <= Rb
    below = Ru <= Rb
    eta[below] = np.exp(lnC0)/(alpha0*beta0)*\
                (au[below]**alpha0-al[below]**alpha0)*(Ru[below]**beta0-Rl[below]**beta0)*f0[below]
    # data where Rl <= Rb and Ru >= Rb
    sandwich = (Rl <= Rb)&(Ru >= Rb)
    eta[sandwich] = np.exp(lnC0)/(alpha0*beta0)*\
                (au[sandwich]**alpha0-al[sandwich]**alpha0)*(Rb**beta0-Rl[sandwich]**beta0)*f0[sandwich]
    eta[sandwich]+= np.exp(lnC1)/(alpha1*beta1)*\
                (au[sandwich]**alpha1-al[sandwich]**alpha1)*(Ru[sandwich]**beta1-Rb**beta1)*\
                f1[sandwich]
    # data where Rl and Ru >= Rb
    above = Rl >= Rb
    eta[above] = np.exp(lnC1)/(alpha1*beta1)*\
                (au[above]**alpha1-al[above]**alpha1)*(Ru[above]**beta1-Rl[above]**beta1)*\
                f1[above]
    
    return eta

def lnlike_breakt(theta,au,al,Ru,Rl,T,x,s):
    """
    Returns the sum of the log-likelihood (Gaussian, independent error bars)
    for the model with one break radius and linear dependent T_eff
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC0, lnC1, alpha0, alpha1, beta0, 
            beta1, Rb, gamma0, gamma1
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        x (ndarray):
            Occurrence rate from data
        s (ndarray):
            Occurrence rate standard deviation from data
    
    Returns:
        f (float):
            Value of the sum of the log-likelihood
    """
    # get model values
    model = breakt_model(au,al,Ru,Rl,T,*theta)

    f = -0.5*(np.sum(np.log(2.*np.pi)+2.*np.log(s)+(x-model)**2/s**2))
    
    return f

def lnprior_breakt(theta):
    """
    Returns the log-prior for the model with one break radius and linear
    dependent T_eff
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC0, lnC1, alpha0, alpha1, beta0,
            beta1, Rb, gamma0, gamma1
    
    Returns:
        f (float):
            Value of the log-prior
    """
    # upper and lower bounds on hyperparameters
    lnC0u, lnC0l = 10., -5.
    lnC1u, lnC1l = 10., -5.
    alpha0u, alpha0l = 2., -2.
    alpha1u, alpha1l = 2., -2.
    beta0u, beta0l = 2., -2.
    beta1u, beta1l = 2., -2.
    Rbl, Rbu = 0.44, 26.
    gamma0u, gamma0l = 100., -100.
    gamma1u, gamma1l = 100., -100.
    
    # get parameter values for this set
    lnC0, lnC1, alpha0, alpha1, beta0, beta1, Rb, gamma0, gamma1 = theta
    # check if parameters are within the bounds
    lnC0_check = lnC0l < lnC0 < lnC0u
    lnC1_check = lnC1l < lnC1 < lnC1u
    alpha0_check = alpha0l < alpha0 < alpha0u
    alpha1_check = alpha1l < alpha1 < alpha1u
    beta0_check = beta0l < beta0 < beta0u
    beta1_check = beta1l < beta1 < beta1u
    Rb_check = Rbl < Rb < Rbu
    gamma0_check = gamma0l < gamma0 < gamma0u
    gamma1_check = gamma1l < gamma1 < gamma1u
    
    all_checks = lnC0_check and lnC1_check and alpha0_check and alpha1_check \
                and beta0_check and beta1_check and Rb_check and gamma0_check \
                and gamma1_check
    if all_checks:
        f = -np.log(lnC0u-lnC0l) - np.log(lnC1u-lnC1l) - np.log(alpha0u-alpha0l) \
            -np.log(alpha1u-alpha1l) - np.log(beta0u-beta0l) - np.log(beta1u-beta1l) \
            -np.log(Rbu-Rbl) - np.log(gamma0u-gamma0l) - np.log(gamma1u-gamma1l)
    else:
        f = -np.inf
    
    return f

def lnprob_breakt(theta,au,al,Ru,Rl,T,x,s):
    """
    Returns the sum of the log-likelihood and log-prior (distribution to sample)
    for the model with one break radius and linear dependent T_eff
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC0, lnC1, alpha0, alpha1, beta0,
            beta1, gamma0, gamma1
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        x (ndarray):
            Occurrence rate from data
        s (ndarray):
            Occurrence rate standard deviation from data
    
    Returns:
        f (float):
            Value of the desired sample distribution
    """
    
    lp = lnprior_breakt(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_breakt(theta,au,al,Ru,Rl,T,x,s)

def breakt2_model(au,al,Ru,Rl,T,lnC0=np.log(0.5),lnC1=np.log(0.6),alpha0=-0.61,\
                 alpha1=-0.5,beta0=-1.16,beta1=-1.0,Rb=3.4,gamma0=0.5,mu0=0.25,\
                 gamma1=0.5,mu1=0.25):
    """
    Model for occurrence rates with one break radius quadratic dependent T_eff
    
    Args:
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        lnC0 (float):
            Occurrence rate density for Earth at T_eff_sun for Rp < Rb
        lnC1 (float):
            Occurrence rate density for Rp >= Rb
        alpha0 (float):
            Power law index for semi-major axis for Rp < Rb
        alpha1 (float):
            Power law index for semi-major axis for Rp >= Rb
        beta0 (float):
            Power law index for planetary radius for Rp < Rb
        beta1 (float):
            Power law index for planetary radius for Rp >= Rb
        Rb (float):
            Break radius
        gamma0 (float):
            Linear coefficient in occurrence dependence on T_eff for Rp < Rb
        mu0 (float):
            Quadratic coefficient in occurrence dependence on T_eff for Rp < Rb
        gamma1 (float):
            Linear coefficient in occurrence dependence on T_eff for Rp >= Rb
        mu1 (float):
            Quadratic coefficient in occurrence dependence on T_eff for Rp >= Rb
    
    Returns:
        eta (ndarray):
            Integrated occurrence values for the semi-major axis and radius bins
    """
    # initialize output
    eta = np.inf*np.ones(au.shape)
    # T_eff quadratic dependence
    f0 = t_dep([gamma0,mu0],T)
    f1 = t_dep([gamma1,mu1],T)

    # data where Ru <= Rb
    below = Ru <= Rb
    eta[below] = np.exp(lnC0)/(alpha0*beta0)*\
                (au[below]**alpha0-al[below]**alpha0)*(Ru[below]**beta0-Rl[below]**beta0)*f0[below]
    # data where Rl <= Rb and Ru >= Rb
    sandwich = (Rl <= Rb)&(Ru >= Rb)
    eta[sandwich] = np.exp(lnC0)/(alpha0*beta0)*\
                (au[sandwich]**alpha0-al[sandwich]**alpha0)*(Rb**beta0-Rl[sandwich]**beta0)*f0[sandwich]
    eta[sandwich]+= np.exp(lnC1)/(alpha1*beta1)*\
                (au[sandwich]**alpha1-al[sandwich]**alpha1)*(Ru[sandwich]**beta1-Rb**beta1)*\
                f1[sandwich]
    # data where Rl and Ru >= Rb
    above = Rl >= Rb
    eta[above] = np.exp(lnC1)/(alpha1*beta1)*\
                (au[above]**alpha1-al[above]**alpha1)*(Ru[above]**beta1-Rl[above]**beta1)*\
                f1[above]
    
    return eta

def lnlike_breakt2(theta,au,al,Ru,Rl,T,x,s):
    """
    Returns the sum of the log-likelihood (Gaussian, independent error bars)
    for the model with one break radius quadratic dependent T_eff
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC0, lnC1, alpha0, alpha1, beta0, 
            beta1, Rb, gamma0, mu0, gamma1, mu1
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        x (ndarray):
            Occurrence rate from data
        s (ndarray):
            Occurrence rate standard deviation from data
    
    Returns:
        f (float):
            Value of the sum of the log-likelihood
    """
    # get model values
    model = breakt2_model(au,al,Ru,Rl,T,*theta)

    f = -0.5*(np.sum(np.log(2.*np.pi)+2.*np.log(s)+(x-model)**2/s**2))
    
    return f

def lnprior_breakt2(theta):
    """
    Returns the log-prior for the model with one break radius quadratic
    dependence on T_eff
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC0, lnC1, alpha0, alpha1, beta0,
            beta1, Rb, gamma0, mu0, gamma1, mu1
    
    Returns:
        f (float):
            Value of the log-prior
    """
    # upper and lower bounds on hyperparameters
    lnC0u, lnC0l = 10., -5.
    lnC1u, lnC1l = 10., -5.
    alpha0u, alpha0l = 2., -2.
    alpha1u, alpha1l = 2., -2.
    beta0u, beta0l = 2., -2.
    beta1u, beta1l = 2., -2.
    Rbl, Rbu = 0.44, 26.
    gamma0u, gamma0l = 100., -100.
    mu0u, mu0l = 500., -500.
    gamma1u, gamma1l = 100., -100.
    mu1u, mu1l = 500., -500.
    
    # get parameter values for this set
    lnC0, lnC1, alpha0, alpha1, beta0, beta1, Rb, gamma0, mu0, gamma1, mu1 = theta
    # check if parameters are within the bounds
    lnC0_check = lnC0l < lnC0 < lnC0u
    lnC1_check = lnC1l < lnC1 < lnC1u
    alpha0_check = alpha0l < alpha0 < alpha0u
    alpha1_check = alpha1l < alpha1 < alpha1u
    beta0_check = beta0l < beta0 < beta0u
    beta1_check = beta1l < beta1 < beta1u
    Rb_check = Rbl < Rb < Rbu
    gamma0_check = gamma0l < gamma0 < gamma0u
    mu0_check = mu0l < mu0 < mu0u
    gamma1_check = gamma1l < gamma1 < gamma1u
    mu1_check = mu1l < mu1 < mu1u
    
    all_checks = lnC0_check and lnC1_check and alpha0_check and alpha1_check \
                and beta0_check and beta1_check and Rb_check and gamma0_check \
                and mu0_check and gamma1_check and mu1_check
    if all_checks:
        f = -np.log(lnC0u-lnC0l) - np.log(lnC1u-lnC1l) - np.log(alpha0u-alpha0l) \
            -np.log(alpha1u-alpha1l) - np.log(beta0u-beta0l) - np.log(beta1u-beta1l) \
            -np.log(Rbu-Rbl) - np.log(gamma0u-gamma0l) - np.log(mu0u-mu0l) \
            -np.log(gamma1u-gamma1l) - np.log(mu1u-mu1l)
    else:
        f = -np.inf
    
    return f

def lnprob_breakt2(theta,au,al,Ru,Rl,T,x,s):
    """
    Returns the sum of the log-likelihood and log-prior (distribution to sample)
    for the model with one break radius and quadratic dependence on T_eff
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC0, lnC1, alpha0, alpha1, beta0,
            beta1, gamma0, mu0, gamma1, mu1
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        x (ndarray):
            Occurrence rate from data
        s (ndarray):
            Occurrence rate standard deviation from data
    
    Returns:
        f (float):
            Value of the desired sample distribution
    """
    
    lp = lnprior_breakt2(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_breakt2(theta,au,al,Ru,Rl,T,x,s)

def breakt3_model(au,al,Ru,Rl,T,lnC0=np.log(0.5),lnC1=np.log(0.6),alpha0=-0.61,\
                 alpha1=-0.5,beta0=-1.16,beta1=-1.0,Rb=3.4,gamma0=0.5,mu0=0.25,\
                 nu0=0.25,gamma1=0.5,mu1=0.25,nu1=0.25):
    """
    Model for occurrence rates with one break radius cubic dependent T_eff
    
    Args:
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        lnC0 (float):
            Occurrence rate density for Earth at T_eff_sun for Rp < Rb
        lnC1 (float):
            Occurrence rate density for Rp >= Rb
        alpha0 (float):
            Power law index for semi-major axis for Rp < Rb
        alpha1 (float):
            Power law index for semi-major axis for Rp >= Rb
        beta0 (float):
            Power law index for planetary radius for Rp < Rb
        beta1 (float):
            Power law index for planetary radius for Rp >= Rb
        Rb (float):
            Break radius
        gamma0 (float):
            Linear coefficient in occurrence dependence on T_eff for Rp < Rb
        mu0 (float):
            Quadratic coefficient in occurrence dependence on T_eff for Rp < Rb
        nu0 (float):
            Cubic coefficient in occurrence dependence on T_eff for Rp < Rb
        gamma1 (float):
            Linear coefficient in occurrence dependence on T_eff for Rp >= Rb
        mu1 (float):
            Quadratic coefficient in occurrence dependence on T_eff for Rp >= Rb
        nu1 (float):
            Cubic coefficient in occurrence dependence on T_eff for Rp < Rb
    
    Returns:
        eta (ndarray):
            Integrated occurrence values for the semi-major axis and radius bins
    """
    # initialize output
    eta = np.inf*np.ones(au.shape)
    # T_eff cubic dependence
    f0 = t_dep([gamma0,mu0,nu0],T)
    f1 = t_dep([gamma1,mu1,nu1],T)

    # data where Ru <= Rb
    below = Ru <= Rb
    eta[below] = np.exp(lnC0)/(alpha0*beta0)*\
                (au[below]**alpha0-al[below]**alpha0)*(Ru[below]**beta0-Rl[below]**beta0)*f0[below]
    # data where Rl <= Rb and Ru >= Rb
    sandwich = (Rl <= Rb)&(Ru >= Rb)
    eta[sandwich] = np.exp(lnC0)/(alpha0*beta0)*\
                (au[sandwich]**alpha0-al[sandwich]**alpha0)*(Rb**beta0-Rl[sandwich]**beta0)*f0[sandwich]
    eta[sandwich]+= np.exp(lnC1)/(alpha1*beta1)*\
                (au[sandwich]**alpha1-al[sandwich]**alpha1)*(Ru[sandwich]**beta1-Rb**beta1)*\
                f1[sandwich]
    # data where Rl and Ru >= Rb
    above = Rl >= Rb
    eta[above] = np.exp(lnC1)/(alpha1*beta1)*\
                (au[above]**alpha1-al[above]**alpha1)*(Ru[above]**beta1-Rl[above]**beta1)*\
                f1[above]
    
    return eta

def lnlike_breakt3(theta,au,al,Ru,Rl,T,x,s):
    """
    Returns the sum of the log-likelihood (Gaussian, independent error bars)
    for the model with one break radius cubic dependent T_eff
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC0, lnC1, alpha0, alpha1, beta0, 
            beta1, Rb, gamma0, mu0, nu0, gamma1, mu1, nu1
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        x (ndarray):
            Occurrence rate from data
        s (ndarray):
            Occurrence rate standard deviation from data
    
    Returns:
        f (float):
            Value of the sum of the log-likelihood
    """
    # get model values
    model = breakt3_model(au,al,Ru,Rl,T,*theta)

    f = -0.5*(np.sum(np.log(2.*np.pi)+2.*np.log(s)+(x-model)**2/s**2))
    
    return f

def lnprior_breakt3(theta):
    """
    Returns the log-prior for the model with one break radius quadratic
    dependence on T_eff
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC0, lnC1, alpha0, alpha1, beta0,
            beta1, Rb, gamma0, mu0, gamma1, mu1
    
    Returns:
        f (float):
            Value of the log-prior
    """
    # upper and lower bounds on hyperparameters
    lnC0u, lnC0l = 10., -5.
    lnC1u, lnC1l = 10., -5.
    alpha0u, alpha0l = 2., -2.
    alpha1u, alpha1l = 2., -2.
    beta0u, beta0l = 2., -2.
    beta1u, beta1l = 2., -2.
    Rbl, Rbu = 0.44, 26.
    gamma0u, gamma0l = 100., -100.
    mu0u, mu0l = 500., -500.
    nu0u, nu0l = 5000., -5000.
    gamma1u, gamma1l = 100., -100.
    mu1u, mu1l = 500., -500.
    nu1u, nu1l = 5000., -5000.
    
    # get parameter values for this set
    lnC0, lnC1, alpha0, alpha1, beta0, beta1, Rb, gamma0, mu0, nu0, \
    gamma1, mu1, nu1 = theta
    # check if parameters are within the bounds
    lnC0_check = lnC0l < lnC0 < lnC0u
    lnC1_check = lnC1l < lnC1 < lnC1u
    alpha0_check = alpha0l < alpha0 < alpha0u
    alpha1_check = alpha1l < alpha1 < alpha1u
    beta0_check = beta0l < beta0 < beta0u
    beta1_check = beta1l < beta1 < beta1u
    Rb_check = Rbl < Rb < Rbu
    gamma0_check = gamma0l < gamma0 < gamma0u
    mu0_check = mu0l < mu0 < mu0u
    nu0_check = nu0l < nu0 < nu0u
    gamma1_check = gamma1l < gamma1 < gamma1u
    mu1_check = mu1l < mu1 < mu1u
    nu1_check = nu1l < nu1 < nu1u
    
    all_checks = lnC0_check and lnC1_check and alpha0_check and alpha1_check \
                and beta0_check and beta1_check and Rb_check and gamma0_check \
                and mu0_check and nu0_check and gamma1_check and mu1_check \
                and nu1_check
    if all_checks:
        f = -np.log(lnC0u-lnC0l) - np.log(lnC1u-lnC1l) - np.log(alpha0u-alpha0l) \
            -np.log(alpha1u-alpha1l) - np.log(beta0u-beta0l) - np.log(beta1u-beta1l) \
            -np.log(Rbu-Rbl) - np.log(gamma0u-gamma0l) - np.log(mu0u-mu0l) \
            -np.log(nu0u-nu0l) - np.log(gamma1u-gamma1l) - np.log(mu1u-mu1l) \
            -np.log(nu1u-nu1l)
    else:
        f = -np.inf
    
    return f

def lnprob_breakt3(theta,au,al,Ru,Rl,T,x,s):
    """
    Returns the sum of the log-likelihood and log-prior (distribution to sample)
    for the model with one break radius and cubic dependence on T_eff
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC0, lnC1, alpha0, alpha1, beta0,
            beta1, gamma0, mu0, nu0, gamma1, mu1, nu1
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        x (ndarray):
            Occurrence rate from data
        s (ndarray):
            Occurrence rate standard deviation from data
    
    Returns:
        f (float):
            Value of the desired sample distribution
    """
    
    lp = lnprior_breakt3(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_breakt3(theta,au,al,Ru,Rl,T,x,s)

def break2_model(au,al,Ru,Rl,T,lnC0=np.log(0.5),lnC1=np.log(0.6),lnC2=np.log(0.7),\
                 alpha0=-0.61,alpha1=-0.5,alpha2=-0.5,beta0=-1.16,beta1=-1.0,\
                 beta2=-1.0,Rb0=3.4,Rb1=5.0):
    """
    Model for occurrence rates with two break radii
    
    Args:
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        lnC0 (float):
            Occurrence rate density for Earth at T_eff_sun for Rp < Rb0
        lnC1 (float):
            Occurrence rate density for Rb0 <= Rp < Rb1
        lnC2 (float):
            Occurrence rate density for Rp >= Rb1
        alpha0 (float):
            Power law index for semi-major axis for Rp < Rb0
        alpha1 (float):
            Power law index for semi-major axis for Rb0 <= Rp < Rb1
        alpha2 (float):
            Power law index for semi-major axis for Rp >= Rb1
        beta0 (float):
            Power law index for planetary radius for Rp < Rb0
        beta1 (float):
            Power law index for planetary radius for Rb0 <= Rp < Rb1
        beta2 (float):
            Power law index for planetary radius for Rp >= Rb1
        Rb0 (float):
            Break radius
        Rb1 (float):
            Break radius
    
    Returns:
        eta (ndarray):
            Integrated occurrence values for the semi-major axis and radius bins
    """
    # initialize output
    eta = np.inf*np.ones(au.shape)
    # data where Ru <= Rb0
    below = Ru <= Rb0
    eta[below] = np.exp(lnC0)/(alpha0*beta0)*\
                (au[below]**alpha0-al[below]**alpha0)*\
                (Ru[below]**beta0-Rl[below]**beta0)
    # data where Rl <= Rb0 and Rb0 <= Ru < Rb1
    pbj = (Rl <= Rb0)&(Ru >= Rb0)&(Ru < Rb1)
    eta[pbj] = np.exp(lnC0)/(alpha0*beta0)*\
                (au[pbj]**alpha0-al[pbj]**alpha0)*\
                (Rb0**beta0-Rl[pbj]**beta0)
    eta[pbj]+= np.exp(lnC1)/(alpha1*beta1)*\
                (au[pbj]**alpha1-al[pbj]**alpha1)*\
                (Ru[pbj]**beta1-Rb0**beta1)
    # data where Rl < Rb0 and Ru >= Rb1
    PBJ = (Rl <= Rb0)&(Ru >= Rb1)
    eta[PBJ] = np.exp(lnC0)/(alpha0*beta0)*\
                (au[PBJ]**alpha0-al[PBJ]**alpha0)*\
                (Rb0**beta0-Rl[PBJ]**beta0)
    eta[PBJ]+= np.exp(lnC1)/(alpha1*beta1)*\
                (au[PBJ]**alpha1-al[PBJ]**alpha1)*\
                (Rb1**beta1-Rb0**beta1)
    eta[PBJ]+= np.exp(lnC2)/(alpha2*beta2)*\
                (au[PBJ]**alpha2-al[PBJ]**alpha2)*\
                (Ru[PBJ]**beta2-Rb1**beta2)
    # data where Rb0 <= Rl < Ru < Rb1
    mid = (Rl >= Rb0)&(Rl < Rb1)&(Ru >= Rb0)&(Ru < Rb1)
    eta[mid] = np.exp(lnC1)/(alpha1*beta1)*\
                (au[mid]**alpha1-al[mid]**alpha1)*\
                (Ru[mid]**beta1-Rl[mid]**beta1)
    # data where Rb0 <= Rl < Rb1 and Ru >= Rb1
    blt = (Rl >= Rb0)&(Rl < Rb1)&(Ru >= Rb1)
    eta[blt] = np.exp(lnC1)/(alpha1*beta1)*\
                (au[blt]**alpha1-al[blt]**alpha1)*\
                (Rb1**beta1-Rl[blt]**beta1)
    eta[blt]+= np.exp(lnC2)/(alpha2*beta2)*\
                (au[blt]**alpha2-al[blt]**alpha2)*\
                (Ru[blt]**beta2-Rb1**beta2)
    # data where Rl and Ru >= Rb1
    above = Rl >= Rb1
    eta[above] = np.exp(lnC2)/(alpha2*beta2)*\
                (au[above]**alpha2-al[above]**alpha2)*\
                (Ru[above]**beta2-Rl[above]**beta2)
    
    return eta

def lnlike_break2(theta,au,al,Ru,Rl,T,x,s):
    """
    Returns the sum of the log-likelihood (Gaussian, independent error bars)
    for the model with two break radii
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC0, lnC1, lnC2, alpha0, alpha1, 
            alpha2, beta0, beta1, beta2, Rb0, Rb1
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        x (ndarray):
            Occurrence rate from data
        s (ndarray):
            Occurrence rate standard deviation from data
    
    Returns:
        f (float):
            Value of the sum of the log-likelihood
    """
    # get model values
    model = break2_model(au,al,Ru,Rl,T,*theta)

    f = -0.5*(np.sum(np.log(2.*np.pi)+2.*np.log(s)+(x-model)**2/s**2))
    
    return f

def lnprior_break2(theta):
    """
    Returns the log-prior for the model with two break radii
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC0, lnC1, lnC2, alpha0, alpha1, 
            alpha2, beta0, beta1, beta2, Rb0, Rb1
    
    Returns:
        f (float):
            Value of the log-prior
    """
    # upper and lower bounds on hyperparameters
    lnC0u, lnC0l = 1., -5.
    lnC1u, lnC1l = 1., -5.
    lnC2u, lnC2l = 1., -5.
    alpha0u, alpha0l = 2., -2.
    alpha1u, alpha1l = 2., -2.
    alpha2u, alpha2l = 2., -2.
    beta0u, beta0l = 5., -5.
    beta1u, beta1l = 5., -5.
    beta2u, beta2l = 5., -5.
    Rb0l, Rb0u = 0.44, 26.
    Rb1l, Rb1u = 0.44, 26.
    
    # get parameter values for this set
    lnC0, lnC1, lnC2, alpha0, alpha1, alpha2, beta0, beta1, beta2, Rb0, Rb1 = theta
    # check if parameters are within the bounds
    lnC0_check = lnC0l < lnC0 < lnC0u
    lnC1_check = lnC1l < lnC1 < lnC1u
    lnC2_check = lnC2l < lnC2 < lnC2u
    alpha0_check = alpha0l < alpha0 < alpha0u
    alpha1_check = alpha1l < alpha1 < alpha1u
    alpha2_check = alpha2l < alpha2 < alpha2u
    beta0_check = beta0l < beta0 < beta0u
    beta1_check = beta1l < beta1 < beta1u
    beta2_check = beta2l < beta2 < beta2u
    Rb0_check = Rb0l < Rb0 < Rb0u
    Rb1_check = Rb1l < Rb1 < Rb1u
    Rb_check = Rb1 > Rb0
    
    all_checks = lnC0_check and lnC1_check and lnC2_check and alpha0_check \
                and alpha1_check and alpha2_check and beta0_check \
                and beta1_check and beta2_check and Rb0_check and Rb1_check \
                and Rb_check
    if all_checks:
        f = -np.log(lnC0u-lnC0l) - np.log(lnC1u-lnC1l) - np.log(lnC2u-lnC2l) \
            -np.log(alpha0u-alpha0l) - np.log(alpha1u-alpha1l) - np.log(alpha2u-alpha2l) \
            -np.log(beta0u-beta0l) - np.log(beta1u-beta1l) - np.log(beta2u-beta2l) \
            -np.log(Rb0u-Rb0l) - np.log(Rb1u-Rb1l)
    else:
        f = -np.inf
    
    return f

def lnprob_break2(theta,au,al,Ru,Rl,T,x,s):
    """
    Returns the sum of the log-likelihood and log-prior (distribution to sample)
    for the model with two break radii
    
    Args:
        theta (tuple):
            Tuple containing fit parameters lnC0, lnC1, lnC2, alpha0, alpha1, 
            alpha2, beta0, beta1, beta2, Rb0, Rb1
        au (ndarray):
            Upper semi-major axis limit in AU
        al (ndarray):
            Lower semi-major axis limit in AU
        Ru (ndarray):
            Upper planetary radius limit in R_Earth
        Rl (ndarray):
            Lower planetary radius limit in R_Earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        x (ndarray):
            Occurrence rate from data
        s (ndarray):
            Occurrence rate standard deviation from data
    
    Returns:
        f (float):
            Value of the desired sample distribution
    """
    
    lp = lnprior_break2(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_break2(theta,au,al,Ru,Rl,T,x,s)

def t_dep(x,T):
    """
    Returns value of T_eff dependence function
    
    Args:
        x (list):
            List containing fit parameters
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
    
    Returns:
        f (ndarray):
            Function dependent on transformed stellar effective temperature
    """
    f = 1.
    for i in xrange(len(x)):
        f += x[i]*T**(i+1)
    
    return f    

def revise_data(M,au,al,Ru,Rl,T,mu,s,inds):
    """
    Returns only the values indicated in inds
    
    Args:
        M (ndarray):
            Stellar masses in M_sun
        au (ndarray):
            Upper bound semi-major axis in AU
        al (ndarray):
            Lower bound semi-major axis in AU
        Ru (ndarray):
            Upper bound planetary radius in R_earth
        Rl (ndarray):
            Lower bound planetary radius in R_earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        mu (ndarray):
            Integrated occurrence rate values
        s (ndarray):
            Standard deviations on occurrence rates
        inds (ndarray):
            Array of indices to keep
    Returns:
        M (ndarray):
            Stellar masses in M_sun
        au (ndarray):
            Upper bound semi-major axis in AU
        al (ndarray):
            Lower bound semi-major axis in AU
        Ru (ndarray):
            Upper bound planetary radius in R_earth
        Rl (ndarray):
            Lower bound planetary radius in R_earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        mu (ndarray):
            Integrated occurrence rate values
        s (ndarray):
            Standard deviations on occurrence rates
    """
    
    return M[inds], au[inds], al[inds], Ru[inds], Rl[inds], T[inds], mu[inds], s[inds]

def grab_data(eta):
    """
    Gets necessary data from EtaBase object for MCMC
    
    Args:
        eta (EtaBase):
            EtaBase object
            
    Returns:
        M (ndarray):
            Stellar masses in M_sun
        au (ndarray):
            Upper bound semi-major axis in AU
        al (ndarray):
            Lower bound semi-major axis in AU
        Ru (ndarray):
            Upper bound planetary radius in R_earth
        Rl (ndarray):
            Lower bound planetary radius in R_earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        mu (ndarray):
            Integrated occurrence rate values
        s (ndarray):
            Standard deviations on occurrence rates
    """
    
    T_eff_sun = 5772.
    # pre-allocate arrays
    M = np.array([])
    au = np.array([])
    al = np.array([])
    Ru = np.array([])
    Rl = np.array([])
    T = np.array([])
    mu = np.array([])
    s = np.array([])
    
    for key in eta.Etas.keys():
        size = eta.Etas[key]['eta'].size
        etas = eta.Etas[key]['eta'].reshape((size,))
        sign = eta.Etas[key]['sign'].reshape((size,))
        sigp = eta.Etas[key]['sigp'].reshape((size,))
        if type(eta.aData['range']) == dict:
            a = eta.aData['range'][key]
        else:
            a = eta.aData['range']
        R = eta.RpData['range']
        aa,RR = np.meshgrid(a,R)
        al_tmp = aa[:-1,:-1].reshape((size,))
        au_tmp = aa[1:,1:].reshape((size,))
        Rl_tmp = RR[:-1,:-1].reshape((size,))
        Ru_tmp = RR[1:,1:].reshape((size,))

        Teff = np.mean(eta.Teff[key])
        M_tmp = eta.Mstar[key]
        Teff *= np.ones(len(etas),)
        M_tmp *= np.ones(len(etas),)
        # dummy values for s_tmp
        s_tmp = 0.5*(sign+sigp)
        
        # remove nan or inf
        inds = np.where(np.isfinite(etas))[0]
        M_tmp,au_tmp,al_tmp,Ru_tmp,Rl_tmp,Teff,etas,s_tmp = revise_data(M_tmp,au_tmp,al_tmp,Ru_tmp,Rl_tmp,Teff,etas,s_tmp,inds)
        sign = sign[inds]
        sigp = sigp[inds]
        
        # remove where lower bound is lower than 0
        low = etas - sign
        inds = np.where(low > 0)[0]
        M_tmp,au_tmp,al_tmp,Ru_tmp,Rl_tmp,Teff,etas,s_tmp = revise_data(M_tmp,au_tmp,al_tmp,Ru_tmp,Rl_tmp,Teff,etas,s_tmp,inds)
        sign = sign[inds]
        sigp = sigp[inds]
        
        # get standard deviation
        # check if sign and sigp are within 20% and keep those values
        vals = np.abs(sigp-sign)/(0.5*(sigp+sign))
        inds2 = np.where(vals <= 0.2)
        M_tmp,au_tmp,al_tmp,Ru_tmp,Rl_tmp,Teff,etas,s_tmp = revise_data(M_tmp,au_tmp,al_tmp,Ru_tmp,Rl_tmp,Teff,etas,s_tmp,inds2)
        sign = sign[inds2]
        sigp = sigp[inds2]
        s_tmp = 0.5*(sign+sigp)
        
        # only include main sequence data (Teff < 7000 K)
        inds = np.where(Teff <= 7000.)[0]
        M_tmp,au_tmp,al_tmp,Ru_tmp,Rl_tmp,Teff,etas,s_tmp = revise_data(M_tmp,au_tmp,al_tmp,Ru_tmp,Rl_tmp,Teff,etas,s_tmp,inds)
        sign = sign[inds]
        sigp = sigp[inds]
        s_tmp = 0.5*(sign+sigp)
        
        # TO REMOVE M STAR DATA, UNCOMMENT THIS!
#        # only include Teff > 4000
#        inds = np.where(Teff >= 4000)[0]
#        M_tmp,au_tmp,al_tmp,Ru_tmp,Rl_tmp,Teff,etas,s_tmp = revise_data(M_tmp,au_tmp,al_tmp,Ru_tmp,Rl_tmp,Teff,etas,s_tmp,inds)
#        sign = sign[inds]
#        sigp = sigp[inds]
#        s_tmp = 0.5*(sign+sigp)

        print('Final data for {} type {}: {}'.format(eta.fname['name'],key,len(etas)))

        T_tmp = Teff/T_eff_sun - 1.
        M = np.hstack((M,M_tmp))
        au = np.hstack((au,au_tmp))
        al = np.hstack((al,al_tmp))
        Ru = np.hstack((Ru,Ru_tmp))
        Rl = np.hstack((Rl,Rl_tmp))
        T = np.hstack((T,T_tmp))
        mu = np.hstack((mu,etas))
        s = np.hstack((s,s_tmp))
        
    return M, au, al, Ru, Rl, T, mu, s

def grab_all_data(etas):
    """
    Gets necessary data from all EtaBase objects in list
    
    Args:
        etas (list):
            List of EtaBase objects
    
    Returns:
        M (ndarray):
            Stellar masses in M_sun
        au (ndarray):
            Upper bound semi-major axis in AU
        al (ndarray):
            Lower bound semi-major axis in AU
        Ru (ndarray):
            Upper bound planetary radius in R_earth
        Rl (ndarray):
            Lower bound planetary radius in R_earth
        T (ndarray):
            Transformed stellar effective temperatures T = T_eff/T_eff_sun - 1
        mu (ndarray):
            Integrated occurrence rate values
        s (ndarray):
            Standard deviations on occurrence rates
    """
    
    # pre-allocate arrays
    M = np.array([])
    au = np.array([])
    al = np.array([])
    Ru = np.array([])
    Rl = np.array([])
    T = np.array([])
    mu = np.array([])
    s = np.array([])
        
    for eta in etas:
        m, AU, AL, rU, rL, t, MU, S = grab_data(eta)
        M = np.hstack((M,m))
        au = np.hstack((au,AU))
        al = np.hstack((al,AL))
        Ru = np.hstack((Ru,rU))
        Rl = np.hstack((Rl,rL))
        T =  np.hstack((T,t))
        mu = np.hstack((mu,MU))
        s = np.hstack((s,S))
    
    return M, au, al, Ru, Rl, T, mu, s

def gen_MCMC(lnprob,args,p0,ndim,walkers,nburn,nsteps):
    """
    Generates MCMC samples for the desired model
    
    Args:
        lnprob (string):
            String containing name of log probability function to sample
        args (tuple):
            Tuple of additional arguments to lnprob function
            (au,al,Ru,Rl,T,x,s)
        p0 (tuple):
            Initial position of each parameter
        ndim (int):
            Number of parameters
        walkers (int):
            Number of walkers for use in emcee
        nburn (int):
            Number of steps for burn in
        nsteps (int):
            Number of steps to take for samples
    
    Returns:
        burnin (ndarray):
            Markov chain for each parameter and walker during burn in
        samples (ndarray):
            Samples from lnprob
    """
    
    # set up emcee.EnsembleSampler
    if lnprob in lnprob_funcs.keys():
        lnprob_func = lnprob_funcs[lnprob]
    else:
        raise Exception('lnprob function not in lnprob_funcs')
        
    sampler = emcee.EnsembleSampler(walkers,ndim,lnprob_func,args=args)
    
    # generate initial points
    q0 = np.array([])
    q0 = np.zeros((walkers,1))
    for p in p0:
        q = p + 1e-4*np.random.randn(walkers).reshape((walkers,1))
        q0 = np.hstack((q0,q))
    
    # run burn in section
    pos, prob, state = sampler.run_mcmc(q0[:,1:],nburn)
    
    burnin = sampler.chain
    
    # plot burn in section
    if ndim <= 5:
        fig1, axes1 = plt.subplots(ndim,1,sharex=True)
        for i in xrange(ndim):
            for j in xrange(walkers):
                axes1[i].plot(sampler.chain[j,:,i],'k-')
            axes1[i].set_xlim(left=0,right=nburn)
        fig1.suptitle(r'{} Walker positions at each step'.format(walkers),fontsize=20)
        fig1.show()
    else:
        fig1, axes1 = plt.subplots(5,1,sharex=True)
        for i in xrange(5):
            for j in xrange(walkers):
                axes1[i].plot(sampler.chain[j,:,i],'k-')
            axes1[i].set_xlim(left=0,right=nburn)
        axes1[-1].set_xlabel(r'MCMC Steps',fontsize=16)
        fig1.suptitle(r'{} Walker positions at each step'.format(walkers),fontsize=20)
        fig1.show()
        
        # 5 < ndim <= 10
        if ndim <= 10:
            subs = ndim - 5
        else:
            subs = 5
        fig2, axes2 = plt.subplots(subs,1,sharex=True)
        if subs > 1:
            for i in xrange(subs):
                for j in xrange(walkers):
                    axes2[i].plot(sampler.chain[j,:,i+5],'k-')
                axes2[i].set_xlim(left=0,right=nburn)
            axes2[-1].set_xlabel(r'MCMC Steps',fontsize=16)
            fig2.suptitle(r'{} Walker positions at each step'.format(walkers),fontsize=20)
            fig2.show()
        else:
            for j in xrange(walkers):
                axes2.plot(sampler.chain[j,:,5],'k-')
            axes2.set_xlim(left=0,right=nburn)
            fig2.suptitle(r'{} Walker positions at each step'.format(walkers),fontsize=20)
            fig2.show()
        
        if ndim > 10:
            subs = ndim - 10
            if subs > 1:
                fig3, axes3 = plt.subplots(subs,1,sharex=True)
                for i in xrange(subs):
                    for j in xrange(walkers):
                        axes3[i].plot(sampler.chain[j,:,i+10],'k-')
                    axes3[i].set_xlim(left=0,right=nburn)
                axes3[-1].set_xlabel(r'MCMC Steps',fontsize=16)
                fig3.suptitle(r'{} Walker positions at each step'.format(walkers),fontsize=20)
                fig3.show()
            else:
                fig3, axes3 = plt.subplots(subs,1,sharex=True)
                for j in xrange(walkers):
                    axes3.plot(sampler.chain[j,:,10],'k-')
                axes3.set_xlabel(r'MCMC Steps',fontsize=16)
                axes3.set_xlim(left=0,right=nburn)
                fig3.suptitle(r'{} Walker positions at each step'.format(walkers),fontsize=20)
                fig3.show()
    
    # get samples
    sampler.reset()
    sampler.run_mcmc(pos,nsteps)
    
    samples = sampler.chain.reshape((-1,ndim))
    
    # plot quick corner plot
    fig4 = corner.corner(samples,scale_hist=True)
    fig4.show()
    
    return burnin, samples

def print_results(data,labels,lnlike,args):
    """
    Prints results from samples
    
    Args:
        data (ndarray):
            MCMC samples
        labels (list):
            List of strings containing variable names for printing out
        lnlike (string):
            Name of log-likelihood function to use
        args (tuple):
            Additional arguments for log-likelihood function
            (au,al,Ru,Rl,T,x,s)
    """
    
    au, al, Ru, Rl, T, mu, s = args
    n = data.shape[1] # number of parameters
    num = len(au) # number of data points
    
    # get median and + and - values from data
    vals = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(data, [16,50,84],axis=0)))
    
    # display this information
    print('Results from MCMC sampling:')
    for i in xrange(data.shape[1]):
        print('{}: {} +{} -{}'.format(labels[i],vals[i][0],vals[i][1],vals[i][2]))
    
    theta = tuple([vals[i][0] for i in xrange(data.shape[1])])
    Lhat = lnlike_funcs[lnlike](theta,au,al,Ru,Rl,T,mu,s)
    BIC = np.log(float(num))*float(n) - 2.*Lhat
    print('Lhat: {}'.format(Lhat))
    print('BIC: {}'.format(BIC))
    
    # get chi-squared
    model = model_funcs[lnlike](au,al,Ru,Rl,T,*theta)
    chis = np.sum(((mu-model)/s)**2)
    print('chi-squared: {}'.format(chis))

def plot_corner(samples,labels,path=None,fmt=None):
    """
    Generate a corner plot for MCMC samples
    
    Args:
        samples (ndarray):
            MCMC samples for corner plot
        labels (list):
            List of strings containing variable labels for corner plot
        path (optional):
            Path to save figure
    """
    
    label_kwargs = {'fontsize':16}
    fig = corner.corner(samples,labels=labels,label_kwargs=label_kwargs,scale_hist=False)
    # set tick fontsize
    ndim = samples.shape[1]
    axes = np.array(fig.axes).reshape((ndim,ndim))
    # left side
    for i in xrange(ndim):
        ax = axes[i,0]
        ax.tick_params(axis='both', bottom='on', top='off', right='off', left='on', which='major', labelsize=14)
        ax2 = axes[-1,i]
        ax2.tick_params(axis='both', bottom='on',top='off', right='off', left='on', which='major', labelsize=14)
    fig.show()
    if path is not None:
        fig.savefig(path,dpi=600,format=fmt,bbox_inches='tight',pad_inches=0.1)

# dictionary of all lnprob
lnprob_funcs = {'simple':lnprob_simple,'simplet':lnprob_simplet,\
                'simplet2':lnprob_simplet2,'simplet3':lnprob_simplet3,\
                'break':lnprob_break,'breakt':lnprob_breakt,\
                'breakt2':lnprob_breakt2,'breakt3':lnprob_breakt3,\
                'break2':lnprob_break2}
# dictionary of all lnlike
lnlike_funcs = {'simple':lnlike_simple,'simplet':lnlike_simplet,\
                'simplet2':lnlike_simplet2,'simplet3':lnlike_simplet3,\
                'break':lnlike_break,'breakt':lnlike_breakt,\
                'breakt2':lnlike_breakt2,'breakt3':lnlike_breakt3,\
                'break2':lnlike_break2}
# dictionary of all models
model_funcs = {'simple':simple_model,'simplet':simplet_model,\
               'simplet2':simplet2_model,'simplet3':simplet3_model,\
               'break':break_model,'breakt':breakt_model,\
               'breakt2':breakt2_model,'breakt3':breakt3_model,\
               'break2':break2_model}