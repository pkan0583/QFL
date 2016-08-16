# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:36:39 2016

@author: Benn
"""

import numpy as np
import scipy as sp
from collections import namedtuple

# Regime switching trend
def RegimeSwitchingTrend_2S(returns, sims, priors):
    
    useInd = np.nonzero(~np.isnan(returns))
    returns = returns[~np.isnan(returns)]
    
    a0 = priors.a0
    a1 = priors.a1
    b0 = priors.b0
    b1 = priors.b1
    psi_mean0 = priors.psi_mean0
    psi_cov0 = priors.psi_cov0
    
    a = np.zeros(sims)
    b = np.zeros(sims)
    sigma = np.zeros(sims)
    psi = np.zeros(sims)
    a[0] = np.random.beta(a0, a1)
    b[0] = np.random.beta(b0, b1)
    psi[0] = np.random.normal(psi_mean0, np.sqrt(psi_cov0))
    
    delta0 = 0.01
    alpha0 = 100
    # f_y = sp.stats.invgamma.pdf(y, alpha0, 0, delta0)
    r = sp.stats.invgamma.rvs(alpha0, size=1, loc=0, scale=delta0)
    sigma[0] = np.sqrt(r)
    
    psi_mean = np.zeros(sims)
    psi_cov = np.zeros(sims)
    
    psi_mean[0] = psi_mean0
    psi_cov[0] = psi_cov0

    print(returns)

    T = len(returns)
    
    states = np.zeros([T, sims])
    stateProbs = np.zeros([T, sims])
    
    for s in range(0, sims-1):
    
        # Forward recursion for state probabilities
        stateProbs[0,0] = b[s] / (a[s] + b[s])
    
        for t in range(1, T):
               
            # Prediction step
            stateProbs[t,s] = stateProbs[t-1,s] * (1-a[s]) + (1-stateProbs[t-1,s]) * b[s]
    
            # Update step
            l1 = np.exp(-(returns[t] ** 2 / (2.0 * sigma[s]**2)))
            l2 = np.exp(-((returns[t] - psi[s])** 2 / (2.0 * sigma[s]**2)))
            
            denom = stateProbs[t,s] * l1 + (1-stateProbs[t,s]) * l2
            stateProbs[t,s] = stateProbs[t,s] * l1 / denom
            
        # Backward recursion for states  (this is RETROSPECTIVE)   
        states[T-1,s] = np.random.binomial(1, 1-stateProbs[t,s])
        for t in range(1, T):
            tau = T-t
            # The future state is 1, prob the prior state is 0
            if states[tau,s] == 1:
                denom = stateProbs[tau-1,s] * a[s] + (1-stateProbs[tau-1,s]) * (1-b[s])
                prob1 = stateProbs[tau-1,s] * a[s] / denom
                states[tau-1,s] = np.random.binomial(1, 1-prob1)
            # The future state is 0, prob the prior state is 0  
            else:
                denom = stateProbs[tau-1,s] * (1-a[s]) + (1-stateProbs[tau-1,s]) * b[s]
                prob1 = stateProbs[tau-1,s] * (1-a[s]) / denom
                states[tau-1,s] = np.random.binomial(1, 1-prob1)
            
        # Switches from 0 to 1    
        ind0 = np.nonzero((states[1:T-1,s] == 1) & (states[0:T-2,s] == 0))[0]
        a0_= a0 + len(ind0)
        a1_ = a1 + T - len(ind0)
        a[s+1] = np.random.beta(a0_, a1_)
        
        # Switches from 1 to 0
        ind1 = np.nonzero((states[1:T-1,s] == 0) & (states[0:T-2,s] == 1))[0] 
        b0_= b0 + len(ind1)
        b1_ = b1 + T - len(ind1)
        b[s+1] = np.random.beta(b0_, b1_)
        
        # Psi (need to use the prior here)
        psi_cov[s+1] = 1.0 / (np.dot(states[:,s], states[:,s].T) + 1.0 / psi_cov0)
        psi_mean[s+1] = psi_cov[s] * (np.dot(states[:,s], returns.T) + psi_mean0 / psi_cov0)
        eps = np.random.normal(0, 1)
        psi[s+1] = psi_mean[s+1] + eps * sigma[s] * np.sqrt(psi_cov[s+1])
        counter = 0
        while (psi[s+1] > 0) & (counter < 10):
            eps = np.random.normal(0, 1)
            psi[s+1] = psi_mean[s+1] - eps * sigma[s] * np.sqrt(psi_cov[s+1])
            counter = counter + 1
        alpha = alpha0 + T
        delta = delta0 + np.dot(returns.T, returns) + psi[0] ** 2 / psi_cov0 # - psi[s+1] ** 2 / psi_cov[s+1]    
        sigma[s+1] = np.sqrt(sp.stats.invgamma.rvs(alpha, size=1, loc=0, scale=delta))
    
    outputTuple = namedtuple("Output", "a b psi sigma states forwardLookingStateProbs usedObservations")
    output = outputTuple(a, b, psi, sigma, states, stateProbs, useInd)
    return output