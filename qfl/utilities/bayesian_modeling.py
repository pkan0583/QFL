# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:36:39 2016

@author: Benn
"""

import pandas as pd
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


class BayesianStateTransitionModel(object):

    @classmethod
    def compute_state_timeseries(cls,
                                 data=None,
                                 params=None,
                                 **kwargs):

        num_states = len(params['pr_s_0'])
        pr_s_p = pd.DataFrame(index=data.index, columns=range(0, num_states))
        pr_s_u = pd.DataFrame(index=data.index, columns=range(0, num_states))
        pr_s_u_b = pd.DataFrame(index=data.index, columns=range(0, num_states))

        pr_s_p.iloc[0] = params['pr_s_0']

        # Forward iteration
        for t in range(0, len(data)):

            if t > 0:

                pr_s_p.iloc[t] = cls.forward_prediction_step(
                    prior_state_prob=pr_s_u.iloc[t-1],
                    params=params,
                    t=t,
                    **kwargs)

            pr_s_u.iloc[t] = cls.forward_update_step(
                prior_state_prob=pr_s_p.iloc[t],
                data=data,
                params=params,
                t=t,
                **kwargs)

        # Backward iteration
        for tau in range(0, len(data)):

            t = len(data) - 1 - tau

            if tau == 0:
                pr_s_u_b.iloc[t] = pr_s_u.iloc[t]

            else:
                pr_s_u_b.iloc[t] = BayesianStateTransitionModel\
                    .backward_update_step_unconditional(
                        state_prob=pr_s_u.iloc[t],
                        future_state_prob=pr_s_u_b.iloc[t+1],
                        params=params)

        return pr_s_u_b, pr_s_u, pr_s_p

    @classmethod
    def likelihood_t(cls,
                     state=None,
                     prior_state_prob=None,
                     data=None,
                     params=None,
                     **kwargs):
        raise NotImplementedError

    @classmethod
    def forward_update_step(cls,
                            prior_state_prob=None,
                            data=None,
                            params=None,
                            t=None,
                            **kwargs):

        num_states = len(prior_state_prob)
        state_prob = np.zeros(num_states)
        ll_s_t = np.zeros(num_states)

        denom = 0.0

        for s in range(0, num_states):

            ll_s_t[s] = cls.likelihood_t(state=s,
                                         data=data,
                                         params=params,
                                         t=t,
                                         prior_state_prob=prior_state_prob,
                                         **kwargs)

            denom += ll_s_t[s] * prior_state_prob[s]

        for s in range(0, num_states):
            state_prob[s] = ll_s_t[s] * prior_state_prob[s] / denom

        return state_prob

    @classmethod
    def forward_prediction_step(cls,
                                prior_state_prob=None,
                                params=None,
                                t=None,
                                **kwargs):

        num_states = len(prior_state_prob)
        pi_t = params['state_transition_matrix']
        state_prob = np.zeros(num_states)

        for s in range(0, num_states):
            state_prob[s] = 0.0
            for s1 in range(0, num_states):
                state_prob[s] += pi_t[s, s1] * prior_state_prob[s1]

        return state_prob

    @classmethod
    def backward_update_step(cls,
                             state_prob=None,
                             future_state=None,
                             t=None,
                             params=None,
                             **kwargs):

        num_states = len(state_prob)
        pi_t = params['state_transition_matrix']
        updated_state_prob = np.zeros(num_states)

        # Loop over possible states, for denominator
        denom = 0.0
        for s in range(0, num_states):
            denom += state_prob[s] * pi_t[future_state, s]

        # Loop over the states we're evaluating probabilities for
        for s in range(0, num_states):
            updated_state_prob[s] = state_prob[s] * pi_t[future_state, s] \
                / denom

        return updated_state_prob

    @classmethod
    def backward_update_step_unconditional(cls,
                                           state_prob=None,
                                           future_state_prob=None,
                                           t=None,
                                           params=None,
                                           **kwargs):

        num_states = len(state_prob)
        updated_state_prob = np.zeros(num_states)

        # Loop over what future states might be
        for s in range(0, num_states):

            updated_state_prob += future_state_prob[s] \
                * cls.backward_update_step(
                    state_prob=state_prob,
                    future_state=s,
                    params=params,
                    t=t,
                    **kwargs
                )

        return updated_state_prob


class BayesianTimeSeriesDataCleaner(BayesianStateTransitionModel):

    @classmethod
    def clean_data(cls,
                   x=None,
                   z=None,
                   **kwargs):

        params = dict()

        # Optional
        params['state_transition_matrix'] = kwargs.get(
            'state_transition_matrix',
            np.array([[0.99, 0.5], [0.01, 0.5]]))
        params['pr_s_0'] = kwargs.get('pr_s_0', np.array([0.99, 0.01]))

        # Data comes in as levels... add a lag of dependent variable
        if z is None:
            z = pd.DataFrame(index=x.index)
        z['x'] = x
        z = z[np.isfinite(z['x'])]
        z['lag'] = z['x'].shift(1)
        del z['x']

        reg = pd.ols(y=x, x=z)
        params['psi'] = pd.Series(index=reg.resid.index)
        params['psi'].iloc[0] = reg.resid.iloc[0]
        for t in range(1, len(params['psi'])):
            params['psi'].iloc[t] = reg.resid.iloc[t] + \
                reg.beta['lag'] * params['psi'].iloc[t - 1]
        params['s2'] = reg.resid.std() * 2.0
        params['rho'] = reg.beta['lag']

        # State probabilities
        pr_s_ub, pr_s_u, pr_s_p = cls.compute_state_timeseries(
            data=reg.resid,
            params=params,
            **kwargs
        )

        # sample (assume last day is at the end)
        df = pd.DataFrame(index=reg.resid.index)
        df['x'] = x
        df['psi'] = params['psi']
        df['state_probs'] = pr_s_ub[1]
        df['x_clean'] = df['x'] - df['state_probs'] * df['psi']

        return df

    @classmethod
    def likelihood_t(cls,
                     state=None,
                     prior_state_prob=None,
                     data=None,
                     params=None,
                     t=None,
                     **kwargs):

        psi = params['psi']
        r = data

        if t == 0:
            psi_l = 0.0
        else:
            psi_l = psi[t-1]

        # Conditional on s(t-1) == 0
        l0 = (1.0 / (2.0 * np.pi * params['s2'])) \
            * np.exp(-0.5 / params['s2'] * (r[t] - psi[t] * state) ** 2.0)

        # Conditional on s(t-1) == 1
        l1 = (1.0 / (2.0 * np.pi * params['s2'])) \
            * np.exp(-0.5 / params['s2'] *
                     (r[t] - psi[t] * state + psi_l * params['rho']) ** 2.0)

        return l0 * prior_state_prob[0] + l1 * prior_state_prob[1]

    @staticmethod
    def sample_psi(x_resid=None, states=None, params=None, shocks=None):

        # Prior for psi is uninformative
        # If s(t-1) = 1 then allow some correlation?

        psi_mean = pd.Series(index=x_resid.index)

        # 0-0
        ind = states.index[(states == 0) & (states.shift(1) == 0)]
        psi_mean.loc[ind] = 0.0

        # 1-1
        ind = states.index[(states == 1) & (states.shift(1) == 1)]
        psi_mean.loc[ind] = x_resid.loc[ind] - x_resid.shift(1).loc[ind]

        # 1-0
        ind = states.index[(states == 1) & (states.shift(1) == 0)]
        psi_mean.loc[ind] = x_resid.loc[ind]

        # 0-1
        ind = states.index[(states == 0) & (states.shift(1) == 1)]
        psi_mean.loc[ind] = -x_resid.shift(1).loc[ind]

        psi_var = params['s2']

        eps = shocks['psi']
        psi_t = eps * psi_var ** 0.5 + psi_mean

        return psi_t


