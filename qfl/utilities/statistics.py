# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:37:32 2016

@author: Benn
"""

import numpy as np
import scipy.stats as sp
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis

# Multivariate nonparametric regression using exponential smoother
# Data should come in as a nparray
# Grid should be a grid of points 
def npreg(dep_var, ind_var, out_grid, bandwidth, other_weights=None):
    
    num_obs = len(dep_var)
    num_vars = np.ndim(ind_var)
    if other_weights is None:
        other_weights = np.ones(len(dep_var)).astype(float)
    
    # Supports 1d or 2d
    
    if num_vars == 1:
        
        num_grid_points = len(out_grid)
        pred_grid = np.zeros(num_grid_points)
        
        for i in range(0, num_grid_points):
            z = np.abs(ind_var - out_grid[i]) / bandwidth
            # weights = np.exp(-z)    
            weights = sp.norm.pdf(z)
            weights[~np.isfinite(weights)] = 0
            weights = weights * other_weights
            weights = weights / np.sum(weights)
            weights = weights / np.sum(weights)
            ind = np.nonzero(np.isfinite(dep_var))[0]
            pred_grid[i] = np.sum(weights[ind] * dep_var[ind])
            
    elif num_vars == 2:
    
        num_grid_points = len(out_grid[0])
        
        pred_grid = np.zeros([num_grid_points, num_grid_points])
         
        for i in range(0, num_grid_points):
            for k in range(0, num_grid_points):
             
                weights = np.ones(num_obs)
                for j in range(0, num_vars):
                    z = np.abs(ind_var[:, j] - out_grid[j][i, k]) / bandwidth
                    # tmpWeights = np.exp(-z)    
                    tmp_weights = sp.norm.pdf(z)
                    tmp_weights[np.isnan(tmp_weights)] = 0
                    tmp_weights = tmp_weights * other_weights
                    tmp_weights = tmp_weights / np.sum(tmp_weights)
                    weights = weights * tmp_weights
                     
                weights = weights / np.sum(weights)
                pred_grid[i,k] = np.sum(weights[~np.isnan(dep_var)] * dep_var[~np.isnan(dep_var)])

    return pred_grid


class ExplicitFactorModel(object):

    @staticmethod
    def build_long_short_factors(component_prices=None,
                                 long_tickers=None,
                                 short_tickers=None,
                                 factor_names=None,
                                 return_multipliers=None,
                                 return_window_days=5,
                                 hedge_vol_neutral=None,
                                 hedge_vol_com=63):

        """
        :param component_prices: Pandas DataFrame indexed on date,
        with columns named for tickers
        :param long_tickers: list of long tickers
        :param short_tickers: list of short tickers
        :param factor_names: list of factor names
        :return:
        """

        component_returns = component_prices \
                            / component_prices.shift(return_window_days) - 1

        component_daily_returns = component_prices \
                            / component_prices.shift(1) - 1

        component_std = component_returns.ewm(com=hedge_vol_com).std()

        factor_returns = pd.DataFrame(index=component_returns.index,
                                      columns=factor_names)

        factor_daily_returns = pd.DataFrame(index=component_returns.index,
                                            columns=factor_names)

        if hedge_vol_neutral is None:
            hedge_vol_neutral = np.ones(len(factor_names))

        for i in range(0, len(factor_names)):

            hedge_ratio = 1.0
            if hedge_vol_neutral[i] == 1:
                hedge_ratio = component_std[short_tickers[i]] \
                              / component_std[long_tickers[i]]

            factor_returns[factor_names[i]] = component_returns[long_tickers[i]]\
                - component_returns[short_tickers[i]] / hedge_ratio

            factor_daily_returns[factor_names[i]] = \
                component_daily_returns[long_tickers[i]]\
                - component_daily_returns[short_tickers[i]] / hedge_ratio

            if return_multipliers is not None:
                factor_returns[factor_names[i]] *= return_multipliers[i]
                factor_daily_returns[factor_names[i]] *= return_multipliers[i]

        return factor_returns, factor_daily_returns

    @staticmethod
    def compute_exposures(factor_returns=None, target_returns=None):

        """

        :param factor_returns: Pandas DataFrame indexed on date,
        with columns = factors
        :param target_returns: a series index on date, or a DataFrame
        indexed on date with columns for the various time series you want
        exposure for
        :return:
        """

        if isinstance(target_returns, pd.Series):
            target_returns = pd.DataFrame(target_returns)

        regressions = dict()
        coefs = pd.DataFrame(index=target_returns.columns,
                             columns=factor_returns.columns)
        t_stats = pd.DataFrame(index=target_returns.columns,
                             columns=factor_returns.columns)
        for col in target_returns.columns:
            regressions[col] = pd.ols(y=target_returns[col], x=factor_returns)
            coefs.loc[col] = regressions[col].beta
            t_stats.loc[col] = regressions[col].t_stat

        return coefs, t_stats, regressions


class RollingFactorModel(object):

    @classmethod
    def run(cls, **kwargs):

        data = kwargs.get('data', None)
        n_components = kwargs.get('n_components', 3)
        minimum_data = kwargs.get('minimum_obs', 21)
        window_length_days = kwargs.get('window_length_days', 512)
        update_weights_interval_days = kwargs.get('update_interval_days', 21)

        estimation_points = np.arange(minimum_data,
                                      len(data),
                                      update_weights_interval_days)

        factor_data_dict = dict()
        factor_weights_dict = dict()
        factor_data_oos_dict = dict()

        for t in estimation_points:

            t_ = max(0, t - window_length_days)
            data_subset = data.iloc[t_:t].copy(deep=True)

            # First kill invalid series during this date range
            data_subset = data_subset[data_subset.columns[
                ~np.isnan(data_subset).all(axis=0)]]

            # Now zero out NA's
            data_subset = data_subset.fillna(value=0)
            data_subset_z = (data_subset - data_subset.mean()) / data_subset.std()

            # Run FA
            # fa = FactorAnalysis(n_components=n_components).fit(data_subset)
            fa = FactorAnalysis(n_components=n_components).fit(data_subset_z)

            factor_data = pd.DataFrame(index=data_subset.index,
                                       data=fa.transform(data_subset_z))

            factor_weights = pd.DataFrame(index=data_subset.columns,
                                          data=fa.components_.transpose(),
                                          columns=range(0, n_components))

            factor_data_dict[t] = factor_data.copy(deep=True)
            factor_weights_dict[t] = factor_weights.copy(deep=True)

            # Out of sample
            if t > estimation_points[0]:

                factor_data_oos_dict[t] = pd.DataFrame(
                    index=data_subset.index,
                    data=prev_fa.transform(data_subset_z[prev_cols]))

            prev_fa = fa
            prev_cols = data_subset_z.columns

        # We want to loop backwards over estimation points to build the factors
        factor_weights_composite = None
        factor_data_composite = None
        factor_data_oos = None
        for k in range(len(estimation_points)-1, 0, -1):

            t = estimation_points[k]

            if factor_weights_composite is None:

                factor_weights_composite = factor_weights_dict[t].copy(deep=True)
                factor_data_composite = factor_data_dict[t].copy(deep=True)
                factor_weights_composite['date'] = data.index[t]
                factor_weights_composite = factor_weights_composite \
                    .reset_index().set_index(['date', 'ticker'])

                factor_data_oos = factor_data_oos_dict[t].copy(deep=True)

            else:

                t_ = estimation_points[k + 1]

                prev_dates = factor_data_dict[t_].index.get_level_values('date')

                new_factor_weights = factor_weights_dict[t].copy(deep=True)
                new_factor_weights['date'] = data.index[t]
                new_factor_weights = new_factor_weights.reset_index() \
                    .set_index(['date', 'ticker'])

                new_factor_data = factor_data_dict[t][
                    factor_data_dict[t].index.get_level_values('date')
                    < prev_dates.min()]

                factor_weights_composite = factor_weights_composite.append(
                    new_factor_weights)
                factor_data_composite = factor_data_composite.append(
                    new_factor_data)

                if t in factor_data_oos_dict:
                    new_factor_data_oos = factor_data_oos_dict[t][
                        factor_data_oos_dict[t].index.get_level_values('date')
                        < prev_dates.min()]

                    factor_data_oos = factor_data_oos.append(new_factor_data_oos)

        # Sort
        factor_weights_composite = factor_weights_composite.sort_index()
        factor_data_composite = factor_data_composite.sort_index()
        factor_data_oos = factor_data_oos.sort_index()

        return factor_weights_composite, factor_data_composite, factor_data_oos


