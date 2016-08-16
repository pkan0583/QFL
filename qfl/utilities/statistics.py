# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:37:32 2016

@author: Benn
"""

import numpy as np
import scipy.stats as sp


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
