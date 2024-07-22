# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numpy.polynomial.hermite import hermgauss
from scipy.stats import logistic, norm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def generate_simulation_data(n, M1, M2, M3, p, alpha, beta, sigma):
    data = pd.DataFrame()
    group_data = []

    for i in range(n):
        X_base = np.random.normal(size=(M1, p))       
        X_related = X_base[: M2] + np.random.normal(scale=0.1, size=(M2, p))   
        X_noise = np.random.normal(size=(M3, p))  
        X_i = np.vstack((X_base, X_related, X_noise))

        omega_i = norm.rvs(loc=0, scale=sigma)
        
        logit_R_i = sum(np.dot(X_i[m], alpha[m]) for m in range(M1)) + omega_i
        R_i = logistic.cdf(logit_R_i)
        R_i_binary = np.random.binomial(1, R_i)
        linear_predictor = sum(np.dot(X_i[m], beta[m]) for m in range(M1)) + omega_i       
        T_star = np.random.exponential(scale=np.exp(-linear_predictor))       
        C = np.random.exponential(scale=1.5)        
        T_i = np.minimum(T_star, C)
        delta_i = 1 if T_star <= C else 0

        group_data.append([R_i_binary, T_i, delta_i] + list(X_i.ravel()))
    
  
    columns=['R_i', 'T_i', 'delta_i'] + [f'X{j}' for j in range(X_i.shape[0]*X_i.shape[1])]
    group_df = pd.DataFrame(group_data, columns=columns)
    data = pd.concat([data, group_df], ignore_index=True)    
          
    return data


def joint_likelihood(params, X, R, T, delta, weights, roots, lambda_, l1_ratio, regularization='none'):
    feature_num = len(X[0])
    alpha = params[:feature_num]
    beta = params[feature_num:2*feature_num]
    sigma = params[-1]
    integral_approx = 0
    
    for r, w in zip(roots, weights):
        omega = sigma * np.sqrt(2) * r
        logit_p = np.dot(X, alpha) + omega
        f = logistic.cdf(logit_p)
        hazard_ratios = np.exp(np.dot(X, beta) + omega)
        likelihoods = hazard_ratios**delta * np.exp(-T*hazard_ratios)
        
        log_fR = np.log(f ** R * (1 - f) ** (1 - R) + 1e-8)
        log_fT = np.log(likelihoods + 1e-8)
        log_re = norm.logpdf(omega, 0, scale=sigma)
        integral_approx += w * (log_fR + log_fT + log_re)

    # Apply regularization
    if regularization == 'lasso':
        penalty = lambda_ * (np.sum(np.abs(alpha)) + np.sum(np.abs(beta)))
    elif regularization == 'ridge':
        penalty = lambda_ * (np.sum(alpha**2) + np.sum(beta**2))
    elif regularization == 'elastic_net':
        lasso_penalty = l1_ratio * lambda_ * (np.sum(np.abs(alpha)) + np.sum(np.abs(beta)))
        ridge_penalty = (1 - l1_ratio) * lambda_ * (np.sum(alpha**2) + np.sum(beta**2))
        penalty = lasso_penalty + ridge_penalty
    else:
        penalty = 0  # No regularization
    
    joint = -np.sum(integral_approx) + penalty
    return joint


def soft_thresholding(x, lambda_):
    return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)

def callback(params):
    params[:-1] = soft_thresholding(params[:-1], lambda_=0.1)
    return params

n_experiments = 1000
n = 2000  
M1 = 3 
M2 = 1
M3 = 1
M = M1 + M2 + M3
p = 2

sigma_simu = 0.5  
alpha_simu = np.random.normal(size=(M, p))
beta_simu = np.random.normal(size=(M, p))
params_simu = np.hstack((alpha_simu.ravel(),beta_simu.ravel(),np.array(sigma_simu)))
params_initial = np.zeros(len(params_simu))
lambda_ = 0.5  
l1_ratio = 0.5  
n_points = 5
roots, weights = hermgauss(n_points)

bounds = [(None, None)] * (len(params_initial) - 1) + [(sigma_simu, None)]  

param_estimates_none = np.zeros((n_experiments, len(params_simu))) 
param_estimates_lasso = np.zeros((n_experiments, len(params_simu))) 
param_estimates_ridge = np.zeros((n_experiments, len(params_simu)))  
param_estimates_net = np.zeros((n_experiments, len(params_simu)))

for i in range(n_experiments):
    print(i)
    simulation_data = generate_simulation_data(n, M1 , M2, M3, p, alpha_simu, beta_simu, sigma_simu)
    
    features = ['X' + str(j) for j in range(p * M)]
    X = simulation_data[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    R = simulation_data['R_i'].values
    T = simulation_data['T_i'].values
    delta = simulation_data['delta_i'].values
    
    # regularization: 'none', 'lasso', 'ridge', 'elastic_net'
    result_none = minimize(joint_likelihood, params_initial, args=(X_scaled, R, T, delta, weights, roots, lambda_, l1_ratio, 'none'), bounds=bounds, method='L-BFGS-B')
    param_estimates_none[i] = result_none.x
    result_lasso = minimize(joint_likelihood, params_initial, args=(X_scaled, R, T, delta, weights, roots, lambda_, l1_ratio, 'lasso'), callback=callback, bounds=bounds, method='L-BFGS-B')
    param_estimates_lasso[i] = result_lasso.x
    result_ridge = minimize(joint_likelihood, params_initial, args=(X_scaled, R, T, delta, weights, roots, lambda_, l1_ratio, 'ridge'), bounds=bounds, method='L-BFGS-B')
    param_estimates_ridge[i] = result_ridge.x
    result_net = minimize(joint_likelihood, params_initial, args=(X_scaled, R, T, delta, weights, roots, lambda_, l1_ratio, 'elastic_net'), bounds=bounds, method='L-BFGS-B')
    param_estimates_net[i] = result_net.x


mse_scores_none = np.zeros(n_experiments)
mse_scores_lasso = np.zeros(n_experiments)
mse_scores_ridge = np.zeros(n_experiments)
mse_scores_net = np.zeros(n_experiments)

for i in range(n_experiments):
    mse_scores_none[i] = (mean_squared_error(params_simu[:M1*p], param_estimates_none[i][:M1*p])+\
                         mean_squared_error(params_simu[M*p:M*p+M1*p], param_estimates_none[i][M*p:M*p+M1*p]))/2
    mse_scores_lasso[i] = (mean_squared_error(params_simu[:M1*p], param_estimates_lasso[i][:M1*p])+\
                           mean_squared_error(params_simu[M*p:M*p+M1*p], param_estimates_lasso[i][M*p:M*p+M1*p]))/2
    mse_scores_ridge[i] = (mean_squared_error(params_simu[:M1*p], param_estimates_ridge[i][:M1*p])+\
                          mean_squared_error(params_simu[M*p:M*p+M1*p], param_estimates_ridge[i][M*p:M*p+M1*p]))/2
    mse_scores_net[i] = (mean_squared_error(params_simu[:M1*p], param_estimates_net[i][:M1*p])+\
                        mean_squared_error(params_simu[M*p:M*p+M1*p], param_estimates_ridge[i][M*p:M*p+M1*p]))/2   

parameter_biases_none = np.hstack((abs(params_simu[:M1*p] - np.mean(param_estimates_none,axis = 0)[:M1*p])\
    ,abs(params_simu[M*p:M*p+M1*p] - np.mean(param_estimates_none,axis = 0)[M*p:M*p+M1*p])))
parameter_biases_lasso = np.hstack((abs(params_simu[:M1*p] - np.mean(param_estimates_lasso,axis = 0)[:M1*p])\
    ,abs(params_simu[M*p:M*p+M1*p] - np.mean(param_estimates_lasso,axis = 0)[M*p:M*p+M1*p])))
parameter_biases_ridge = np.hstack((abs(params_simu[:M1*p] - np.mean(param_estimates_ridge,axis = 0)[:M1*p])\
    ,abs(params_simu[M*p:M*p+M1*p] - np.mean(param_estimates_ridge,axis = 0)[M*p:M*p+M1*p])))
parameter_biases_net = np.hstack((abs(params_simu[:M1*p] - np.mean(param_estimates_net,axis = 0)[:M1*p])\
    ,abs(params_simu[M*p:M*p+M1*p] - np.mean(param_estimates_net,axis = 0)[M*p:M*p+M1*p])))    

mean_mse_none = np.mean(mse_scores_none)
mean_bias_none = np.mean(parameter_biases_none)
mean_mse_lasso = np.mean(mse_scores_lasso)
mean_bias_lasso = np.mean(parameter_biases_lasso)
mean_mse_ridge = np.mean(mse_scores_ridge)
mean_bias_ridge = np.mean(parameter_biases_ridge)
mean_mse_net = np.mean(mse_scores_net)
mean_bias_net = np.mean(parameter_biases_net)

print("Mean MSE None:", mean_mse_none)
print("Mean Parameter Bias None", mean_bias_none)
print("Mean MSE Lasso:", mean_mse_lasso)
print("Mean Parameter Bias Lasso", mean_bias_lasso)
print("Mean MSE Ridge:", mean_mse_ridge)
print("Mean Parameter Bias Ridge", mean_bias_ridge)
print("Mean MSE Net:", mean_mse_net)
print("Mean Parameter Bias Net", mean_bias_net)