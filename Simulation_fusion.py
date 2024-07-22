# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numpy.polynomial.hermite import hermgauss
from scipy.stats import logistic, norm
from sklearn.linear_model import LogisticRegression
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler

def generate_simulation_data(K, n_k, M1, M2, M3, p, alpha, beta, sigma):
    data = pd.DataFrame()
    for k in range(1, K+1):
        subgroup_data = []
        sigma_k = sigma[k - 1]

        for i in range(n_k):  
            X_ki_base = np.random.normal(size=(M1, p))               
            X_ki_related = X_ki_base[: M2]  + np.random.normal(scale=0.2, size=(M2, p))    
            X_ki_noise = np.random.normal(size=(M3, p))  
            X_ki = np.vstack((X_ki_base, X_ki_related, X_ki_noise))
            omega_ki = norm.rvs(loc=0, scale=sigma_k)
            
            logit_R_ki = sum(np.dot(X_ki[m], alpha[k-1][m]) for m in range(M1)) + omega_ki
            R_ki = logistic.cdf(logit_R_ki)
            R_ki_binary = np.random.binomial(1, R_ki)

            linear_predictor = sum(np.dot(X_ki[m], beta[k-1][m]) for m in range(M1)) + omega_ki
            T_star = np.random.exponential(scale=np.exp(-linear_predictor))
            C = np.random.exponential(scale=1.5)
            T_ki = np.minimum(T_star, C)
            delta_ki = 1 if T_star <= C else 0
            
            subgroup_data.append([R_ki_binary, T_ki, delta_ki] + list(X_ki.ravel()))
        
        subgroup_df = pd.DataFrame(subgroup_data, \
                                   columns=['R_ki', 'T_ki', 'delta_ki'] + [f'X{j}' for j in range(M * p)])
        subgroup_df['Subgroup'] = k
        data = pd.concat([data, subgroup_df], ignore_index=True)            
    return data

def fused_likelihood(params, data_groups, weights, roots, lambda_, gamma, tau, M, p):
    feature_num = M * p
    K = len(data_groups)
    likelihood_sum = 0
    reg_sum = 0
    group_diff_sum = 0
    sigmas = params[-K:]
       
    # Iterate over each group
    for k in range(K):
        subgroup_data = data_groups[k]
        X, R, T, delta = subgroup_data
        alpha_k = params[k * feature_num:(k + 1) * feature_num]
        beta_k = params[(K + k) * feature_num:(K + k + 1) * feature_num]
        sigma_k = sigmas[k]
        
        integral_approx = 0
        for r, w in zip(roots, weights):
            omega = sigma_k * np.sqrt(2) * r
            logit_p = X @ alpha_k + omega
            prob = logistic.cdf(logit_p)
            hazard_ratios = np.exp(X @ beta_k + omega)
            likelihoods = hazard_ratios**delta * np.exp(-T * hazard_ratios)
            log_fR = np.log(prob ** R * (1 - prob) ** (1 - R) + 1e-8)
            log_fT = np.log(likelihoods + 1e-8)
            log_re = norm.logpdf(omega, 0, scale=sigma_k)
            integral_approx += w * (log_fR + log_fT + log_re)

        likelihood_sum += np.sum(integral_approx)
        reg_sum += lambda_ * (np.sum(alpha_k**2) + np.sum(beta_k**2))
        
        # Fusion penalty between groups
        for j in range(k + 1, K):
            alpha_j = params[j * feature_num:(j + 1) * feature_num]
            beta_j = params[(K + j) * feature_num:(K + j + 1) * feature_num]
            group_diff_sum += tau * (np.sum((alpha_k - alpha_j) ** 2) + np.sum((beta_k - beta_j) ** 2))

    total_penalty = reg_sum + gamma * group_diff_sum
    return -likelihood_sum + total_penalty

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

# Define the simulation parameters
n_experiments = 500
K = 3  # Number of subgroups
n_k = 1000  # Number of individuals per subgroup
# Number of subclones
M1 = 3  
M2 = 1
M3 = 1
M = M1 + M2 + M3
p = 2  # Number of predictors per subclone

sigma_simu = np.array([0.5] * K) 
alpha_simu_base = np.random.normal(size=(M, p))
alpha_simu = np.array([alpha_simu_base + np.random.normal(scale=0.15, size=(M, p)) for _ in range(K)])
beta_simu_base = np.random.normal(size=(M, p))
beta_simu = np.array([beta_simu_base + np.random.normal(scale=0.15, size=(M, p)) for _ in range(K)]  )
params_simu = np.hstack((alpha_simu.reshape((K, M * p)),beta_simu.reshape((K, M * p)),\
                         sigma_simu.reshape(-1,1)))

#Gauss-Hermite
n_points = 5
roots, weights = hermgauss(n_points)
lambda_ = 0.5
l1_ratio = 0.5
gamma = 0.5
tau = 1

initial_params = np.hstack((params_simu[:,:M*p].ravel(),params_simu[:,M*p:2*M*p].ravel(),params_simu[:,-1]))
bounds1 = [(None, None)] * (2 * K * M * p) + [(sigma_simu[0], None)] * K
bounds2 = [(None, None)] * (2 * M * p) + [(0.5, None)]  

param_estimates = np.zeros((n_experiments, 2 * K * M * p + K)) 
logistic_coefs = np.zeros((n_experiments,  K * M * p)) 
cox_coefs = np.zeros((n_experiments,  K * M * p )) 
logistic_coefs_all = np.zeros((n_experiments, p * M)) 
cox_coefs_all = np.zeros((n_experiments, p * M)) 
param_estimates_all = np.zeros((n_experiments, 2*M*p+1)) 

for i in range(n_experiments):
    print(i)    
    # Generate synthetic data
    simulation_data = generate_simulation_data(K, n_k, M1, M2, M3, p, alpha_simu, beta_simu, sigma_simu)
    
    # Split data by subgroup and standardize features
    data_groups = []
    for k in range(1, K + 1):
        subgroup_data = simulation_data[simulation_data['Subgroup'] == k]
        features = ['X' + str(j) for j in range(p * M)]
        X = subgroup_data[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        R = subgroup_data['R_ki'].values
        T = subgroup_data['T_ki'].values
        delta = subgroup_data['delta_ki'].values
        data_groups.append((X_scaled, R, T, delta))
    
        # Logistic Regression
        log_reg = LogisticRegression(max_iter=500)
        log_reg.fit(X_scaled, R)
        logistic_coefs[i, (k-1) * M * p:k * M * p] = log_reg.coef_.flatten()
    
        # Cox PH Regression
        survival_data = pd.DataFrame(data=np.c_[T, delta, X_scaled], columns=['T_ki', 'delta_ki'] + features)
        cph = CoxPHFitter()
        cph.fit(survival_data, duration_col='T_ki', event_col='delta_ki')
        cox_coefs[i,(k-1) * M * p:k * M * p] = cph.params_.values
        
        
    result = minimize(fused_likelihood, initial_params, args=(data_groups, weights, roots, lambda_, gamma, tau, M, p),
                      bounds=bounds1, method='L-BFGS-B')
    if result.success:
        print(f"Optimization succeeded in experiment {i}.")
        param_estimates[i] = result.x
    else:
        print(f"Optimization failed in experiment {i}: {result.message}")
    
    features = ['X' + str(j) for j in range(p * M)]
    X_all = simulation_data[features].values
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)
    R_all = simulation_data['R_ki'].values
    T_all = simulation_data['T_ki'].values
    delta_all = simulation_data['delta_ki'].values
           
    result_all = minimize(joint_likelihood, params_simu[0], args=(X_all_scaled, R_all, T_all, delta_all, weights, roots, lambda_, l1_ratio, 'ridge'), 
                             bounds=bounds2, method='L-BFGS-B')
    param_estimates_all[i] = result_all.x