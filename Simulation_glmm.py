# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numpy.polynomial.hermite import hermgauss
from scipy.stats import logistic, norm
from sklearn.linear_model import LogisticRegression
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

def generate_simulation_data(n, M, p, alpha, beta, sigma):
    data = pd.DataFrame()
    group_data = []
    
    for i in range(n):
        X_i = np.random.normal(size=(M, p))       

        omega_i = norm.rvs(loc=0, scale=sigma)
        
        logit_R_i = sum(np.dot(X_i[m], alpha[m]) for m in range(M)) + omega_i
        R_i = logistic.cdf(logit_R_i)
        R_i_binary = np.random.binomial(1, R_i)
     
        linear_predictor = sum(np.dot(X_i[m], beta[m]) for m in range(M)) + omega_i       
        T_star = np.random.exponential(scale=np.exp(-linear_predictor))       
        C = np.random.exponential(scale=1.5)        
        T_i = np.minimum(T_star, C)
        delta_i = 1 if T_star <= C else 0
        
        group_data.append([R_i_binary, T_i, delta_i] + list(X_i.ravel()))
    
  
    columns=['R_i', 'T_i', 'delta_i'] + [f'X{j}' for j in range(X_i.shape[0]*X_i.shape[1])]
    group_df = pd.DataFrame(group_data, columns=columns)
    data = pd.concat([data, group_df], ignore_index=True)    
          
    return data


def joint_likelihood(params, X, R, T, delta, weights, roots):
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
    
    joint = -np.sum(integral_approx)
    return joint

n_experiments = 1000
n = 1000  
M = 3  
p = 2  
sigma_simu = 1.5   
alpha_simu = np.random.normal(size=(M, p))
beta_simu = np.random.normal(size=(M, p))
params_simu = np.hstack((alpha_simu.ravel(),beta_simu.ravel(),np.array(sigma_simu)))
params_initial = np.zeros(len(params_simu))
n_points = 10
roots, weights = hermgauss(n_points)
bounds = [(None, None)] * (len(params_initial) - 1) + [(sigma_simu, None)]  

param_estimates = np.zeros((n_experiments, len(params_simu)))  
logistic_coefs = np.zeros((n_experiments, p * M)) 
cox_coefs = np.zeros((n_experiments, p * M))  

for i in range(n_experiments):
    simulation_data = generate_simulation_data(n, M, p, alpha_simu, beta_simu, sigma_simu)
    
    features = ['X' + str(j) for j in range(p * M)]
    X = simulation_data[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    R = simulation_data['R_i'].values
    T = simulation_data['T_i'].values
    delta = simulation_data['delta_i'].values
    
    result = minimize(joint_likelihood, params_initial, args=(X_scaled, R, T, delta, weights, roots), bounds=bounds, method='L-BFGS-B')
    param_estimates[i] = result.x
    
    # Logistic Regression
    log_reg = LogisticRegression(max_iter=500)
    log_reg.fit(X_scaled, R)
    logistic_coefs[i] = log_reg.coef_.flatten()

    # Cox PH Regression
    survival_data = pd.DataFrame(data=np.c_[T, delta, X_scaled], columns=['T_i', 'delta_i'] + features)
    cph = CoxPHFitter()
    cph.fit(survival_data, duration_col='T_i', event_col='delta_i')
    cox_coefs[i] = cph.params_.values
   
accuracy_scores = np.zeros(n_experiments)
auc_scores = np.zeros(n_experiments)
mse_scores = np.zeros(n_experiments)
parameter_biases = np.zeros((n_experiments, len(params_simu)))

mse_scores_log = np.zeros(n_experiments)
parameter_biases_log = np.zeros((n_experiments, M*p))
mse_scores_cox = np.zeros(n_experiments)
parameter_biases_cox = np.zeros((n_experiments, M*p))

for i in range(n_experiments):
    predictions = param_estimates[i][:M*p].dot(X_scaled.T)
    predicted_classes = (predictions > 0).astype(int)
    accuracy_scores[i] = accuracy_score(R, predicted_classes)
    
    if len(np.unique(R)) > 1:  
        auc_scores[i] = roc_auc_score(R, predictions)
    
    mse_scores[i] = mean_squared_error(params_simu, param_estimates[i])
    mse_scores_log[i] = mean_squared_error(alpha_simu.ravel(), logistic_coefs[i])
    mse_scores_cox[i] = mean_squared_error(beta_simu.ravel(), cox_coefs[i])

    parameter_biases[i] = abs(params_simu - param_estimates[i])
    parameter_biases_log[i] = abs(alpha_simu.ravel() - logistic_coefs[i])
    parameter_biases_cox[i] = abs(beta_simu.ravel() - cox_coefs[i])

mean_accuracy = np.mean(accuracy_scores)
mean_auc = np.mean(auc_scores)
mean_mse = np.mean(mse_scores)
mean_bias = np.mean(parameter_biases)
mean_mse_log = np.mean(mse_scores_log)
mean_bias_log = np.mean(parameter_biases_log)
mean_mse_cox = np.mean(mse_scores_cox)
mean_bias_cox = np.mean(parameter_biases_cox)

print("Mean Accuracy:", mean_accuracy)
print("Mean AUC-ROC:", mean_auc)
print("Mean MSE:", mean_mse)
print("Mean Parameter Bias", mean_bias)
print("Mean MSE log:", mean_mse_log)
print("Mean Parameter Bias for log:", mean_bias_log)
print("Mean MSE cox:", mean_mse_cox)
print("Mean Parameter Bias for cox:", mean_bias_cox)

