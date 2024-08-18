# -*- coding: utf-8 -*-
"""
Created on Mon May 27 22:47:44 2024

@author: zjwyx
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numpy.polynomial.hermite import hermgauss
from scipy.stats import logistic, norm
from sklearn.preprocessing import MinMaxScaler

# Function to prepare feature matrices from genomic data
def prepare_feature_matrices(df):
    feature_matrices = {}
    max_clone_label = int(df['clone'].max())
    p = 2  # Number of predictors per subclone
    patient_ids = df['PATIENT_ID'].unique()
    
    for patient_id in patient_ids:
        patient_data = df[df['PATIENT_ID'] == patient_id]
        summary = patient_data.groupby('clone').agg(
            clone_tmb=('clone', 'size'),
            CCF_mean=('CCF', 'mean')
        ).reset_index()
        matrix = np.zeros((max_clone_label, p))
        for _, row in summary.iterrows():
            clone_index = int(row['clone']) - 1
            matrix[clone_index, :] = [row['clone_tmb'], row['CCF_mean']]
        feature_matrices[patient_id] = matrix.ravel()
    
    return feature_matrices, max_clone_label, p

# Function to merge feature matrices with clinical data
def create_feature_dataframe(feature_matrices, df_clinical):
    features_df = pd.DataFrame.from_dict(feature_matrices, orient='index')
    features_df.columns = [f'feature_{i+1}' for i in range(features_df.shape[1])]
    return df_clinical.join(features_df, on='PATIENT_ID')

# Function to calculate Euclidean distances between data groups
def calculate_feature_distances(data_groups):
    distances, max_distance = {}, 0
    for k, group_k in enumerate(data_groups):
        mean_k = np.mean(group_k[0], axis=0)
        for j, group_j in enumerate(data_groups[k+1:], k+1):
            distance = np.linalg.norm(mean_k - np.mean(group_j[0], axis=0))
            distances[(k+1, j+1)] = distance
            max_distance = max(max_distance, distance)
    return distances, max_distance

# Functions for KL divergence calculations
def kl_divergence(mu1, sigma1, mu2, sigma2):
    epsilon = 1e-5
    regularized_sigma1 = sigma1 + epsilon * np.eye(sigma1.shape[0])
    regularized_sigma2 = sigma2 + epsilon * np.eye(sigma2.shape[0])
    inv_sigma2 = np.linalg.inv(regularized_sigma2)
    tr_term = np.trace(inv_sigma2 @ regularized_sigma1)
    diff_mu = mu2 - mu1
    quad_term = diff_mu.T @ inv_sigma2 @ diff_mu
    log_det_term = np.log(np.linalg.det(regularized_sigma2) / np.linalg.det(regularized_sigma1))
    return 0.5 * (tr_term + quad_term - len(mu1) + log_det_term)

def symmetric_kl(mu1, sigma1, mu2, sigma2):
   return 0.5 * (kl_divergence(mu1, sigma1, mu2, sigma2) + kl_divergence(mu2, sigma2, mu1, sigma1))

# Calculate Taus based on distance or KL divergence
def calculate_kl_distances(mus, sigmas):
    kl_distances = {}
    for i in range(len(mus)):
        for j in range(i + 1, len(mus)):
            kl_dist = symmetric_kl(mus[i], sigmas[i], mus[j], sigmas[j])
            kl_distances[(i+1, j+1)] = kl_distances[(j+1, i+1)] = kl_dist
    return kl_distances


def calculate_taus(data_groups, mus, sigmas, tau_type):
    if tau_type == "none":
        taus = {(i+1, j+1): 1 for i in range(len(data_groups)) for j in range(i, len(data_groups))}
    elif tau_type == "distance":
        distances, max_distance = calculate_feature_distances(data_groups)
        taus = {key: 1 - value / max_distance for key, value in distances.items()}
    elif tau_type == "kl_divergence":
        kl_distances = calculate_kl_distances(mus, sigmas)
        max_kl = max(kl_distances.values())
        taus = {key: 1 - value / max_kl for key, value in kl_distances.items()}
    return taus

def fused_likelihood(params, data_groups, max_clone_label, p, weights, roots, lambda_, gamma, tau_type, taus):
    feature_num = max_clone_label * p
    K = len(data_groups)
    likelihood_sum = 0
    reg_sum = 0
    group_diff_sum = 0
       
    # Iterate over each group
    for k in range(K):
        subgroup_data = data_groups[k]
        X, R, T, delta = subgroup_data
        alpha_k = params[k * feature_num:(k + 1) * feature_num]
        beta_k = params[(K + k) * feature_num:(K + k + 1) * feature_num]
        sigma_k = 0.5
        
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
        
        # Apply fusion penalty
        for j in range(k + 1, K):
            alpha_j = params[j * feature_num:(j + 1) * feature_num]
            beta_j = params[(K + j) * feature_num:(K + j + 1) * feature_num]
            if tau_type == "distance" or tau_type == "kl_divergence":
                tau_value = taus.get((k, j), 1)  # Default tau to 1 if not specified
            else:
                tau_value = 1  # uniform fusion penalty
            group_diff_sum += tau_value * (np.sum((alpha_k - alpha_j) ** 2) + np.sum((beta_k - beta_j) ** 2))

    total_penalty = reg_sum + gamma * group_diff_sum
    return -likelihood_sum + total_penalty

def joint_likelihood(params, X, R, T, delta, weights, roots, lambda_, l1_ratio, regularization='none'):
    feature_num = len(X[0])
    alpha = params[:feature_num]
    beta = params[feature_num:2*feature_num]
    sigma = 0.5
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

# Load and prepare data
def load_and_prepare_data(clinical_path, genomic_path):
    df_clinical = pd.read_excel(clinical_path, sheet_name='Clinical')
    df_genomic = pd.read_excel(genomic_path, sheet_name='Genomic')
    feature_matrices, max_clone_label, p = prepare_feature_matrices(df_genomic)
    df = create_feature_dataframe(feature_matrices, df_clinical)
    return df.dropna(), max_clone_label, p

# Setup data for model fitting
def setup_data_groups(df, feature_columns):
    K = df['Study ID'].max()
    data_groups, mus, sigmas = [], [], []
    
    for k in range(1, K + 1):
        subgroup_data = df[df['Study ID'] == k]
        X = subgroup_data[feature_columns].values
        mus.append(np.mean(X, axis=0))
        sigmas.append(np.cov(X, rowvar=False))
        R, T, delta = subgroup_data['ORR'].values, subgroup_data['PFS'].values, subgroup_data['Status'].values
        data_groups.append((X, R, T, delta))
    
    return data_groups, mus, sigmas, K

#example
df, max_clone_label, p = load_and_prepare_data('nsclc.xlsx', 'nsclc.xlsx')

feature_columns1 = [col for col in df.columns if 'feature_' in col and int(col.split('_')[-1]) % 2 != 0]
df_normalized = df.copy()

scaler = MinMaxScaler()

for name, group in df.groupby('Study ID'):
    normalized_data = scaler.fit_transform(group[feature_columns1])
    df_normalized.loc[group.index, feature_columns1] = normalized_data  
df_normalized[feature_columns1] = df_normalized[feature_columns1].astype(float)


feature_columns = [f'feature_{i+1}' for i in range(max_clone_label * p)]
data_groups, mus, sigmas, K = setup_data_groups(df_normalized, feature_columns)


#Gauss-Hermite
n_points = 5
roots, weights = hermgauss(n_points)
# Set parameters
lambda_, gamma, l1_ratio = 0.5, 0.5, 0.5
tau_type = "kl_divergence"  # "none", "distance", or "kl_divergence"
taus = calculate_taus(data_groups, mus, sigmas, tau_type)

result = minimize(fused_likelihood, np.random.normal(size=(2 * K * max_clone_label * p )),
                  args=(data_groups, max_clone_label, p, weights, roots, lambda_, gamma, tau_type, taus),
                  method='L-BFGS-B', options={'maxiter': 10000})
if result.success:
    print("Optimization succeeded.")
    param_fusion = result.x
else:
    print(f"Optimization failed: {result.message}")
