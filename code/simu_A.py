import numpy as np
import pandas as pd
import statistics
from sklearn.datasets import make_spd_matrix
from sklearn.preprocessing import StandardScaler

def generate_covariates(n, p, corr_matrix=None):
    """
    Generate covariates with specified correlation structure.
    :param n: Number of samples
    :param p: Number of covariates
    :param corr_matrix: Correlation matrix (p x p)
    :return: Covariates matrix (n x p)
    """
    if corr_matrix is None:
        corr_matrix = np.eye(p)
    mean = np.zeros(p)
    corr_matrix[0, 4] = corr_matrix[4, 0] = 0.2  # X1 和 X2 之间的相关性
    corr_matrix[2, 6] = corr_matrix[6, 2] = 0.9  # X3 和 X4 之间的相关性
    corr_matrix[3, 8] = corr_matrix[8, 3] = 0.2  # X5 和 X6 之间的相关性
    corr_matrix[4, 9] = corr_matrix[9, 4] = 0.9  # X7 和 X8 之间的相关性

    X = np.random.multivariate_normal(mean, corr_matrix, n)
    return X

def generate_exposure(X, beta, scenario):
    """
    Generate binary exposure based on logistic model.
    :param covariates: Covariates matrix (n x p)
    :param beta: Coefficients for logistic model
    :param scenario: Scenario specification (A to G)
    :return: Exposure vector (n,)
    """
    n, p = X.shape
    if scenario == 'A':
        f_W = X[:, :4].dot(beta[:4])
    elif scenario == 'B':
        f_W = X[:, :4].dot(beta[:4]) + X[:, 0]**2 * beta[4]
    elif scenario == 'C':
        f_W = X[:, :4].dot(beta[:4]) + np.sum(X[:, :3]**2 * beta[4:7], axis=1)
    elif scenario == 'D':
        f_W = X[:, :4].dot(beta[:4]) + np.sum(X[:, :3] * X[:, 3:6] * beta[4:7], axis=1)
    elif scenario == 'E':
        f_W = X[:, :4].dot(beta[:4]) + np.sum(X[:, :3] * X[:, 3:6] * beta[4:7], axis=1) + X[:, 0]**2 * beta[7]
    elif scenario == 'F':
        f_W = X[:, :4].dot(beta[:4]) + np.sum(X[:, :5] * X[:, 5:10] * beta[4:14], axis=1)
    if scenario == 'G':
        f_X = np.column_stack((X[:, :4], X[:, 4:7]**2, X[:, 0] * X[:, 5], X[:, 0] * X[:, 6], X[:, 1] * X[:, 4], X[:, 1] * X[:, 6], X[:, 2] * X[:, 4], X[:, 2] * X[:, 5], X[:, 2] * X[:, 6], X[:, 3] * X[:, 4], X[:, 3] * X[:, 5], X[:, 3] * X[:, 6]))
    else:
        raise ValueError("Invalid scenario")
    
    exposure_prob = 1 / (1 + np.exp(-f_X.dot(beta)))
    #print(statistics.mean(exposure_prob))
    exposure = np.random.binomial(1, exposure_prob)
    return exposure

def generate_outcome(X, exposure, alpha, gamma):
    """
    Generate continuous outcome.
    :param X: Covariates matrix (n x p)
    :param exposure: Exposure vector (n,)
    :param alpha: Coefficients for covariates
    :param gamma: Effect of exposure
    :return: Outcome vector (n,)
    """
    outcome = X.dot(alpha) + gamma * exposure + np.random.normal(0, 0.1, X.shape[0])
    return outcome

def simulate_data(n, scenario):
    """
    Simulate data for a given scenario and sample size.
    :param n: Sample size
    :param scenario: Scenario specification (A to G)
    :return: DataFrame with covariates, exposure, and outcome
    """
    p = 10  # Number of covariates
    X = generate_covariates(n, p)
    
    # Define coefficients for logistic model
    beta = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,0.1,0.1])
    
    # Generate exposure
    exposure = generate_exposure(X, beta, scenario)
    
    # Define coefficients for outcome model
    alpha = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    gamma = -0.4  # Effect of exposure
    
    # Generate outcome
    outcome = generate_outcome(X, exposure, alpha, gamma)
    
    # Create DataFrame
    data = pd.DataFrame(X, columns=[f'W{i+1}' for i in range(p)])
    data['T'] = exposure
    data['Y'] = outcome
    
    return data

# Simulate data for different scenarios and sample sizes
scenarios = ['G']
sample_sizes = [500]
num_datasets = 100

for scenario in scenarios:
    for n in sample_sizes:
        for i in range(num_datasets):
            data = simulate_data(n, scenario)
            data.to_csv(f'data_A/data_scenario_{scenario}_n_{n}_dataset_{i}.csv', index=False)
            print(f'Saved data_scenario_{scenario}_n_{n}_dataset_{num_datasets-1}.csv')