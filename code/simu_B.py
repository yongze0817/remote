import numpy as np
import pandas as pd
from scipy.stats import norm
import pywt

# Set the random seed for reproducibility
np.random.seed(0)

# Number of datasets and samples per dataset
num_datasets = 100
n = 2000

# # Define the directory to save the CSV files
# output_dir = 'datasets'

# # Create the directory if it does not exist
# import os
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# Transform Z into observed covariates X through nonlinear functions
def transform_Z(Z):
    X = np.zeros((len(Z), 10))
    X[:, 0] = np.exp(Z[:, 0] / 2)
    X[:, 1] = Z[:, 1] / (1 + np.exp(Z[:, 0]))
    X[:, 2] = (Z[:, 0] * Z[:, 2] / 25 + 0.6) ** 3
    X[:, 3] = (Z[:, 1] + Z[:, 3] + 20) ** 2
    X[:, 4:] = Z[:, 4:]
    return X

# Propensity score model (Model 1)
def propensity_score(Z):
    # return 1 / (1 + np.exp(-Z[:, 0] - 0.1 * Z[:, 3]))

    Z_tuta = (Z[:, 1] + Z[:, 3] + Z[:, 5] + Z[:, 7] + Z[:, 9]) / 5
    # 使用Daubechies 4-tap小波的缩放函数
    cA2 = pywt.wavedec(Z_tuta, 'db4', level=2)[-1]
    # 将cA2扩展到与Z相同的长度
    cA2_expanded = np.interp(np.arange(Z.shape[0]), np.arange(cA2.shape[0]), cA2)
    return 1 / (1 + np.exp(-Z[:, 0] - 0.1 * Z[:, 3] + cA2_expanded))

    # 简化的Weierstrass函数
    def weierstrass(Z):
        return Z[:, 2]**2 + Z[:, 4]**2 + Z[:, 6]**2
    return 1 / (1 + np.exp(-Z[:, 0] - 0.1 * Z[:, 3] + weierstrass(Z)))

# Outcome model
def outcome_model(Z, T):
    return 200 + 10 * T + (1.5 * T - 0.5) * (27.4 * Z[:, 0] + 13.7 * Z[:, 1] + 13.7 * Z[:, 2] + 13.7 * Z[:, 3]) + norm.rvs(size=len(T))

# Generate and save datasets
for i in range(num_datasets):
    # Generate a ten-dimensional random vector Z from the standard normal distribution
    p = 10
    Z = np.random.normal(size=(n, p))
    
    # Transform Z into observed covariates X
    X = transform_Z(Z)
    
    # Propensity score
    ps = propensity_score(Z)
    
    # Treatment assignment
    T = np.random.binomial(1, ps, size=n)
    
    # Simulate the outcome
    Y = outcome_model(Z, T)
    
    # Create a DataFrame
    data = pd.DataFrame(X, columns=[f'W{i+1}' for i in range(p)])
    data['T'] = T
    data['Y'] = Y
    
    data.to_csv(f"data_B3_2000/dataset_{i}.csv", index=False)
    
print(f"Saved dataset {i}")