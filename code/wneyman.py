import numpy as np
import pandas as pd

# 假设数据集已经加载，并且有以下变量：
simulate_data_A = pd.read_csv('./data_A/data_scenario_G_n_500_dataset_0.csv')
X = simulate_data_A.drop(columns=['T', 'Y'])
W = simulate_data_A['T']
Y_obs = simulate_data_A['Y']

# Y_obs: 观测到的结果
# W: 处理指示变量（0表示控制组，1表示处理组）
# X: 协变量
# lambda_i: 权重，对于特定的估计器，这是协变量和分配向量的已知函数

# 计算mu_c(x)和mu_t(x)，这里需要根据具体数据进行定义
def calculate_mu(X, Y_obs, W, group):
    if group == 0:
        return np.mean(Y_obs[W == 0])
    else:
        return np.mean(Y_obs[W == 1])

mu_c = calculate_mu(X, Y_obs, W, 0)
mu_t = calculate_mu(X, Y_obs, W, 1)

# 计算sigma_c^2(x)和sigma_t^2(x)，这里需要根据具体数据进行定义
def calculate_sigma(X, Y_obs, W, group):
    if group == 0:
        return np.var(Y_obs[W == 0])
    else:
        return np.var(Y_obs[W == 1])

sigma_c_squared = calculate_sigma(X, Y_obs, W, 0)
sigma_t_squared = calculate_sigma(X, Y_obs, W, 1)

# 计算mu_i和sigma_i^2
def calculate_mu_sigma(Y_obs, W, mu_c, mu_t, sigma_c_squared, sigma_t_squared):
    mu_i = np.where(W == 0, mu_c, mu_t)
    sigma_i_squared = np.where(W == 0, sigma_c_squared, sigma_t_squared)
    return mu_i, sigma_i_squared

mu_i, sigma_i_squared = calculate_mu_sigma(Y_obs, W, mu_c, mu_t, sigma_c_squared, sigma_t_squared)

# 计算条件抽样方差
def calculate_conditional_variance(lambda_i, W, sigma_i_squared, N_t, N_c):
    variance = (np.sum(lambda_i[W == 1]**2 * sigma_i_squared[W == 1]) / N_t**2 +
               np.sum(lambda_i[W == 0]**2 * sigma_i_squared[W == 0]) / N_c**2)
    return variance

# 假设N_t和N_c是已知的
N_t = np.sum(W == 1)
N_c = np.sum(W == 0)

# 计算lambda_i，这里需要根据具体估计器来定义
lambda_i = np.random.rand(len(Y_obs))  # 这里只是示例，实际需要根据估计器计算

conditional_variance = calculate_conditional_variance(lambda_i, W, sigma_i_squared, N_t, N_c)
print("条件抽样方差:", conditional_variance)