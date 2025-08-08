import numpy as np
import pandas as pd
import multiprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skopt import gp_minimize
from skopt.learning import RandomForestRegressor
from skopt.space import Real, Categorical, Space
from skopt.utils import use_named_args
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.neighbors import NearestNeighbors
from scipy.special import jv

import json
class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，用于处理numpy数据类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)
        

def estimate_ess(Y, W, X, alpha, support_indices):
        weights = np.zeros(X.shape[0])
        weights[support_indices] = np.abs(alpha)
        alpha = weights / np.sum(weights)
        
        treated_ess = (np.sum(alpha[W == 1]))**2 / np.sum(alpha[W == 1]**2)
        control_ess = (np.sum(alpha[W == -1]))**2 / np.sum(alpha[W == -1]**2)

        return treated_ess + control_ess
    
def estimate_diff(Y, W, X, alpha, support_indices):

    # 计算权重
    weights = np.zeros(X.shape[0])
    weights[support_indices] = np.abs(alpha)
    alpha = weights / np.sum(weights)
    
    treated_mean = np.mean(Y[W == 1])
    control_mean = np.mean(Y[W == -1])
    
    treated_std = np.std(Y[W == 1], ddof=1)
    control_std = np.std(Y[W == -1], ddof=1)

    normed_diff = np.abs(treated_mean - control_mean) / np.sqrt((treated_std**2 + control_std**2) / 2)

    return normed_diff

def estimate_ate(Y, W, X, alpha, support_indices):
    
    # 计算权重
    weights = np.zeros(X.shape[0])
    weights[support_indices] = np.abs(alpha)
    alpha = weights / np.sum(weights)
    
    treated_mean = np.sum(Y[W == 1] * alpha[W == 1]) / np.sum(alpha[W == 1])
    control_mean = np.sum(Y[W == -1] * alpha[W == -1]) / np.sum(alpha[W == -1])
    return treated_mean - control_mean



def compute_wnayman(Y_obs, W, X, alpha, support_indices):
    
    # 计算权重
    weights = np.zeros(X.shape[0])
    weights[support_indices] = np.abs(alpha)
    lambda_i = weights / np.sum(weights)
    
    # 计算mu_c(x)和mu_t(x)，这里需要根据具体数据进行定义
    mu_c = np.mean(Y_obs[W == -1])
    mu_t = np.mean(Y_obs[W == 1])

    # 计算sigma_c^2(x)和sigma_t^2(x)，这里需要根据具体数据进行定义
    sigma_c_squared = np.var(Y_obs[W == -1])
    sigma_t_squared = np.var(Y_obs[W == 1])

    # 计算mu_i和sigma_i^2
    mu_i = np.where(W == -1, mu_c, mu_t)
    sigma_i_squared = np.where(W == -1, sigma_c_squared, sigma_t_squared)

    N_t = np.sum(W == 1)
    N_c = np.sum(W == -1)

    # 计算条件抽样方差
    conditional_variance = np.sum(lambda_i[W == 1]**2 * sigma_i_squared[W == 1]) + np.sum(lambda_i[W == -1]**2 * sigma_i_squared[W == -1])
    
    stderr = np.sqrt(conditional_variance)


    print("条件抽样方差:", conditional_variance)
    return stderr

class ConvergenceStopper:
    def __init__(self, patience=10, tol=1e-6):
        self.patience = patience
        self.tol = tol
        self.best_value = np.inf
        self.no_improvement = 0
    
    def __call__(self, result):
        if len(result.x_iters) < 10:  # 至少10个点才开始检查
            return False
        
        current_min = np.min(result.func_vals)
        
        if self.best_value - current_min > self.tol:
            self.best_value = current_min
            self.no_improvement = 0
        else:
            self.no_improvement += 1
            
        return self.no_improvement >= self.patience

class BOStopper:
    def __init__(self, conv_patience=10, conv_tol=1e-6,
                 ci_width=0.2, ci_patience=10, max_iter=100):
        self.conv_stopper = ConvergenceStopper(conv_patience, conv_tol)
        # self.ci_stopper = ConfidenceIntervalStopper(ci_width, ci_patience)
        self.max_iter = max_iter
    
    def __call__(self, result):
        # 最大迭代次数检查
        if len(result.x_iters) >= self.max_iter:
            return True
            
        # 收敛检查
        if self.conv_stopper(result):
            return True
        
        # 置信区间检查
        # if self.ci_stopper(result):
        #     return True
            
        return False

def bessel_kernel(x, y):
    gamma=1
    nu=0
    return jv(nu, gamma * np.linalg.norm(x[:, np.newaxis] - y, axis=2))


# ========== Step 2: 类权重 C_i ==========
def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    class_weights = {cls: total / (len(classes) * cnt) for cls, cnt in zip(classes, counts)}
    return np.array([class_weights[yi] for yi in y])


# ========== Step 3: 模糊权重 s_i ==========
def compute_fuzzy_membership(X, y, sigma=0.1):
    s = np.zeros(len(X))
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        X_class = X[idx]
        center = X_class.mean(axis=0)
        distances = np.linalg.norm(X_class - center, axis=1)
        t1, tl = distances.min(), distances.max()
        normalized = (distances - t1) / (tl - t1 + 1e-8)
        s[idx] = (1 - sigma) * (normalized ** 2) + sigma
    return s


def bayesian_opt(Y, X, W, kernel, c_upper): 
    # Define the hyperparameter space
    dim_C = Real(low=1e-4, high=c_upper, prior='log-uniform', name='C')
    dim_class_weight_1 = Real(low=0.1, high=2.0, prior='uniform', name='class_weight_1')
    dim_class_weight_neg1 = Real(low=0.1, high=2.0, prior='uniform', name='class_weight_neg1')
    # gamma = Real(low=1e-4, high=1e1, prior='log-uniform', name='gamma')

    dimensions = [dim_C]
    # Define the target score function
    @use_named_args(dimensions=dimensions)
    def objective_1(**params):
        #class_weight = {1: params['class_weight_1'], -1: params['class_weight_neg1']}
        clf = SVC(C=params['C'], kernel=kernel)

        C_i = compute_class_weights(W_train)
        sigma = 0.1  # You can adjust this value as needed
        s_i = compute_fuzzy_membership(X_train, W_train, sigma)
        weights = C_i * s_i
        
        # return score  # We negate because we want to maximize accuracy
    
        clf.fit(X_train, W_train, sample_weight=weights)
        score = clf.score(X_test, W_test)
        # ess = estimate_ess(Y, W, X, alpha, support_indices)
        return -score
    
    @use_named_args(dimensions=dimensions)
    def objective_2(**params):
        #class_weight = {1: params['class_weight_1'], -1: params['class_weight_neg1']}
        clf = SVC(C=params['C'], kernel=kernel)

        C_i = compute_class_weights(W_test)
        sigma = 0.1  # You can adjust this value as needed
        s_i = compute_fuzzy_membership(X_test, W_test, sigma)
        weights = C_i * s_i
        
        # return score  # We negate because we want to maximize accuracy
    
        clf.fit(X_test, W_test, sample_weight=weights)
        score = clf.score(X_train, W_train)
        # ess = estimate_ess(Y, W, X, alpha, support_indices)
        return -score


    #print(result)
    # Extract optimized hyperparameters
    def crossfitting(result, Y, X, W):
        best_params = result.x
        best_score = result.fun  # Negate to get the actual score

        C_i = compute_class_weights(W)
        sigma = 0.1  # You can adjust this value as needed
        s_i = compute_fuzzy_membership(X, W, sigma)
        weights = C_i * s_i

        C = best_params[0]
        # gamma = best_params[1]
        #class_weight = {1: best_params[1], -1: best_params[2]}

        best_model = SVC(C=C, kernel=kernel)
        best_model.fit(X, W, sample_weight=weights)
        support_indices = best_model.support_
        alpha = np.abs(best_model.dual_coef_[0])
        ATE = estimate_ate(Y, W, X, alpha, support_indices)
        DIM = estimate_diff(Y, W, X, alpha, support_indices)
        ESS = estimate_ess(Y, W, X, alpha, support_indices)
        STD = compute_wnayman(Y, W, X, alpha, support_indices)
        
        print("Best Hyperparameters:", best_params)
        print("Best Cross-Validation Score:", best_score)
        print("ATE:", ATE)
        print("STD:", STD)

        return(ATE, DIM, ESS, C, STD)
    
    # Initialize the Bayesian Optimization
    n_initial_points = 5
    n_calls = 50

    # 定义综合停止器
    stopper_1 = BOStopper(
        conv_patience=10,
        conv_tol=1e-6,
        # ci_width=0.2,
        # ci_patience=10,
        max_iter=100
    )
    stopper_2 = BOStopper(
        conv_patience=10,
        conv_tol=1e-6,
        # ci_width=0.2,
        # ci_patience=10,
        max_iter=100
    )

    # rf = RandomForestRegressor(n_estimators=100)

    Y_train, Y_test, X_train, X_test, W_train, W_test = train_test_split(Y, X, W, test_size=0.5, random_state=42)

    result_1 = gp_minimize(objective_1, dimensions, n_initial_points=n_initial_points, callback=[stopper_1], random_state=42, verbose=True)
    result_1 = crossfitting(result_1, Y_test, X_test, W_test)

    result_2 = gp_minimize(objective_2, dimensions, n_initial_points=n_initial_points, callback=[stopper_2], random_state=42, verbose=True)
    result_2 = crossfitting(result_2, Y_train, X_train, W_train)

    ATE = (result_1[0] + result_2[0])/2
    DIM = (result_1[1] + result_2[1])/2
    ESS = result_1[2] + result_2[2]
    C = (result_1[3] + result_2[3])/2
    STD = (result_1[4] + result_2[4])/2

    return(ATE, DIM, ESS, C, STD)



# Define the dataset and hyperparameter space
num_datasets = 1
simulation_results_A = {n: [] for n in range(num_datasets)}


kernels = ['rbf']
# pi_range = np.linspace(0.0, 1.0, 11)

# pi_range = [0.0]
# results = {pi: [] for pi in pi_range}

c_uppers = [1e+1, 1e+2, 1e+3]

results = {c_upper: [] for c_upper in c_uppers}


for n in range(num_datasets):
    # simulate_data = pd.read_csv('../data_A_2000/data_scenario_G_n_2000_dataset_{}.csv'.format(n))
    simulate_data = pd.read_csv('../data_B/dataset_{}.csv'.format(n))

    X = simulate_data.drop(columns=['T', 'Y'])
    T = simulate_data['T']
    Y = simulate_data['Y']

    W = 2 * T - 1

    for kernel in kernels:
        for c_upper in c_uppers:
            result = bayesian_opt(Y, X, W, kernel, c_upper) 

            results[c_upper].append(result)


        data = results

        data_str = json.dumps(data, cls=NumpyEncoder, indent=4)


        # with open(f'../results/{kernel}/simu_B2_100_2000_f.json', 'w') as f:
        #     f.write(data_str)
