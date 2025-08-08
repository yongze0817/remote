import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import optuna

# 加载鸢尾花数据集
num_datasets = 1
import pandas as pd
n=1
simulate_data = pd.read_csv('./data_B2_2000/dataset_{}.csv'.format(n))

X = simulate_data.drop(columns=['T', 'Y'])
T = simulate_data['T']
Y = simulate_data['Y']
y = 2 * T - 1

# 数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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

from scipy.special import jv
def bessel_kernel(x, y, sigma=1.0, v=1.0):
    norm = np.linalg.norm(x - y)
    if norm == 0:
        return 1.0
    return jv(v + 1, sigma * norm) / (norm ** (-v - 1))

# 自定义核函数的矩阵计算
def bessel_kernel_matrix(X, Y, sigma=1.0, v=1.0):
    X = np.atleast_2d(X)  # 确保 X 是二维数组
    Y = np.atleast_2d(Y)  # 确保 Y 是二维数组
    n_samples_X, n_features = X.shape
    n_samples_Y, _ = Y.shape
    K = np.zeros((n_samples_X, n_samples_Y))
    for i in range(n_samples_X):
        for j in range(n_samples_Y):
            K[i, j] = bessel_kernel(X[i], Y[j], sigma, v)
    return K

# 定义目标函数
def objective(trial):
    # 定义超参数搜索空间
    C = trial.suggest_loguniform('C', 1e-3, 10)
    kernel = trial.suggest_categorical('kernel', ['bessel'])
    gamma = trial.suggest_loguniform('gamma', 1e-4, 1)
    degree = trial.suggest_int('degree', 1, 5)
    sigma = trial.suggest_loguniform('sigma', 1e-4, 1)
    v = trial.suggest_int('v', 1, 5)

    # 创建 SVM 模型
    if kernel == 'bessel':
        model = SVC(
            C=C,
            kernel=lambda X, Y: bessel_kernel_matrix(X, Y, sigma=sigma, v=v),
            random_state=42
        )
    else:
        model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            random_state=42
        )

    # 使用交叉验证评估模型性能
    score = cross_val_score(model, X_train, y_train, n_jobs=1, cv=3, scoring='accuracy')
    return np.mean(score)

# 创建 Optuna 研究对象
study = optuna.create_study(direction='maximize')

# 运行优化
study.optimize(objective, n_trials=2)

# 输出最佳超参数
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)



C = study.best_trial.params['C']
gamma = study.best_trial.params['gamma']
kernel = study.best_trial.params['kernel']
sigma = study.best_trial.params['sigma']
v = study.best_trial.params['v']

if kernel == 'bessel':
    kernel=lambda X, Y: bessel_kernel_matrix(X, Y, sigma=sigma, v=v)
else:
    kernel = kernel

degree = study.best_trial.params['degree']
#class_weight = {1: best_params[1], -1: best_params[2]}

best_model = SVC(C=C, kernel=kernel, gamma=gamma)
best_model.fit(X, y)
support_indices = best_model.support_
alpha = np.abs(best_model.dual_coef_[0])
ATE = estimate_ate(Y, y, X, alpha, support_indices)
DIM = estimate_diff(Y, y, X, alpha, support_indices)
ESS = estimate_ess(Y, y, X, alpha, support_indices)
STD = compute_wnayman(Y, y, X, alpha, support_indices)

print(ATE, DIM, ESS, C, STD)