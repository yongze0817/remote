import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

# 定义搜索空间
param_bounds = {'C': (1e-6, 1e6), 'gamma': (1e-6, 1e6)}

# 高斯过程模型
def gp_model(X, y):
    kernel = C(1.0, (0.1, 1000000.0)) * Matern(length_scale=1.0, nu=2.5)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gpr.fit(X, y)
    return gpr

# 获取候选解
def get_candidates(bounds):
    C = np.logspace(np.log10(bounds['C'][0]), np.log10(bounds['C'][1]), 30)
    gamma = np.logspace(np.log10(bounds['gamma'][0]), np.log10(bounds['gamma'][1]), 30)
    candidates = np.array(np.meshgrid(C, gamma)).T.reshape(-1, 2)
    return candidates

# 评估函数
def evaluate(params, X, y):
    svm = SVC(C=params[0], gamma=params[1], probability=True)
    score = cross_val_score(svm, X, y, cv=5).mean()
    return score

# 蒙特卡洛PRB
def mc_prb(candidates, X, y, epsilon, delta_mod, delta_est):
    Z = np.array([evaluate(candidate, X, y) >= epsilon for candidate in candidates])
    return Z

# 获取下一个查询点
def get_next_query(model, bounds):
    # 使用采集函数获取下一个查询点，这里简化为随机选择
    next_query = np.array([np.random.uniform(bounds['C'][0], bounds['C'][1]),
                        np.random.uniform(bounds['gamma'][0], bounds['gamma'][1])])
    return next_query

# 贝叶斯优化
def bo_with_monte_carlo_prb(X, y, param_bounds, T, epsilon, delta_mod, delta_est):
    delta_est_schedule = np.linspace(delta_est, 0.01, T)
    data = np.array([[param_bounds['C'][0], param_bounds['gamma'][0]], [param_bounds['C'][1], param_bounds['gamma'][1]]])
    scores = np.array([0.5, 0.5])  # 初始化分数

    for t in range(T):
        model = gp_model(data[:, 0:1], data[:, 1])  # 修正参数传递
        candidates = get_candidates(param_bounds)
        Z = mc_prb(candidates, X, y, epsilon, delta_mod, delta_est_schedule[t])
        if np.any(Z >= 1 - delta_mod):
            break
        next_query = get_next_query(model, param_bounds)
        data = np.vstack((data, [next_query, evaluate(next_query, X, y)]))
    return data[np.argmax(scores)]

# 示例用法
if __name__ == "__main__":
    # 加载数据集
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)

    # 参数设置
    T = 10  # 最大迭代次数
    epsilon = 0.9  # 目标分数
    delta_mod = 0.1  # 模型误差界限
    delta_est = 0.1  # 估计误差界限

    # 运行BO with Monte Carlo PRB
    best_params = bo_with_monte_carlo_prb(X, y, param_bounds, T, epsilon, delta_mod, delta_est)
    print("找到的最优参数：", best_params)