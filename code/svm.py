import numpy as np
import pandas as pd 
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from skopt import BayesSearchCV


# 生成模拟数据
simulate_data_A = pd.read_csv('data_A/data_scenario_G_n_500_dataset_0.csv')
X = simulate_data_A.drop(columns=['T', 'Y'])
T = simulate_data_A['T']

# 将处理变量转换为 -1 和 1
W = 2 * T - 1



##SVM DUAL
def compute_svm_weights(X, W, kernel='linear', C=1.0):
    """
    计算 SVM 平衡权重
    :param X: 协变量矩阵 (N, p)
    :param T: 处理变量 (N,)
    :param kernel: 核函数类型 ('linear', 'poly', 'rbf')
    :param C: 正则化参数
    :return: 平衡权重 (N,)
    """
    
    # 创建 SVM 模型
    svm = SVC(kernel=kernel, C=C, probability=True)
    svm.fit(X, W)
    
    # 获取对偶系数
    alpha = svm.dual_coef_[0]
    
    # 计算权重
    weights = np.abs(alpha) / np.sum(np.abs(alpha))
    
    return weights



# ##PATH ALGORITHM
# def svm_path(X, W, C_range):
#     """
#     Compute the SVM regularization path.
#     :param X: Feature matrix (N, p)
#     :param y: Target vector (N,)
#     :param C_range: List of regularization parameters
#     :return: List of SVM models and corresponding coefficients
#     """
#     models = []
#     coefficients = []
    
#     for C in C_range:
#         svm = SVC(kernel='linear', C=C, probability=True)
#         svm.fit(X, W)
#         models.append(svm)
#         coefficients.append(svm.coef_.flatten())
    
#     return models, coefficients

# # Define a range of regularization parameters
# C_range = np.logspace(-3, 3, 10)

# # Compute the regularization path
# models, coefficients = svm_path(X, W, C_range)

# # Plot the regularization path
# plt.figure(figsize=(10, 6))
# for i in range(X.shape[1]):
#     plt.plot(C_range, [coef[i] for coef in coefficients], label=f'Feature {i+1}')

# plt.xscale('log')
# plt.xlabel('Regularization parameter (C)')
# plt.ylabel('Coefficient value')
# plt.title('SVM Regularization Path')
# plt.legend()
# plt.grid(True)
# plt.show()


param_space = {'C': (1e-3, 1e3, 'log-uniform')}
bayes_search = BayesSearchCV(SVC(kernel='linear'), param_space, n_iter=100, cv=5, scoring='accuracy')
bayes_search.fit(X, T)

print("Best parameters found: ", bayes_search.best_params_)





# from sklearn.metrics import accuracy_score

# # Evaluate the models
# accuracies = []
# for model in models:
#     y_pred = model.predict(X_test)
#     accuracies.append(accuracy_score(y_test, y_pred))

# # Plot the accuracy vs regularization parameter
# plt.figure(figsize=(10, 6))
# plt.plot(C_range, accuracies, marker='o')
# plt.xscale('log')
# plt.xlabel('Regularization parameter (C)')
# plt.ylabel('Accuracy')
# plt.title('Model Accuracy vs Regularization Parameter')
# plt.grid(True)
# plt.show()



# 计算权重
weights = compute_svm_weights(X, T, kernel='rbf', C=1.0)
print("平衡权重:", weights)



def compute_balance_metrics(X, T, weights):
    """
    Compute balance metrics (e.g., standardized difference-in-means)
    :param X: Feature matrix (N, p)
    :param T: Treatment vector (N,)
    :param weights: Weights for each sample (N,)
    :return: Balance metrics
    """
    treated_mean = np.average(X[T == 1], weights=weights[T == 1], axis=0)
    control_mean = np.average(X[T == 0], weights=weights[T == 0], axis=0)
    diff_in_means = treated_mean - control_mean
    std_diff = diff_in_means / np.std(X, axis=0)
    return std_diff

# Compute balance metrics for each regularization parameter
balance_metrics = []
for model in models:
    alpha = np.abs(model.dual_coef_[0])
    weights = alpha / np.sum(alpha)
    balance_metrics.append(compute_balance_metrics(X, T, weights))

# Plot balance metrics
plt.figure(figsize=(10, 6))
for i in range(X.shape[1]):
    plt.plot(C_range, [metric[i] for metric in balance_metrics], label=f'Feature {i+1}')

plt.xscale('log')
plt.xlabel('Regularization parameter (C)')
plt.ylabel('Standardized Difference-in-Means')
plt.title('Balance Metrics vs Regularization Parameter')
plt.legend()
plt.grid(True)
plt.show()



##ATE ESTIMATION
def estimate_ate(Y, T, weights):
    """
    使用加权差异估计器估计 ATE
    :param Y: 结果变量 (N,)
    :param T: 处理变量 (N,)
    :param weights: 平衡权重 (N,)
    :return: ATE 估计值
    """
    treated_mean = np.sum(Y[T == 1] * weights[T == 1]) / np.sum(weights[T == 1])
    control_mean = np.sum(Y[T == 0] * weights[T == 0]) / np.sum(weights[T == 0])
    return treated_mean - control_mean

# 示例结果变量
Y = np.random.randn(100)  # 随机结果变量

# 估计 ATE
ate_estimate = estimate_ate(Y, T, weights)
print("ATE 估计值:", ate_estimate)

