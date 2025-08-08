import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.special import jv
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import os

# 定义贝塞尔核函数
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

# 定义高斯核函数
def gaussian_kernel_matrix(X, Y, sigma=1.0):
    return np.exp(-np.linalg.norm(X[:, np.newaxis] - Y, axis=2) ** 2 / (2 * sigma ** 2))

# 生成示例数据
num_datasets = 1
import pandas as pd
n=1
simulate_data = pd.read_csv('./data_B2_2000/dataset_{}.csv'.format(n))

X = simulate_data.drop(columns=['T', 'Y'])
T = simulate_data['T']
Y = simulate_data['Y']
y = 2 * T - 1
# X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义贝塞尔核的目标函数
@use_named_args(dimensions=[Real(0.1, 2.0, name='sigma'), Real(0.1, 5.0, name='v'), Real(0.1, 10.0, name='C')])
def bessel_objective(**params):
    clf = svm.SVC(kernel=lambda x, y: bessel_kernel_matrix(x, y, params['sigma'], params['v']), C=params['C'])
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return -score  # 贝叶斯优化是最大化目标函数，所以取负值

# 定义高斯核的目标函数
@use_named_args(dimensions=[Real(0.1, 2.0, name='sigma'), Real(0.1, 10.0, name='C')])
def gaussian_objective(**params):
    clf = svm.SVC(kernel='rbf', gamma=1/(2*params['sigma']**2), C=params['C'])
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return -score  # 贝叶斯优化是最大化目标函数，所以取负值

# 定义参数空间
bessel_space = [Real(0.1, 2.0, name='sigma'), Real(0.1, 5.0, name='v'), Real(0.1, 10.0, name='C')]
gaussian_space = [Real(0.1, 2.0, name='sigma'), Real(0.1, 10.0, name='C')]

# 使用 gp_minimize 进行优化
bessel_result = gp_minimize(bessel_objective, bessel_space, n_calls=20, random_state=42)
gaussian_result = gp_minimize(gaussian_objective, gaussian_space, n_calls=20, random_state=42)

# 输出优化结果
print(f"贝塞尔核最佳参数: sigma={bessel_result.x[0]}, v={bessel_result.x[1]}, C={bessel_result.x[2]}")
print(f"贝塞尔核最佳准确率: {-bessel_result.fun:.4f}")
print(f"高斯核最佳参数: sigma={gaussian_result.x[0]}, C={gaussian_result.x[1]}")
print(f"高斯核最佳准确率: {-gaussian_result.fun:.4f}")

# 使用最佳参数训练 SVM
best_sigma_bessel, best_v_bessel, best_C_bessel = bessel_result.x
best_sigma_gaussian, best_C_gaussian = gaussian_result.x

clf_bessel = svm.SVC(kernel=lambda x, y: bessel_kernel_matrix(x, y, best_sigma_bessel, best_v_bessel), C=best_C_bessel)
clf_bessel.fit(X_train, y_train)
y_pred_bessel = clf_bessel.predict(X_test)
accuracy_bessel = accuracy_score(y_test, y_pred_bessel)

clf_gaussian = svm.SVC(kernel='rbf', gamma=1/(2*best_sigma_gaussian**2), C=best_C_gaussian)
clf_gaussian.fit(X_train, y_train)
y_pred_gaussian = clf_gaussian.predict(X_test)
accuracy_gaussian = accuracy_score(y_test, y_pred_gaussian)

print(f"贝塞尔核 SVM 的准确率: {accuracy_bessel:.4f}")
print(f"高斯核 SVM 的准确率: {accuracy_gaussian:.4f}")

# 绘制决策边界
def plot_decision_boundary(clf, X, y, title, output_path):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

# 保存贝塞尔核 SVM 的决策边界
output_folder = "output_images"
output_path_bessel = os.path.join(output_folder, "svm_bessel_kernel_optimized.png")
plot_decision_boundary(clf_bessel, X, y, "SVM with Bessel Kernel (Optimized Parameters)", output_path_bessel)

# 保存高斯核 SVM 的决策边界
output_path_gaussian = os.path.join(output_folder, "svm_gaussian_kernel_optimized.png")
plot_decision_boundary(clf_gaussian, X, y, "SVM with Gaussian Kernel (Optimized Parameters)", output_path_gaussian)

print(f"贝塞尔核 SVM 的决策边界已保存到 {output_path_bessel}")
print(f"高斯核 SVM 的决策边界已保存到 {output_path_gaussian}")