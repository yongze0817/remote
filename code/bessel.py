import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_classification
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
    n_samples_X, n_features = X.shape
    n_samples_Y, _ = Y.shape
    K = np.zeros((n_samples_X, n_samples_Y))
    for i in range(n_samples_X):
        for j in range(n_samples_Y):
            K[i, j] = bessel_kernel(X[i], Y[j], sigma, v)
    return K

# 生成示例数据
X, y = make_classification(n_samples=100, n_features=4, n_informative=2, n_redundant=0, random_state=42)

# 定义目标函数
@use_named_args(dimensions=[Real(0.1, 2.0, name='sigma'), Real(0.1, 5.0, name='v')])
def objective(**params):
    clf = svm.SVC(kernel=lambda x, y: bessel_kernel_matrix(x, y, params['sigma'], params['v']))
    clf.fit(X, y)
    score = clf.score(X, y)
    return -score  # 贝叶斯优化是最大化目标函数，所以取负值

# 定义参数空间
space = [Real(0.1, 2.0, name='sigma'), Real(0.1, 5.0, name='v')]

# 使用 gp_minimize 进行优化
result = gp_minimize(objective, space, n_calls=20, random_state=42)

# 输出优化结果
print(f"Best parameters: sigma={result.x[0]}, v={result.x[1]}")
print(f"Best score: {-result.fun}")

# 使用最佳参数训练 SVM
best_sigma, best_v = result.x
clf = svm.SVC(kernel=lambda x, y: bessel_kernel_matrix(x, y, best_sigma, best_v))
clf.fit(X, y)

# 绘制决策边界
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# 由于数据是多维的，我们只取前两个特征进行可视化
Z = clf.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros((xx.size, 2))])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM with Bessel Kernel (Optimized Parameters)')
plt.show()

# 保存图像到指定文件夹
# output_folder = "output_images"
# os.makedirs(output_folder, exist_ok=True)  # 确保文件夹存在
# output_path = os.path.join(output_folder, "svm_bessel_kernel_optimized.png")
# plt.savefig(output_path)
# plt.close()

# print(f"图像已保存到 {output_path}")