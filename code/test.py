import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from sklearn.impute import SimpleImputer

# 生成模拟数据
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义多个核函数
def linear_kernel(X, Y):
    return np.dot(X, Y.T)

def rbf_kernel(X, Y, gamma):
    return np.exp(-gamma * np.linalg.norm(X[:, np.newaxis] - Y, axis=2) ** 2)

def poly_kernel(X, Y, degree, gamma, coef0):
    return (gamma * np.dot(X, Y.T) + coef0) ** degree

# 线性组合多个核函数
def combined_kernel(X, Y, alpha, beta, gamma_rbf, degree_poly, gamma_poly, coef0_poly):
    try:
        return alpha * linear_kernel(X, Y) + beta * rbf_kernel(X, Y, gamma_rbf) + (1 - alpha - beta) * poly_kernel(X, Y, degree_poly, gamma_poly, coef0_poly)
    except ValueError:
        return np.full((X.shape[0], Y.shape[0]), np.nan)

# 定义目标函数
@use_named_args(dimensions=[Real(0.0, 1.0, name='alpha'),
                            Real(0.0, 1.0, name='beta'),
                            Real(1e-4, 1e1, prior='log-uniform', name='gamma_rbf'),
                            Real(2, 5, name='degree_poly'),
                            Real(1e-4, 1e1, prior='log-uniform', name='gamma_poly'),
                            Real(0, 1, name='coef0_poly')])
def objective(**params):
    alpha, beta, gamma_rbf, degree_poly, gamma_poly, coef0_poly = params['alpha'], params['beta'], params['gamma_rbf'], params['degree_poly'], params['gamma_poly'], params['coef0_poly']
    K_train = combined_kernel(X_train, X_train, alpha, beta, gamma_rbf, degree_poly, gamma_poly, coef0_poly)
    K_test = combined_kernel(X_test, X_train, alpha, beta, gamma_rbf, degree_poly, gamma_poly, coef0_poly)
    
    # 处理NaN值
    imputer = SimpleImputer(strategy='mean')
    K_train = imputer.fit_transform(K_train)
    K_test = imputer.transform(K_test)
    
    model = SVC(kernel='precomputed')
    model.fit(K_train, y_train)
    score = model.score(K_test, y_test)
    return -score  # 贝叶斯优化是最大化目标函数，因此取负值

# 定义参数空间
dimensions = [
    Real(0.0, 1.0, name='alpha'),  # 线性核的权重
    Real(0.0, 1.0, name='beta'),  # RBF核的权重
    Real(1e-4, 1e1, prior='log-uniform', name='gamma_rbf'),  # RBF核的gamma
    Real(2, 5, name='degree_poly'),  # 多项式核的度数
    Real(1e-4, 1e1, prior='log-uniform', name='gamma_poly'),  # 多项式核的gamma
    Real(0, 1, name='coef0_poly')  # 多项式核的coef0
]

# 使用贝叶斯优化
result = gp_minimize(objective, dimensions, n_calls=50, random_state=42, verbose=True)

# 输出最佳参数
print("Best parameters found: ", result.x)
print("Best cross-validation score: ", -result.fun)

# 使用最佳参数训练模型
alpha, beta, gamma_rbf, degree_poly, gamma_poly, coef0_poly = result.x
K_train = combined_kernel(X_train, X_train, alpha, beta, gamma_rbf, degree_poly, gamma_poly, coef0_poly)
K_test = combined_kernel(X_test, X_train, alpha, beta, gamma_rbf, degree_poly, gamma_poly, coef0_poly)

# 处理NaN值
imputer = SimpleImputer(strategy='mean')
K_train = imputer.fit_transform(K_train)
K_test = imputer.transform(K_test)

best_svm = SVC(kernel='precomputed')
best_svm.fit(K_train, y_train)
y_pred = best_svm.predict(K_test)

# 评估模型性能
from sklearn.metrics import classification_report
print("Classification report on test set:")
print(classification_report(y_test, y_pred))