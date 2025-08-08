import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV
from skopt.space import Real

# ========== Step 1: 生成不平衡数据 ==========
X, y = make_classification(n_samples=1000, n_features=10, weights=[0.8, 0.2], random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)

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

# ========== Step 4: 定义优化目标函数 ==========
def fsvm_objective(C, gamma, sigma):
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    acc_scores = []
    sv_ratios = []

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 权重
        C_i = compute_class_weights(y_train)
        s_i = compute_fuzzy_membership(X_train, y_train, sigma)
        weights = C_i * s_i

        model = SVC(C=C, gamma=gamma, kernel='rbf')
        model.fit(X_train, y_train, sample_weight=weights)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        sv_ratio = len(model.support_) / len(X_train)

        acc_scores.append(acc)
        sv_ratios.append(sv_ratio)

    # 目标函数：高精度 + 惩罚支持向量占比
    acc_mean = np.mean(acc_scores)
    sv_penalty = np.mean(sv_ratios)
    return acc_mean - 0.2 * sv_penalty  # α = 0.2，可调

# ========== Step 5: 使用 skopt 进行贝叶斯搜索 ==========
from skopt.utils import use_named_args

# 搜索空间
space = [
    Real(1e-2, 1e2, prior='log-uniform', name='C'),
    Real(1e-3, 1e1, prior='log-uniform', name='gamma'),
    Real(0.01, 0.5, name='sigma')
]

@use_named_args(space)
def objective(**params):
    return -fsvm_objective(**params)  # skopt 是最小化目标 → 取负

from skopt import gp_minimize
result = gp_minimize(objective, space, n_calls=25, random_state=42, verbose=True)

# ========== Step 6: 输出最优参数并训练最终模型 ==========
best_C, best_gamma, best_sigma = result.x
print("\nBest Parameters:")
print(f"C = {best_C:.4f}, gamma = {best_gamma:.4f}, sigma = {best_sigma:.4f}")

# 最终训练模型
C_i = compute_class_weights(y)
s_i = compute_fuzzy_membership(X, y, sigma=best_sigma)
weights = C_i * s_i

final_model = SVC(C=best_C, gamma=best_gamma, kernel='rbf')
final_model.fit(X, y, sample_weight=weights)

print("Final Support Vector Ratio:", len(final_model.support_) / len(X))
