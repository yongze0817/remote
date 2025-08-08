import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.multi_objective import qLogExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

def estimate_ess(Y, W, X, alpha, support_indices):
    weights = np.zeros(X.shape[0])
    weights[support_indices] = np.abs(alpha)
    weights = weights / np.sum(weights)
    treated_ess = (np.sum(weights[W == 1]))**2 / np.sum(weights[W == 1]**2)
    control_ess = (np.sum(weights[W == -1]))**2 / np.sum(weights[W == -1]**2)
    return treated_ess + control_ess

def estimate_ate(Y, W, X, alpha, support_indices):
    weights = np.zeros(X.shape[0])
    weights[support_indices] = np.abs(alpha)
    weights = weights / np.sum(weights)
    treated_mean = np.sum(Y[W == 1] * weights[W == 1]) / np.sum(weights[W == 1])
    control_mean = np.sum(Y[W == -1] * weights[W == -1]) / np.sum(weights[W == -1])
    return treated_mean - control_mean

def estimate_diff(Y, W, X, alpha, support_indices):
    weights = np.zeros(X.shape[0])
    weights[support_indices] = np.abs(alpha)
    weights = weights / np.sum(weights)
    treated_mean = np.mean(Y[W == 1])
    control_mean = np.mean(Y[W == -1])
    treated_std = np.std(Y[W == 1], ddof=1)
    control_std = np.std(Y[W == -1], ddof=1)
    return np.abs(treated_mean - control_mean) / np.sqrt((treated_std**2 + control_std**2) / 2)

def compute_wnayman(Y_obs, W, X, alpha, support_indices):
    weights = np.zeros(X.shape[0])
    weights[support_indices] = np.abs(alpha)
    lambda_i = weights / np.sum(weights)
    mu_c = np.mean(Y_obs[W == -1])
    mu_t = np.mean(Y_obs[W == 1])
    sigma_c_squared = np.var(Y_obs[W == -1])
    sigma_t_squared = np.var(Y_obs[W == 1])
    mu_i = np.where(W == -1, mu_c, mu_t)
    sigma_i_squared = np.where(W == -1, sigma_c_squared, sigma_t_squared)
    conditional_variance = np.sum(lambda_i[W == 1]**2 * sigma_i_squared[W == 1]) + \
                         np.sum(lambda_i[W == -1]**2 * sigma_i_squared[W == -1])
    return np.sqrt(conditional_variance)

def bayesian_opt_multi(Y, X, W, kernel, pi):
    # 转换为numpy数组确保类型正确
    Y_np = np.array(Y, dtype=np.float64)
    X_np = np.array(X, dtype=np.float64)
    W_np = np.array(W, dtype=np.int32)
    
    # 定义参数空间边界
    bounds = torch.tensor([
        [1e-3, 0.1, 0.1],   # C_min, class_weight_1_min, class_weight_neg1_min
        [1e+3, 2.0, 2.0]     # C_max, class_weight_1_max, class_weight_neg1_max
    ], dtype=torch.float64)
    
    # 定义评估函数 (原objective_multi)
    def evaluate(params):
        params_np = params.numpy()
        clf = SVC(
            C=float(params_np[0]),
            kernel=kernel,
            class_weight={1: float(params_np[1]), -1: float(params_np[2])},
            random_state=42
        )
        
        try:
            # 使用交叉验证
            scores = cross_val_score(clf, X_np, W_np, cv=5)
            score = np.mean(scores)
            
            # 获取支持向量信息
            clf.fit(X_np, W_np)
            alpha = np.abs(clf.dual_coef_[0]) if hasattr(clf, 'dual_coef_') else np.zeros(1)
            ess = estimate_ess(Y_np, W_np, X_np, alpha, clf.support_)
            
            return torch.tensor([-score, -1/ess], dtype=torch.float64)  # 负号因为要最小化
        except Exception as e:
            print(f"评估失败: {str(e)}")
            return torch.tensor([0.0, 0.0], dtype=torch.float64)
    
    # 初始采样
    train_X = draw_sobol_samples(bounds, n=5, q=1).squeeze(1)
    train_Y = torch.stack([evaluate(x) for x in train_X])
    
    # 优化循环
    for iteration in range(20):
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        
        ref_point = torch.tensor([-3.0, -3.0], dtype=torch.float64)
        partitioning = DominatedPartitioning(ref_point=ref_point, Y=train_Y)
        
        acq_func = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning
        )
        
        candidate, _ = optimize_acqf(
            acq_func,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,
        )
        
        new_Y = evaluate(candidate)
        train_X = torch.cat([train_X, candidate])
        train_Y = torch.cat([train_Y, new_Y.unsqueeze(0)])
    
    # 获取Pareto前沿
    pareto_mask = pareto.is_non_dominated(train_Y)
    pareto_front = train_Y[pareto_mask]
    pareto_params = train_X[pareto_mask]
    
    # 根据pi选择最佳折中点
    scores = -pareto_front[:, 0].numpy()  # 转换为正分数
    esses = -1/pareto_front[:, 1].numpy()  # 转换为正ESS
    combined_scores = (1-pi)*scores + pi*esses
    best_idx = np.argmax(combined_scores)
    
    best_params = pareto_params[best_idx]
    C = best_params[0].item()
    class_weight = {1: best_params[1].item(), -1: best_params[2].item()}
    
    # 训练最终模型
    final_model = SVC(C=C, kernel=kernel, class_weight=class_weight, random_state=42)
    final_model.fit(X_np, W_np)
    
    # 计算评估指标
    support_indices = final_model.support_
    alpha = np.abs(final_model.dual_coef_[0]) if hasattr(final_model, 'dual_coef_') else np.zeros(1)
    ATE = estimate_ate(Y_np, W_np, X_np, alpha, support_indices)
    DIM = estimate_diff(Y_np, W_np, X_np, alpha, support_indices)
    ESS = estimate_ess(Y_np, W_np, X_np, alpha, support_indices)
    STD = compute_wnayman(Y_np, W_np, X_np, alpha, support_indices)
    
    print("\n优化结果:")
    print(f"C: {C:.4f}, class_weight: {class_weight}")
    print(f"分类准确率: {scores[best_idx]:.4f}, ESS: {esses[best_idx]:.4f}")
    print(f"ATE: {ATE:.4f}, DIM: {DIM:.4f}, STD: {STD:.4f}")
    
    return ATE, DIM, ESS, C, class_weight, STD

# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    X = np.random.rand(100, 5)
    W = np.random.choice([-1, 1], size=100)
    Y = 0.5 * W + np.random.normal(size=100)
    
    # 运行优化
    results = bayesian_opt_multi(
        Y=Y,
        X=X,
        W=W,
        kernel='rbf',
        pi=0.5  # 平衡分类准确率和ESS的权重
    )