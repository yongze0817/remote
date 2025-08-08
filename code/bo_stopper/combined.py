from skopt import gp_minimize
from skopt.callbacks import DeltaXStopper, DeltaYStopper
import numpy as np
from scipy import stats

# 示例目标函数
def objective(params):
    x, y = params
    return (x-1)**2 + (y+2)**2 + np.random.normal(0, 0.1)

# 搜索空间
space = [(-5.0, 5.0), (-5.0, 5.0)]

class ConvergenceStopper:
    def __init__(self, patience=5, tol=1e-4):
        self.patience = patience
        self.tol = tol
        self.best_value = np.inf
        self.no_improvement = 0
    
    def __call__(self, result):
        current_min = np.min(result.func_vals)
        
        if self.best_value - current_min > self.tol:
            self.best_value = current_min
            self.no_improvement = 0
        else:
            self.no_improvement += 1
            
        return self.no_improvement >= self.patience

class ConfidenceIntervalStopper:
    def __init__(self, min_width=0.1, patience=3):
        self.min_width = min_width
        self.patience = patience
        self.satisfied_count = 0
    
    def __call__(self, result):
        if len(result.x_iters) < 10:  # 至少10个点才开始检查
            return False
            
        # 获取当前最优点
        best_idx = np.argmin(result.func_vals)
        best_x = result.x_iters[best_idx]
        
        # 使用GPR预测(需要访问内部模型)
        if hasattr(result, 'models') and result.models:
            model = result.models[-1]
            mean, std = model.predict(np.array(best_x).reshape(1, -1), 
                                    return_std=True)
            ci_width = 1.96 * 2 * std  # 95% CI总宽度
            
            if ci_width < self.min_width:
                self.satisfied_count += 1
            else:
                self.satisfied_count = 0
                
            return self.satisfied_count >= self.patience
        return False

class CombinedStopper:
    def __init__(self, conv_patience=5, conv_tol=1e-4,
                 ci_width=0.2, ci_patience=3, max_iter=100):
        self.conv_stopper = ConvergenceStopper(conv_patience, conv_tol)
        self.ci_stopper = ConfidenceIntervalStopper(ci_width, ci_patience)
        self.max_iter = max_iter
    
    def __call__(self, result):
        # 最大迭代次数检查
        if len(result.x_iters) >= self.max_iter:
            return True
            
        # 收敛检查
        if self.conv_stopper(result):
            return True
            
        # 置信区间检查
        if self.ci_stopper(result):
            return True
            
        return False

# 定义综合停止器
stopper = CombinedStopper(
    conv_patience=10,
    conv_tol=1e-4,
    ci_width=0.2,
    ci_patience=10,
    max_iter=50
)

# 运行优化
result = gp_minimize(
    objective,
    space,
    n_calls=50,  # 最大调用次数(会被stopper提前终止)
    n_random_starts=10,  # 初始随机点
    callback=[stopper],  # 使用我们的停止器
    random_state=42,
    verbose=True
)

print(f"最佳参数: {result.x}")
print(f"最佳值: {result.fun}")
print(f"总评估次数: {len(result.func_vals)}")