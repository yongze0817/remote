import numpy as np
from scipy import stats
from bayes_opt import BayesianOptimization
from bayes_opt.util import NotUniqueError

class RobustConvergenceStopper:
    def __init__(self, patience=5, tol=1e-4, min_evals=3):
        """
        patience: 连续多少次无显著改进则停止
        tol:  认为显著改进的最小变化量
        min_evals: 最少需要多少次评估才开始检查收敛
        """
        self.patience = patience
        self.tol = tol
        self.min_evals = min_evals
        self.best_value = -np.inf
        self.no_improvement_count = 0
    
    def __call__(self, optimizer):
        try:
            # 检查是否有足够的评估数据
            if len(optimizer._space.target) < self.min_evals:
                return False
                
            current_value = np.max(optimizer._space.target)
            
            if current_value - self.best_value > self.tol:
                self.best_value = current_value
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
                
            return self.no_improvement_count >= self.patience
        except (ValueError, AttributeError):
            return False

class RobustBayesianOptimization(BayesianOptimization):
    def maximize(self, init_points=5, n_iter=25, **kwargs):
        """重写maximize方法确保初始点存在"""
        if len(self._space.target) == 0 and init_points == 0:
            init_points = 5  # 确保至少有5个初始点
        super().maximize(init_points=init_points, n_iter=n_iter, **kwargs)

# 使用示例
def black_box_function(x, y):
    return -x**2 - (y-1)**2 + 1 + np.random.normal(0, 0.1)

# 使用我们增强版的优化器
optimizer = RobustBayesianOptimization(
    f=black_box_function,
    pbounds={'x': (-2, 2), 'y': (-3, 3)},
    random_state=1,
)

# 创建更健壮的停止器
stopper = RobustConvergenceStopper(
    patience=10,
    tol=1e-4,
    min_evals=5  # 至少5次评估后才开始检查收敛
)

# 安全的优化流程
try:
    # 初始评估
    optimizer.maximize(init_points=5, n_iter=0)
    
    # 迭代优化
    iteration = 0
    max_iterations = 50
    
    while iteration < max_iterations and not stopper(optimizer):
        optimizer.maximize(init_points=0, n_iter=1)
        iteration += 1
        print(f"Iter {iteration}: Current max {optimizer.max['target']:.3f}")
    
    print("优化完成，最终结果:")
    print(optimizer.max)

except Exception as e:
    print(f"优化过程中出错: {str(e)}")
    print("当前最佳结果:")
    if len(optimizer._space.target) > 0:
        print(optimizer.max)
    else:
        print("无有效评估数据")