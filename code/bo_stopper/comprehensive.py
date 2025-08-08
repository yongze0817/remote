import numpy as np
from scipy import stats
from bayes_opt import BayesianOptimization

class ConfidenceIntervalStopper:
    def __init__(self, min_interval_width=0.1, patience=3):
        self.min_width = min_interval_width
        self.patience = patience
        self.satisfied_count = 0
    
    def __call__(self, optimizer):
        # 检查是否有评估数据
        if len(optimizer._space.target) == 0:
            return False
            
        best_idx = np.argmax(optimizer._space.target)
        best_x = optimizer._space.params[best_idx]
        
        try:
            # 适配不同版本的BayesianOptimization
            if hasattr(optimizer, '_gp'):
                mean, std = optimizer._gp.predict(best_x.reshape(1, -1), return_std=True)
            elif hasattr(optimizer, 'model'):
                mean, std = optimizer.model.predict(best_x.reshape(1, -1), return_std=True)
            else:
                return False
                
            interval_width = 3.92 * std  # 95%置信区间宽度(1.96*2)
            
            if interval_width < self.min_width:
                self.satisfied_count += 1
            else:
                self.satisfied_count = 0
                
            return self.satisfied_count >= self.patience
        except:
            return False

class ImprovementProbabilityStopper:
    def __init__(self, min_improvement_prob=0.05, patience=5, n_samples=100):
        self.min_prob = min_improvement_prob
        self.patience = patience
        self.n_samples = n_samples
        self.low_prob_count = 0
    
    def __call__(self, optimizer):
        if len(optimizer._space.target) == 0:
            return False
            
        current_max = optimizer._space.target.max()
        X_samples = optimizer._space.random_sample()
        
        improvement_probs = []
        try:
            for x in X_samples:
                if hasattr(optimizer, '_gp'):
                    mean, std = optimizer._gp.predict(x.reshape(1, -1), return_std=True)
                elif hasattr(optimizer, 'model'):
                    mean, std = optimizer.model.predict(x.reshape(1, -1), return_std=True)
                else:
                    return False
                    
                if std < 1e-8:  # 避免除以0
                    prob = 0
                else:
                    z = (mean - current_max) / std
                    prob = 1 - stats.norm.cdf(z)
                improvement_probs.append(prob)
            
            max_improvement_prob = np.max(improvement_probs)
            
            if max_improvement_prob < self.min_prob:
                self.low_prob_count += 1
            else:
                self.low_prob_count = 0
                
            return self.low_prob_count >= self.patience
        except:
            return False

class ComprehensiveCIStopper:
    def __init__(self, interval_width=0.1, imp_prob=0.1, 
                 max_iter=100, patience=5):
        self.interval_stopper = ConfidenceIntervalStopper(interval_width, patience)
        self.prob_stopper = ImprovementProbabilityStopper(imp_prob, patience)
        self.max_iter = max_iter
        self.iteration = 0
    
    def __call__(self, optimizer):
        self.iteration += 1
        
        if self.iteration >= self.max_iter:
            return True
            
        # 只在有评估数据时检查
        if len(optimizer._space.target) > 0:
            if self.interval_stopper(optimizer):
                return True
                
            if self.prob_stopper(optimizer):
                return True
                
        return False

# 使用示例
def black_box_function(x, y):
    return -x**2 - (y-1)**2 + 1 + np.random.normal(0, 0.1)

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds={'x': (-2, 2), 'y': (-3, 3)},
    random_state=1,
)

stopper = ComprehensiveCIStopper(
    interval_width=0.2,
    imp_prob=0.05,
    max_iter=50,
    patience=10
)

# 初始评估点
optimizer.maximize(init_points=5, n_iter=0)

# 迭代优化
while not stopper(optimizer):
    optimizer.maximize(init_points=0, n_iter=1)
    print(f"Iter {stopper.iteration}: Current max {optimizer.max['target']:.3f}")

print("优化完成，最终结果:")
print(optimizer.max)