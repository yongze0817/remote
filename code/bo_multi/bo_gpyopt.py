import numpy as np
import GPyOpt
from GPyOpt.methods import MultiObjectiveBayesianOptimization
import matplotlib.pyplot as plt

# 定义多目标函数 (ZDT1测试函数)
def zdt1(x):
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    f1 = x[:, 0]
    g = 1 + 9 * np.sum(x[:, 1:], axis=1) / (x.shape[1] - 1)
    h = 1 - np.sqrt(f1 / g)
    f2 = g * h
    return np.vstack((f1, f2)).T

# 定义优化问题
domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (0, 1)},
          {'name': 'var_2', 'type': 'continuous', 'domain': (0, 1)}]

# 初始采样点
initial_X = np.random.rand(5, 2)
initial_Y = zdt1(initial_X)

# 多目标优化
optimizer = MultiObjectiveBayesianOptimization(f=zdt1, 
                                             domain=domain,
                                             initial_design_numdata=5,
                                             X=initial_X,
                                             Y=initial_Y,
                                             acquisition_type='EHVI',  # 使用Expected Hypervolume Improvement
                                             normalize_Y=False)       # 禁用Y的标准化

# 运行优化
optimizer.run_optimization(max_iter=15)

# 获取Pareto前沿
pareto_front = optimizer.get_optimal_set()

# 可视化
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c='r', label='Pareto Front')
plt.xlabel('f1')
plt.ylabel('f2')
plt.title('Multi-objective Bayesian Optimization')
plt.legend()
plt.show()