import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.multi_objective import qLogExpectedHypervolumeImprovement  # 修改这里
from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective import pareto
import matplotlib.pyplot as plt

# 设置默认数据类型
torch.set_default_dtype(torch.float64)

# 定义目标函数
def my_objective(x):
    x1, x2 = x[:, 0], x[:, 1]
    f1 = x1**2 + x2**2
    f2 = (x1-1)**2 + (x2-1)**2
    return torch.stack([f1, f2], dim=-1)

# 初始化数据
train_X = torch.rand(5, 2, dtype=torch.float64)
train_Y = my_objective(train_X)
train_Y_std = standardize(train_Y)

# 构建模型
model = SingleTaskGP(train_X, train_Y_std)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

# 设置参考点
ref_point = torch.tensor([-2.0, -2.0], dtype=torch.float64)

# 创建partitioning对象
partitioning = DominatedPartitioning(ref_point=ref_point, Y=train_Y)

# 使用推荐的采集函数
acq_func = qLogExpectedHypervolumeImprovement(  # 修改这里
    model=model,
    ref_point=ref_point,
    partitioning=partitioning
)

# 优化采集函数
bounds = torch.stack([torch.zeros(2, dtype=torch.float64), 
                     torch.ones(2, dtype=torch.float64)])
candidate, _ = optimize_acqf(
    acq_function=acq_func,
    bounds=bounds,
    q=1,
    num_restarts=5,
    raw_samples=20,
)

# 获取Pareto前沿
pareto_mask = pareto.is_non_dominated(train_Y)
pareto_front = train_Y[pareto_mask]

# 可视化
plt.figure(figsize=(8, 6))
plt.scatter(train_Y[:, 0], train_Y[:, 1], color='blue', label='Samples')
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], color='red', label='Pareto Front')
plt.xlabel("Objective 1")
plt.ylabel("Objective 2")
plt.title("Multi-objective BO with qLogEHVI")  # 更新标题
plt.legend()
plt.show()