import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# 设置随机种子以便复现
np.random.seed(0)

# 方法名称
methods = ['SVM', 'KOM', 'KCB', 'CARD', 'GLM', 'RFRST']

# 核函数类型
kernels = ['Linear', 'Polynomial', 'RBF']

# 生成模拟数据，每个方法和每个核函数生成10组数据
data = {
    method: [np.random.normal(loc=10, scale=5, size=(10,)) for _ in kernels]
    for method in methods
}
print(data)
# 准备绘制箱线图的数据
boxplot_data = []
labels = []

# 为每种核函数生成一个颜色和填充模式
colors = ['lightblue', 'lightgreen', 'lightcoral']
hatches = ['/', '\\', 'x']

for method in methods:
    for i, kernel in enumerate(kernels):
        boxplot_data.append(data[method][i])
        labels.append(f"{method} ({kernel})")

print(boxplot_data)
# 绘制箱线图
fig, ax = plt.subplots(figsize=(12, 6))
for i, kernel in enumerate(kernels):
    # 计算当前核函数的箱型图位置
    kernel_data = [boxplot_data[j] for j in range(i, len(boxplot_data), len(kernels))]
    positions = [m * (len(kernels) + 1) + i + 0.5 for m in range(len(methods))]
    ax.boxplot(kernel_data, positions=positions, widths=0.6,
               patch_artist=True,
               boxprops=dict(facecolor=colors[i], hatch=hatches[i]))

# 设置标题和标签
ax.set_title('Boxplot of ATE by Method and Kernel')
ax.set_xlabel('Method')
ax.set_ylabel('ATE')

# 添加参考线
ax.axhline(y=10, color='r', linestyle='--', label='Reference line')

# 添加图例
legend_patches = [Patch(facecolor=colors[i], hatch=hatches[i], label=kernel) for i, kernel in enumerate(kernels)]
ax.legend(handles=legend_patches, title='Kernel')

# 设置X轴标签
ax.set_xticks([m * (len(kernels) + 1) + len(kernels) // 2 for m in range(len(methods))])
ax.set_xticklabels(methods, rotation=45)  # 旋转X轴标签，以便更好地显示

# 显示图表
plt.grid(True, axis='y')
plt.tight_layout()  # 调整布局以防止标签被截断
plt.show()