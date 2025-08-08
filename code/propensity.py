import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.interpolate import interp1d

def propensity_score_linear(Z):
    Z1 = Z[:, 0]
    Z4 = Z[:, 3]
    propensity_score = 1 / (1 + np.exp(-Z1 - 0.1 * Z4))
    return propensity_score

def propensity_score_wavelet(Z):
    # 使用Daubechies 4-tap小波的缩放函数
    Z1 = Z[:, 0]
    Z4 = Z[:, 3]
    Z_tulta = (Z[:, 1] + Z[:, 3] + Z[:, 5] + Z[:, 7] + Z[:, 9]) / 5
    # cA2 = pywt.wavedec(Z_tilde, 'db4', level=2)[-1]
    # cA2_expanded = np.interp(np.arange(Z.shape[0]), np.arange(cA2.shape[0]), cA2)

    # 定义 Daubechies 4-tap 小波
    wavelet_name = 'db2'

    # 获取尺度函数和对应的时间轴
    phi, psi, x = pywt.Wavelet(wavelet_name).wavefun(level=5)

    # 创建插值函数
    phi_interp = interp1d(x, phi, kind='cubic', fill_value="extrapolate")

    propensity_score = 1 / (1 + np.exp(-Z1 - 0.1 * Z4 + phi_interp(Z_tulta)))
    return propensity_score


def propensity_score_weierstrass(Z):
    Z1 = Z[:, 0]
    Z4 = Z[:, 3]
    Z_tulta = (Z[:, 1] + Z[:, 3] + Z[:, 5] + Z[:, 7] + Z[:, 9]) / 5
    # 假设 Weierstrass 函数的实现如下，这里仅为示例
    def weierstrass(x, a=0.5, b=3, n_terms=100):
        """
        计算 Weierstrass 函数
        :param x: 输入向量 x
        :param a: 常数 a
        :param b: 常数 b
        :param n_terms: 级数的项数
        :return: Weierstrass 函数的值
        """
        result = np.sum([a**n * np.cos(b**n * np.pi * x) for n in range(n_terms)], axis=0)
        return result
    cA2_expanded = weierstrass(Z_tulta)
    propensity_score = 1 / (1 + np.exp(-Z1 - 0.1 * Z4 + cA2_expanded))
    return propensity_score

# 生成模拟数据
np.random.seed(0)
Z = np.random.normal(size=(1000, 10))

# 计算倾向得分
ps_linear = propensity_score_linear(Z)
ps_wavelet = propensity_score_wavelet(Z)
ps_weierstrass = propensity_score_weierstrass(Z)

# 绘制倾向得分分布图
plt.figure(figsize=(12, 6))
plt.hist(ps_linear, bins=30, alpha=0.5, label='Linear Model', density=True)
plt.hist(ps_wavelet, bins=30, alpha=0.5, label='Wavelet Model', density=True)
plt.hist(ps_weierstrass, bins=30, alpha=0.5, label='Weierstrass Model', density=True)
plt.title('Distribution of Propensity Scores')
plt.xlabel('Propensity Score')
plt.ylabel('Density')
plt.legend()
plt.show()