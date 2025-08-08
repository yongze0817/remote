import numpy as np

def calculate_phi(M, y_I, Q_M_I, c_new_I, c_old_I):
    # 计算phi
    return -np.linalg.inv(M).dot(np.vstack((y_I, Q_M_I)).dot(c_new_I - c_old_I))

def calculate_psi(y_i, Q_i_M, phi, Q_i_I, c_new_I, c_old_I):
    # 计算psi
    return np.vstack((y_i, Q_i_M)).dot(phi) + Q_i_I.dot(c_new_I - c_old_I)

def calculate_delta_theta(phi, psi, alpha, C, y, f_x, d_m_i, phi_i_plus_1):
    # 计算delta_theta
    terms = []
    for i in range(len(alpha)):
        for j in range(len(y)):
            term1 = -alpha[i] / phi[i+1]
            term2 = (C[i] - alpha[i]) / (phi[i+1] - d_m_i)
            term3 = (1 - y[j] * f_x[j]) / psi[j]
            terms.append(min(term1, term2, term3))
    return min(terms)

def update_parameters(alpha, b, c, delta_theta):
    # 更新参数
    alpha += delta_theta
    c += delta_theta
    return alpha, b, c

def update_sets(M, delta, y_i_up, y_i_low, g_i_up, g_i_low):
    # 更新集合
    if delta > 0:
        b = y_i_up * g_i_up
        M = [i_up]
    else:
        b = y_i_low * g_i_low
        M = [i_low]
    return M

def set_bias_term(y_i_up, y_i_low, g_i_up, g_i_low):
    # 设置偏置项
    if delta > 0:
        return y_i_up * g_i_up
    else:
        return y_i_low * g_i_low

def delta_alpha():
    # 计算delta(alpha)
    pass  # 需要具体的计算逻辑

def trace_until_equal(delta_theta, y_i, g_i):
    # 追踪直到u(delta_theta) = l(delta_theta)
    u_value = max(y_i * (g_i + delta_g_i(delta_theta)))
    l_value = min(y_i * (g_i + delta_g_i(delta_theta)))
    while u_value != l_value:
        delta_theta = (u_value + l_value) / 2
        u_value = max(y_i * (g_i + delta_g_i(delta_theta)))
        l_value = min(y_i * (g_i + delta_g_i(delta_theta)))
    return delta_theta

def u(delta_theta, y_i, g_i):
    # 计算u(delta_theta)
    return max(y_i * (g_i + delta_g_i(delta_theta)))

def l(delta_theta, y_i, g_i):
    # 计算l(delta_theta)
    return min(y_i * (g_i + delta_g_i(delta_theta)))

def delta_g_i(delta_theta):
    # 计算delta_g_i
    return -delta_theta * sum(Q_ij * d_j)

# 主函数
def wsvm_path(alpha, b, c_old, M, O, I, L, c_new):
    theta = 0
    c = c_old
    while theta != 1:
        if M is None:
            delta_theta = empty_margin()
        else:
            phi = -np.linalg.inv(M).dot(np.vstack((y_I, Q_M_I)).dot(c_new_I - c_old_I))
            psi = calculate_psi(y_i, Q_i_M, phi, Q_i_I, c_new[I], c_old[I])
            delta_theta = calculate_delta_theta(phi, psi, alpha, C, y, f_x, d_m_i, phi_i_plus_1)
        
        if theta + delta_theta > 1:
            delta_theta = 1 - theta
        
        alpha, b, c = update_parameters(alpha, b, c, delta_theta)
        theta += delta_theSta
        
        M = update_sets(M, delta_theta, y_i_up, y_i_low, g_i_up, g_i_low)
        O = update_sets(O)
        I = update_sets(I)
        # update_L(L)
        L = update_sets(L)
    
    return c

def empty_margin():
    if delta_alpha() != 0:
        b = set_bias_term(y_i_up, y_i_low, g_i_up, g_i_low)
        delta_theta = 0
    else:
        delta_theta = trace_until_equal()
    return delta_theta



simulate_data_A = pd.read_csv('C:/Users/rmaat\Desktop/MSc/data_A/data_scenario_G_n_500_dataset_{}.csv'.format(n))
X = simulate_data_A.drop(columns=['T', 'Y'])
T = simulate_data_A['T']
Y = simulate_data_A['Y']
W = 2 * T - 1


def gaussian_kernel(x, x_prime):
    """
    Compute the Gaussian kernel.
    
    Parameters:
    x (numpy array): Input vector.
    x_prime (numpy array): Input vector.
    gamma (float): Hyper-parameter controlling the kernel width.
    p (int): Dimensionality of the input vectors.
    
    Returns:
    float: The value of the Gaussian kernel.
    """
    gamma = 1
    p = 2
    distance = np.linalg.norm(x - x_prime) ** 2
    return np.exp(-gamma * distance / p)



# 初始化参数
n_samples = X.shape[0]
M = set()  # 在边界上的样本索引
O = set()  # 在边界外的样本索引
I = set()  # 在边界内的样本索引
L = None  # Cholesky因子，需要根据Q_M计算



# 计算核矩阵Q
Q = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(n_samples):
        Q[i, j] = y[i] * y[j] * gaussian_kernel(X[i], X[j])



# 其他参数
d_m_i = np.zeros(n_samples)  # 与样本相关的距离度量
phi_i_plus_1 = np.zeros(n_samples)  # phi的下一个值，用于计算delta_theta
y_i_up = np.max(y)  # y的最大值，用于更新b和M
y_i_low = np.min(y)  # y的最小值，用于更新b和M
g_i_up = np.max(X, axis=0)  # X的最大值，用于更新b
g_i_low = np.min(X, axis=0)  # X的最小值，用于更新b
