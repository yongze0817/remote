import numpy as np
import matplotlib.pyplot as plt
import os

# 定义核函数
def linear_kernel(x, y, c=1.0):
    return np.dot(x, y) + c

def polynomial_kernel(x, y, alpha=1.0, c=1.0, d=2):
    return (alpha * np.dot(x, y) + c) ** d

def gaussian_kernel(x, y, sigma=1.0):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))

def exponential_kernel(x, y, sigma=1.0):
    return np.exp(-np.linalg.norm(x - y) / (2 * sigma ** 2))

def laplacian_kernel(x, y, sigma=1.0):
    return np.exp(-np.linalg.norm(x - y) / sigma)

def anova_kernel(x, y, sigma=1.0, d=2):
    return np.sum(np.exp(-sigma * (x - y) ** 2) ** d)

def hyperbolic_tangent_kernel(x, y, alpha=1.0, c=1.0):
    return np.tanh(alpha * np.dot(x, y) + c)

def rational_quadratic_kernel(x, y, c=1.0):
    return 1 - (np.linalg.norm(x - y) ** 2) / (np.linalg.norm(x - y) ** 2 + c)

def multiquadric_kernel(x, y, c=1.0):
    return np.sqrt(np.linalg.norm(x - y) ** 2 + c ** 2)

def inverse_multiquadric_kernel(x, y, theta=1.0):
    return 1 / np.sqrt(np.linalg.norm(x - y) ** 2 + theta ** 2)

def circular_kernel(x, y, sigma=1.0):
    norm = np.linalg.norm(x - y)
    if norm < sigma:
        return (2 / np.pi) * np.arccos(-norm / sigma) - (2 / np.pi) * (norm / sigma) * np.sqrt(1 - (norm / sigma) ** 2)
    else:
        return 0

def spherical_kernel(x, y, sigma=1.0):
    norm = np.linalg.norm(x - y)
    if norm < sigma:
        return 1 - (3 / 2) * (norm / sigma) + (1 / 2) * (norm / sigma) ** 3
    else:
        return 0

def wave_kernel(x, y, theta=1.0):
    norm = np.linalg.norm(x - y)
    return (theta / norm) * np.sin(norm / theta)

def power_kernel(x, y, d=2):
    return -np.linalg.norm(x - y) ** d

def log_kernel(x, y, d=2):
    return -np.log(np.linalg.norm(x - y) ** d + 1)

def spline_kernel(x, y):
    product = 1.0
    for xi, yi in zip(x, y):
        term = 1 + xi * yi + xi * yi * min(xi, yi) - (xi + yi) / 2 * min(xi, yi) ** 2 + min(xi, yi) ** 3 / 3
        product *= term
    return product

def b_spline_kernel(x, y, p=1):
    return np.prod([np.sinc((xi - yi) / (p + 1)) ** 2 for xi, yi in zip(x, y)])

def bessel_kernel(x, y, sigma=1.0, v=1.0):
    from scipy.special import jv
    norm = np.linalg.norm(x - y)
    return jv(v + 1, sigma * norm) / (norm ** (-v - 1))

def cauchy_kernel(x, y, sigma=1.0):
    return 1 / (1 + np.linalg.norm(x - y) ** 2 / sigma)

def chi_square_kernel(x, y):
    return 1 - np.sum((x - y) ** 2 / (0.5 * (x + y)))

def histogram_intersection_kernel(x, y):
    return np.sum(np.minimum(x, y))

def generalized_histogram_intersection_kernel(x, y, alpha=1.0, beta=1.0):
    return np.sum(np.minimum(np.abs(x) ** alpha, np.abs(y) ** beta))

def generalized_t_student_kernel(x, y, d=2):
    return 1 / (1 + np.linalg.norm(x - y) ** d)

def bayesian_kernel(x, y):
    # Placeholder for Bayesian kernel implementation
    return np.sum(x * y)

def wavelet_kernel(x, y, a=1.0, c=0.0):
    norm = np.linalg.norm(x - y)
    return (a / norm) * np.sin(norm / a)


# 绘制核函数的二维图
def plot_kernel(kernel_func, title, **kwargs):
    # x = np.linspace(-5, 5, 100)
    # y = np.linspace(-5, 5, 100)
    # X, Y = np.meshgrid(x, y)
    # Z = np.zeros_like(X)
    
    # for i in range(len(x)):
    #     for j in range(len(y)):
    #         Z[i, j] = kernel_func(np.array([X[i, j], Y[i, j]]), np.array([0, 0]), **kwargs)

    # plt.figure(figsize=(6, 6))
    # plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    # plt.colorbar()
    # plt.title(title)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

    
    X = np.linspace(-5, 5, 100)
    Y = X  # y = x for the linear plot
    Z = np.zeros_like(X)
    
    for i in range(len(X)):
        Z[i] = kernel_func(np.array([X[i], Y[i]]), np.array([0, 0]), **kwargs)\
        
    plt.figure(figsize=(6, 6))
    plt.plot(X, Z, label=title)
    plt.xlabel('x')
    plt.ylabel('Kernel Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # 保存图像到指定文件夹
    output_folder = "output_images"
    os.makedirs(output_folder, exist_ok=True)  # 确保文件夹存在
    output_path = os.path.join(output_folder, "svm_bessel_kernel.png")
    plt.savefig(output_path)
    plt.close()

    print(f"图像已保存到 {output_path}")
    plt.show()


# 绘制核函数的线性图
# def plot_kernels():
    x = np.linspace(-5, 5, 100)
    y = x  # y = x for the linear plot
    kernels = [
        (linear_kernel, "Linear Kernel", {"c": 1.0}),
        (polynomial_kernel, "Polynomial Kernel", {"alpha": 1.0, "c": 1.0, "d": 2}),
        (gaussian_kernel, "Gaussian Kernel", {"sigma": 1.0}),
        (exponential_kernel, "Exponential Kernel", {"sigma": 1.0}),
        (laplacian_kernel, "Laplacian Kernel", {"sigma": 1.0}),
        (anova_kernel, "ANOVA Kernel", {"sigma": 1.0, "d": 2}),
        (hyperbolic_tangent_kernel, "Hyperbolic Tangent Kernel", {"alpha": 1.0, "c": 1.0}),
        (rational_quadratic_kernel, "Rational Quadratic Kernel", {"c": 1.0}),
        (multiquadric_kernel, "Multiquadric Kernel", {"c": 1.0}),
        (inverse_multiquadric_kernel, "Inverse Multiquadric Kernel", {"theta": 1.0}),
        (circular_kernel, "Circular Kernel", {"sigma": 1.0}),
        (spherical_kernel, "Spherical Kernel", {"sigma": 1.0}),
        (wave_kernel, "Wave Kernel", {"theta": 1.0}),
        (power_kernel, "Power Kernel", {"d": 2}),
        (log_kernel, "Log Kernel", {"d": 2}),
        (spline_kernel, "Spline Kernel", {}),
        (b_spline_kernel, "B-Spline Kernel", {"p": 1}),
        (bessel_kernel, "Bessel Kernel", {"sigma": 1.0, "v": 1.0}),
        (cauchy_kernel, "Cauchy Kernel", {"sigma": 1.0}),
        (chi_square_kernel, "Chi-Square Kernel", {}),
        (histogram_intersection_kernel, "Histogram Intersection Kernel", {}),
        (generalized_histogram_intersection_kernel, "Generalized Histogram Intersection Kernel", {"alpha": 1.0, "beta": 1.0}),
        (generalized_t_student_kernel, "Generalized T-Student Kernel", {"d": 2}),
        (bayesian_kernel, "Bayesian Kernel", {}),
        (wavelet_kernel, "Wavelet Kernel", {"a": 1.0, "c": 0.0})
    ]

    fig, axs = plt.subplots(5, 5, figsize=(20, 20))
    fig.tight_layout(pad=3.0)

    for i, (kernel_func, title, kwargs) in enumerate(kernels):
        Z = np.zeros_like(x)
        for j in range(len(x)):
            Z[j] = kernel_func(np.array([x[j], y[j]]), np.array([0, 0]), **kwargs)
        
        ax = axs[i // 5, i % 5]
        ax.plot(x, Z, label=title)
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('Kernel Value')
        ax.legend()
        ax.grid(True)

    plt.show()

# 调用函数绘制所有核函数的线性图
# plot_kernels()

# # 示例绘制
plot_kernel(linear_kernel, "Linear Kernel", c=1.0)
plot_kernel(polynomial_kernel, "Polynomial Kernel", alpha=1.0, c=1.0, d=2)
plot_kernel(gaussian_kernel, "Gaussian Kernel", sigma=1.0)
plot_kernel(exponential_kernel, "Exponential Kernel", sigma=1.0)
plot_kernel(laplacian_kernel, "Laplacian Kernel", sigma=1.0)
plot_kernel(anova_kernel, "ANOVA Kernel", sigma=1.0, d=2)
plot_kernel(hyperbolic_tangent_kernel, "Hyperbolic Tangent Kernel", alpha=1.0, c=1.0)
plot_kernel(rational_quadratic_kernel, "Rational Quadratic Kernel", c=1.0)
plot_kernel(multiquadric_kernel, "Multiquadric Kernel", c=1.0)
plot_kernel(inverse_multiquadric_kernel, "Inverse Multiquadric Kernel", theta=1.0)
plot_kernel(circular_kernel, "Circular Kernel", sigma=1.0)
plot_kernel(spherical_kernel, "Spherical Kernel", sigma=1.0)
plot_kernel(wave_kernel, "Wave Kernel", theta=1.0)
plot_kernel(power_kernel, "Power Kernel", d=2)
plot_kernel(log_kernel, "Log Kernel", d=2)
plot_kernel(spline_kernel, "Spline Kernel")
plot_kernel(b_spline_kernel, "B-Spline Kernel", p=1)
plot_kernel(bessel_kernel, "Bessel Kernel", sigma=1.0, v=1.0)
plot_kernel(cauchy_kernel, "Cauchy Kernel", sigma=1.0)
plot_kernel(chi_square_kernel, "Chi-Square Kernel")
plot_kernel(histogram_intersection_kernel, "Histogram Intersection Kernel")
plot_kernel(generalized_histogram_intersection_kernel, "Generalized Histogram Intersection Kernel", alpha=1.0, beta=1.0)
plot_kernel(generalized_t_student_kernel, "Generalized T-Student Kernel", d=2)
plot_kernel(bayesian_kernel, "Bayesian Kernel")
plot_kernel(wavelet_kernel, "Wavelet Kernel", a=1.0, c=0.0)