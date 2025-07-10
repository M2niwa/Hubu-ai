import numpy as np
import matplotlib.pyplot as plt
import datetime
from helpers import *

height, weight, gender = load_data(sub_sample=False, add_outlier=False)
x, mean_x, std_x = standardize(height)
b, A = build_model_data(x, weight)

'''
任务1
'''
def calculate_objective(Axmb):
    '''Calculate the mean squared error for vector Axmb = Ax - b.'''
    # 计算均方误差
    mse = np.mean(Axmb ** 2)
    return mse

'''
任务2
'''
def calculate_L(b, A):
    """Calculate the smoothness constant for f"""
    # 计算 A^T * A
    AtA = np.dot(A.T, A)

    # 计算 A^T * A 的最大特征值
    eigenvalues = np.linalg.eigvals(AtA)
    L = np.max(eigenvalues)

    return L
'''
任务3
'''
def compute_gradient(b, A, x):
    """Compute the gradient."""
    # 计算 Ax - b
    Axmb = np.dot(A, x) - b

    # 计算梯度
    grad = np.dot(A.T, Axmb)

    return grad, Axmb
'''
任务4
'''


def gradient_descent(b, A, initial_x, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store x and objective func. values
    xs = [initial_x]
    objectives = []
    x = initial_x
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and objective function
        # ***************************************************
        # 计算 Ax - b
        Axmb = np.dot(A, x) - b

        # 计算梯度
        grad = np.dot(A.T, Axmb)

        # 计算目标函数值（均方误差）
        obj = np.mean(Axmb ** 2)

        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update x by a gradient descent step
        # ***************************************************
        # 更新 x
        x = x - gamma * grad

        # store x and objective function value
        xs.append(x)
        objectives.append(obj)
        print("Gradient Descent({bi}/{ti}): objective={l}".format(
            bi=n_iter, ti=max_iters - 1, l=obj))

    return objectives, xs

'''
测试
'''

# Define the parameters of the algorithm.
max_iters = 50

gamma = 0.1

# Initialization
x_initial = np.zeros(A.shape[1])

# Start gradient descent.
start_time = datetime.datetime.now()
gradient_objectives_naive, gradient_xs_naive = gradient_descent(b, A, x_initial, max_iters, gamma)
end_time = datetime.datetime.now()

# Print result
exection_time = (end_time - start_time).total_seconds()
print("Gradient Descent: execution time={t:.3f} seconds".format(t=exection_time))

'''
任务6
'''
# 定义算法参数
max_iters = 50

# ***************************************************
# INSERT YOUR CODE HERE
# TODO: a better learning rate using the smoothness of f
# ***************************************************
# 计算平滑性常数L
L = calculate_L(b, A)

# 计算最优学习率
gamma = 1 / L

# Initialization
x_initial = np.zeros(A.shape[1])

# Start gradient descent.
start_time = datetime.datetime.now()
gradient_objectives, gradient_xs = gradient_descent(b, A, x_initial, max_iters, gamma)
end_time = datetime.datetime.now()

# Print result
exection_time = (end_time - start_time).total_seconds()
print("Gradient Descent: execution time={t:.3f} seconds".format(t=exection_time))


data = np.loadtxt("Concrete_Data.csv",delimiter=",")

A = data[:,:-1]
b = data[:,-1]
A, mean_A, std_A = standardize(A)

'''
任务7
'''
# 计算 A 的算子范数（即 A^TA 的最大特征值的平方根）
A_norm = np.linalg.norm(A, ord=2)

# 计算 b 的欧几里得范数（即 b 的2-范数）
b_norm = np.linalg.norm(b, ord=2)

# ***************************************************
# INSERT YOUR CODE HERE
# TODO: Compute the bound on the gradient norm
# ***************************************************
grad_norm_bound = A_norm * b_norm


max_iters = 50
'''
任务8
'''
# ***************************************************
# INSERT YOUR CODE HERE
# TODO: Compute learning rate based on bounded gradient
# ***************************************************
gamma = 1 / grad_norm_bound

# Initialization
x_initial = np.zeros(A.shape[1])

# Start gradient descent.
start_time = datetime.datetime.now()
bd_gradient_objectives, bd_gradient_xs = gradient_descent(b, A, x_initial, max_iters, gamma)
end_time = datetime.datetime.now()


# Print result
exection_time = (end_time - start_time).total_seconds()
print("Gradient Descent: execution time={t:.3f} seconds".format(t=exection_time))

# Averaging the iterates as is the case for bounded gradients case
bd_gradient_objectives_averaged = []
for i in range(len(bd_gradient_xs)):
    if i > 0:
        bd_gradient_xs[i] = (i * bd_gradient_xs[i-1] + bd_gradient_xs[i])/(i + 1)
    grad, err = compute_gradient(b, A, bd_gradient_xs[i])
    obj = calculate_objective(err)
    bd_gradient_objectives_averaged.append(obj)


max_iters = 50

'''
任务9
'''


def calculate_L(b, A):
    """Calculate the smoothness constant for f"""
    # 计算 A^T * A
    AtA = np.dot(A.T, A)

    # 计算 A^T * A 的最大特征值
    eigenvalues = np.linalg.eigvals(AtA)
    L = max(eigenvalues)

    return L


# ***************************************************
# INSERT YOUR CODE HERE
# TODO: a better learning rate using the smoothness of f
# ***************************************************
# 计算平滑性常数 L
L = calculate_L(b, A)

# 计算最优学习率
gamma = 1 / L

# Initialization
x_initial = np.zeros(A.shape[1])

# Start gradient descent.
start_time = datetime.datetime.now()
gradient_objectives, gradient_xs = gradient_descent(b, A, x_initial, max_iters, gamma)
end_time = datetime.datetime.now()

# Print result
exection_time = (end_time - start_time).total_seconds()
print("Gradient Descent: execution time={t:.3f} seconds".format(t=exection_time))


plt.figure(figsize=(8, 8))
plt.xlabel('Number of steps')
plt.ylabel('Objective Function')
plt.plot(range(len(gradient_objectives)), gradient_objectives,'r', label='gradient descent with 1/L stepsize')
plt.plot(range(len(bd_gradient_objectives)), bd_gradient_objectives,'b', label='gradient descent assuming bounded gradients')
plt.plot(range(len(bd_gradient_objectives_averaged)), bd_gradient_objectives_averaged,'g', label='gradient descent assuming bounded gradients with averaged iterates')
plt.legend(loc='upper right')
plt.show()