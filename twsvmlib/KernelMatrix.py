import numpy as np

"""
因为涉及到的核运算是
K(矩阵，向量) 或 K(矩阵，矩阵)
因此下面的接口按照这样来设计

输入的两个矩阵/向量只要在维度上匹配矩阵乘法的要求即可
"""

def linear_kernel(X, y):
    """
    线性核函数
    :param X: 矩阵
    :param y: 向量或矩阵
    :return: 核函数计算结果
    """
    if X.shape[1] != y.shape[0]:
        raise ValueError(f"矩阵 X 的列数 {X.shape[1]} 必须等于矩阵/向量 y 的行数 {y.shape[0]}")
    return X @ y


def rbf_kernel(X, y, gamma=1.0):
    """
    高斯核（RBF）核函数
    :param X: 矩阵
    :param y: 向量或矩阵
    :公式为:exp(-gamma * ||x - y||^2)
    :param gamma: 核函数的带宽参数，默认为 1.0
    :return: 核函数计算结果
    """
    if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
        # 处理 y 为向量的情况
        if y.ndim == 2:
            y = y.flatten()
        if X.shape[1] != len(y):
            raise ValueError(f"矩阵 X 的列数 {X.shape[1]} 必须等于向量 y 的长度 {len(y)}")
        # 一维向量默认按行广播
        diff = X - y
        squared_norms = np.sum(diff ** 2, axis=1)
        res = np.exp(-gamma * squared_norms)
        return res
    else:
        # 处理 y 为矩阵的情况
        if X.shape[1] != y.shape[0]:
            raise ValueError(f"矩阵 X 的列数 {X.shape[1]} 必须等于矩阵 y 的行数 {y.shape[0]}")
        y = y.T  # 转置 y 以方便后续计算
        X_reshaped = X[:, np.newaxis, :]
        y_reshaped = y[np.newaxis, :, :]
        diff = X_reshaped - y_reshaped
        squared_norms = np.sum(diff ** 2, axis=-1)
        res = np.exp(-gamma * squared_norms)
        return res


def poly_kernel(X, y, gamma=1.0, r=0.0, d=3):
    """
    多项式核函数
    :param X: 矩阵
    :param y: 向量或矩阵
    :公式为:(gamma * x^T y + r)^d
    :param gamma: 核系数，默认为 1.0
    :param r: 偏移量，默认为 0.0
    :param d: 多项式阶数，默认为 3
    :return: 核函数计算结果
    """
    if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
        # 处理 y 为向量的情况
        if y.ndim == 2:
            y = y.flatten()
        if X.shape[1] != len(y):
            raise ValueError(f"矩阵 X 的列数 {X.shape[1]} 必须等于向量 y 的长度 {len(y)}")
        dot_product = X @ y
        res = (gamma * dot_product + r) ** d
        return res
    else:
        # 处理 y 为矩阵的情况
        if X.shape[1] != y.shape[0]:
            raise ValueError(f"矩阵 X 的列数 {X.shape[1]} 必须等于矩阵 y 的行数 {y.shape[0]}")
        dot_matrix = X @ y
        res = (gamma * dot_matrix + r) ** d
        return res
    