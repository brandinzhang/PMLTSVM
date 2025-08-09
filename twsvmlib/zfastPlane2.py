import numpy as np
from cvxopt import matrix, solvers
import scipy.linalg


def solve(R, S, c2, lamda2, gamma_a, lamda_a=0):
    """
    优化后的高效求解函数 (solve_v2)
    使用Cholesky分解替代直接矩阵求逆，显著提高计算速度
    """

    m1 = S.shape[0]
    n = R.shape[1]
    
    # ===== 矩阵计算优化部分 =====

    # 预计算R^T R
    R_T_R = R.T @ R
    
    # 计算A1 = R^T R + λI 并进行Cholesky分解
    A1 = R_T_R + lamda2 * np.eye(n)
    L1, lower1 = scipy.linalg.cho_factor(A1, lower=True)
    
    # 求解线性系统：A1 * temp = S.T
    temp = scipy.linalg.cho_solve((L1, lower1), S.T)
    H = S @ temp  # 计算H
    


    
    # 同样方式处理第二个矩阵（仅当λa不为0时）
    if lamda_a != 0:
        A2 = R_T_R + lamda_a * np.eye(n)
        L2, lower2 = scipy.linalg.cho_factor(A2, lower=True)
        temp2 = scipy.linalg.cho_solve((L2, lower2), S.T)
        H_k = S @ temp2
    else:
        H_k = np.zeros_like(H)
    

    
    # ===== Cholesky分解优化部分 =====
    # 添加小扰动确保正定性
    H_pert = H + 1e-5 * np.eye(H.shape[0])
    
    # 使用Cholesky分解代替显式逆矩阵计算

    L = scipy.linalg.cholesky(H_pert, lower=True)

    
    # 使用线性求解代替显式逆矩阵计算
    def solve_triangular_system(v):
        """解下三角系统Lx = v"""
        # 确保向量v是一维数组
        v = np.asarray(v).flatten()
        
        # 使用更稳健的scipy.linalg.solve而不是solve_triangular
        perturbed_L = L + 1e-5 * np.eye(L.shape[0])
        return scipy.linalg.solve(perturbed_L, v)
    
    # ===== 中间结果计算 =====
    diff_mat = H_k - H if lamda_a != 0 else -H
    v_vector = diff_mat @ gamma_a
    
    # 使用三角形系统求解替代显式逆矩阵
    solved_vector = solve_triangular_system(v_vector)
    norm_L_inv = np.linalg.norm(solved_vector) / 2
    
    # 计算范数向量 - 更高效的方法
    norm_L_rows = np.linalg.norm(L, axis=1)
    
    AA = c2 * ((H + H_k) / 2) @ gamma_a
    BB = c2 * norm_L_rows * norm_L_inv
    
    F_da = AA - BB
    F_xiao = AA + BB
    
    # ===== 索引处理 =====
    epsilon = 1e-8
    idx1 = F_xiao < 1 - epsilon
    idx0 = F_da > 1 + epsilon
    idx_else = ~(idx1 | idx0)
    
    gamma = np.ones(m1)
    gamma[idx0] = 0
    
    # ===== 二次规划准备 =====
    # 更新m1为剩余的数量
    m1_remain = np.sum(idx_else)
    if m1_remain == 0:
        z = -temp @ gamma
        u1 = z[:len(z)-1].flatten()
        b1 = z[len(z)-1]
        return u1, b1, gamma, m1_remain
    
    # 高效提取子矩阵
    HRR = H[idx_else, :][:, idx_else]
    HDR = H[idx1 | idx0, :][:, idx_else]
    gammaD = gamma[idx1 | idx0]
    

    
    # 使用内存高效的矩阵创建
    P_matrix = c2 * HRR
    q_vector = c2 * gammaD @ HDR - np.ones(m1_remain)
    
    # 设置二次规划问题
    P = matrix(P_matrix, tc='d')
    q = matrix(q_vector, tc='d')
    
    # 约束矩阵 G x <= h
    G = matrix(np.vstack((-np.eye(m1_remain), np.eye(m1_remain))), tc='d')
    h = matrix(np.hstack((np.zeros(m1_remain), np.ones(m1_remain))).T, tc='d')
    
    # 设置求解器选项
    solvers.options['show_progress'] = False
    solvers.options['abstol'] = 1e-8
    solvers.options['reltol'] = 1e-8
    solvers.options['feastol'] = 1e-8
    
    # 求解二次规划
    sol = solvers.qp(P, q, G, h)
    
    # ===== 结果提取 =====
    gamma_else = np.array(sol['x']).ravel()
    gamma[idx_else] = gamma_else
    z = -temp @ gamma
    u1 = z[:len(z)-1].flatten()
    b1 = z[len(z)-1]
    

    
    return u1, b1, gamma, m1_remain