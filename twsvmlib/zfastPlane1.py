import numpy as np
from cvxopt import matrix, solvers
import scipy.linalg


def solve(R, S, c1, lamda1, alpha_a, lamda_a=0):
    """
    优化后的高效求解函数
    使用Cholesky分解替代直接矩阵求逆，显著提高计算速度
    """

    m1 = R.shape[0]
    n = S.shape[1]
    
    # ===== 矩阵计算优化部分 =====
    # 使用Cholesky分解替代直接求逆（关键优化）

    # 预计算S^T S
    S_T_S = S.T @ S
    
    # 计算A1 = S^T S + λI 并进行Cholesky分解
    A1 = S_T_S + lamda1 * np.eye(n)
    L1, lower1 = scipy.linalg.cho_factor(A1, lower=True)
    
    # 求解线性系统：A1 * temp = R.T
    temp = scipy.linalg.cho_solve((L1, lower1), R.T)
    H = R @ temp  # 计算H_k
    


    
    # 同样方式处理第二个矩阵（仅当λa不为0时）
    if lamda_a != 0:
        A2 = S_T_S + lamda_a * np.eye(n)
        L2, lower2 = scipy.linalg.cho_factor(A2, lower=True)
        temp2 = scipy.linalg.cho_solve((L2, lower2), R.T)
        H_k = R @ temp2
    else:
        H_k = np.zeros_like(H)
    


    
    # ===== Cholesky分解优化部分 =====
    # 添加更强的扰动确保正定性
    diag_mean = np.mean(np.abs(np.diag(H)))
    perturbation = max(1e-5, diag_mean * 1e-4)
    H_pert = H + perturbation * np.eye(H.shape[0]) + np.diag(np.abs(H).sum(axis=1) * 1e-6)
    
    # 使用Cholesky分解代替显式逆矩阵计算
    try:
        L = scipy.linalg.cholesky(H_pert, lower=True, check_finite=True)
    except np.linalg.LinAlgError as e:
        # 如果Cholesky分解失败，使用伪逆作为备选方案
        print(f"Warning: Cholesky decomposition failed: {e}")
        # 添加更强的扰动确保正定性
        H_pert = H_pert + np.eye(H_pert.shape[0]) * (np.abs(np.diag(H_pert)).max() * 1e-3)
        L = scipy.linalg.cholesky(H_pert, lower=True, check_finite=False)
        
    # 使用线性求解代替显式逆矩阵计算
    def solve_triangular_system(v):
        """解下三角系统Lx = v，添加更多稳定性保障"""
        # 额外扰动确保不会有奇异值问题
        perturbed_L = L.copy()
        min_diag = np.min(np.abs(np.diag(perturbed_L)))
        if min_diag < 1e-6:
            np.fill_diagonal(perturbed_L, np.diag(perturbed_L) + 1e-6)
        try:
            return scipy.linalg.solve_triangular(perturbed_L, v, lower=True, check_finite=True)
        except Exception as e:
            # 如果仍然失败，使用伪逆作为备选方案
            print(f"Warning: Triangular solve failed, using pseudo-inverse: {e}")
            return np.linalg.lstsq(perturbed_L, v, rcond=1e-10)[0]

    # ===== 中间结果计算 =====
    diff_mat = H_k - H if lamda_a != 0 else -H
    v_vector = diff_mat @ alpha_a
    
    # 使用三角形系统求解替代显式逆矩阵
    solved_vector = solve_triangular_system(v_vector)
    norm_L_inv = np.linalg.norm(solved_vector) / 2
    # ===== 中间结果计算 =====
    diff_mat = H_k - H if lamda_a != 0 else -H
    v_vector = diff_mat @ alpha_a
    
    # 使用三角形系统求解替代显式逆矩阵
    solved_vector = solve_triangular_system(v_vector)
    norm_L_inv = np.linalg.norm(solved_vector) / 2
    
    # 计算范数向量 - 更高效的方法
    norm_L_rows = np.linalg.norm(L, axis=1)
    
    AA = c1 * ((H + H_k) / 2) @ alpha_a
    BB = c1 * norm_L_rows * norm_L_inv
    
    F_da = AA - BB
    F_xiao = AA + BB
    
    # ===== 索引处理 =====
    epsilon = 1e-8
    idx1 = F_xiao < 1 - epsilon
    idx0 = F_da > 1 + epsilon
    idx_else = ~(idx1 | idx0)
    
    alpha = np.ones(m1)
    alpha[idx0] = 0
    
    # ===== 二次规划准备 =====
    # 更新m1为剩余的数量
    m1_remain = np.sum(idx_else)
    if m1_remain == 0:
        z = -temp @ alpha
        u1 = z[:len(z)-1].flatten()
        b1 = z[len(z)-1]
        return u1, b1, alpha, m1_remain
    
    # 高效提取子矩阵
    HRR = H[idx_else, :][:, idx_else]
    HDR = H[idx1 | idx0, :][:, idx_else]
    alphaD = alpha[idx1 | idx0]
    


    
    # ===== 二次规划求解 =====

    
    # 使用内存高效的矩阵创建
    P_matrix = c1 * HRR
    q_vector = c1 * alphaD @ HDR - np.ones(m1_remain)
    
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
    alpha_else = np.array(sol['x']).ravel()
    alpha[idx_else] = alpha_else
    z = -temp @ alpha
    u1 = z[:len(z)-1].flatten()
    b1 = z[len(z)-1]
    

    
    return u1, b1, alpha, m1_remain