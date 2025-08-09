import numpy as np
from cvxopt import matrix, solvers
from scipy.linalg import cholesky
def solve(R, S, c2, lamda_2, gamma_a, ca=0):

    m1 = S.shape[0]

    # 预处理 (S^T S + lamda_2 I)^(-1) R^T
    temp = np.linalg.inv(R.T @ R + lamda_2 * np.eye(R.shape[1])) @ S.T

    # 构造二次项矩阵P
    H = S @ temp
    
    
    J = np.linalg.cholesky(H+1e-5*np.eye(H.shape[0])) 


    T1 = (c2+ca)/2*(H @ gamma_a)
    T2 = np.linalg.norm(J, axis=1)*np.linalg.norm((c2-ca)/2*(J.T @ gamma_a))
    F_da = T1 - T2 
    F_xiao = T1 + T2
    epsilon = 1e-6
    idx1 = F_xiao < 1 - epsilon
    idx0 = F_da > 1 + epsilon
    idx_else = np.logical_not(np.logical_or(idx1, idx0))
    
    gamma = np.ones(m1)
    gamma[idx0] = 0

    # 更新 m1为remain的数量
    m1 = np.sum(idx_else)

    # min 0.5 *c2* x^T HRR x + (c2 gammaD*HDR-e)*x
     
    HRR = H[np.ix_(idx_else, idx_else)]
    HDR = H[np.ix_(idx1|idx0, idx_else)]
    gammaD = gamma[idx1|idx0]

    P = matrix(c2*HRR,tc = 'd')
    q = matrix(c2*gammaD@HDR - np.ones(m1),tc = 'd')

    zeros = np.zeros(m1)
    v = 1.0*np.ones(m1)
    h = matrix(np.hstack((zeros, v)).reshape(-1, 1), tc='d')
    
    G1 = -np.eye(m1)
    G2 = np.eye(m1)
    G = matrix(np.vstack((G1,G2)),tc = 'd')

    A = None
    b = None

    # 设置精度参数（示例值）

    solvers.options['show_progress'] = False  # 关闭求解过程输出
    solvers.options['abstol'] = 1e-8
    solvers.options['reltol'] = 1e-8
    solvers.options['feastol'] = 1e-8
    sol = solvers.qp(P,q,G,h,A,b)
    gamma_else = np.array(sol['x']).flatten()
    gamma[idx_else]=gamma_else
    z = -temp@gamma
    # return u1,b1,u是二维列向量
    u1 = z[:len(z)-1].flatten()
    b1 = z[len(z)-1]
    
    return u1,b1,gamma,m1