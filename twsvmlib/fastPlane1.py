import numpy as np
from cvxopt import matrix, solvers
from scipy.linalg import cholesky
def solve(R, S, c1, lamda1, alpha_a, ca=0):

    m1 = R.shape[0]

    # 预处理 (S^T S + lamda1 I)^(-1) R^T
    temp = np.linalg.inv(S.T @ S + lamda1 * np.eye(S.shape[1])) @ R.T

    # 构造二次项矩阵P
    H = R @ temp
    
    
    J = np.linalg.cholesky(H+1e-5*np.eye(H.shape[0])) 


    AA = (c1+ca)/2*(H @ alpha_a)
    BB = np.linalg.norm(J, axis=1)*np.linalg.norm((c1-ca)/2*(J.T @ alpha_a))
    F_da = AA - BB
    F_xiao = AA + BB
    epsilon = 1e-6
    idx1 = F_xiao < 1 - epsilon
    idx0 = F_da > 1 + epsilon
    idx_else = np.logical_not(np.logical_or(idx1, idx0))


    alpha = np.ones(m1)
    alpha[idx0] = 0


    # 更新 m1为remain的数量
    m1 = np.sum(idx_else)

    # min 0.5 *c1* x^T HRR x + (c1 alphaD*HDR-e)*x
     
    HRR = H[np.ix_(idx_else, idx_else)]
    HDR = H[np.ix_(idx1|idx0, idx_else)]
    alphaD = alpha[idx1|idx0]

    P = matrix(c1*HRR,tc = 'd')
    q = matrix(c1*alphaD@HDR - np.ones(m1),tc = 'd')

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
    alpha_else = np.array(sol['x']).flatten()
    alpha[idx_else]=alpha_else
    z = -temp@alpha
    # return u1,b1,u是二维列向量
    u1 = z[:len(z)-1].flatten()
    b1 = z[len(z)-1]
    

    return u1,b1,alpha,m1