import numpy as np
from cvxopt import matrix, solvers

def solve(R,S,c1,Epsi1,v = None):
    # temp = (S^TS)^-1R^T
    m1 = R.shape[0]
    temp = c1*np.linalg.inv(S.T@S+Epsi1*np.eye(S.shape[1]))@R.T
    P = matrix(R@(c1*np.linalg.inv(S.T@S+Epsi1*np.eye(S.shape[1]))@R.T),tc = 'd')
    q = matrix(-np.ones(m1),tc = 'd')

    zeros = np.zeros(m1)
    
    if v is None:
        v = 1.0*np.ones(m1)
        
    else:
        # 转换为nparray
        v = np.array(v)
        v = c1*v+1.0*c1*np.ones(m1)
        
    
    h = matrix(np.hstack((zeros, v)).reshape(-1, 1), tc='d')
    
    G1 = -np.eye(m1)
    G2 = np.eye(m1)
    G = matrix(np.vstack((G1,G2)),tc = 'd')

    A = None
    b = None

    solvers.options['show_progress'] = False
        # 设置求解精度参数
    
    solvers.options['abstol'] = 1e-8
    solvers.options['reltol'] = 1e-8
    solvers.options['feastol'] = 1e-8
    sol = solvers.qp(P,q,G,h,A,b)
    alpha = np.array(sol['x']).flatten()
    z = -(c1*np.linalg.inv(S.T@S+Epsi1*np.eye(S.shape[1]))@R.T)@alpha
    # return u1,b1,u是二维列向量
    u1 = z[:len(z)-1].flatten()
    b1 = z[len(z)-1]
    
    return u1,b1,alpha.flatten()
