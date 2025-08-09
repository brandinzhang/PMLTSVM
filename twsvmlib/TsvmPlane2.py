import numpy as np
from cvxopt import matrix, solvers

def solve(R,S,c2,Epsi2,v = None):
    # temp = (R^TR)^-1S^T
    m2 = S.shape[0]
    temp = c2*np.linalg.inv(R.T@R+Epsi2*np.eye(R.shape[1]))@S.T
    
    P = matrix(S@(c2*np.linalg.inv(R.T@R+Epsi2*np.eye(R.shape[1]))@S.T),tc = 'd')
    q = matrix(-np.ones(m2),tc = 'd')

    zeros = np.zeros(m2)
    # 修改 h 的生成方式，使其成为单列矩阵
    if v is None:    
        v = 1.0*np.ones(m2)
    else:
        v = 1.0*c2*v
    h = matrix(np.hstack((zeros, v)).reshape(-1, 1), tc='d')

    G1 = -np.eye(m2)
    G2 = np.eye(m2)
    G = matrix(np.vstack((G1,G2)),tc = 'd')

    A = None
    b = None

    solvers.options['show_progress'] = False
    solvers.options['abstol'] = 1e-8
    solvers.options['reltol'] = 1e-8
    solvers.options['feastol'] = 1e-8
    sol = solvers.qp(P,q,G,h,A,b)
    beta = np.array(sol['x']).flatten()
    z = (c2*np.linalg.inv(R.T@R+Epsi2*np.eye(R.shape[1]))@S.T)@beta
    # return u2,b2
    u2 = z[:len(z)-1].flatten()
    b2 = z[len(z)-1]
    
    return u2,b2,beta.flatten()
