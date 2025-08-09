import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from . import KernelMatrix as km
from . import TsvmPlane1
from . import TsvmPlane2
from . import Read as R

class TwinSvm(BaseEstimator, ClassifierMixin):
    def __init__(self, c1=1, c2=1, Epsi1=0.01, Epsi2=0.01, kernel='linear', degree=2, gamma=1.0, r=0, palpha=1.0):
        self.c1 = c1
        self.c2 = c2
        self.Epsi1 = Epsi1
        self.Epsi2 = Epsi2
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.r = r
        self.palpha = palpha
        self.a = -1
        self.b = 0

    def _compute_f(self, X):
        C = np.vstack((self.A, self.B))
        if self.kernel == 'linear':
            xc = km.linear_kernel(X, C.T)
        elif self.kernel == 'poly':
            xc = km.poly_kernel(X, C.T, self.gamma, self.r, self.degree)
        elif self.kernel == 'rbf':
            xc = km.rbf_kernel(X, C.T, self.gamma)
        else:
            xc = km.linear_kernel(X, C.T)

        d1 = np.abs(xc @ self.u1 + self.b1)
        d2 = np.abs(xc @ self.u2 + self.b2)

        temp1 = (xc @ self.u1 + self.b1) / np.sqrt(np.dot(self.u1, self.u1))
        temp2 = (xc @ self.u2 + self.b2) / np.sqrt(np.dot(self.u2, self.u2))
        temp3 = 2 * np.dot(self.u1, self.u2) / (np.linalg.norm(self.u1)**2 * np.linalg.norm(self.u2)**2 + 1e-10)
        temp3 = np.clip(temp3, -2.0, 2.0)

        num1 = np.abs(temp1 + temp2)
        den1 = np.sqrt(np.maximum(2 + temp3, 0) + 1e-10)
        D_pos = num1 / den1
        num2 = np.abs(temp1 - temp2)
        den2 = np.sqrt(np.maximum(2 - temp3, 0) + 1e-10)
        D_neg = num2 / den2

        d = np.minimum(D_pos, D_neg)
        D = np.maximum(D_pos, D_neg)

        f = np.zeros(d1.shape)
        temp = d * (d / (D + 1e-6)) ** self.palpha
        idx1 = d1 < d2
        idx2 = d1 > d2
        f[idx1] = temp[idx1]
        f[idx2] = -temp[idx2]

        return f, d2

    def fit(self, X, Y):
        A = X[Y == 1]
        B = X[Y == 0]
        C = np.vstack((A, B))
        m1 = A.shape[0]
        m2 = B.shape[0]
        e1 = np.ones((m1, 1))
        e2 = np.ones((m2, 1))

        if self.kernel == 'linear':
            K1 = km.linear_kernel(A, C.T)
            K2 = km.linear_kernel(B, C.T)
        elif self.kernel == 'poly':
            K1 = km.poly_kernel(A, C.T, self.gamma, self.r, self.degree)
            K2 = km.poly_kernel(B, C.T, self.gamma, self.r, self.degree)
        elif self.kernel == 'rbf':
            K1 = km.rbf_kernel(A, C.T, self.gamma)
            K2 = km.rbf_kernel(B, C.T, self.gamma)
        else:
            K1 = km.linear_kernel(A, C.T)
            K2 = km.linear_kernel(B, C.T)

        S = np.c_[K1, e1]
        R = np.c_[K2, e2]
        
        self.u1, self.b1, _ = TsvmPlane1.solve(R, S, self.c1, self.Epsi1)
        self.u2, self.b2, _ = TsvmPlane2.solve(R, S, self.c2, self.Epsi2)
        self.A = A
        self.B = B
        return self
    

    def fast_fit(self, X, Y):
        A = X[Y == 1]
        B = X[Y == 0]
        C = np.vstack((A, B))
        m1 = A.shape[0]
        m2 = B.shape[0]
        e1 = np.ones((m1, 1))
        e2 = np.ones((m2, 1))

        if self.kernel == 'linear':
            K1 = km.linear_kernel(A, C.T)
            K2 = km.linear_kernel(B, C.T)
        elif self.kernel == 'poly':
            K1 = km.poly_kernel(A, C.T, self.gamma, self.r, self.degree)
            K2 = km.poly_kernel(B, C.T, self.gamma, self.r, self.degree)
        elif self.kernel == 'rbf':
            K1 = km.rbf_kernel(A, C.T, self.gamma)
            K2 = km.rbf_kernel(B, C.T, self.gamma)
        else:
            K1 = km.linear_kernel(A, C.T)
            K2 = km.linear_kernel(B, C.T)

        S = np.c_[K1, e1]
        R = np.c_[K2, e2]
        
        self.u1, self.b1, _ = TsvmPlane1.solve(R, S, self.c1, self.Epsi1)
        self.u2, self.b2, _ = TsvmPlane2.solve(R, S, self.c2, self.Epsi2)
        self.A = A
        self.B = B
        return self

    def predict(self, X):
        f, delta = self._compute_f(X)
        self.delta = delta
        return (f >= 0).astype(int)

    def get_score(self, X):
        f, _ = self._compute_f(X)
        return 1 / (1 + np.exp(self.a * f + self.b))

    def fit_sigmoid(self, X, Y):
        f, _ = self._compute_f(X)
        m = len(Y)
        t = np.zeros(m)
        mpos = np.sum(Y == 1)
        mneg = m - mpos

        for i in range(m):
            if Y[i] == 1:
                t[i] = (mpos + 1) / (mpos + 2)
            else:
                t[i] = 1 / (mneg + 2)

        a, b = 0.0, np.log((mneg + 1) / (mpos + 1))

        for _ in range(20):
            fApB = a * f + b
            p = 1 / (1 + np.exp(fApB))
            q = 1 - p
            d1 = t - p
            d2 = p * q
            g1 = np.sum(f * d1)
            g2 = np.sum(d1)
            h11 = np.sum(f * f * d2)
            h22 = np.sum(d2)
            h21 = np.sum(f * d2)
            det = h11 * h22 - h21 * h21
            if det == 0:
                break
            dA = -(h22 * g1 - h21 * g2) / det
            dB = -(h11 * g2 - h21 * g1) / det
            a += dA
            b += dB
            if np.abs(dA) < 1e-6 and np.abs(dB) < 1e-6:
                break

        self.a = a
        self.b = b

    def get_delta(self):
        return self.delta

    def get_support_vectors(self):
        idx1 = self.alpha > 1e-5
        idx2 = self.beta > 1e-5
        sv1 = self.A[idx2.flatten()]
        sv2 = self.B[idx1.flatten()]
        return np.vstack((sv1, sv2))

class pMLTSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, c1=1, c2=1, Epsi1=0.01, Epsi2=0.01, kernel='linear', degree=2, gamma=1.0, r=0, dname='flags', palpha=1.0):
        self.c1 = c1
        self.c2 = c2
        self.Epsi1 = Epsi1
        self.Epsi2 = Epsi2
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.r = r
        self.palpha = palpha
        self.dataset = dname
        self.models = []

    def fit(self, X, Y):
        n_labels = Y.shape[1]
        for i in range(n_labels):
            y = Y[:, i]
            model = TwinSvm(c1=self.c1, c2=self.c2, Epsi1=self.Epsi1, Epsi2=self.Epsi2,
                            kernel=self.kernel, degree=self.degree, gamma=self.gamma, r=self.r, palpha=self.palpha)
            if np.all(y == 0) or np.all(y == 1):
                X_all, y_all = R.read(self.dataset)
                y_i = y_all[:, i]
                if np.all(y == 0):
                    x_1 = X_all[y_i == 1]
                    X[0, :] = x_1[0, :]
                    y[0] = 1
                else:
                    x_0 = X_all[y_i == 0]
                    X[0, :] = x_0[0, :]
                    y[0] = 0
            model.fit(X, y)
            self.models.append(model)
        return self

    def fast_fit(self, X, Y):
        n_labels = Y.shape[1]
        for i in range(n_labels):
            y = Y[:, i]
            model = TwinSvm(c1=self.c1, c2=self.c2, Epsi1=self.Epsi1, Epsi2=self.Epsi2,
                            kernel=self.kernel, degree=self.degree, gamma=self.gamma, r=self.r, palpha=self.palpha)
            if np.all(y == 0) or np.all(y == 1):
                X_all, y_all = R.read(self.dataset)
                y_i = y_all[:, i]
                if np.all(y == 0):
                    x_1 = X_all[y_i == 1]
                    X[0, :] = x_1[0, :]
                    y[0] = 1
                else:
                    x_0 = X_all[y_i == 0]
                    X[0, :] = x_0[0, :]
                    y[0] = 0
            model.fast_fit(X, y)
            self.models.append(model)
        return self

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return np.array(predictions).T

    def score(self, X):
        scores = [model.get_score(X) for model in self.models]
        return np.array(scores).T

    def get_delta_k(self):
        deltas = [model.get_delta() for model in self.models]
        return np.array(deltas)
    def score_mlknn(self, X, Y_train, X_train, k=10):
        from sklearn.neighbors import NearestNeighbors
        S = self.score(X)  # 原TSVM输出的Sigmoid概率分数
        n, l = S.shape

        # 用邻居去修正
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X_train)
        dists, idx = nbrs.kneighbors(X)

        # 准备输出
        S_new = np.zeros_like(S)

        for j in range(l):  # 每一个标签
            for i in range(n):  # 每一个样本
                neighbors = idx[i]
                y_neighbors = Y_train[neighbors, j]  # 邻居中这个标签的出现情况
                c = np.sum(y_neighbors)

                # 简单MLKNN贝叶斯公式（可以加拉普拉斯平滑）
                p1 = (c + 1) / (k + 2)  # 邻居中是1的概率
                p0 = 1 - p1

                # 修正原来的得分
                s = S[i, j]
                s_new = s * p1 / (s * p1 + (1 - s) * p0 + 1e-10)  # 防止除0
                S_new[i, j] = s_new

        return S_new
