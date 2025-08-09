import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from . import KernelMatrix as km
from . import TsvmPlane1
from . import Read as R



class TwinSvm(BaseEstimator, ClassifierMixin):
    def __init__(self,c1=1,Epsi1=0.01,kernel='linear',degree=2,gamma=1.0,r=0):
        """
        初始化 TwinSvm 分类器的参数。
        :param c1: 第一个平面的惩罚参数
        :param c2: 第二个平面的惩罚参数
        :param Epsi1: 第一个平面的正则化系数
        :param Epsi2: 第二个平面的正则化系数
        :param kernel: 核函数类型，可选 'linear', 'poly', 'rbf'
        :param degree: 多项式核的阶数
        :param gamma: 核函数的参数
        :param r: 多项式核的偏移量
        """
        self.c1 = c1
        self.Epsi1 = Epsi1
        self.u1 = None
        self.b1 = None
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.r = r
    
    def fit(self,X,Y):
        """
        训练 TwinSvm 分类器。
        :param X: 特征矩阵
        :param y: 标签向量
        :return: 训练好的分类器实例
        """
        # Data = sorted(zip(Y,X), key=lambda pair: pair[0], reverse = True)
        
        A = X[Y == 1]  
        B = X[Y == 0]
        C = np.vstack((A,B))
        m1 = A.shape[0]
        m2 = B.shape[0]
        e1 = np.ones((m1,1))
        e2 = np.ones((m2,1))
        if self.kernel == 'linear':
            K1 = km.linear_kernel(A,C.T)
            K2 = km.linear_kernel(B,C.T)
        elif self.kernel == 'poly':
            K1 = km.poly_kernel(A,C.T,self.gamma,self.r,self.degree)
            K2 = km.poly_kernel(B,C.T,self.gamma,self.r,self.degree)
        elif self.kernel == 'rbf':
            K1 = km.rbf_kernel(A,C.T,self.gamma)
            K2 = km.rbf_kernel(B,C.T,self.gamma)
        else:
            K1 = km.linear_kernel(A,C.T)
            K2 = km.linear_kernel(B,C.T)
        S = np.c_[K1,e1]
        R = np.c_[K2,e2]
        self.u1,self.b1,self.alpha = TsvmPlane1.solve(R,S,self.c1,self.Epsi1)
        self.A = A
        self.B = B
        return self
    def predict(self, X):
        """
        对输入数据进行预测。
        :param X: 待预测的特征矩阵
        :return: 预测的标签向量
        """
        # 首先将 A 和 B 合并成 C
        C = np.vstack((self.A, self.B))
        if self.kernel == 'linear':
            K_x_C = km.linear_kernel(X, C.T)
        elif self.kernel == 'poly':
            K_x_C = km.poly_kernel(X, C.T, self.gamma, self.r, self.degree)
        elif self.kernel == 'rbf':
            K_x_C = km.rbf_kernel(X, C.T, self.gamma)
        else:
            K_x_C = km.linear_kernel(X, C.T)

        # 计算到两个平面的绝对值距离
        distance1 = np.abs(K_x_C @ self.u1 + self.b1)


        # 根据距离选择类别，距离小的类别为预测结果
        predictions = (distance1 <= 1).astype(int)

        # 返回的是一维向量
        return predictions

    def get_score(self, X):
        """
        返回输入数据的分数
        取-distance作为最终分数
        """
        # 首先将 A 和 B 合并成 C
        C = np.vstack((self.A, self.B))
        if self.kernel == 'linear':
            K_x_C = km.linear_kernel(X, C.T)
        elif self.kernel == 'poly':
            K_x_C = km.poly_kernel(X, C.T, self.gamma, self.r, self.degree)
        elif self.kernel == 'rbf':
            K_x_C = km.rbf_kernel(X, C.T, self.gamma)
        else:
            K_x_C = km.linear_kernel(X, C.T)

        # 计算到平面1的绝对值距离
        distance1 = np.abs(K_x_C @ self.u1 + self.b1)

        # 返回一维向量，距离的负值作为分数
        return -distance1


    def set_params(self, **params):
        """
        设置估计器的参数。
        :param params: 包含参数名和参数值的字典
        :return: 分类器实例
        """
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Invalid parameter {param} for estimator {self.__class__.__name__}.")
        return self
    
    def get_support_vectors(self):
        """
        获取支持向量。
        :return: 支持向量矩阵
        """
        mask1 = self.alpha > 1e-5
        mask1 = mask1.flatten()
        sv2 = self.B[mask1]
        return sv2


class kMLTSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, c1=1, Epsi1=0.01,kernel='linear', degree=2, gamma=1.0, r=0, dname='flags'):
        self.c1 = c1
        self.Epsi1 = Epsi1
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.r = r
        self.models = []
        self.dataset = dname

    def fit(self, X, Y):
        n_labels = Y.shape[1]
        for i in range(n_labels):
            y = Y[:, i]
            model = TwinSvm(c1=self.c1, Epsi1=self.Epsi1, 
                            kernel=self.kernel, degree=self.degree, gamma=self.gamma, r=self.r)
            # 若y全为0
            if(np.all(y==0)):
                X_all,y_all = R.read(self.dataset)
                y_i = y_all[:,i]
                x_1 = X_all[y_i==1]
                X[0,:] = x_1[0,:]
                y[0] = 1
            if(np.all(y==1)):
                X_all,y_all = R.read(self.dataset)
                y_i = y_all[:,i]
                x_0 = X_all[y_i==0]
                X[0,:] = x_0[0,:]
                y[0] = 0
            model.fit(X, y)
            self.models.append(model)
        return self

    def predict(self, X):
        predictions = []
        for model in self.models:
            # 或得对当下标签预测的一维向量
            pred = model.predict(X)
            predictions.append(pred)
        return np.array(predictions).T

    def score(self, X):
        scores = []
        for model in self.models:
            # 或得对当下标签预测的一维向量
            pred = model.get_score(X)
            scores.append(pred)
        return np.array(scores).T