import pylab as pl
import numpy as np
from numpy import linalg
from cvxopt import matrix, solvers # untuk menyelesaikan permesalahan nilai optimum di matriks pada quadratic programming

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class SVM(object):
    def __init__(self, kernel, galat=1e-5, C=None):
        super(SVM, self).__init__()
        if kernel=="linear":
            self.kernel = linear_kernel
        elif kernel=="polynomial":
            self.kernel = polynomial_kernel
        else:
            self.kernel = gaussian_kernel
        self.C = C
        self.galat = galat
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        print "Ukuran matriks data yang akan dilatih =",X.shape
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        P = matrix(np.outer(y,y) * K)
        q = matrix(np.ones(n_samples) * -1)
        A = matrix(y, (1,n_samples))
        b = matrix(0.0)

        if self.C is None:
            G = matrix(np.diag(np.ones(n_samples) * -1))
            h = matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = matrix(np.hstack((tmp1, tmp2)))

        # solve Quadratic Programming problem
        print "Solusi Optimal: "
        solution = solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # non zero lagrange multipliers
        sv = a > self.galat
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print "%d support vectors out of %d data latih" % (len(self.a), n_samples)

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def testing(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def akurasi(self,fitur):
        return 0.1  # nilai confidence masih belum dicari

    def predict(self, X):
        return np.sign(self.testing(X))

    def plot_margin(self, X1_train, X2_train, clf):
        def f(x, w, b, c=0):
            return (-w[0] * x - b + c) / w[1]

        pl.plot(X1_train[:,0], X1_train[:,1], "ro")
        pl.plot(X2_train[:,0], X2_train[:,1], "bo")
        pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

        # w.x + b = 0
        a0 = -4; a1 = f(a0, clf.w, clf.b)
        b0 = 4; b1 = f(b0, clf.w, clf.b)
        pl.plot([a0,b0], [a1,b1], "k")

        # w.x + b = 1
        a0 = -4; a1 = f(a0, clf.w, clf.b, 1)
        b0 = 4; b1 = f(b0, clf.w, clf.b, 1)
        pl.plot([a0,b0], [a1,b1], "k--")

        # w.x + b = -1
        a0 = -4; a1 = f(a0, clf.w, clf.b, -1)
        b0 = 4; b1 = f(b0, clf.w, clf.b, -1)
        pl.plot([a0,b0], [a1,b1], "k--")

        pl.axis("tight")
        pl.show()
