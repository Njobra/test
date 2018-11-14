import numpy as np
from numpy.linalg import inv


def fit_locdisp_mlfp(e, p, ni, trsh):


    d = min(e.shape) #dimension

    mi_hfp = np.zeros((d, 1))
    s2_hfp = np.zeros((d, d))
    for i in range(len(p)):
        mi_hfp = mi_hfp + p[i]*e[i].reshape(2, 1)
    for i in range(len(p)):
        s2_hfp = s2_hfp + p[i] * np.matmul(e[i].reshape(2, 1) - mi_hfp, (e[i].reshape(2, 1) - mi_hfp).T)
    mi = []
    mi.append(mi_hfp)
    s2 = []
    s2.append((ni-2)/ni * s2_hfp if ni > 2 else s2_hfp)
    w = np.zeros(len(p))
    q = np.zeros(len(p))

    while(True):
        for i in range(len(p)):
            w[i] = (ni + d)/(ni + np.matmul(np.matmul((e[i].reshape(2, 1) - mi[-1]).T, inv(s2[-1])),
                                            e[i].reshape(2, 1) - mi[-1]))
        x = 0
        x = np.dot(w, p)
        for i in range(len(p)):
            q[i] = (p[i] * w[i])/x

        mi_ = np.zeros((d, 1))
        for i in range(len(p)):
            mi_ = mi_ + q[i] * e[i].reshape(2, 1)
        mi.append(mi_)
        s2_ = np.zeros((d, d))
        for i in range(len(p)):
            s2_ = s2_ + q[i] * np.matmul(e[i].reshape(2,1) - mi[-1], (e[i].reshape(2,1) - mi[-1]).T)
        s2.append(s2_)

        if np.linalg.norm(mi[-2] - mi[-1])/np.linalg.norm(mi[-1]) < trsh and \
                                np.linalg.norm(np.subtract(s2[-1], s2[-2]), ord='fro')/\
                                np.linalg.norm(s2[-1], ord='fro') < trsh:
            return mi[-1], s2[-1]

nu = 100
t_ = 1000
trsh = 10**(-9)
semp = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=1000)
pt_vector = (1/t_)*np.ones(t_)

print(fit_locdisp_mlfp(semp, pt_vector, nu, trsh))