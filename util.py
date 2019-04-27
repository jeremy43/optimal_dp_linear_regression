import numpy as np
from cvxpy import *
delta = 1e-6
eps = 1
log_term = np.log(2/delta)

a = np.maximum(2/eps*(1+log_term), 0)
d = 11


def obtain_gamma(lamda, eps, delta):
    """
    claculate the maximum gamma that remains feasible to the privacy budget
    :param lamda: smallest eigenvalue
    :param eps:
    :param delta:
    :return: corresponding gamma
    """
    log_term = np.log(2 / delta)
    c1 = 1 + log_term # since \beta is aroung 1/4, L is 1, c1 = L + \beta
    c2 = log_term
    gamma = (-np.sqrt(lamda * c2) + np.sqrt(lamda * c2 - 4 * (c1 - eps * lamda))) ** 2 / 4
    return gamma

def limit(alpha, eps):

    func_min = lambda a1, gamma1: a1 + 1/gamma1*d*np.log(d)
    func_assure = lambda a1, gamma1 : (c1+gamma1)/a1 + np.sqrt(gamma1*c2/a1)
    c1 = 1 + log_term
    c2 = log_term
    cur_min = 10000
    opt_alpha = alpha
    opt_gamma = 0.1
    while alpha < 1000:
        gamma = (-np.sqrt(alpha*c2)+np.sqrt(alpha*c2-4*(c1-eps*alpha)))**2/4
        #print('gamma={} eps={}'.format(gamma,func_assure(alpha,gamma)))
        #gamma = np.maximum(gamma,0.1)
        if func_min(alpha, gamma) < cur_min and func_assure(alpha,gamma) < eps+0.1:
            opt_alpha = alpha
            opt_gamma = gamma
            cur_min = func_min(alpha, gamma)
            print('alpha={} min_={} gamma= {}'.format(alpha, cur_min,gamma))

        alpha +=1
    return opt_alpha, opt_gamma
#limit(a,eps)

def linear(X,y):

    dim = X.shape[1]
    theta = Variable(shape  = dim)
    loss = norm(y-X*theta,2)
    problem = Problem(Minimize(loss))
    problem.solve(verbose=True)
    theta = theta.value
    predict = np.dot(X, theta)
    H = np.dot(X.T, X)
    Xy = np.dot(X.T, y)
    theta_t = np.linalg.solve(H, Xy)
    mse_t = np.mean((y-np.dot(X, theta_t))**2)
    mse = np.mean((y-predict)**2)
    print('mse_true={} mse={}'.format(mse_t, mse))
def cvx_objpert(X, y, eps, delta):
    """

    :param X: Input X with n records
    :param Y: {1 or -1}
    :param eps:
    :param delta:
    :return: logistic parameter theta in k class
    prepocession dataset into num_cls classes (one vs all model), in each class y in (1, -1)

    to compute objpert for logistic regression
    L (lipschiz) = 1
    lamda = 1/4 (strong convex term)
    Lamda = 2*lamda/eps
    """
    theta = []
    num_cls = len(np.unique(y))
    theta = []
    new_col = np.ones([len(X), 1])
    X = np.append(X, new_col, axis=1)  # add bias term
    dim = X.shape[1]
    n = X.shape[0]
    L = 1
    lamda = 1/4
    Lamda = 2*lamda/eps
    b = np.sqrt(8*np.log(2/delta) + 4*eps)*L/eps * np.random.standard_normal(dim)

    for idx, cls in enumerate(np.unique(y)):
        idx = np.where(y == cls)
        cur_y = -np.ones(shape = y.shape)
        cur_y[idx] = 1
        w = Variable(shape = dim)

        loss = sum(logistic(-multiply(cur_y, X*w))) + Lamda/(2)*norm(w,2) +sum(multiply(b,w))
        constraints =[norm(w,2) <=1]
        problem = Problem(Minimize(loss), constraints)
        problem.solve(verbose=True)
        opt = problem.value
      #  print(opt)
        cur_theta = w.value
        #cur_theta = np.expand_dims(cur_theta, axis =0) #add dim for evaluate
        theta.append(cur_theta)
    theta = np.array(theta)
    return theta

