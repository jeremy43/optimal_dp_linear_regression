
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon

import numpy as np
import random
import math

# import the privacy accountant
#from pyDiffPriv import dpacct

from scipy.optimize import minimize_scalar

# Set privacy parameter ahead of time

def nonprivate(X,y,eps,delta):
    H =  np.dot(X.T, X)
    Xy = np.dot(X.T, y)
    theta = np.linalg.solve(H, Xy)
    return theta

def trivial_solution(X,y,eps,delta):
    [n, d] = X.shape
    theta = np.zeros((d,1))
    # tmp = np.dot(X.T, y)
    # tmp2 = np.dot(X.T,X) + np.eye(d)
    # theta = tmp2/tmp
    return theta

def NoisySGD(X,y, eps,delta):
    L  = 1 # by constraining the space to unit ball
    [n,d] = X.shape
    sigma = np.sqrt(32*np.log(n/delta)*np.log(1/delta)) * L /eps
    theta = np.zeros((d,1))
    theta_avg=np.zeros((d,1))

    loss_seq = []
    mov_avg_loss = .0
    beta = 0.001

#    DPobject = dpacct.CGFAcct(500)
    delta = 1e-6

    T = int(np.minimum(n**2,1e8)) # we try not to run more than one hundred million iterations.
    #idx = np.random.randint(0, n, T )

    for i in range(T):
        t=random.randint(0,n-1)
        data = X[t,:].reshape((d,1))
        label = y[t]

        #eta = 0.1/T
        eta = 0.01/math.sqrt((i+1)*(L**2+d*sigma**2))

        l = float(np.dot(data.T,theta)-label)
        grad = l*data
        Z= np.random.standard_normal(d).reshape((d,1))

        theta[:] -= eta * (grad + sigma * Z)

        mov_avg_loss = beta * l**2 + (1-beta)*mov_avg_loss
        est_loss = mov_avg_loss / (1 - (1-beta)**(i+1))
        loss_seq.append(est_loss)

        theta_avg = theta * 1/(i+1) + theta_avg*i/(i+1)

    return theta_avg, loss_seq






def OutPert(X,y,eps,delta):
    #  Best tuning parameter
    [n, d] = X.shape
    L=1 # need to infer L
    #

    lamb  =  np.sqrt(n)*L*np.sqrt(np.log(2/delta))/eps # such that it is at least as large as the noise
    H =  np.dot(X.T, X)+lamb*np.eye(d)
    Xy = np.dot(X.T,y)
    theta = np.linalg.solve(H, Xy)
    thetahat = theta + L*np.sqrt(np.log(2/delta))/eps/lamb * np.random.standard_normal(d)
    return thetahat



def SuffPert(X,y, eps,delta, lamb=0, xbound=1,ybound=1):
    #  Best tuning parameter
    [n, d] = X.shape
    #

    #lamb  =  np.sqrt(n)*L*np.sqrt(np.log(2/delta))/eps # such that it is at least as large as the noise
    Z = np.random.standard_normal((d,d))
    H =  np.dot(X.T, X)+lamb*np.eye(d) + xbound**2 * np.sqrt(np.log(2/delta))/eps*(Z+np.transpose(Z))
    Xy = np.dot(X.T,y) + 2 * ybound * xbound * np.sqrt(np.log(2/delta))/eps * np.random.standard_normal((d))
    theta = np.linalg.solve(H, Xy)
    return theta

def AdaSuffPert(X,y, eps,delta, xbound = 1, ybound = 1):
    #  Best tuning parameter
    [n, d] = X.shape
    #

    # calculate the smallest eigenvalue
    lambmin = np.linalg.cond(np.dot(X.T, X), p=-2) + 3*np.sqrt(np.log(6/delta))/eps - 3*np.log(6/delta)/eps
    lambmin = np.maximum(0.0, lambmin)

    lamb = np.maximum(2*np.sqrt(d)*np.sqrt(np.log(6/delta))/(eps/3) - lambmin, 0.0)

    return SuffPert(X,y,eps*2/3,delta*2/3,lamb=lamb, xbound=xbound, ybound=ybound)


def AdaOPS(X,y, eps,delta, xbound=1, ybound=1):
    #  Best tuning parameter
    [n, d] = X.shape
    logfactor= np.log(6/delta)
    logfactor2= np.log(n)
    lambmin = np.linalg.cond(np.dot(X.T, X), p=-2) + 4 * np.sqrt(logfactor) / eps * np.random.standard_normal(1) - 4 * logfactor / eps
    lambmin = np.maximum(0.0, lambmin)
    epsbar = eps/2 -eps**2/8*(1/logfactor + (1+logfactor)/logfactor)

    C1 = (d/2 + np.sqrt(d*logfactor2) + logfactor2)*np.log(2/delta)/epsbar**2
    C2 = np.log(2/delta)/eps


    def F(lamb):
        return C1*(1+1/(lamb + lambmin))**(2*C2)/(lamb + lambmin) + lamb

    Results = minimize_scalar(F, method='bounded', bounds=(0.0, 10*np.sqrt(d)*np.log(6/delta)/(eps/3)))
    if Results.success:
        lamb = Results.x
    else:
        print("AdaOPS Error:Optimization failed.")
        lamb = np.sqrt(d)*np.sqrt(np.log(6/delta))/(eps/3)

    H =  np.dot(X.T, X)+lamb*np.eye(d)
    Xy = np.dot(X.T,y)
    theta = np.linalg.solve(H, Xy)
    sensitivity = np.log(1+xbound**2/(lamb+lambmin))
    Delta = np.log(1+np.linalg.norm(theta)*xbound) \
            + sensitivity*np.sqrt(logfactor)/(eps/4) * np.random.standard_normal(1)\
            + sensitivity*logfactor/(eps/4)

    L = (np.exp(Delta)-1)
    a= 1/logfactor + (1+ logfactor)/(logfactor*L**2)
    epstilde = (-2 + np.sqrt(4 + 4*a))/(2*a)
    gamma = (lambmin+lamb)*epstilde**2/logfactor/L**2

    return OPS_v2(X,y,gamma,lamb)


def JL_OLS(X,y,eps,delta):
    return None


# -------------------------------------------------
# Calculate privacy while running a fixed (data independent, privately calibrated) randomized algorithm.

# Draw the accuracy, epsilon curve as we change the randomized algorithm


# draw heat map for all combinations of "noise-level, early stopping and so on"

def NoisySGD_v2(X,y,eps,delta):
    # Output solution, and the corresponding eps, delta
    L  = 1 # by constraining the space to unit ball
    [n,d] = X.shape
    sigma = np.minimum(5.0, 5/eps)

    thresh = L

    theta = np.zeros((d,1))
    theta_avg=np.zeros((d,1))

    DPobject = dpacct.CGFAcct(500)

    loss_seq = []
    mov_avg_loss = .0
    beta = 0.001

    T = int(np.minimum(n**2,1e8)) # we try not to run more than one hundred million iterations.
    #idx = np.random.randint(0, n, T )

    for i in range(T):
        t=random.randint(0,n-1)
        data = X[t,:].reshape((d,1))
        label = y[t]

        eta = 1/n
        #eta = 1/math.sqrt((i+1)*(thresh**2+d*sigma**2))

        l = float(np.dot(data.T,theta)-label)
        grad = l*data
        gradnorm= np.linalg.norm(grad,2)
        if gradnorm > thresh:
            grad = grad/gradnorm
        Z= np.random.standard_normal(d).reshape((d,1))

        theta[:] -= eta * (grad + sigma * Z)

        mov_avg_loss = beta * l**2 + (1-beta)*mov_avg_loss
        est_loss = mov_avg_loss / (1 - (1-beta)**(i+1))
        loss_seq.append(est_loss)

        DPobject.update_cgf_subsamplegaussian(1.0 / n, sigma / thresh)

        eps1 = DPobject.get_eps(delta)
        #print(eps1)
        if eps1>eps:
            return theta_avg
        else:
            theta_avg = theta * 1 / (i + 1) + theta_avg * i / (i + 1)
    return theta_avg

# Draw heat map, for an exponential grid of gamma and lambda.

def OPS_v2(X,y, gamma,lamb):
    # Output solution and the corresponding CGF moment?
    # For linear regression it reduces to a univariate Gaussian
    [n, d] = X.shape
    L=1
    H =  np.dot(X.T, X)+lamb*np.eye(d)
    Xy = np.dot(X.T,y)
    theta = np.linalg.solve(H, Xy)
    Lf = np.linalg.cholesky(H)
    # Lf * Lf.T  = H
    z = np.linalg.solve(Lf, np.random.standard_normal(d))

    thetahat = theta + gamma*z

    return thetahat

    # need to do SGD

