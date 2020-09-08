
from pyunlocbox import functions, solvers
import imp
import Ggradient
imp.reload(Ggradient)
from Ggradient import *
import numpy as np
import Nodeneighbor
imp.reload(Nodeneighbor)
from Nodeneighbor import *
import matplotlib.pyplot as plt
import random
def optGFB(D,GK0,triangles,fxy,E,maxit,x1,step,tau1,tau2,tau3,flag):
####
    N=len(E)

    f2 = functions.func()
    f2._eval = lambda x: 0
    f2._prox = lambda x,T:np.clip(x, 0,0.7)#
###
    T1=Nodeneighbor(triangles,N)
    g = lambda x: Ggradient(x,T1)
    f3 = functions.norm_l1(A=g,At=None, dim=1, y=np.zeros(N),lambda_=tau2)#['EVAL', 'PROX']
######
    yy2=np.ravel(fxy)
####    
    N_cov=np.zeros((2*N,2*N))
    for j in np.arange(0,2*N,2):
        N_cov[j,j]=3
        N_cov[j+1,j+1]=1
    gamma=GK0@N_cov@GK0.T#
    gamma+= np.eye(gamma.shape[1])*1e-5
    gamma_inv=np.linalg.inv(gamma)
    f8=functions.func(lambda_=tau1)
    f8._eval = lambda x: (yy2-D@x).T@gamma_inv@(yy2-D@x)*1e+7
    f8._grad = lambda x: -D.T@gamma_inv@(yy2-D@x)*1e+7
#######
    #step = 0.08#0.5 /tau1# (np.linalg.norm(func(x0),2)**2/np.linalg.norm(x0,2)**2) #0.5/tau#2e3/scale
    solver2 = solvers.generalized_forward_backward(step=step*0.2)#generalized_forward_backward(step=step*0.1)#step*0.1)douglas_rachford
    # without f3 -->singular matrix
    ret2 = solvers.solve([f8,f3,f2], x1, solver2, rtol=1e-15, maxit=maxit)
    objective = np.array(ret2['objective'])
    sol=ret2['sol']
    
    import matplotlib.pyplot as plt

    if flag==1:
        _ = plt.figure(figsize=(10,4))
        _ = plt.subplot(121)
        _ = plt.plot(E, 'o', label='Original E')#-np.ones(N)*0.1
        _ = plt.plot(ret2['sol'], 'xr', label='Reconstructed E')
        _ = plt.grid(True)
        _ = plt.title('Achieved reconstruction')
        _ = plt.legend(numpoints=1)
        _ = plt.xlabel('Signal dimension number')
        _ = plt.ylabel('Signal value')

        _ = plt.subplot(122)
        _ = plt.semilogy(objective[:, 0], '-.',label='l2-norm')#
        _ = plt.semilogy(np.sum(objective, axis=1), label='Global objective')
        _ = plt.grid(True)
        _ = plt.title('Convergence')
        _ = plt.legend(numpoints=1)
        _ = plt.xlabel('Iteration number')
        _ = plt.ylabel('Objective function value')
        _ =plt.show()
    return sol
