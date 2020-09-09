
from timeit import default_timer as timer
import numpy as np
import importlib
import random
import optGFB
importlib.reload(optGFB)
from optGFB import *
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
def proxsolve(E, um, matTens, Tens, triangles, vertices, fxy, ymeas_noise_coef, x1, vmin, vmax, filename, Ntlevel, SNR, Itr1, Itr2,savefig,MA):
    start=timer()
    tau1=0.95#0.92#0.96#0.90
    tau2=0.05#0.02#0.03#0.09
    tau3=0.01#0.06#0.01#0.01
    step=0.5#0.08
    N=len(E)
    Emax=np.max(E)
    Dm=matTens@um
    D2=(Dm.T)/3
    solmat=np.zeros((N,Itr2))
    
    for j in range(Itr2):#9 for one inclusion with step 0.08#17 for step 0.07
        flag=0
        if (j%20==0): flag=1
        GK0=np.squeeze(Tens@x1/3)#
        sol=optGFB(D2,GK0,triangles,fxy,E,Itr1,x1, step, tau1, tau2, tau3,flag)#No wighting is allowed due to error
        x1=sol
        solmat[:,j]=sol
        if np.all(np.isnan(sol)) or np.mean(sol)>1:#sol.any()==nan:        
            sol=solmat[:,j-1]
            break
        print('iter2=',j)
        #print('flag:',flag)
    print("--- %s seconds ---" % (timer()-start))  
    
    plt.figure(figsize=(10,10))
    x = np.array(vertices[:,0])
    y =  np.array(vertices[:,1])   
    x_new = np.linspace(x.min(),x.max(),N)
    y_new = np.linspace(y.min(),y.max(),N)
    X, Y = np.meshgrid(x_new, y_new)
    E_interp=np.around(griddata((x,y),np.ravel(sol),(X,Y),method='linear'),decimals=10)#+0.1*np.ones(N)
    E_true=np.around(griddata((x,y),np.ravel(E),(X,Y),method='linear'),decimals=10)
    #plt.subplot(121);
    plt.imshow(np.abs(E_interp),vmin=vmin,vmax=vmax)#;plt.title('Reconstructed Young\' modulus ' + r'$\mathbf{\mu}$'+' using \n noisy measurements ('+str(Ntlevel)+'% noise level)',fontsize=15);
    plt.colorbar()
    #plt.subplot(122);plt.imshow(np.abs(E_true),vmin=vmin,vmax=vmax);plt.title('Reconstructed Young\' modulus ' + r'$\mathbf{\mu}$'+' using \n noiseless measurements',fontsize=15);plt.colorbar()
    if savefig: plt.savefig(filename+'noise'+str(Ntlevel)+'snr'+str(SNR)+'E'+str(Emax)+'.png')
    return np.ravel(sol)+0.1*np.ones(N)