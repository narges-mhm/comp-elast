from scipy.interpolate import griddata
import numpy as np
import random
from timeit import default_timer as timer
#import IP
import matplotlib.pyplot as plt
def im2bw(Ig,level):
    S=np.copy(Ig)
    S[Ig > level] = 1
    S[Ig <= level] = 0
    return(S)
def init(um, disp1, E, vertices, ymeas_noise_coef,Randinit,MA,scale):
    start = timer() 
    N=len(E)
    np.random.seed(1)
    varibx0=ymeas_noise_coef*2#
    stdx0=varibx0*np.abs(E)

    umx=um[0::2]
    umy=um[1::2]
    x = np.array(vertices[:,0])
    y =  np.array(vertices[:,1])   
    x_new = np.linspace(x.min(),x.max(),N)
    y_new = np.linspace(y.min(),y.max(),N)
    X, Y = np.meshgrid(x_new, y_new)
    umx_interp=np.around(griddata((x,y),np.ravel(umx),(X,Y),method='linear'),decimals=10)
    #umy_interp=np.around(griddata((x,y),np.ravel(umy),(X,Y),method='linear'),decimals=10)
    I1=im2bw(np.abs(umx_interp),0.12)
    xx, yy = np.indices((I1.shape[0], I1.shape[1]))
    x1=int(I1.shape[0]/2)
    y1=int(I1.shape[1]/2)
    dia=np.where(I1[x1,:]==0)
    r1=(np.max(dia[0])-np.min(dia[0])+20)/2
    mask_circle1 = (xx - x1) ** 2 + (yy - y1) ** 2 < r1 ** 2
    I3=mask_circle1*1
    mask=(1-I1)*I3
    x1=scale*mask+np.random.normal(0,stdx0,N)+0.1*np.ones(N)#++np.ravel(w)
    x00=np.zeros(len(E))
    for j in range(len(E)):
        idx=vertices[j,:]/30e-3*N
        x00[j]=x1[int(np.floor(idx[0]))-1,int(np.floor(idx[1]))-1]

    SNRx0=10*np.log10(np.linalg.norm(x1)**2/np.sum(stdx0**2))
    print('SNRx0:',SNRx0)
    print("--- %s seconds ---" % (timer()-start)) 
    return x00


