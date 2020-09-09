import numpy as np
import random

def addnoise(disp1,ymeas_noise_coef):
    dispx=disp1[0::2]#-np.mean(disp1[0::2])
    dispy=disp1[1::2]#-np.mean(disp1[1::2])
    N=len(dispx)
    M=2*N
    n_iter=1
    np.random.seed(1)
    um=np.zeros(2*N)
    um_mat=np.zeros((M,n_iter))
    Noisemat=np.zeros((M,n_iter))

    xmeas_noise_coef=1.7*ymeas_noise_coef#e-3# the power is 1.7**2=3 times the power in y irection
    stdx=xmeas_noise_coef*np.abs(dispx)
    umx = dispx + np.random.normal(0,stdx,N)
    umx = umx + np.random.normal(0,(xmeas_noise_coef*(max(umx)-min(umx))),N)
    #ymeas_noise_coef=1e-2
    stdy=ymeas_noise_coef*np.abs(dispy)
    umy = dispy + np.random.normal(0,stdy,N)
    umy = umy + np.random.normal(0,(ymeas_noise_coef*(max(umy)-min(umy))),N)
    um[0::2]=umx
    um[1::2]=umy
    SNR=10*np.log10(np.linalg.norm(um)**2/(np.linalg.norm(um-disp1)**2))#np.linalg.norm(nx)**2+np.linalg.norm(ny)**2
    SNR=np.around(SNR,decimals=1)
    #SNR2=10*np.log10(np.sum(um ** 2)/np.sum((um-disp1) ** 2))
    Nxlevel=np.linalg.norm(umx-dispx)/np.linalg.norm(umx)
    Nylevel=np.linalg.norm(umy-dispy)/np.linalg.norm(umy)
    Ntlevel=np.linalg.norm(um-disp1)/np.linalg.norm(um)
    Ntlevel=np.around(Ntlevel*1e2,decimals=1)
    print('SNR:',SNR)
    print('Nx:',Nxlevel)
    print('Ny:',Nylevel)
    print('Nt:',Ntlevel)
    return  um, Ntlevel,SNR