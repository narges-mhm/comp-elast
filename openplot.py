from timeit import default_timer as timer
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
def openplot(filename,triname,noiselevel,vmin,vmax, savefig,oneE):
    Data=loadmat(filename)
    solmat=Data['theta']
    sol=solmat[:,-1]*1e-3*2
    Etrue=np.ravel(Data['Esim'])*1e-3*2
    Emax=np.round(np.max(Etrue),decimals=1)
    N=len(Etrue)
    tri=loadmat(triname)
    x = np.array(tri['points'][:,0])#vertices[:,0])
    y =  np.array(tri['points'][:,1])#vertices[:,1])   
    x_new = np.linspace(x.min(),x.max(),N)
    y_new = np.linspace(y.min(),y.max(),N)
    X, Y = np.meshgrid(x_new, y_new)
    fig=plt.figure(figsize=(10,10))
    EGN_interp=np.around(griddata((x,y),np.ravel(sol),(X,Y),method='linear'),decimals=10)
    if oneE:
        plt.imshow(np.abs(EGN_interp),vmin=vmin,vmax=vmax)#;plt.title('Reconstructed Young\' modulus ' + r'$\mathbf{\mu}$'+' using noisy \n measurements ('+str(noiselevel)+'% noise level) with OpenQSEI',fontsize=18);
        plt.colorbar();plt.axis('off')
    else:
        Ntlevel=5.1
        plt.imshow(np.abs(EGN_interp),vmin=vmin,vmax=vmax)#;plt.title('Reconstructed Young\' modulus (' + r'$\mathbf{\mu}_{true}=$'+str(Emax)+') using noisy \n measurements ('+str(Ntlevel)+'% noise level) with OpenQSEI',fontsize=16);
        plt.colorbar();plt.axis('off')
    if savefig: fig.savefig(filename+'.png')#plt.savefig(filename+'.png')
    return sol


