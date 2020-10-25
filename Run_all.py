### Runing Code

from calc_Ae import *
from genBe import *
from global_stiffness3 import *
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from Meshneighbor import *
from Ggradient import *
from Nodeneighbor import *
from boundaries2 import *
from boundariesTens import *
from timeit import default_timer as timer
from scipy.io import loadmat
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
#from matplotlib.mlab import griddata
import scipy.io as sio
from pyunlocbox import functions, solvers
from scipy.fftpack import dct
import imp
import random
import numpy as np
import importlib
import openfile;importlib.reload(openfile);from openfile import *
import addnoise;importlib.reload(addnoise);from addnoise import *
import init;importlib.reload(init);from init import *
import proxsolve;importlib.reload(proxsolve);from proxsolve import *
import openplot;importlib.reload(openplot);from openplot import *
import calc_cnr;importlib.reload(calc_cnr);from calc_cnr import *
import matlab_cnr;importlib.reload(matlab_cnr);from matlab_cnr import *;

## Oneinc5E Simmulation Results
### reconstruction vs. noise level
import numpy as np
import importlib
import openfile;importlib.reload(openfile);from openfile import *
import addnoise;importlib.reload(addnoise);from addnoise import *
import init;importlib.reload(init);from init import *
import proxsolve;importlib.reload(proxsolve);from proxsolve import *
import openplot;importlib.reload(openplot);from openplot import *
filename='10Nd_E5'
Randinit=0
scale=0.5
MA=0
Itr1=200#400
Itr2=500
savefig=0
vmin=0
vmax=0.6#
#ncoeff=[1e-3,5e-3,1e-2,1.5e-2,2e-2,2.5e-2,3e-2]#5e-4
ncoeff=[1e-3]
E, disp1, Tens, matTens, triangles, vertices, fxy=openfile(filename)
N=len(E)
solmatf=np.zeros((N,len(ncoeff)))
pnoiselevel=np.zeros(len(ncoeff))
psnr=np.zeros(len(ncoeff))
for j, ymeas_noise_coef in enumerate(ncoeff):
    print('j is:',j)
    um, Ntlevel,SNR=addnoise(disp1,ymeas_noise_coef)
    pnoiselevel[j]=Ntlevel
    psnr[j]=SNR
    x1=init(um, disp1, E, vertices, ymeas_noise_coef,Randinit,MA,scale)
    solmatf[:,j]=proxsolve(E, um, matTens, Tens, triangles, vertices, fxy, ymeas_noise_coef, x1, vmin,vmax,filename, Ntlevel, SNR, Itr1, Itr2,savefig,MA)#
sio.savemat('one5E_sol_Itr1'+str(Itr1)+'Itr2'+str(Itr2)+'step03.mat', {'sol': solmatf,'pnl':  pnoiselevel,'psnr': psnr})

Data=loadmat('one5E_sol_img2bw0.12_MA0.mat')
solmat=Data['sol']
pnl=Data['pnl'][0]
psnr=Data['psnr'][0]
x = np.array(vertices[:,0])
y =  np.array(vertices[:,1])   
x_new = np.linspace(x.min(),x.max(),N)
y_new = np.linspace(y.min(),y.max(),N)
X, Y = np.meshgrid(x_new, y_new)
savefig=1
for j in range(len(pnl)):
    sol=solmat[:,j]
    Ntlevel=pnl[j]
    snr=psnr[j]
    Emax=np.max(E)
    E_interp=np.around(griddata((x,y),np.ravel(sol)+0.1*ones(len(E)),(X,Y),method='linear'),decimals=10)
ncoeff=[1e-3,5e-3,1e-2,1.5e-2,2e-2,2.5e-2,3e-2]
pnoiselevel=Data['pnl']
import calc_cnr;importlib.reload(calc_cnr);from calc_cnr import *
oneE=1
cnrp,rmsp=calc_cnr(E,solmat,oneE)
print('cnrp',cnrp)
print('rmsp',rmsp)

### reconstruction vs. noise level matlab & performance computation (CNR,RMS)

import matlab_cnr;importlib.reload(matlab_cnr);from matlab_cnr import *;
directory = './matlabsim/E5tv/'
directory0 = './matlabsim/E5notv/'
cnrm,rmsm=matlab_cnr(directory,1)#thresh=0.4
cnrm0,rmsm0=matlab_cnr(directory0,0)##thresh=0.2
print('cnrm',cnrm)
print('rmsm',rmsm)
print('cnrm0',cnrm0)

#sio.savemat('E5comp.mat', {'cnrp':cnrp,'rmsp':rmsp,'cnrm':cnrm,'rmsm':rmsm,'cnrm0':cnrm0,'rmsm0':rmsm0})
triname='./matlabsim/tri'
savefig=0
oneE=1
mnoiselevel=pnoiselevel[0]
for j,file in enumerate(os.listdir(directory)):
    filename=directory+file
    #print(filename)
    #openplot(filename,triname,mnoiselevel[j],vmin,vmax, savefig,oneE)

### Matlab & Python performance comparison
import calc_cnr;importlib.reload(calc_cnr);from calc_cnr import *
from scipy import interpolate
oneE=1
cnrp,rmsp=calc_cnr(E,solmat,oneE)
xnew = np.linspace(mnoiselevel.min(), mnoiselevel.max(), 50) 
a1_BSpline = interpolate.make_interp_spline(mnoiselevel,cnrp)
cnrp_new = a1_BSpline(xnew)
a2_BSpline = interpolate.make_interp_spline(mnoiselevel,cnrm)
cnrm_new = a2_BSpline(xnew)
a3_BSpline = interpolate.make_interp_spline(mnoiselevel,cnrm0)
cnrm0_new = a3_BSpline(xnew)

_ =plt.figure(figsize=(13,3))
_ = plt.subplot(121)
_ = plt.plot(xnew,cnrp_new,'--',label='statistiical_tv')
_ = plt.plot(xnew,cnrm_new,'-.',label='OpenQSEI_tv')
_ = plt.plot(xnew,cnrm0_new,'-',label='OpenQSEI_ws')
_ = plt.xlabel('noise level %');plt.ylabel('CNR(dB)')
_ = plt.legend(numpoints=1,fontsize=8);plt.title('Contrat to noise ratio for one inclusion with ' + r'$\mathbf{\mu}$'+'=50 KPa',fontsize=10)
_ = plt.grid(True)

a1_BSpline = interpolate.make_interp_spline(mnoiselevel,rmsp)
rmsp_new = a1_BSpline(xnew)
a2_BSpline = interpolate.make_interp_spline(mnoiselevel,rmsm)
rmsm_new = a2_BSpline(xnew)
a3_BSpline = interpolate.make_interp_spline(mnoiselevel,rmsm0)
rmsm0_new = a3_BSpline(xnew)

_ = plt.subplot(122)
_ = plt.plot(xnew,rmsp_new,'--',label='statistical_tv')
_ = plt.plot(xnew,rmsm_new,'-.',label='OpenQSEI_tv')
_ = plt.plot(xnew,rmsm0_new,'-',label='OpenQSEI_ws')
_ = plt.xlabel('noise level %');plt.ylabel('RMS');plt.legend(numpoints=1,fontsize=8);plt.title('RMS error for one inclusion with ' + r'$\mathbf{\mu}$'+'=50 KPa',fontsize=10)
_ =plt.grid('on')
_ =plt.show()



#########################################
## Oneinc multi-E Reconstruction
import numpy as np
import importlib
import openfile;importlib.reload(openfile);from openfile import *
import addnoise;importlib.reload(addnoise);from addnoise import *
import init;importlib.reload(init);from init import *
import proxsolve;importlib.reload(proxsolve);from proxsolve import *
import openplot;importlib.reload(openplot);from openplot import *
directory='./EData/'
Randinit=0
#ma=[0]
MA=0
Itr1=1250#400
Itr2=15
savefig=0
vmin=0
vmax=0.6
ymeas_noise_coef=1e-2
N=730
Emat=np.zeros((N,4))
solmatf=np.zeros((N,4))
nl=np.zeros(4)
for j,file in enumerate(os.listdir(directory)):
    filename=directory+file
    print(filename)
    E, disp1, Tens, matTens, triangles, vertices, fxy=openfile(filename)
    scale=np.max(E)
    um, Ntlevel,SNR=addnoise(disp1,ymeas_noise_coef)
    nl[j]=Ntlevel
    Emat[:,j]=E   
    #vmax=np.max(E)+0.1
    x1=init(um, disp1, E, vertices, ymeas_noise_coef,Randinit,MA,scale)
    solmatf[:,j]=proxsolve(E, um, matTens, Tens, triangles, vertices, fxy, ymeas_noise_coef, x1, vmin,vmax,filename, Ntlevel, SNR, Itr1, Itr2,savefig,MA)#
sio.savemat('3try8_multiE_img2bw0.12_snr'+str(SNR)+'.mat', {'sol': solmatf,'Emat':  Emat,'nl':nl})

### plotting reconstructed E from saved matrix
Data=loadmat('3try8_multiE_img2bw0.15_snr29.1.mat')
solmat=Data['sol']
Emat=Data['Emat']
x = np.array(vertices[:,0])
y =  np.array(vertices[:,1])   
x_new = np.linspace(x.min(),x.max(),N)
y_new = np.linspace(y.min(),y.max(),N)
X, Y = np.meshgrid(x_new, y_new)
savefig=0
Ntlevel=5.1
for j in range(4):
    sol=solmat[:,j]
    E=Emat[:,j]
    Emax=np.round(np.max(E),decimals=1)
    E_interp=np.around(griddata((x,y),np.ravel(sol),(X,Y),method='linear'),decimals=10)
    E_true=np.around(griddata((x,y),np.ravel(E),(X,Y),method='linear'),decimals=10)
    fig=plt.figure(figsize=(8,4))
    plt.subplot(121);plt.imshow(np.abs(E_interp),vmin=vmin,vmax=vmax);plt.title('Reconstructed Young\' modulus (' + r'$\mathbf{\mu}_{true}=$'+str(Emax)+') \n using noisy measurements ('+str(Ntlevel)+'% noise level)',fontsize=4)#;plt.colorbar();plt.axis('off');plt.show()
    #if savefig: fig.savefig('E'+str(Emax)+'snr25.png')
    plt.subplot(122);plt.imshow(np.abs(E_true),vmin=vmin,vmax=vmax)

### reconstruction vs. multiE matlab & performance computation (CNR,RMS)
import calc_cnr;importlib.reload(calc_cnr);from calc_cnr import *
import matlab_cnr;importlib.reload(matlab_cnr);from matlab_cnr import *;
oneE=0
Data=loadmat('3try8_multiE_img2bw0.15_snr29.1.mat')
solmat=Data['sol']
Emat=Data['Emat']
cnrp,rmsp=calc_cnr(Emat,solmat,oneE)
directory = './matlabsim/multiE_matlab/multiE_tv/'
directory0 = './matlabsim/multiE_matlab/multiE_notv/'

#directory0 = './matlabsim/E5notv/'
cnrm,rmsm=matlab_cnr(directory,1)
cnrm0,rmsm0=matlab_cnr(directory0,0)
print('rmsp',rmsp)
print('cnrm',cnrm)
print('rmsm',rmsm)
print('cnrm0',cnrm0)
#print('mnoiselevel',mnoiselevel)
sio.savemat('Emulticomp30.mat', {'cnrp':cnrp,'rmsp':rmsp,'cnrm':cnrm,'rmsm':rmsm,'cnrm0':cnrm0,'rmsm0':rmsm0})
triname='./matlabsim/tri'
savefig=0
#mnoiselevel=pnoiselevel[0]
for j,file in enumerate(os.listdir(directory)):
    filename=directory+file
    openplot(filename,triname,5.1,vmin,vmax, savefig,oneE)

#oneE=0
#cnrp,rmsp=calc_cnr(E,solmat,oneE)
D=loadmat('Emulticomp30.mat')
cnrp=np.ravel(D['cnrp'])
cnrm=np.ravel(D['cnrm'])
cnrm0=np.ravel(D['cnrm0'])
rmsp=np.ravel(D['rmsp'])
rmsm=np.ravel(D['rmsm'])
rmsm0=np.ravel(D['rmsm0'])
from scipy import interpolate
x=np.arange(0.2,0.6,0.1)
xnew = np.linspace(x.min(), x.max(), 10) 
a1_BSpline = interpolate.make_interp_spline(x,cnrp)
cnrp_new = a1_BSpline(xnew)
a2_BSpline = interpolate.make_interp_spline(x,cnrm)
cnrm_new = a2_BSpline(xnew)
a3_BSpline = interpolate.make_interp_spline(x,cnrm0)
cnrm0_new = a3_BSpline(xnew)

_ =plt.figure(figsize=(13,3))
_ = plt.subplot(121)
_ = plt.plot(xnew,cnrp_new,'--',label='Proposed')
_ = plt.plot(xnew,cnrm_new,'-.',label='OpenQSEI_tv')
_ = plt.plot(xnew,cnrm0_new,'-',label='OpenQSEI_ws')
_ = plt.xlabel('Inclusion Young\'s modulus '+ r'$\mathbf{\mu}$'+'(10KPa)');plt.ylabel('CNR(dB)')
_ = plt.legend(numpoints=1,fontsize=8)#;plt.title('Contrat to noise ratio for one inclusion with SNR 35dB (1.8% noise level)',fontsize=10)
#_ = plt.grid(True)

a1_BSpline = interpolate.make_interp_spline(x,rmsp)
rmsp_new = a1_BSpline(xnew)
a2_BSpline = interpolate.make_interp_spline(x,rmsm)
rmsm_new = a2_BSpline(xnew)
a3_BSpline = interpolate.make_interp_spline(x,rmsm0)
rmsm0_new = a3_BSpline(xnew)
_ = plt.subplot(122)
_ = plt.plot(xnew,rmsp_new,'--',label='Proposed')
_ = plt.plot(xnew,rmsm_new,'-.',label='OpenQSEI_tv')
_ = plt.plot(xnew,rmsm0_new,'-',label='OpenQSEI_ws')
_ = plt.xlabel('Inclusion Young\'s modulus '+ r'$\mathbf{\mu}$'+'(10KPa)');plt.ylabel('RMS');plt.legend(numpoints=1,fontsize=8)#;plt.title('RMS error for one inclusion with SNR 35dB (1.8% noise level)',fontsize=10)
_ = plt.grid(True)
_ =plt.show()

