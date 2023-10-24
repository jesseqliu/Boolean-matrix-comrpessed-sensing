
import numpy as np
import pdb
import random
import cvxpy as cp
from scipy import io

m=40
n=40
L=m*n
k=2
measure_range=15
nonzero_range=1
test=50
recA=np.zeros((measure_range,nonzero_range,test))
recB=np.zeros((measure_range,nonzero_range,test))
recZ=np.zeros((measure_range,nonzero_range,test))
densiZ=np.zeros((measure_range,nonzero_range,test))
densiZ1=np.zeros((measure_range,nonzero_range,test))
recZ_group=np.zeros((measure_range,nonzero_range,test))
def get_random_matrices(M, N, K, num_measure,p_x_1 = .5, p_y_1 = .5,  p_nonzero=0.01):
    A = (np.random.rand(M, K) < p_x_1).astype(int)
    B = (np.random.rand(N, K) < p_y_1).astype(int)
    Z = (A.dot(B.T) > 0).astype(int)
    H=(np.random.rand(num_measure,M, N) < p_nonzero).astype(int)
    y_true=np.zeros(num_measure)
    for i in range(num_measure):
        y_true[i]=(np.sum(np.multiply(H[i,:,:],Z))>0).astype(int)
    mats = {'A':A, 'B':B, 'Z':Z, 'y_true':y_true,'H':H}
    return mats
def rec_error(Z, Zh):
    return np.sum(np.abs(Z - Zh))/float(np.prod(Z.shape))
def density(X):
    return np.sum(X)/float(np.prod(X.shape))   
         
for jj1 in range (measure_range):
    for jj2 in range(1): 
        for jj3 in range(test):            
            num_measure=50*(jj1+1)
            p_xy=pow((1-pow(0.95,1/3)),0.5)
            key=0
            while (key==0):
                mats=get_random_matrices(m, n, k, num_measure,p_x_1 = 0.05, p_y_1 = 0.05, p_nonzero=0.1)
                if density(mats['Z']) >0:
                    key=1
            
            y_true=mats['y_true']
            H=mats['H']
            a=num_measure
            
            t=cp.Variable((m*k,n))
            A=cp.Variable((m,k))
            B=cp.Variable((k,n))
            Z=cp.Variable((m,n))
            KK=[A>=0,A<=1]
            KK=np.append(KK,[B>=0,B<=1])
            KK=np.append(KK,[t>=0,t<=1])
            KK=np.append(KK,[Z>=0,Z<=1])
            for i1 in range(m):
                for i2 in range(n):
                    tsum=0
                    for i3 in range(k):
                        KK=np.append(KK,[t[i1*k+i3,i2]>=0,A[i1,i3]+B[i3,i2]-1<=t[i1*k+i3,i2],
                                      t[i1*k+i3,i2]<=A[i1,i3],t[i1*k+i3,i2]<=B[i3,i2]])
                        KK=np.append(KK,[Z[i1,i2]>=t[i1*k+i3,i2]])
                        tsum=tsum+t[i1*k+i3,i2]
                    KK=np.append(KK,[Z[i1,i2]<=tsum])
            
            for i1 in range(a):
                Hx=np.nonzero(H[i1,:,:])[0]
                Hy=np.nonzero(H[i1,:,:])[1]
                Zsum=0
                for i2 in range(np.count_nonzero(H[i1,:,:])):
                    Zsum=Zsum+Z[Hx[i2],Hy[i2]]
                if y_true[i1]==0:
                    KK=np.append(KK,Zsum==0)
                else:
                    KK=np.append(KK,Zsum>=1)
                        
            prob = cp.Problem(cp.Minimize(cp.sum(A)+cp.sum(B)),KK)            
            
            ans = prob.solve()
            
            A1=(A.value>0.5).astype(int)
            B1=(B.value>0.5).astype(int)
            Z1 = (Z.value > 0.5).astype(int)
            
            print(ans)
            print("rec_error of A:", rec_error(A1,mats['A']))
            recA[jj1,jj2,jj3]=rec_error(A1,mats['A'])
            print("rec_error of B:", rec_error(B1.T,mats['B']))
            recB[jj1,jj2,jj3]=rec_error(B1.T,mats['B'])
            print("rec_error of Z:", rec_error(Z1,mats['Z']))
            recZ[jj1,jj2,jj3]=rec_error(Z1,mats['Z'])
            print("density of Z:",density(mats['Z']))
            densiZ[jj1,jj2,jj3]=density(mats['Z'])
            print("density of reconsrtuction of Z:",density(Z1))
            densiZ1[jj1,jj2,jj3]=density(Z1)
                        
            H_group=np.zeros((num_measure,L))
            for ii in range(num_measure):
                H_group[ii,:]=mats['H'][ii,:,:].flatten()               
            Z_group=cp.Variable((L))
            KK_group=[Z_group>=0,Z_group<=1]
            for i1 in range(num_measure):
                Hx_group=np.nonzero(H_group[i1,:])[0]
                Zsum_group=0
                for i2 in range(np.count_nonzero(H_group[i1,:])):
                    Zsum_group=Zsum_group+Z_group[Hx_group[i2]]
                if y_true[i1]==0:
                    KK_group=np.append(KK_group,Zsum_group==0)
                else:
                    KK_group=np.append(KK_group,Zsum_group>=1)
            prob = cp.Problem(cp.Minimize(cp.sum(Z_group)),KK_group)
            ans = prob.solve()
            if ans>100000:
                Z2=np.zeros(L)
            else:
                Z2= (Z_group.value > 0.5).astype(int)
            Zvector=mats['Z'].flatten()
            print("rec_error of Z:", rec_error(Z2,Zvector))
            recZ_group[jj1,jj2,jj3]=rec_error(Z2,Zvector)
            
io.savemat('mn40k2together1.mat', {'densiZ': densiZ, 'densiZ1': densiZ1,'recA':recA,'recB':recB,'recZ':recZ,'recZ_group':recZ_group})    