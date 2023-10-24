# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 20:35:15 2023

@author: LQ
"""

import numpy as np
import sys
import pdb
import pdb
import random
from scipy import io


class MatrixCompletion(object):
    
    def __init__(self,
                 M,
                 N,
                 K,
                 O_measure, #observed measurement
                 co_measure,#coordinate of measurements
                 num_measure,
                 num_nonzero,
                 sample_co,
                 max_iter,
                 min_sum = True,                 
                 tol = 1e-3,#tolerance for message updates
                 learning_rate = .2, #damping parameter
                  #maximum number of message passing updates
                 verbose = False,
                 p_x_1 = .5, 
                 p_y_1 = .5, 
         
                 p_1_given_1 = 0.99,
                 p_0_given_0 = 0.99, 
                ):
        
        assert(p_x_1 < 1 and p_x_1 > 0)
        assert(p_y_1 < 1 and p_y_1 > 0)
        assert(p_1_given_1 > .5 and p_1_given_1 < 1)
        assert(p_0_given_0 > .5 and p_0_given_0 < 1)                
        
        self.O_measure = O_measure
        self.co_measure=co_measure
        self.sample_co=sample_co
        self.row=np.array(sample_co)[:,0]
        self.col=np.array(sample_co)[:,1]
        self.M,self.N,self.K = M,N,K
        self.verbose = verbose

        #assert(self.K < min(self.M,self.N))
        #if mask is not None:
        #    assert(mask.shape[0] == self.M and mask.shape[1] == self.N)
        #    self.mask = mask.astype(bool)
        #else:
        #    self.mask = np.ones(mat.shape, dtype=bool)
            
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.min_sum = min_sum
        self.num_edges = num_measure*num_nonzero   
        self.num_measure=num_measure
        self.num_nonzero=num_nonzero
        self.update_adj_list()
        
        # will be used frequently
        self.pos_edges = np.nonzero(O_measure)[0]
        self.neg_edges = np.nonzero(1 - O_measure)[0]
        self.range_edges = np.arange(self.num_measure)
        self.cx = np.log(p_x_1) - np.log(1 - p_x_1)
        self.cy = np.log(p_y_1) - np.log(1 - p_y_1)
        self.co1 = np.log(p_1_given_1) - np.log(1. - p_0_given_0) #log(p(1|1)/p(1|0))
        self.co0 = np.log(1. - p_1_given_1) - np.log(p_0_given_0) ##log(p(0|1)/p(0|0))

    
    def init_msgs_n_marginals(self):
        self.marg_x = np.zeros((self.M, self.K))
        self.marg_y = np.zeros((self.N, self.K))
        
        self.in_x = np.zeros((self.num_edges, self.K)) #message going towards variable X: phi in the papger
        self.new_in_x = np.zeros((self.num_edges, self.K)) #the new one        
        self.out_x = np.log((np.random.rand(self.num_edges, self.K)))#/self.M #message leaving variable x: phi_hat in the paper 
        self.in_y = np.zeros((self.num_edges, self.K)) #message leaving variable y: psi in the paper
        self.new_in_y = np.zeros((self.num_edges, self.K))
        self.out_y = np.log(np.random.rand(self.num_edges, self.K))#/self.N #psi_hat in the paper
        
        self.in_z = np.zeros((self.num_edges, self.K)) #gamma in the paper
        self.out_z = np.zeros((self.num_edges, self.K))
        self.out_z1 = np.zeros((self.num_measure,self.num_nonzero*self.K)) #gamma_hat in the paper
        
        
    def update_adj_list(self):
        ''' nbM: list of indices of nonzeros organized in rows
        nbM: list of indices of nonzeros organized in columns
        '''
        '''
        Mnz,Nnz = np.nonzero(self.mask)
        M = self.M
        N = self.N
        nbM = [[] for i in range(M)] 
        nbN = [[] for i in range(N)]

        for z in range(len(Mnz)):
            nbN[Nnz[z]].append(z)
            nbM[Mnz[z]].append(z)

        for i in range(M):
            nbM[i] = np.array(nbM[i], dtype=int)
        for i in range(N):
            nbN[i] = np.array(nbN[i], dtype=int)
            
        self.rows = nbM
        self.cols = nbN
        '''
        self.record=0
    
    def run(self):
        self.init_msgs_n_marginals()
        iters = 1
        diff_msg = np.inf

        while (diff_msg > self.tol and iters <= self.max_iter) or iters < 5:
            self.update_min_sum()#(outX, outY, inZ, outZ, newInX, newInY, posEdges, negEdges,  opt)
            diff_msg = np.max(np.abs(self.new_in_x - self.in_x))
            self.in_x *= (1. - self.learning_rate)
            self.in_x += self.learning_rate * (self.new_in_x)
            self.in_y *= (1. - self.learning_rate)
            self.in_y += self.learning_rate * (self.new_in_y)
            self.update_margs()
            if self.verbose:
                print ("iter %d, diff:%f" %(iters, diff_msg))
                self.record=iters
            else:
                print (".",
                sys.stdout.flush())
                
            iters += 1

        #recover X and Y from marginals and reconstruct Z
        self.X = (self.marg_x > 0).astype(int)
        self.Y = (self.marg_y > 0).astype(int)
        self.Z = (self.X.dot(self.Y.T) > 0).astype(int)

        
        
    def update_min_sum(self):
        self.in_z1 = np.minimum(np.minimum(self.out_x + self.out_y, self.out_x), self.out_y) #gamma update in the paper
        self.in_z=np.zeros((self.num_measure,self.num_nonzero*self.K))
        
        for i in range(self.num_measure):
            self.in_z[i,:]=self.in_z1[self.num_nonzero*i:self.num_nonzero*(i+1),:].reshape(1,self.num_nonzero*self.K)
       
        inz_pos = np.maximum(0.,self.in_z) # calculate it now, because we're chaning inz
        #find the second larges element along the 1st axis (there's also a 0nd! axis)
        inz_max_ind = np.argmax(self.in_z, axis=1)
        inz_max = np.maximum(-self.in_z[self.range_edges, inz_max_ind],0)
        self.in_z[self.range_edges, inz_max_ind] = -np.inf
        inz_max_sec = np.maximum(-np.max(self.in_z, axis=1),0) # update for gamma_hat in the paper
        sum_val = np.sum(inz_pos, axis=1)
        #penalties/rewards for confoming with observations
        sum_val[self.pos_edges] += self.co1
        sum_val[self.neg_edges] += self.co0
        
        tmp_inz_max = inz_max.copy()
        inz_pos =  sum_val[:, np.newaxis] - inz_pos
        
        for k in range(self.num_nonzero*self.K):
            self_max_ind = np.nonzero(inz_max_ind == k)[0]#find the indices where the max incoming message is from k
            tmp_inz_max[self_max_ind] = inz_max_sec.take(self_max_ind)#replace the value of the max with the second largest value
            self.out_z1[:, k] = np.minimum( tmp_inz_max, inz_pos[:,k])#see the update for gamma_hat
            tmp_inz_max[self_max_ind] = inz_max.take(self_max_ind)#fix tmp_iz_max for the next iter
        
        self.out1=(self.out_z1[0,:]).reshape(self.num_nonzero,self.K)
        for i in range(self.num_measure-1):
            self.out1=np.row_stack((self.out1,self.out_z1[i+1,:].reshape(self.num_nonzero,self.K)))
        self.out_z=self.out1
        # update in_x and in_y: phi_hat and psi_hat in the paper
        self.new_in_x = np.maximum(self.out_z + self.out_y, 0) - np.maximum(self.out_y,0)
        self.new_in_y = np.maximum(self.out_z + self.out_x, 0) - np.maximum(self.out_x,0)

    

    def update_margs(self):
        #updates for phi and psi
        for m in range(self.M):
            row_index=np.where(self.row==m)            
            self.marg_x[m,:] = np.sum(self.in_x.take(row_index,axis=0), axis=1) + self.cx
            self.out_x[row_index, :] = -self.in_x.take(row_index,axis=0) + self.marg_x[m,:]

        for n in range(self.N):
            col_index=np.where(self.col==n)
            self.marg_y[n, :] = np.sum(self.in_y.take(col_index, axis=0), axis=1) + self.cy
            self.out_y[col_index, :] = -self.in_y.take(col_index, axis=0) + self.marg_y[n,:]
def log_ratio(x):
    return np.log(x) - np.log(1. - x)

def get_random_matrices(M, N, K, p_x_1 = .5, p_y_1 = .5, p_flip = 0,p_observe=0.1, p_measure = .1,num_nonzero=2):
    X = (np.random.rand(M, K) < p_x_1).astype(int)
    Y = (np.random.rand(N, K) < p_y_1).astype(int)
    Z = (X.dot(Y.T) > 0).astype(int)
    mask = np.random.rand(M,N) < p_observe
    O = Z.copy()
    sample_list=np.empty(shape=(M*N),dtype=tuple)
    num_measure=int(p_measure*M*N)
    co_measure=np.empty(shape=(num_measure,num_nonzero),dtype=tuple)
    O_measure=np.zeros(num_measure)
    for i in range (M):
        for j in range(N):
            sample_list[i*M+j]=(i,j)
    sample_list=list(sample_list)
    sample_co = random.sample(sample_list,num_nonzero*num_measure)
    for i in range(num_measure):
        co_measure[i,]=sample_co[num_nonzero*i:num_nonzero*(i+1)]
        for j in range(num_nonzero):
            O_measure[i]=O_measure[i]+Z[co_measure[i,j]]
    O_measure=(O_measure>0).astype(int)
    flip = np.random.rand(M,N) < p_flip
    O[flip] = 1 - O[flip]
    mats = {'X':X, 'Y':Y, 'Z':Z, 'O':O, 'mask':mask, 'sample_co':sample_co,'co_measure':co_measure,'O_measure':O_measure}
    return mats


def hamming(X,Y):
    return np.sum(np.abs(X - Y) > 1e-5)

def density(X):
    return np.sum(X)/float(np.prod(X.shape))


def rec_error(Z, Zh):
    return np.sum(np.abs(Z - Zh))/float(np.prod(Z.shape))


def read_csv(fname, delimiter = ','):
    mat = np.genfromtxt(fname, delimiter=delimiter)
    return mat


test=10
nonzero_range=10
px=13
nn=10

M = 200
N = 200
#rank
K = (i1+3)
max_iter=200

p_measure=0.005*(i+1)
p_flip = 0
num_nonzero=jjj+13
p_observe = .1
p_x_1=0.05
p_y_1=0.05

num_measure=int(p_measure*M*N)

mat_dic = get_random_matrices(M, N, K,p_x_1 = p_x_1, p_y_1 = p_y_1, p_flip = p_flip, p_observe = p_observe, p_measure=p_measure,num_nonzero=num_nonzero)


print ("running inference")

comp = MatrixCompletion(M,N,K,mat_dic['O_measure'],mat_dic['co_measure'],num_measure,num_nonzero,mat_dic['sample_co'], min_sum = True, verbose = True,p_x_1 = p_x_1, p_y_1 = p_y_1,max_iter=max_iter)
comp.run()


print (rec_error(comp.Z, mat_dic['Z'])
