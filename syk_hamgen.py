#!/usr/bin/env python
# coding: utf-8

# In[1]:


#! /usr/bin/env python
import numpy as np
import scipy.linalg as LA
import scipy.optimize as SO
import random
import math
import itertools
import statistics
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import tqdm 
#print("MAKE SURE YOU CHANGE THE FILE NAME")
#input("Press enter to continue")

def expectation(rand_vec,rand_mat):
    return np.vdot(rand_vec,np.dot(rand_mat,rand_vec))/(LA.norm(rand_vec))**2


# In[2]:


def herm(ww):
    if(np.allclose(ww, np.conj(ww).T)):
        return "Hermitian"
    elif(np.allclose(ww, -1* np.conj(ww).T)):
        return "Anti Hermitian"
    else:
        return "Not hermitian"


# In[3]:


def check_sparcity(a):
    k = np.hstack(a)
    mat = np.zeros(2**N)
    for i in range(len(k)):
        if abs(k[i])>1e-5:
            mat[i] = 1
    nonzero = sum(mat)
    return nonzero/len(k)


# In[4]:


#

def create_gamma(dim):
    sig1 = np.array([[0,1],[1,0]])
    sig2 = np.array([[0 , -1j],[1j , 0]])
    sig3 = np.array([[-1 , 0j ],[0j, 1]])
    gamma = []
    gamma.append(sig1)
    gamma.append(sig2)
    for i in range(2,dim-1,2):
        for l in range(i):
            gamma[l] = np.kron(gamma[l],sig3)
        iden = np.identity(2**int(i/2))
        gamma.append(np.kron(iden,sig1))
        gamma.append(np.kron(iden,sig2))
    return gamma


# In[5]:


def create_free_H(N,gamma,stdev):
    H = np.zeros([2**int(N/2),2**int(N/2)])
    for i in itertools.combinations(gamma,q):
        ans = np.identity(2**int(N/2))
        if not (len(i) == q):
            print("ERROR")
            exit(0)
        for k in i:
            ans = np.matmul(ans,k)
        ans = random.gauss(0,stdev)*ans
        H = np.add(H,ans)
    return H


# In[6]:


def add_coupling_H(N,gamma,free,g,s,J=10):
    if g == 0:
      return free
    bs = []
    if s[0]:
        bs = np.array([1,0])
    else:
        bs = np.array([0,1])

    for k in range(1,len(s)):
        if s[k]:
            bs = np.kron(bs,np.array([1,0]))
        else:
            bs = np.kron(bs,np.array([0,1]))   
            

    #fil = 'bs_'+str(N)+'_g'+str(g)
    #np.save(fil,bs)
    modif = np.zeros([2**int(N/2),2**int(N/2)])
    for k in range(int(N/2)):
        if s[k]:
            modif = np.add(modif,-1*np.matmul(gamma[2*k - 1],gamma[2*k]))
        else:
            modif = np.add(modif,np.matmul(gamma[2*k - 1],gamma[2*k]))

    H = np.add(free,-1j*g*J*modif)
    return H


# In[7]:


def test(N):
    gamma = create_gamma(N)
    length =  sum(1 for _ in itertools.combinations(gamma,q))
    ran_num = 10.*stdev*np.random.random(length,) - 5.*stdev
    prob = np.exp(-1.*np.sum(np.square(ran_num))/((stdev**2)) + 150*N)
    print(-1.*np.sum(np.square(ran_num))/((stdev**2)) + 150*N)
    print(prob)


# In[8]:


q = 4

def answer(gamma,N,gg,s=None,J=10):
    stdev = math.sqrt(12.*(J**2)/(N**3))
    iterations = 10*int(113379904./(N**6))
    #iterations = 5
    print(N,gg,J,stdev,iterations)
    def eigs_f(N):
        free = create_free_H(N,gamma,stdev)
        H = add_coupling_H(N,gamma,free,gg,s)
        eigs = LA.eigvalsh(H,eigvals=(0,3))
        return eigs
    
    return np.mean([eigs_f(N) for _ in range(iterations)],axis=0)    

def J_variation(Jlist):
    for N in range(20,22,2):
      gamma = create_gamma(N)
      for jj in Jlist:
        fil = 'syk_eigs_'+str(N)+'_J'+str(jj)
        eigs = answer(gamma,N,0,J=jj)
        np.save(fil,eigs)

def plots(g):

    for N in range(22,24,2):
        print('--------------------------------')
        print(N)
        gamma = create_gamma(N)
        for gg in g:
            s = np.random.randint(2,size =int(N/2))

            #Saving the spin state

            bs = []
            if s[0]:
                bs = np.array([1,0])
            else:
                bs = np.array([0,1])

            for k in range(1,len(s)):
                if s[k]:
                    bs = np.kron(bs,np.array([1,0]))
                else:
                    bs = np.kron(bs,np.array([0,1]))   
                    

            fil = 'bs_'+str(N)+'_g'+str(gg)
            np.save(fil,bs)
            #

            fil = 'syk_eigs_'+str(N)+'_g'+str(gg)
            eigs = answer(gamma,N,gg,s)
            np.save(fil,eigs)
    

#plots([-0.5,-0.4,-0.3,-0.2,-0.1,0.,0.1,0.2,0.3,0.4,0.5])
#plots([-0.6,-0.7,-0.8,-0.9,0.6,0.7,0.8,0.9])
k = [0.7 + 0.01*i for i in range(0,10,1)]
#plots(k)
J_variation(k)

#plots([-0.11,-0.12,-0.13,-0.14,-0.15,-0.17,-0.18,-0.19])
#plots([0.005,-0.004,-0.003,-0.002,-0.001,0.001,0.002,0.003,0.004,0.005])
#plots([0.05,0.04,0.4,0.5])
#J_variation([0.051,0.052,0.053,0.054,0.055,0.056,0.057,0.058,0.059])
#J_variation([4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9])
#J_variation([5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9])
#J_variation([0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89])
#J_variation([1,2,3,4,5,6,7,8,9])
#J_variation([10,20,30,40,50,60,70,80,90])
#J_variation([60,70,80,90])

def verify_orthogonality(v):
    k = v.T
    for i in itertools.combinations(k,2):
        if np.vdot(k[0],k[1]) < 1e-10:
            pass
        else:
            print("Not Orthogonal",k[0],k[1])
            return
    print("Orthogonal")
def check_for_variational_principle(coup,itera=100000):
    free,prob = create_free_H(gamma)
    H = add_coupling_H(free,coup)
    print(herm(H))
    w,v = LA.eigh(H)
    verify_orthogonality(v)
    for _ in range(itera):
        rand = np.random.rand(2**int(N/2))
        if expectation(rand,H)< w[0]:
            print("Error in",_,"and",rand,(expectation(rand,H)-w[0])/w[0])
            print("G",w[0])
            print(v[:,0],np.vdot(rand,v[:,0]))
            break
    print("Success")
    print("G",w[0])
    print("-----------------------------------------------------")






