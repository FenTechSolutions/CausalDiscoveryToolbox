'''
Created on 2013/01/26

@author: myamada
'''
import numpy as np

def kernel_Delta_norm(xin1, xin2):
    n1 = xin1.shape[1]
    n2 = xin2.shape[1]

    K = np.zeros((n1,n2))
    ulist = np.unique(xin1)

    for ind in ulist:
        c1 = np.sqrt(np.sum(xin1 == ind))
        c2 = np.sqrt(np.sum(xin2 == ind))
        ind1 = np.where(xin1 == ind)[1]
        ind2 = np.where(xin2 == ind)[1]
        K[np.ix_(ind1,ind2)] = 1/c1/c2

    return K

def kernel_Delta(xin1, xin2):
    n1 = xin1.shape[1]
    n2 = xin2.shape[1]

    K = np.zeros((n1,n2))
    ulist = np.unique(xin1)

    for ind in ulist:
        ind1 = np.where(xin1 == ind)[1]
        ind2 = np.where(xin2 == ind)[1]
        K[np.ix_(ind1,ind2)] = 1

    return K

def kernel_Gaussian(xin1,xin2,sigma):
    
    n1 = xin1.shape[1]
    n2 = xin2.shape[1]
    
    xin12 = np.sum(np.power(xin1,2),0)
    xin22 = np.sum(np.power(xin2,2),0)
    
    dist2 = np.tile(xin22,(n1,1)) + np.tile(xin12,(n2,1)).transpose() - 2*np.dot(xin1.T, xin2)
    K = np.exp(-dist2/(2*np.power(sigma,2)))
    
    return K
