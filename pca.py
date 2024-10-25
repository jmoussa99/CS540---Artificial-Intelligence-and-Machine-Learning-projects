# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 18:42:53 2020

@author: Jamal Moussa
"""
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
   dataset = loadmat(filename)
   
   x = dataset['fea']
   
   x = np.array(x)
   u = np.mean(x, axis=0)
   x = x - u
   
   return x

def get_covariance(x):
    n = len(x)
    cov = np.dot(np.transpose(x), x)

    return cov / (n - 1)

"""adopted from https://stackoverflow.com
/questions/8092920/sort-eigenvalues-and-
associated-eigenvectors-after-using-numpy-linalg-eig-in-pyt"""
def get_eig(S,m):
    
    evals, evectors = np.linalg.eigh(S)
    i = evals.argsort()[-m:][::-1]
    evals = np.diag(evals[i])
    evectors = evectors[:,i]
    
    return (evals, evectors)
    
def project_image(image, U):
    k = len(U[0])

    proj = 0
    for i in range(k):
        proj += np.dot(image,U[:,i]) * U[:,i]
        
    return proj

def display_image(orig, proj):
    orig = np.transpose(orig.reshape((32,32)))
    proj = np.transpose(proj.reshape((32,32)))
    
    fig, ax = plt.subplots(1,2)
    
    ax[0].set_title("Original")
    orig = ax[0].imshow(orig, aspect='equal')
    
    ax[1].set_title("Projection")
    proj = ax[1].imshow(proj, aspect='equal')
    
    fig.colorbar(orig)
    fig.colorbar(proj)
    
    plt.show()
    