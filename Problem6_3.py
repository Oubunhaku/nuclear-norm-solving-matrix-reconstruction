from random import sample as sample
import numpy as np  
import matplotlib.pyplot as plt
from numpy.linalg import svd,norm

m = 20
n = 40
r = 2
ninc = 100

def sigmoid(X):  
    return 1.0/(1+np.exp(-X))

def sft(X,tao):
    U,d,V = svd(X,full_matrices=False)
    d = np.maximum(d-tao,0)
    D = np.diag(d)
    return np.matmul(np.matmul(U,D),V)

Z = np.matmul(np.random.randn(m,r),np.random.randn(r, n))
Z = sigmoid(Z)
Q = sample(range(m*n), ninc)
A = Z.copy()
mask = np.ones((m,n)).reshape(m*n)
mask[Q] = 0
mask = mask.reshape(m,n)

A = np.multiply(Z,mask)


#Initialize and set hyperparameters
tao = 40
delta = 0.2
pro_M = norm(A,'nuc')
k_0 = np.round(tao/(delta*pro_M))+1
Y_k = delta*k_0*A
Z_k = sft(Y_k,tao)
k = 0

#Iteration  Scheme
while norm(np.multiply((Z_k-Z),mask),'nuc') >1e-4 and k <=400:
    Z_k = sft(Y_k,tao)
    Y_k = Y_k + delta*np.multiply((Z-Z_k),mask)
    k += 1
plt.figure()
plt.imshow(Z)
plt.show()   
plt.imshow(Z_k)
plt.show()  
