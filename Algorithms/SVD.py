#Algorithm 1a: Householder reduction to bidiagonal form:
import numpy as np
def householder(x):
    vector = np.array(x)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    vector[0] += np.copysign(norm, vector[0])
    return vector / np.linalg.norm(vector)

def householderReduction(A):
    B = np.array(A)
    m, n = B.shape
    U = np.identity(m)
    V = np.identity(n)

    for k in range(n):
        u = householder(B[k:, k])
        B[k:,k:] -= 2 * np.outer(u, np.matmul(u,  B[k:, k:]))
        U[:,k:] -= 2 * np.outer(np.matmul(U[:, k:],u), u)
        B[k+1:, k] = 0

        if k< n-2:
            v = householder(B[k, k+1:])
            B[k:,k+1:] -= 2 * np.outer(np.matmul(B[k:, k+1:], v), v)
            V[:,k+1:] -= 2 * np.outer(np.matmul(V[:, k+1:], v), v)
            B[k, k+2:] = 0

    return U,B,V

threshold = 10^-10
def checkSize(B):
    superdiag = np.diagonal(B, offset=1)
    zeroIndex = np.flatnonzero(superdiag == 0)
    nonzeroIndex = np.flatnonzero(superdiag)
    if len(zeroIndex) == 0:
        return 0, -1
    if len(nonzeroIndex) == 0:
        return -1,0 
    
    print(zeroIndex, nonzeroIndex)
    if nonzeroIndex[-1] == len(superdiag) - 1:
        return -1, zeroIndex[-1]
    
    a = np.searchsorted(zeroIndex, nonzeroIndex[-1]) 
    matrix22para = zeroIndex[a-1]+1 
    return matrix22para, nonzeroIndex[-1]+1

def GolubReinschSVD(U,B,V):
    n,m = B.shape
    iterate = min(n,m)
    for i in range(iterate):
        if abs(B[i,i+1]) < threshold:
            B[i,i+1]=0
    a,b = checkSize(B)
    if b == 0:
        exit
    for i in range(iterate):
        golubKahanSVD(B,a,b)
        

def golubKahanSVD(B,a,b):
    C = np.empty([2, 2])
    C[0,0] = B[b-2,b-3]^2 + B[b-2,b-2]^2
    C[0,1] = B[b-2,b-1]*B[b-2,b-2]
    C[1,0] = np.copy(C[0,1])
    C[1,1] = B[b-1,b-1]^2 + B[b-2,b-1]^2

    eigs = np.linalg.eigvals(C)
    mean = eigs[np.argmin(np.abs(eigs - C[1, 1]))]

    

