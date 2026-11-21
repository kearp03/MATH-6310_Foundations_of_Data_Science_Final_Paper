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
    B = np.array(A, dtype=float)
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

threshold = 1e-8
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

def GolubReinschSVD(U,B,V, iterations):
    n,m = B.shape
    iterate = min(n,m)
    for it in range(iterations):
        for i in range(iterate-1):
            if abs(B[i,i+1]) < threshold:
                B[i,i+1]=0
        a,b = checkSize(B)
        if b == 0:
            exit
        for i in range(a,b-1):
            golubKahanSVD(B,i,b)

def givensRotationMatrix(a, b):
    if b == 0:
        c = np.copysign(1.0, a)
        s = 0.0
    elif a == 0:
        c = 0.0
        s = -np.copysign(1.0, b)
    elif abs(a) > abs(b):
        t = b / a
        u = np.copysign(np.sqrt(1 + t * t), a)
        c = 1 / u
        s = -c * t
    else:
        t = a / b
        u = np.copysign(np.sqrt(1 + t * t), b)
        s = -1 / u
        c = t / u
    givenMatrix = np.zeros((2,2))
    givenMatrix[0,0] = givenMatrix[1,1] = c
    givenMatrix[0,1] = -s
    givenMatrix[1,0] = s

    return givenMatrix

def givensRotation(B, i, b, alpha,beta):
    if i == b-2:
        givenMatrix = givensRotationMatrix(alpha, beta)
        B[i:i+2, i:i+2] = np.matmul(B[i:i+2, i:i+2], givenMatrix)
    
        givenMatrix = givensRotationMatrix(B[i,i], B[i+1,i]).T
        B[i:i+2, i:i+2] = np.matmul(givenMatrix, B[i:i+2, i:i+2])
        return
    givenMatrix = givensRotationMatrix(alpha, beta)
    B[i:i+3, i:i+2] = np.matmul(B[i:i+3, i:i+2], givenMatrix)

    givenMatrix = givensRotationMatrix(B[i,i], B[i+1,i]).T
    B[i:i+2, i:i+3] = np.matmul(givenMatrix, B[i:i+2, i:i+3])
    return




def golubKahanSVD(B,i,b):
    if B[i,i] == 0:
        return givensRotation(B,i,b, 0, 1)
    C = np.empty([2, 2])
    C[0,0] = B[b-2,b-3]**2 + B[b-2,b-2]**2
    C[0,1] = B[b-2,b-1]*B[b-2,b-2]
    C[1,0] = np.copy(C[0,1])
    C[1,1] = B[b-1,b-1]**2 + B[b-2,b-1]**2

    eigs = np.linalg.eigvals(C)
    mean = eigs[np.argmin(np.abs(eigs - C[1, 1]))]

    alpha = B[i,i]^2 - mean
    beta = B[i,i]*B[i,i+1]

    return givensRotation(B,i,b,alpha,beta)


    

