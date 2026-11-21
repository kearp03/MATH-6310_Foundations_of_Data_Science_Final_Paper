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

def orderSVD(U,B,V, iterate):
    diag = np.diag(B).copy()
    U[:, :iterate] *= np.where(diag < 0, -1.0, 1.0)
    diag = np.abs(diag)

    order = np.argsort(diag)[::-1]
    U[:, :iterate] = U[:, order]
    V[:, :iterate] = V[:, order]

    B = np.zeros_like(B)
    B[np.arange(iterate), np.arange(iterate)] = diag[order]
    return U, B, V

def checkSize(B):
    superdiag = np.diagonal(B, offset=1)
    zeroIndex = np.flatnonzero(superdiag == 0)
    nonzeroIndex = np.flatnonzero(superdiag)

    if len(nonzeroIndex) == 0:
            return 0,0
    if len(zeroIndex) == 0:
        return 0, nonzeroIndex[-1]+2
    if nonzeroIndex[-1] == len(superdiag) - 1:
        return zeroIndex[-1] + 1, nonzeroIndex[-1]+2
    
    a = np.searchsorted(zeroIndex, nonzeroIndex[-1]) 
    matrix22para = zeroIndex[a-1] + 1 if a > 0 else 0
    return matrix22para, nonzeroIndex[-1]+2

def golubReinschSVD(U,B,V, iterations, threshold):
    n,m = B.shape
    iterate = min(n,m)
    for it in range(iterations):
        for i in range(iterate-1):
            if abs(B[i,i+1]) < threshold:
                B[i,i+1]=0
        a,b = checkSize(B)
        if b == 0:
            return orderSVD(U,B,V, iterate)
        golubKahanSVD(U,B,V,a,b)
    return orderSVD(U,B,V, iterate)

def givensRotationMatrix(a, b):
    if b == 0:
        c = np.copysign(1, a)
        s = 0
    elif a == 0:
        c = 0
        s = -np.copysign(1, b)
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
    givenMatrix[0,0] = c
    givenMatrix[1,1] = c
    givenMatrix[0,1] = -s
    givenMatrix[1,0] = s


    return givenMatrix

def golubKahanSVD(U,B,V,a,b):
    C = np.empty([2, 2])
    g = B[b-3,b-2] if b-3 >= a else 0.0 

    C[0,0] = g**2 + B[b-2,b-2]**2
    C[0,1] = B[b-2,b-1]*B[b-2,b-2]
    C[1,0] = C[0,1]
    C[1,1] = B[b-1,b-1]**2 + B[b-2,b-1]**2

    eigs = np.linalg.eigvalsh(C)
    mean = eigs[np.argmin(np.abs(eigs - C[1, 1]))]

    alpha = B[a,a]**2 - mean
    beta = B[a,a]*B[a,a+1]

    for j in range(a, b-1):
        lo = max(j-1, a)          
        hi = min(j+3, b)          
        
        givenMatrix = givensRotationMatrix(alpha, beta).T
        B[lo:j+2, j:j+2] = np.matmul(B[lo:j+2, j:j+2], givenMatrix)
        V[:, j:j+2] = np.matmul(V[:, j:j+2], givenMatrix)

        alpha = B[j, j]
        beta  = B[j+1, j]

        givenMatrix = givensRotationMatrix(alpha, beta)
        B[j:j+2, j:hi] = np.matmul(givenMatrix, B[j:j+2, j:hi])
        U[:, j:j+2] = np.matmul(U[:, j:j+2], givenMatrix.T)

        if (j<b-2):
            alpha= B[j,j+1]
            beta = B[j,j+2]

def householderReductionHO(A):
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

def orderHOSVD(U,B, iterate):
    diag = np.diag(B).copy()
    U[:, :iterate] *= np.where(diag < 0, -1.0, 1.0)
    diag = np.abs(diag)

    order = np.argsort(diag)[::-1]
    U[:, :iterate] = U[:, order]

    B = np.zeros_like(B)
    B[np.arange(iterate), np.arange(iterate)] = diag[order]
    return U, B


def golubReinschHOSVD(U,B, iterations, threshold):
    n,m = B.shape
    iterate = min(n,m)
    for it in range(iterations):
        for i in range(iterate-1):
            if abs(B[i,i+1]) < threshold:
                B[i,i+1]=0
        a,b = checkSize(B)
        if b == 0:
            return orderSVD(U,B, iterate)
        golubKahanSVD(U,B,a,b)
    return orderSVD(U,B, iterate)

def givensRotationMatrix(a, b):
    if b == 0:
        c = np.copysign(1, a)
        s = 0
    elif a == 0:
        c = 0
        s = -np.copysign(1, b)
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
    givenMatrix[0,0] = c
    givenMatrix[1,1] = c
    givenMatrix[0,1] = -s
    givenMatrix[1,0] = s


    return givenMatrix

def golubKahanSVD(U,B,V,a,b):
    C = np.empty([2, 2])
    g = B[b-3,b-2] if b-3 >= a else 0.0 

    C[0,0] = g**2 + B[b-2,b-2]**2
    C[0,1] = B[b-2,b-1]*B[b-2,b-2]
    C[1,0] = C[0,1]
    C[1,1] = B[b-1,b-1]**2 + B[b-2,b-1]**2

    eigs = np.linalg.eigvalsh(C)
    mean = eigs[np.argmin(np.abs(eigs - C[1, 1]))]

    alpha = B[a,a]**2 - mean
    beta = B[a,a]*B[a,a+1]

    for j in range(a, b-1):
        lo = max(j-1, a)          
        hi = min(j+3, b)          
        
        givenMatrix = givensRotationMatrix(alpha, beta).T
        B[lo:j+2, j:j+2] = np.matmul(B[lo:j+2, j:j+2], givenMatrix)

        alpha = B[j, j]
        beta  = B[j+1, j]

        givenMatrix = givensRotationMatrix(alpha, beta)
        B[j:j+2, j:hi] = np.matmul(givenMatrix, B[j:j+2, j:hi])
        U[:, j:j+2] = np.matmul(U[:, j:j+2], givenMatrix.T)

        if (j<b-2):
            alpha= B[j,j+1]
            beta = B[j,j+2]