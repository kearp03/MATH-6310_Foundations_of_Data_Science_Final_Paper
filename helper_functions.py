import numpy as np
import math

def tensor_to_matrix(tensor:np.ndarray, mode:int) -> np.ndarray:
    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)

def matrix_to_tensor(matrix:np.ndarray, mode:int, shape:tuple) -> np.ndarray:
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return np.moveaxis(matrix.reshape(full_shape), 0, mode)

def frobenius_norm(A:np.ndarray) -> float:
    return math.sqrt(np.sum(A**2))

def matrix_reconstruction(sigma:np.ndarray, U:np.ndarray, V:np.ndarray, rank:int) -> np.ndarray:
    return U[:,:rank]@np.diag(sigma[:rank])@np.transpose(V[:,:rank])

def mode_wise_product(tensor:np.ndarray, matrix:np.ndarray, mode:int):
    result = np.tensordot(tensor, matrix, axes=([mode], [1]))
    return np.moveaxis(result, -1, mode)

def tensor_reconstruction(S:np.ndarray, U1:np.ndarray, U2:np.ndarray, U3:np.ndarray) -> np.ndarray:
    A = mode_wise_product(S, U1, 0)
    A = mode_wise_product(A, U2, 1)
    return mode_wise_product(A, U3, 2)