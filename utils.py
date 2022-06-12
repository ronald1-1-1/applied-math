import numpy as np
import scipy


def identity_matrix(n, format="csr"):
    """
    Единичная квадратная матрица размера n в разреженном виде
    :param format: csr по умолчанию
    :param n: размерность матрицы nxn
    :return: матрица E
    """
    from lab3.matrix import Matrix
    return Matrix(content=scipy.sparse.identity(n, format=format))


def empty_matrix(n, format="lil"):
    """
    Пустой нулевой двумерный массив
    :param format: lil по умолчанию
    :param n: количество строк
    :return: []
    """
    mtrx = scipy.sparse.__dict__[format + "_matrix"]
    return mtrx((n, n))


def generate_vector(n):
    return np.random.rand(n)

