import numpy as np

from matrix import Matrix
from utils import empty_matrix


def generate_x_vecotr(n):
    return np.array(list(range(1, n + 1)))


def generate_diagonal_domination_matrix(a, k):
    """
    Генерация матрицы с диагональным преобладанием
    """
    n = len(a)
    A_k = empty_matrix(n)

    for i in range(n):
        t1 = -sum(a[i][k] for k in range(i))
        t2 = -sum(a[i][k] for k in range(i + 1, n))
        t = t1 + t2
        for j in range(n):
            if i != j:
                A_k[i, j] = a[i, j]
            else:
                A_k[i, j] = t + pow(10.0, -k)

    A_k = A_k.tocsr()
    return A_k


def generate_hilbert(n):
    """
    Генерация матрицы Гильберта
    :param n: размер
    :return: Матрица
    """
    A_k = empty_matrix(n)
    for i in range(n):
        for j in range(n):
            A_k[i, j] = 1.0 / (i + j + 1.0)

    return Matrix(content=A_k.tocsr())


def get_f_k(F_k: Matrix):
    result = F_k.content.tocsr() * generate_x_vecotr(F_k.n)
    return result