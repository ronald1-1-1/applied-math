import numpy as np
import numpy as np
import scipy.sparse as sps


def lower_trivial_system_solution(A, b):
    """
    Найти тривиальное решение уравнения Ax=B
    A - нижнетреугольная матрица, хранящаяся в разреженном виде
    b - вектор в правой части
    """
    x = np.zeros(len(b))
    x[0] = b[0]

    for i in range(1, len(b)):
        x[i] = b[i] - A[i] * x

    return x


def upper_trivial_system_solution(A, b):
    """
    Найти тривиальное решение уравнения Ax=B
    A - верхнетреугольная матрица, хранящаяся в разреженном виде
    b - вектор в правой части
    """
    N = len(b)
    x = np.zeros(N)
    x[-1] = b[-1] / A[-1, -1]

    for i in reversed(range(N - 1)):
        x[i] = (b[i] - A[i] * x) / A[i, i]

    return x
