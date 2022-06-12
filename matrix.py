import numpy as np
import scipy
import scipy.sparse as sps

from Operation import lower_trivial_system_solution, upper_trivial_system_solution
from utils import identity_matrix, empty_matrix


class Matrix:
    """
        Главный класс любой матрицы
    """

    def __init__(self, n=None, content: sps.csr_matrix = None, array: [] = None):
        self.n = n
        if content is not None:
            self.content = content
            self.n = content.shape[0]
        if array is not None:
            self.content = sps.csr_matrix(array)
            self.n = len(array)

    def generate_rand(self) -> None:
        """
            Заполнение матрицы случайынми числами
        """
        self.content = sps.rand(self.n, self.n, density=0.1, format='csr', dtype=np.int16)
        self.content.setdiag(1)

    def decomposition_lu(self) -> []:
        """
        Нахождение LU-Разложения
        :return: [L, U]
        """
        n = len(self.content.indptr) - 1

        U = sps.csr_matrix(sps.rand(n, n, density=0.0))
        L = sps.csr_matrix(sps.rand(n, n, density=0.0))
        iteration_counter = 0
        for i in range(n):
            for j in range(n):
                U[0, i] = self.content[0, i]
                L[i, 0] = self.content[i, 0] / U[0, 0]

                s = 0.0

                for k in range(i):
                    s += L[i, k] * U[k, j]
                    iteration_counter += 1
                U[i, j] = self.content[i, j] - s

                if i > j:
                    L[j, i] = 0
                else:
                    s = 0.0
                    for k in range(i):
                        s += L[j, k] * U[k, i]
                    L[j, i] = (self.content[j, i] - s) / U[i, i]
            iteration_counter += 1
        return [Matrix(n=n, content=L), Matrix(n=n, content=U), iteration_counter]

    def system_solution(self, b):
        """
        Найти решение системы Ax=b
        A - невырожденная матрица, хранящаяся в разреженном виде
        b - вектор в правой части
        returns:
        None - если решения не существует
        """
        L, U = [matrix.content for matrix in self.decomposition_lu() if matrix is Matrix]
        return upper_trivial_system_solution(U, lower_trivial_system_solution(L, b))

    def invert(self):
        """
        Инвертирование матрицы
        :return: Новая инвертированная матрица
        """
        B = identity_matrix(self.n)
        X = empty_matrix(self.n)
        for i in range(self.n):
            X[i] = self.system_solution(B.content.getcol(i).toarray())
        return Matrix(content=X.tocsr().transpose())







