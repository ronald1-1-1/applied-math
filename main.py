import jacobi
from data_gen import generate_hilbert 
import numpy as np

matrix = np.matrix([[2, 1, 2], [1, 3, 0], [2, 0, 3]])
#matrix = np.matrix([[100, 100, 100, 100, 100], [100, 50, 17, -14, 100], [100, 14, 50, 10, 100], [100, -14, 10, 50, 100], [100, 100, 100, 100, 100]])
print(generate_hilbert(3).toarray())
# lam, x = jacobi.jacobi(matrix, 1.0e-18)
# print(x)
# print(lam)
