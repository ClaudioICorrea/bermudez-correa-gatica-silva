import numpy as np
from dolfin import *

# Definir la matriz A
A = np.array([[1, 2], [3, 4]])

# Calcular la norma matricial de A
norm_A = np.linalg.norm(A, ord=2)

# Imprimir la norma matricial de A
print("La norma matricial de A es:", norm_A)