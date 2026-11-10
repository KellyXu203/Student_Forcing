
import numpy as np

A = np.array([[4., 1.],
              [1., 3.]])

b = np.array([1., 2.])

# Solve the linear system
x = np.linalg.solve(A, b)

print("Solution:", x)
print("Check Ax:", A @ x)
print("Error:", np.linalg.norm(A @ x - b))

