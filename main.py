
import numpy as np

import jax
import jax.numpy as jnp

A = np.array([[4., 1.],
              [1., 3.]])

b = np.array([1., 2.])

# Solve the linear system
x = np.linalg.solve(A, b)

print("Solution:", x)
print("Check Ax:", A @ x)
print("Error:", np.linalg.norm(A @ x - b))




A = jnp.array([[4., 1.],
               [1., 3.]])
b = jnp.array([1., 2.])

# Define loss function
def loss_fn(x):
    return 0.5 * jnp.sum((A @ x - b) ** 2)

# Automatically calculate gradient
grad_fn = jax.grad(loss_fn)

x = jnp.zeros(2)
learning_rate = 0.05

for i in range(200):
    gradient = grad_fn(x)
    x = x - learning_rate * gradient

print("JAX solution:", x)
print("Exact solution:", jnp.linalg.solve(A, b))


