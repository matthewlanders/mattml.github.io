import numpy as np

gamma = 0.9
R = np.array([10, 5])
P = np.array([[0.8, 0.2], [0.6, 0.4]])
I = np.eye(2)

V = np.linalg.inv(I - gamma * P) @ R
print(V)
