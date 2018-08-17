import numpy as np
from numpy.linalg import matrix_power

# We define an adjacency matrix for a graph, set all the weights to 1

G = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

G2 = matrix_power(G, 2)

G3 = matrix_power(G, 3)

G4 = matrix_power(G, 4)
G5 = matrix_power(G, 5)

# Since we just want the paths starting at nodes 0, 1 and 2 we count as follows:
paths1 = np.sum(G[0:3, :])
paths2 = np.sum(G2[0:3, :])
paths3 = np.sum(G3[0:3, :])
paths4 = np.sum(G4[0:3, :])
paths5 = np.sum(G5[0:3, :])