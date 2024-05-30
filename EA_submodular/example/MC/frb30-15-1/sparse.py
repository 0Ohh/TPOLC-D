import numpy as np


edges = []
with open('frb30-15-1_sparse.mis', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if np.random.rand() > 0.3:
            edges.append(line)
    f.close()

with open('frb30-15-1_sparse.mis', 'w') as f:
    for e in edges:
        f.write(e)
    f.close()


