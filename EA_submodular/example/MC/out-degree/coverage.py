import numpy as np

graph = np.zeros([450, 450], dtype='int')
with open('原版frb30-15-1.mis', 'r') as f:
    lines = f.readlines()
    for l in lines:
        a, b = l.split(' ')
        a, b = int(a)-1, int(b)-1
        graph[a, b] = 1
        graph[b, a] = 1
    f.close()

degrees = np.sum(graph, axis=0)   # degree of each node
quadra_costs = np.square(degrees) + 1

def Cqua(x):
    # todo Quadra + 1
    return np.dot(x, quadra_costs).sum()

def C1(x):
    return np.dot(x, degrees+1).sum()

def Card(x):
    return x.sum()

def F(x):
    m = np.multiply(x, graph)
    neis = np.any(m > 0, axis=1).sum()
    return neis



