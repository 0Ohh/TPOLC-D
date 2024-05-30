import numpy as np

graph = np.zeros([800, 800], dtype='int')


with open('graph.txt') as f:
    while 1:
        line = f.readline()
        if line == '':
            break
        a, b = line.split(' ')
        a = int(a)
        b = int(b)
        graph[a][b] = 1
        graph[b][a] = 1


def get_n():
    return 800

def FS(x):
    nx = (1 - x).reshape(-1, 1)
    m1 = np.multiply(x, graph)
    m2 = np.multiply(nx, m1)
    return m2.sum()

