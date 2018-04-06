import numpy as np
import itertools
import random
ws = np.array([[0, 7, 6, 7, 5, 0],
               [0, 5, 4, 5, 4, 0],
               [2, 0, 2, 1, 2, 0],
               [0, 5, 6, 6, 0, 5],
               [0, 3, 5, 4, 5, 0]])

nzero = np.nonzero(ws)
print(ws[0][1])
index = [[0,0] for _ in range(len(nzero[0]))]
count = 0
for i in nzero[0]:
    index[count][0]=i
    count += 1
count = 0
for i in nzero[1]:
    index[count][1]=i
    count += 1
index = np.asarray(index)
frets = []
for j in range(len(index)):
    frets.append(ws[[index[j][0]], [index[j][1]]])
frets = list(frets)
print(frets)
print(index)
print (ws[index[1][0]][index[1][1]])