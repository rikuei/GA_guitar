import numpy as np



ws = np.matrix('0 0 0 7 0 0;'
               '0 10 0 0 0 0;'
               '0 8 0 0 0 0;'
               '0 7 0 0 0 0;'
               '0 7 9 8 0 9;'
               '0 0 7 0 0 7;'
               '0 5 0 0 0 0;'
               '0 7 0 0 0 0;'
               '0 0 7 0 0 0;'
               '0 0 0 0 0 3;'
               '0 0 2 2 0 2;'
               '0 5 0 0 0 0;'
               '0 3 2 4 2 0;'
               '0 7 0 0 0 0')




nzero = np.nonzero(ws)
index = [[0,0] for _ in range(len(nzero[0]))]
count = 0
for i in nzero[0]:
    index[count][0]=i
    count += 1
count = 0
for i in nzero[1]:
    index[count][1]=i
    count += 1

ws = np.asarray(ws)

print(index)

DNA_SIZE = len(index)            # DNA length
POP_SIZE = 100           # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)
MUTATION_RATE = 0.003
N_GENERATIONS = 200


def get_fitness(pred): return pred + 1e-3 - np.min(pred)

pop = np.random.randint(1, 5, size=(POP_SIZE, DNA_SIZE))   # initialize the pop DNA

def F(pop):
    pf, mf, rf, lf = [], [], [], []
    count = 0
    print(pop[0])
    for i_ in range(POP_SIZE):
        for j_ in range(DNA_SIZE):
            ind = index[j_]
            #print(ind)
            if pop[i_][j_] == 1:
                pf.append([i_, ind[1], ws[ind[0],ind[1]]])
            elif pop[i_][j_] == 2:
                mf.append([i_, ind[1], ws[ind[0],ind[1]]])
            elif pop[i_][j_] == 3:
                rf.append([i_, ind[1], ws[ind[0],ind[1]]])
            else:
                lf.append([i_, ind[1], ws[ind[0],ind[1]]])
    print(pf)
    mp, mm, mr, ml = 0, 0, 0, 0
    ep, em, er, el = [], [], [], []
    epp, emm, err, ell= [], [], [], []
    for i in range(99):
        fret= 0
        str = 0
        while pf[mp][0]==i:
            ap = list(map(lambda x: x[0] - x[1], zip(pf[mp], pf[mp + 1])))
            ap = list(map(abs, ap))
            if ap[0] == 0:
                str = str + ap[1]
                fret= fret + ap[2]
            mp += 1
        ep.append(str+fret)
    for i in range(len(ep)):
        epp.append([np.argsort(ep)[::1][i],np.sort(ep)[::1][i]])
    print(epp)

    fit=0
    for i in range(100) :
        for j in range(100):
            if pf[i][0]==i:
                c0 = pf[j+1][1]
                c1 = pf[j][1]
                d0 = pf[j+1][2]
                d1 = pf[j][2]
                fit= fit+abs(c0 - c1) + abs(d0 - d1)
    for i in range(100):
        for j in range(100):
            if mf[i][0]==i:
                c0 = mf[j+1][1]
                c1 = mf[j][1]
                d0 = mf[j+1][2]
                d1 = mf[j][2]
                fit= fit+abs(c0 - c1) + abs(d0 - d1)
    for i in range(100):
        for j in range(100):
            if rf[i][0]==i:
                c0 = rf[j+1][1]
                c1 = rf[j][1]
                d0 = rf[j+1][2]
                d1 = rf[j][2]
                fit= fit+abs(c0 - c1) + abs(d0 - d1)
    for i in range(100):
        for j in range(100):
            if lf[i][0]==i:
                c0 = lf[j+1][1]
                c1 = lf[j][1]
                d0 = lf[j+1][2]
                d1 = lf[j][2]
                fit= fit+abs(c0 - c1) + abs(d0 - d1)
    return fit

def select(pop, fit):
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]

def crossover(parent, pop):
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)                             # select another individual from pop
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)   # choose crossover points
        parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
    return parent


def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child

for _ in range(N_GENERATIONS):
    F_values = pop  # compute function value by extracting DNA

    fitness = F(pop)
    print(fitness)
    print("Most fitted DNA: ", pop[np.argmax(fitness), :])
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child