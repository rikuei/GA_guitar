import numpy as np
import itertools
import random

ws = np.array([[0, 7, 6, 7, 5, 0],
               [0, 5, 4, 5, 4, 0],
               [2, 0, 2, 1, 2, 0],
               [0, 5, 6, 6, 0, 5],
               [0, 3, 5, 4, 5, 0]])

nzero = np.nonzero(ws)
index = [[0, 0] for _ in range(len(nzero[0]))]
count = 0
for i in nzero[0]:
    index[count][0] = i
    count += 1
count = 0
for i in nzero[1]:
    index[count][1] = i
    count += 1
frets = []
for j in range(len(index)):
    frets.append(ws[[index[j][0]], [index[j][1]]])
index = np.asarray(index)
judge_dupl = list(nzero[0])
judge_dupl = sum([[judge_dupl.count(e)] for e in set(judge_dupl) if judge_dupl.count(e) > 1], [])
j_d = np.zeros((len(judge_dupl), len(index)))
for i in range(len(judge_dupl)):
    for j in range(judge_dupl[i]):
        j_d[i][sum(judge_dupl[:i]) + j] = 1
j_d = np.array(j_d.astype(bool))

DNA_SIZE = len(index)
POP_SIZE = 10000
CROSS_RATE = 0.5
MUTATION_RATE = 0.02
N_GENERATIONS = 10000

pop = np.random.randint(1, 5, size=(POP_SIZE, DNA_SIZE))

def conv_pop_to_finger(pop_):
    pf, mf, rf, lf = [], [], [], []
    finger_t = []
    for i_ in range(POP_SIZE):
        for j_ in range(DNA_SIZE):
            ind = index[j_]
            if pop_[i_][j_] == 1:
                pf.append([i_, ind[1], ws[ind[0], ind[1]]])
                finger_t.append([[i_], [1, ind[0]]])
            elif pop_[i_][j_] == 2:
                mf.append([i_, ind[1], ws[ind[0], ind[1]]])
                finger_t.append([[i_], [2, ind[0]]])
            elif pop_[i_][j_] == 3:
                rf.append([i_, ind[1], ws[ind[0], ind[1]]])
                finger_t.append([[i_], [3, ind[0]]])
            else:
                lf.append([i_, ind[1], ws[ind[0], ind[1]]])
                finger_t.append([[i_], [4, ind[0]]])
    return pf, mf, rf, lf, finger_t

def get_fitness(f):
    count = 0
    e = []
    for i in range(POP_SIZE):
        fret, str = 0, 0
        while f[count][0] == i:
            if count > len(f) - 2:
                break
            ap = list(map(lambda x: x[0] - x[1], zip(f[count], f[count + 1])))
            ap = list(map(abs, ap))
            if ap[0] == 0:
                str = str + ap[1]
                fret = fret + ap[2]
            count += 1
        e.append((str + fret) ** 2)
    e = np.asarray(e)
    return e

def fix_fitness(f, pop):
    for dna in range(POP_SIZE):
        for dupl in j_d:
            fidx = []
            ddupl = list(pop[dna] * dupl)
            for fin in range(4):
                count = ddupl.count(fin + 1)
                if count > 1:
                    f[dna] = f[dna] * 2
                elif (fin + 1) in ddupl:
                    fidx.append(ddupl.index(fin + 1))
            if len(fidx) == 2:
                if frets[fidx[0]] > frets[fidx[1]] or (frets[fidx[0]] == frets[fidx[1]] and index[fidx[0]][1] < index[fidx[1]][1]):
                    f[dna] = f[dna] * 2
            elif len(fidx) == 3:
                if frets[fidx[0]] > frets[fidx[1]] or (frets[fidx[0]] == frets[fidx[1]] and index[fidx[0]][1] < index[fidx[1]][1]):
                    f[dna] = f[dna] * 2
                elif frets[fidx[0]] > frets[fidx[2]] or (frets[fidx[0]] == frets[fidx[2]] and index[fidx[0]][1] < index[fidx[2]][1]):
                    f[dna] = f[dna] * 2
                elif frets[fidx[1]] > frets[fidx[2]] or (frets[fidx[1]] == frets[fidx[2]] and index[fidx[1]][1] < index[fidx[2]][1]):
                    f[dna] = f[dna] * 2
            elif len(fidx) == 4:
                if frets[fidx[0]] > frets[fidx[1]] or (frets[fidx[0]] == frets[fidx[1]] and index[fidx[0]][1] < index[fidx[1]][1]):
                    f[dna] = f[dna] * 2
                elif frets[fidx[0]] > frets[fidx[2]] or (frets[fidx[0]] == frets[fidx[2]] and index[fidx[0]][1] < index[fidx[2]][1]):
                    f[dna] = f[dna] * 2
                elif frets[fidx[0]] > frets[fidx[3]] or (frets[fidx[0]] == frets[fidx[3]] and index[fidx[0]][1] < index[fidx[3]][1]):
                    f[dna] = f[dna] * 2
                elif frets[fidx[1]] > frets[fidx[2]] or (frets[fidx[1]] == frets[fidx[2]] and index[fidx[1]][1] < index[fidx[2]][1]):
                    f[dna] = f[dna] * 2
                elif frets[fidx[1]] > frets[fidx[3]] or (frets[fidx[1]] == frets[fidx[3]] and index[fidx[1]][1] < index[fidx[3]][1]):
                    f[dna] = f[dna] * 2
                elif frets[fidx[2]] > frets[fidx[3]] or (frets[fidx[2]] == frets[fidx[3]] and index[fidx[2]][1] < index[fidx[3]][1]):
                    f[dna] = f[dna] * 2
    return f

def select(pop, fitness):
    x = (fitness[:, 1] / sum(fitness[:, 1]))
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=x[::-1])
    return pop[idx]

def crossover(parent, pop):
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)
        parent[cross_points] = pop[i_, cross_points]
    return parent


def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            if child[point] == 1:
                child[point] = 2
            elif child[point] == 2:
                child[point] = 3
            elif child[point] == 3:
                child[point] = 4
            else:
                child[point] = 1
    return child

for _ in range(N_GENERATIONS):
    print(_+1)
    pf, mf, rf, lf, finger_t = conv_pop_to_finger(pop)
    pfe = get_fitness(pf)
    mfe = get_fitness(mf)
    rfe = get_fitness(rf)
    lfe = get_fitness(lf)
    f = pfe + mfe + rfe + lfe
    f = fix_fitness(f, pop)
    fitness = []
    for i in range(len(f)):
        fitness.append([np.argsort(f)[::1][i], np.sort(f)[::1][i]])
    fitness = np.asarray(fitness)
    if _ > 0:
        pop[fitness[-1, 0]] = elite
    elite = pop[fitness[0, 0]]
    print("Most fitted DNA: ", elite, ' fitness: ', fitness[0, 1])
    pop = select(pop, fitness)
    pop[random.randint(0, POP_SIZE - 1)] = elite
    pop_copy = pop.copy()
    ecost = 0
    x = random.randint(0, POP_SIZE - 1)
    for parent in pop:
        if (parent != elite).any():
            child = crossover(parent, pop_copy)
            child = mutate(child)
            parent[:] = child