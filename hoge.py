import numpy as np
import itertools
import random

ws = np.array([[0, 1, 0, 2, 0, 0],
               [0, 1, 2, 1, 2, 0],
               [0, 1, 1, 0, 2, 0],
               [0, 1, 0, 2, 0, 0],
               [0, 6, 6, 5, 0, 5],
               [0, 5, 5, 3, 5, 0],
               [0, 4, 4, 3, 4, 0],
               [0, 5, 4, 5, 3, 0],
               [0, 3, 2, 2, 3, 0],
               [0, 1, 2, 2, 0, 1],
               [0, 3, 2, 3, 1, 0],
               [0, 5, 7, 7, 0, 7],
               [0, 8, 7, 6, 0, 6]])

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
#print('j_d: ', j_d)
print('judge_dupl: ', judge_dupl)
for i in range(len(judge_dupl)):
    for j in range(judge_dupl[i]):
        j_d[i][sum(judge_dupl[:i]) + j] = 1
j_d = np.array(j_d.astype(bool))

DNA_SIZE = len(index)
POP_SIZE = 3000
CROSS_RATE = 0.5
MUTATION_RATE = 1.0 / DNA_SIZE
N_GENERATIONS = 10000

time_idx = []
for i in range(len(judge_dupl)):
    for j in range(judge_dupl[i]):
        time_idx.append(i)
print(time_idx)


pop = np.random.randint(1, 5, size=(POP_SIZE, DNA_SIZE))

def conv_pop_to_finger(pop_):
    pf, mf, rf, lf = [], [], [], []
    finger_t = []
    for i_ in range(POP_SIZE):
        for j_ in range(DNA_SIZE):
            ind = index[j_]
            if pop_[i_][j_] == 1:
                pf.append([i_, ind[1], ws[ind[0], ind[1]], time_idx[j_]])
                finger_t.append([[i_], [1, ind[0]]])
            elif pop_[i_][j_] == 2:
                mf.append([i_, ind[1], ws[ind[0], ind[1]], time_idx[j_]])
                finger_t.append([[i_], [2, ind[0]]])
            elif pop_[i_][j_] == 3:
                rf.append([i_, ind[1], ws[ind[0], ind[1]], time_idx[j_]])
                finger_t.append([[i_], [3, ind[0]]])
            else:
                lf.append([i_, ind[1], ws[ind[0], ind[1]], time_idx[j_]])
                finger_t.append([[i_], [4, ind[0]]])
    return pf, mf, rf, lf, finger_t

def get_fin_2_fret(pop):
    fret, f2f = [], []
    for i, dna in pop:
        for j in range(index):
            fret.append(ws[[index[j][0]], [index[j][1]]])
            pos = np.asarray(pos)
            fret = list(itertools.chain.from_iterable(pos))
    return fret

def get_fitness(f):
    count = 0
    e = []
    for i in range(POP_SIZE):
        fret, str, press = 0, 0, 0
        while f[count][0] == i:
            if count > len(f) - 2:
                break
            ap = list(map(lambda x: x[0] - x[1], zip(f[count], f[count + 1])))
            ap = list(map(abs, ap))
            if ap[0] == 0:
                if ap[3] > 1:
                    press = press + 2
                else:
                    str = str + ap[1]
                    fret = fret + ap[2]
                    if ap[3] > 1:
                        press = press + 2
            count += 1
        e.append((str + fret + press) ** 2)
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
                    f[dna] = f[dna] +1000               #セーハの場合を除外するため
                elif (fin + 1) in ddupl:
                    fidx.append(ddupl.index(fin + 1))
            if len(fidx) == 2:
                if frets[fidx[0]] > frets[fidx[1]] or (frets[fidx[0]] == frets[fidx[1]] and index[fidx[0]][1] < index[fidx[1]][1]):
                    f[dna] = f[dna] + 1000
            elif len(fidx) == 3:
                if frets[fidx[0]] > frets[fidx[1]] or (frets[fidx[0]] == frets[fidx[1]] and index[fidx[0]][1] < index[fidx[1]][1]):
                    f[dna] = f[dna] + 1000
                elif frets[fidx[0]] > frets[fidx[2]] or (frets[fidx[0]] == frets[fidx[2]] and index[fidx[0]][1] < index[fidx[2]][1]):
                    f[dna] = f[dna] + 1000
                elif frets[fidx[1]] > frets[fidx[2]] or (frets[fidx[1]] == frets[fidx[2]] and index[fidx[1]][1] < index[fidx[2]][1]):
                    f[dna] = f[dna] + 1000
            elif len(fidx) == 4:
                if frets[fidx[0]] > frets[fidx[1]] or (frets[fidx[0]] == frets[fidx[1]] and index[fidx[0]][1] < index[fidx[1]][1]):
                    f[dna] = f[dna] + 1000
                elif frets[fidx[0]] > frets[fidx[2]] or (frets[fidx[0]] == frets[fidx[2]] and index[fidx[0]][1] < index[fidx[2]][1]):
                    f[dna] = f[dna] + 1000
                elif frets[fidx[0]] > frets[fidx[3]] or (frets[fidx[0]] == frets[fidx[3]] and index[fidx[0]][1] < index[fidx[3]][1]):
                    f[dna] = f[dna] + 1000
                elif frets[fidx[1]] > frets[fidx[2]] or (frets[fidx[1]] == frets[fidx[2]] and index[fidx[1]][1] < index[fidx[2]][1]):
                    f[dna] = f[dna] + 1000
                elif frets[fidx[1]] > frets[fidx[3]] or (frets[fidx[1]] == frets[fidx[3]] and index[fidx[1]][1] < index[fidx[3]][1]):
                    f[dna] = f[dna] + 1000
                elif frets[fidx[2]] > frets[fidx[3]] or (frets[fidx[2]] == frets[fidx[3]] and index[fidx[2]][1] < index[fidx[3]][1]):
                    f[dna] = f[dna] + 1000
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

def crossover_trad(pop):
    crossover_size = int(POP_SIZE * CROSS_RATE)
    for i in range(crossover_size):
        p1_idx = np.random.randint(0,POP_SIZE)
        p2_idx = np.random.randint(0,POP_SIZE-1)
        if (p1_idx <= p2_idx):
            p2_idx += 1
        cross_point = np.random.randint(1,DNA_SIZE-1)
        for p in range(cross_point, DNA_SIZE):
            tmp = pop[p1_idx][p]
            pop[p1_idx][p] = pop[p2_idx][p]
            pop[p2_idx][p] = tmp

    return pop


def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = np.random.randint(1,5)
            # if child[point] == 1:
            #     child[point] = 2
            # elif child[point] == 2:
            #     child[point] = 3
            # elif child[point] == 3:
            #     child[point] = 4
            # else:
            #     child[point] = 1
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
    for parent in pop:
        if (parent != elite).any():
            child = crossover(parent, pop_copy)
            child = mutate(child)
            parent[:] = child