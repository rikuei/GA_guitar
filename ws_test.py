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

index = []
pos = np.array([[4,7],[2,10],[2,8],[2,7],[[2,7],[3,9],[4,8],[6,9]],[[3,7],[6,7]],[2,5],[2,7],[3,7],[6,3]])
lhand = np.random.randint(1,5)
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

print(index)
DNA_SIZE = len(index)
POP_SIZE = 100
CROSS_RATE = 0.8
MUTATION_RATE = 0.003
N_GENERATIONS = 200

def get_fitness(pred): return pred + 1e-3 - np.min(pred)

def select(pop, fitness):    # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]

def crossover(parent, pop):     # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)                             # select another individual from pop
        cross_points = np.random.randint(1, 5, size=DNA_SIZE).astype(np.bool)   # choose crossover points
        parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
    return parent

def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = np.random.randint(1,5)
    return child

pop = np.random.randint(1, 5, size=(POP_SIZE, DNA_SIZE))

for _ in range(N_GENERATIONS):
    F_values = pop  # compute function value by extracting DNA


    # GA part (evolution)
    fitness = get_fitness(F_values)
    print("Most fitted DNA: ", pop[np.argmax(fitness), :])
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child       # parent is replaced by its chi

print(pop)