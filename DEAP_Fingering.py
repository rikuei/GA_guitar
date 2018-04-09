import numpy as np
import random

from deap import base, tools, creator

tabulature = np.array([[0, 7, 6, 7, 5, 0],
                       [0, 5, 4, 5, 4, 0],
                       [2, 0, 2, 1, 2, 0],
                       [0, 5, 6, 6, 0, 5],
                       [0, 3, 5, 4, 5, 0]])

tab_flatten = tabulature.flatten()
nonzero_index = np.nonzero(tab_flatten)[0]
individual_size = len(nonzero_index)


def getSingleFingering(finger, solution):
    str_fret = []
    for i in range(len(solution)):
        fingers_in_code = solution[i]
        for s in range(len(fingers_in_code)):
            if fingers_in_code[s] == finger:
                str_fret.append((s, tabulature[i][s]))
    return np.array(str_fret).T


def calcFingerCost(str_fret):
    cost = 0
    if 1 < len(str_fret):
        cost = np.sum(np.sum(np.diff(str_fret)**2, axis=1))
    return cost


def calcPenaltyFingerDup(solution):
    penalty = 0
    for fingers_in_code in solution:
        f = fingers_in_code[fingers_in_code > 0]
        uf = np.unique(f)
        penalty += 300 * (len(f) - len(uf))
    return penalty


def calcPenaltyFingerPos(solution):
    penalty = 0
    for f in range(len(solution)):
        arg_sort = np.argsort(solution[f])
        fret = tabulature[f]
        for i in range(len(arg_sort)-1):
            for j in range(i+1, len(arg_sort)):
                if fret[arg_sort[i]] > fret[arg_sort[j]]:
                    penalty += 100
    return penalty


def convIndivi2Tab(individual):
    solution = np.array(np.zeros(len(tab_flatten)), dtype=np.int32)
    for i in range(individual_size):
        solution[nonzero_index[i]] = individual[i]
    return solution.reshape(tabulature.shape)


def calcFitness(individual):
    solution = convIndivi2Tab(individual)
    fitness = 0
    for fin in range(1, 5):
        str_fret = getSingleFingering(fin, solution)
        fitness += calcFingerCost(str_fret)
    fitness += calcPenaltyFingerDup(solution)
    fitness += calcPenaltyFingerPos(solution)
    return fitness,


creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool", np.random.randint, 1, 5)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, individual_size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", calcFitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=1, up=4, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    random.seed(64)

    pop = toolbox.population(n=5000)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 100

    print("Start of evolution")

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    for g in range(NGEN):
        print("-- Generation %i --" % g)

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

        best_ind = tools.selBest(pop, 1)[0]
        print("Best individual is %s" % best_ind)

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    print(convIndivi2Tab(best_ind))


if __name__ == "__main__":
    main()
