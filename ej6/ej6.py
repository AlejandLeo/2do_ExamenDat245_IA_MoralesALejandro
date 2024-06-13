import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

distancia = numpy.array=[
    [0 , 7 , 9 , 8 , 20],
    [7 , 0 , 10 , 4 , 11],
    [9 , 10 , 0 , 15 , 5],
    [8 , 4 , 15 , 0 , 17],
    [20 , 11 , 5 , 17 , 0]
    ]

col=['A','B','C','D','E']
fil=['A','B','C','D','E']

#Problem parameter
nroDist = 5

def evalDistance(individual):
  suma=0
  for i in range(len(individual)):
    if i==len(individual)-1:
      suma=suma+distancia[individual[i]][ individual[0]]
    else:
      suma=suma+distancia[individual[i]][ individual[i+1]]
  return suma,


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

#Since there is only one queen per line,
#individual are represented by a permutation
toolbox = base.Toolbox()
toolbox.register("permutation", random.sample, range(nroDist), nroDist)

#Structure initializers
#An individual is a list that represents the position of each queen.
#Only the line is stored, the column is the index of the number in the list.
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.permutation)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalDistance)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=2.0)
toolbox.register("select", tools.selTournament, tournsize=3)

def main(seed=0):
    random.seed(seed)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Avg", numpy.mean)
    stats.register("Std", numpy.std)
    stats.register("Min", numpy.min)
    stats.register("Max", numpy.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, stats=stats,
                        halloffame=hof, verbose=True)

    return pop, stats, hof

if __name__ == "__main__":
    pop, stats, hof=main()
    for i in hof:
        hofA=[]
        y=[]
        for j in i:
            y.append(fil[j])
        hofA.append(y)
    for i in range(len(hof)):
        print(hof[i],hofA[i],evalDistance(hof[i]))
