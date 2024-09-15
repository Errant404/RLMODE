import random
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
from pymoo.problems.multi.ctp import CTP1
from pymoo.problems import get_problem
from deap import base, creator, tools

# Problem definition
PROBLEM = "ctp1"
NOBJ = 2
NVAR = 10
problem = get_problem("ctp1", n_var=NVAR)
BOUND_LOW, BOUND_UP = problem.bounds()

# Algorithm parameters
MU = 100
NGEN = 100
CXPB = 1.0
MUTPB = 1.0
T = 0.9
num_actions = 3
num_states = 3

# Create classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJ)
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

# Initialize Q-table
Q_TABLE = np.zeros((num_states, num_actions))  # 3 states, 3 actions
F_TABLE = np.random.random(size=MU)
CR_TABLE = np.random.random(size=MU)

# Toolbox initialization
def uniform(low, up, size):
    return [random.uniform(low[i], up[i]) for i in range(size)]

def differential_mutation(population, F):
    r1, r2, r3 = random.sample(population, 3)
    mutant = [x1 + F * (x2 - x3) for x1, x2, x3 in zip(r1, r2, r3)]

    return mutant

def binomial_crossover(mutant, ind, CR):
    dim = len(ind)
    j_rand = random.randint(0, dim - 1)
    trial = [mutant[j] if random.random() <= CR or j == j_rand else ind[j] for j in range(dim)]

    return trial

def softmax(Q_values, T):
    exp_values = np.exp(Q_values / T)
    return exp_values / np.sum(exp_values)

def select_action(state, T):
    probabilities = softmax(Q_TABLE[state], T)
    return np.random.choice(num_actions, p=probabilities)

def update_q_table(state, action, reward, next_state):
    learning_rate = 0.1
    discount_factor = 0.9
    old_value = Q_TABLE[state, action]
    next_max = np.max(Q_TABLE[next_state])
    new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
    Q_TABLE[state, action] = new_value

def update_control_parameters(index, action):
    if action == 0:
        F_f, CR_f = -0.1, 0.1
    elif action == 1:
        F_f, CR_f = 0.1, 0.1
    else:
        F_f, CR_f = 0, 0

    F_TABLE[index] = F_TABLE[index] + F_f
    CR_TABLE[index] = CR_TABLE[index] + CR_f

def rlmode_variation(population, toolbox, state):
    offspring = []

    def is_dominate(x, y):
        return x.fitness.dominates(y.fitness)

    for i in range(len(population)):
        ind = population[i]
        # TODO: Implement individual-based F and CR
        mutant = toolbox.mutate(population, F_TABLE[i])
        trial = toolbox.crossover(mutant, ind, CR_TABLE[i])
        trial = creator.Individual(trial)
        trial.fitness.values = toolbox.evaluate(trial)

        offspring.append(trial)

        if is_dominate(trial, ind):
            reward = 1
        else:
            if is_dominate(ind, trial):
                reward = -1
            else:
                reward = 0

        next_state = [2, 0, 1][reward]
        action = select_action(state, T)
        update_control_parameters(i, action)
        update_q_table(state, action, reward, next_state)
        state = next_state

    return offspring

toolbox = base.Toolbox()
toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NVAR)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", problem.evaluate)
toolbox.register("mutate", differential_mutation)
toolbox.register("crossover", binomial_crossover)
toolbox.register("variation", rlmode_variation)
toolbox.register("select", tools.selNSGA2)

# pool = multiprocessing.Pool()
# toolbox.register("map", pool.map)


def main(seed=None):
    random.seed(seed)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)
    state = random.randint(0, 2)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Compile statistics about the population
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        offspring = toolbox.variation(pop, toolbox, state)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, MU)

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    return pop, logbook


pop, stats = main()
pop_fit = np.array([ind.fitness.values for ind in pop])

pf = problem.pareto_front()

# Calculate and print IGD
from deap.benchmarks.tools import igd

print(f"IGD: {igd(pop_fit, pf)}")

# pool.close()