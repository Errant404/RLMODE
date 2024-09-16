import numpy as np
import random

from pymoo.problems import get_problem
from pymoo.indicators.igd import IGD
from pymoo.visualization.scatter import Scatter
from functools import cmp_to_key

problem = get_problem("ctp1", n_var=5)
pf = problem.pareto_front()
ind = IGD(pf)

def softmax(values, T=1.0):
    exp_values = np.exp(values / T)
    return exp_values / np.sum(exp_values)

class Qlearning:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        probabilities = softmax(self.q_table[state])
        return np.random.choice(self.num_actions, p=probabilities)

    def update(self, state, action, reward, next_state):
        next_max = np.max(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (reward + self.gamma * next_max - self.q_table[state, action])


def get_offspring(pop, F, CR, index):
    num_pop = pop.shape[0]
    num_var = pop.shape[1]
    offspring = np.zeros(num_var)
    idxs = np.random.choice(num_pop, 3, replace=False)
    a, b, c = pop[idxs]
    mutant = a + F[index] * (b - c)
    jrand = np.random.randint(num_var)
    for j in range(num_var):
        if np.random.rand() < CR[index] or j == jrand:
            offspring[j] = mutant[j]
        else:
            offspring[j] = pop[index][j]
        if offspring[j] > BOUND_UP[j] or offspring[j] < BOUND_LOW[j]:
            offspring[j] = random.uniform(BOUND_LOW[j], BOUND_UP[j])
    return offspring


def is_dominate(x, y):
    # x["F"]: objs values
    # x["G"]: ieqs constraints
    # x["H"]: eqs constraints
    # x: offspring, y: parent
    # return: 1 if x dominates y, -1 if y dominates x, 0 if x and y are non-dominated
    x_dict = dict()
    y_dict = dict()
    x_dict["F"] = problem.evaluate(x, return_values_of=["F"])
    y_dict["F"] = problem.evaluate(y, return_values_of=["F"])
    if problem.n_ieq_constr > 0:
        x_dict["G"] = problem.evaluate(x, return_values_of=["G"])
        y_dict["G"] = problem.evaluate(y, return_values_of=["G"])
    else:
        x_dict["G"], y_dict["G"] = np.zeros(0), np.zeros(0)
    if problem.n_eq_constr > 0:
        x_dict["H"] = problem.evaluate(x, return_values_of=["H"])
        y_dict["H"] = problem.evaluate(y, return_values_of=["H"])
    else:
        x_dict["H"], y_dict["H"] = np.zeros(0), np.zeros(0)

    def is_feasible(dict):
        return np.all(dict["G"] <= 0) and np.all(dict["H"] == 0)

    if is_feasible(x_dict) and not is_feasible(y_dict):
        return 1
    elif not is_feasible(x_dict) and is_feasible(y_dict):
        return -1
    elif not is_feasible(x_dict) and not is_feasible(y_dict):
        cv_x = np.sum(np.maximum(0, x_dict["G"])) + np.sum(np.abs(x_dict["H"]))
        cv_y = np.sum(np.maximum(0, y_dict["G"])) + np.sum(np.abs(y_dict["H"]))
        if cv_x < cv_y:
            return 1
        elif cv_x > cv_y:
            return -1
        else:
            return 0
    elif is_feasible(x_dict) and is_feasible(y_dict):
        if np.all(x_dict["F"] >= y_dict["F"]) and np.any(x_dict["F"] > y_dict["F"]):
            return 1
        elif np.all(x_dict["F"] <= y_dict["F"]) and np.any(x_dict["F"] < y_dict["F"]):
            return -1
        else:
            return 0


random.seed(0)
np.random.seed(0)
num_states = 3
num_actions = 3
qlearning = Qlearning(num_states, num_actions)
BOUND_LOW, BOUND_UP = problem.bounds()

num_gen = 100
num_pop = 100
state = random.randint(0, num_states - 1)
F_table = np.random.random(num_pop)
CR_table = np.random.random(num_pop)
F_j = np.zeros(num_pop)
CR_j = np.zeros(num_pop)

x_max = np.array([BOUND_UP])
x_min = np.array([BOUND_LOW])
pop = np.random.rand(num_pop, problem.n_var) * (x_max - x_min) + x_min
for i in range(num_gen):
    offsprings = []
    for j in range(num_pop):
        F_table[j] = F_table[j] + F_j[j]
        CR_table[j] = CR_table[j] + CR_j[j]
        if F_table[j] > 1 or F_table[j] < 0:
            F_table[j] = random.random()
        if CR_table[j] > 1 or CR_table[j] < 0:
            CR_table[j] = random.random()

        offspring = get_offspring(pop, F_table, CR_table, j)

        action = qlearning.choose_action(state)
        F_j[j] = [-0.1, 0.1, 0][action]
        CR_j[j] = [0.1, 0.1, 0][action]

        reward = [0, 1, -1][is_dominate(offspring, pop[j])]
        next_state = [2, 0, 1][reward]
        qlearning.update(state, action, reward, next_state)
        state = next_state
        offsprings.append(offspring)

    pop = np.append(pop, offsprings, axis=0)
    pop = np.array(sorted(pop, key=cmp_to_key(is_dominate)))[0:num_pop]

    # The result found by the algorithm
    A = problem.evaluate(pop, return_values_of=["F"])
    print("IGD", ind(A))

# plot the result
Scatter(legend=True).add(pf, label="Pareto-front").add(A, label="Result").show()