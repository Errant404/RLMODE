import matplotlib.pyplot as plt
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.indicators.igd import IGD
from pymoode.survival import RankAndCrowding
from rlmode import RLMODE

problem = get_problem("ctp1")
pf = problem.pareto_front()

NGEN = 250
POPSIZE = 100
SEED = 1

rlmode = RLMODE(
    pop_size=POPSIZE, variant="DE/rand/1/bin", de_repair="bounce-back",
    survival=RankAndCrowding(crowding_func="cd"),
)

res_rlmode = minimize(
    problem,
    rlmode,
    ('n_gen', NGEN),
    seed=SEED,
    save_history=False,
    verbose=False,
)

igd = IGD(pf=problem.pareto_front(), zero_to_one=True)
print("IGD of GDE3 with normal crowding distances: ", igd.do(res_rlmode.F))

fig, ax = plt.subplots(figsize=[6, 5], dpi=70)
ax.scatter(pf[:, 0], pf[:, 1], color="navy", label="True Front")
ax.scatter(res_rlmode.F[:, 0], res_rlmode.F[:, 1], color="firebrick", label="RLMODE")
ax.set_ylabel("$f_2$")
ax.set_xlabel("$f_1$")
ax.legend()
fig.tight_layout()
plt.show()