import numpy as np
# pymoo imports
from pymoo.core.population import Population
from pymoo.operators.sampling.rnd import random
from pymoo.util.dominator import get_relation

# pymoode imports
from pymoode.algorithms.base.differential import MODE
from pymoode.survival import RankAndCrowding

# custom imports
from variant import RlmodeVariant
from qlearning import Qlearning

# =========================================================================================================
# Implementation
# =========================================================================================================

class RLMODE(MODE):

    def __init__(self,
                 pop_size=100,
                 variant="DE/rand/1/bin",
                 CR=(0, 1.0),
                 F=(0, 1.0),
                 gamma=1e-4,
                 de_repair="bounce-back",
                 survival=RankAndCrowding(),
                 **kwargs):
        """
        GDE3 is an extension of DE to multi-objective problems using a mixed type survival strategy.
        It is implemented in this version with the same constraint handling strategy of NSGA-II by default.

        Derived algorithms GDE3-MNN and GDE3-2NN use by default survival RankAndCrowding with metrics 'mnn' and '2nn'.

        For many-objective problems, try using NSDE-R, GDE3-MNN, or GDE3-2NN.

        For Bi-objective problems, survival = RankAndCrowding(crowding_func='pcd') is very effective.

        Kukkonen, S. & Lampinen, J., 2005. GDE3: The third evolution step of generalized differential evolution. 2005 IEEE congress on evolutionary computation, Volume 1, pp. 443-450.

        Parameters
        ----------
        pop_size : int, optional
            Population size. Defaults to 100.

        variant : str, optional
            Differential evolution strategy. Must be a string in the format: "DE/selection/n/crossover", in which, n in an integer of number of difference vectors, and crossover is either 'bin' or 'exp'. Selection variants are:

                - 'ranked'
                - 'rand'
                - 'best'
                - 'current-to-best'
                - 'current-to-best'
                - 'current-to-rand'
                - 'rand-to-best'

            The selection strategy 'ranked' might be helpful to improve convergence speed without much harm to diversity. Defaults to 'DE/rand/1/bin'.

        CR : float, optional
            Crossover parameter. Defined in the range [0, 1]
            To reinforce mutation, use higher values. To control convergence speed, use lower values.

        F : iterable of float or float, optional
            Scale factor or mutation parameter. Defined in the range (0, 2]
            To reinforce exploration, use higher values; for exploitation, use lower values.

        gamma : float, optional
            Jitter deviation parameter. Should be in the range (0, 2). Defaults to 1e-4.

        de_repair : str, optional
            Repair of DE mutant vectors. Is either callable or one of:

                - 'bounce-back'
                - 'midway'
                - 'rand-init'
                - 'to-bounds'

            If callable, has the form fun(X, Xb, xl, xu) in which X contains mutated vectors including violations, Xb contains reference vectors for repair in feasible space, xl is a 1d vector of lower bounds, and xu a 1d vector of upper bounds.
            Defaults to 'bounce-back'.

        genetic_mutation, optional
            Pymoo's genetic mutation operator after crossover. Defaults to NoMutation().

        repair : Repair, optional
            Pymoo's repair operator after mutation. Defaults to NoRepair().

        survival : Survival, optional
            Pymoo's survival strategy.
            Defaults to RankAndCrowding() with crowding distances ('cd').
            In GDE3, the survival strategy is applied after a one-to-one comparison between child vector and corresponding parent when both are non-dominated by the other.
        """

        super().__init__(
            pop_size=pop_size,
            variant=variant,
            CR=CR,
            F=F,
            gamma=gamma,
            de_repair=de_repair,
            survival=survival,
            **kwargs,
        )

        self.mating = RlmodeVariant(
            variant=variant, CR=CR, F=F, gamma=gamma,
            de_repair=de_repair, **kwargs,
        )
        self.current_state = np.random.randint(3, size=pop_size)
        self.qlearning = Qlearning(n_states=3, n_actions=3, alpha=0.1, gamma=0.9)

    def _advance(self, infills=None, **kwargs):

        assert infills is not None, "This algorithms uses the AskAndTell interface thus 'infills' must to be provided."

        # The individuals that are considered for the survival later and final survive
        survivors = []
        # now for each of the infill solutions
        for k in range(len(self.pop)):
            # Get the offspring an the parent it is coming from
            off, parent = infills[k], self.pop[k]
            # Check whether the new solution dominates the parent or not
            rel = get_relation(parent, off)
            next_state = [2, 0, 1][rel]
            if not hasattr(self.pop[k], "state"):
                self.pop[k].state = np.random.randint(3)
            # Update the Q-table
            self.qlearning.update_q_table(self.pop[k].state, next_state, rel)
            self.pop[k].state = next_state
            survivors.append([parent, off])

        # Create the population
        survivors = Population.create(*survivors)
        # Perform a survival to reduce to pop size
        self.pop = self.survival.do(self.problem, survivors, n_survive=self.n_offsprings)