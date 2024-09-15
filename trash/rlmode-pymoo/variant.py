# pymoode imports
from operators import RLMODEX, RLMODEM

from pymoode.operators.variant import DifferentialVariant, _fix_deprecated_pm_kwargs

class RlmodeVariant(DifferentialVariant):
    def __init__(self,
                 variant="DE/rand/1/bin",
                 CR=(0, 1.0),
                 F=(0, 1.0),
                 gamma=1e-4,
                 de_repair="bounce-back",
                 genetic_mutation=None,
                 **kwargs):
        super().__init__(**kwargs)
        kwargs, genetic_mutation = _fix_deprecated_pm_kwargs(kwargs, genetic_mutation)
        _, selection_variant, n_diff, crossover_variant, = variant.split("/")
        n_diffs = int(n_diff)
        # Define differential evolution mutation
        self.de_mutation = RLMODEM(F=F, gamma=gamma, de_repair=de_repair, n_diffs=n_diffs)
        # Define crossover strategy (DE mutation is included)
        self.crossover = RLMODEX(variant=crossover_variant, CR=CR, at_least_once=True)