from pymoode.operators.dex import DEX
from pymoode.operators.dem import DEM

class RLMODEX(DEX):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)


class RLMODEM(DEM):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)