from .ANM import ANM
from .CDS import CDS
# from .Jarfo import Jarfo
from .IGCI import IGCI
from .GNN import GNN
from .Bivariate_fit import BivariateFit
from ...utils.Settings import SETTINGS
if SETTINGS.torch is not None:
    from .RCC import RCC
    from .NCC import NCC
