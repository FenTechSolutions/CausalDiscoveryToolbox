from ...utils.Settings import SETTINGS
from ...utils.R import RPackages
if SETTINGS.r_is_available:
    print(RPackages.pcalg)
