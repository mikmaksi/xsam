from enum import Enum

from importlib_resources import files

PACKAGE_PATH = files("xsam")
ANGSTROM_TO_NM = 0.1


class SIMILARITY_FUNCTION(str, Enum):
    COSINE = "cosine"
    PEARSON = "correlation"
    JSD = "jensenshannon"
    EMD = "earth-mover-distance"


class WAVELENGTH_TYPE(str, Enum):
    COPPER = "CuKa"


class TERMINATION_CONDITION(str, Enum):
    MAX_PHASES = "max_phases"
    INTENSITY_CUTOFF = "intensity_cutoff_reached"
    NO_MATCHES = "no_matches"
