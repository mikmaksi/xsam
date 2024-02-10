from monty.serialization import dumpfn, loadfn

from xsam.constants import WAVELENGTH_TYPE, SIMILARITY_FUNCTION, SIGNAL_TYPE
from xsam.pydantic_config import Model


class SpectrumSettings(Model):
    """
    TODO
    """

    min_angle: float = 15.0
    max_angle: float = 85.0
    domain_size: float = 25.0
    n_points: int = 4501
    wavelength: WAVELENGTH_TYPE = WAVELENGTH_TYPE.COPPER


class SearchMatchSettings(Model):
    """
    TODO
    """

    max_phases: int = 3
    signal_cutoff: float = 0.05
    signal_type: SIGNAL_TYPE = SIGNAL_TYPE.MAX_INTENSITY
    min_kernel: float = 0.30
    spectrum_settings: SpectrumSettings
    similarity_function: SIMILARITY_FUNCTION = SIMILARITY_FUNCTION.COSINE

    def to_json(self, path: str) -> None:
        dumpfn(self.dict(), path, indent=4)

    @classmethod
    def from_json(cls, path: str) -> "SearchMatchSettings":
        model_dict = loadfn(path)
        return cls(**model_dict)
