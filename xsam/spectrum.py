import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import plotnine as pn
from pydantic import field_serializer, field_validator, model_validator
from pymatgen.analysis.diffraction.core import DiffractionPattern
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import filtfilt, resample
from skimage import restoration

from xsam.constants import ANGSTROM_TO_NM
from xsam.exceptions import XsamException
from xsam.pydantic_config import Model
from xsam.settings import SpectrumSettings
from xsam import logger

from tqdm import tqdm


class LinePattern(Model):
    """
    TODO
    """

    x: np.ndarray
    y: np.ndarray
    calculator: XRDCalculator
    spectrum_settings: SpectrumSettings

    @field_serializer("x", "y")
    def serialize_np_array(self, np_array: np.ndarray):
        return np_array.tolist()

    @field_validator("x", "y", mode="before")
    @classmethod
    def convert_to_np_array(cls, value: list) -> np.ndarray:
        if isinstance(value, list):
            value = np.array(value)
        return value

    @field_serializer("calculator")
    def serialize_xrd_calculator(self, calculator: XRDCalculator):
        return {
            "wavelength": calculator.wavelength,
            "debye_waller_factors": calculator.debye_waller_factors,
            "symprec": calculator.symprec,
        }

    @field_validator("calculator", mode="before")
    @classmethod
    def convert_to_xrd_calculator(cls, value: dict) -> XRDCalculator:
        if isinstance(value, dict):
            value = XRDCalculator(**value)
        return value

    @classmethod
    def from_cif(
        cls, path: str, spectrum_settings: SpectrumSettings, max_intensity: Union[float, None] = 1.0
    ) -> "LinePattern":
        structure = Structure.from_file(path)
        calculator = XRDCalculator(wavelength=spectrum_settings.wavelength)
        if max_intensity is None:
            pattern: DiffractionPattern = calculator.get_pattern(
                structure, scaled=False, two_theta_range=(spectrum_settings.min_angle, spectrum_settings.max_angle)
            )
        else:
            pattern: DiffractionPattern = calculator.get_pattern(
                structure, scaled=True, two_theta_range=(spectrum_settings.min_angle, spectrum_settings.max_angle)
            )
            pattern.y = pattern.y * max_intensity / 100.0  # 100 is default for calculator.get_pattern
        return cls(x=pattern.x, y=pattern.y, calculator=calculator, spectrum_settings=spectrum_settings)


class Spectrum(Model):
    phase: str
    x: np.ndarray
    y: np.ndarray
    spectrum_settings: Optional[SpectrumSettings]
    line_pattern: Optional[LinePattern] = None  # will not exist if from_csv

    @field_serializer("x", "y")
    def serialize_np_array(self, np_array: np.ndarray):
        return np_array.tolist()

    @field_validator("x", "y", mode="before")
    @classmethod
    def convert_to_np_array(cls, value: list) -> np.ndarray:
        if isinstance(value, list):
            value = np.array(value)
        return value

    def __eq__(self, other: "Spectrum") -> bool:
        if self.phase != other.phase:
            return False
        if self.spectrum_settings != other.spectrum_settings:
            return False
        if not np.isclose(self.x, other.x, rtol=1.0e-7, atol=1.0e-10).all():
            return False
        if not np.isclose(self.y, other.y, rtol=1.0e-7, atol=1.0e-10).all():
            return False
        return True

    def resample(self, down_n_points: int) -> None:
        resampled_y, resampled_x = resample(x=self.y, num=down_n_points, t=self.x)
        self.y = resampled_y
        self.x = resampled_x

    def smooth(self, smoothing_a: float = 1.0, smoothing_n: int = 20) -> None:
        smoothing_b = [1.0 / smoothing_n] * smoothing_n
        smoothed_y = filtfilt(smoothing_b, smoothing_a, self.y)
        self.y = smoothed_y

    def normalize(self, max_intensity: float = 1.0, min_intensity: Optional[float] = None) -> float:
        if min_intensity is not None:
            self.y = self.y - min_intensity
        normalization_constant = max_intensity / self.y.max()
        self.y = self.y * normalization_constant
        return normalization_constant

    def background_subtract(self, rolling_ball_radius: float = 800.0) -> None:
        """
        note: radius of 800 is tunned to max intensity of 255
        """
        background = restoration.rolling_ball(self.y, radius=rolling_ball_radius)
        self.y = self.y - background

    @staticmethod
    def calc_peak_width(two_theta, domain_size: float, wavelength_nm: float, K: float = 0.9):
        """
        calculate standard deviation based on angle (two theta) and domain size (domain_size)
        Args:
            two_theta: angle in two theta space
            domain_size: domain size in nm
            wavelength_nm: wavelength in nm
            K: shape factor
        Returns:
            standard deviation for gaussian kernel
        """
        # calculate FWHM based on the Scherrer equation
        theta = np.radians(two_theta / 2.0)  # Bragg angle in radians
        beta = (K * wavelength_nm) / (np.cos(theta) * domain_size)  # convert to radians

        # convert FWHM to std deviation of gaussian
        sigma = np.sqrt(1 / (2 * np.log(2))) * 0.5 * np.degrees(beta)
        return sigma**2

    @classmethod
    def from_xy(cls, path: str, spectrum_settings: SpectrumSettings) -> "Spectrum":
        """
        TODO
        """
        # read the xy data
        xy_data = np.loadtxt(path)
        x = xy_data[:, 0]
        y = xy_data[:, 1]

        # check that the loaded data falls into the correct angle range
        if min(x) < spectrum_settings.min_angle:
            raise XsamException(
                "Spectrum outside of specified angle range (min).",
                {"data_limit": min(x), "min_angle": spectrum_settings.min_angle},
            )
        if max(x) > spectrum_settings.max_angle:
            raise XsamException(
                "Spectrum outside of specified angle range (max).",
                {"data_limit": max(x), "max_angle": spectrum_settings.max_angle},
            )

        # conform to a certain number of data-points that can be compatible with downstream processing
        interpolator_function = CubicSpline(x, y)
        interp_x = np.linspace(spectrum_settings.min_angle, spectrum_settings.max_angle, spectrum_settings.n_points)
        interp_y = interpolator_function(interp_x)

        # create spectrum
        spectrum = cls(phase=Path(path).name, x=interp_x, y=interp_y, spectrum_settings=spectrum_settings)

        # smooth and normalize 0 to 255
        spectrum.smooth()
        _ = spectrum.normalize(max_intensity=255.0, min_intensity=spectrum.y.min())

        # subtract background and normalize 0 to 1
        spectrum.background_subtract()
        _ = spectrum.normalize(min_intensity=spectrum.y.min())

        return spectrum

    @classmethod
    def from_cif(cls, path: str, spectrum_settings: SpectrumSettings) -> "Spectrum":
        """
        TODO
        """
        # load the line pattern
        line_pattern = LinePattern.from_cif(path, spectrum_settings)

        # map angle to closest data point step
        xs = np.linspace(spectrum_settings.min_angle, spectrum_settings.max_angle, spectrum_settings.n_points)
        ys = np.zeros([len(line_pattern.x), len(xs)])
        for i, angle in enumerate(line_pattern.x):
            index = np.argmin(np.abs(angle - xs))
            ys[i, index] = line_pattern.y[i]

        # smooth each line in the pattern into a peak using a gaussian filter
        step_size = (spectrum_settings.max_angle - spectrum_settings.min_angle) / spectrum_settings.n_points
        wavelength_nm = line_pattern.calculator.wavelength * ANGSTROM_TO_NM
        for i in range(len(ys)):
            row = ys[i, :]
            x = xs[np.argmax(row)]
            peak_width = cls.calc_peak_width(x, domain_size=spectrum_settings.domain_size, wavelength_nm=wavelength_nm)
            ys[i, :] = gaussian_filter1d(row, np.sqrt(peak_width) * (1 / step_size), mode="constant")

        # combine signals
        ys = np.sum(ys, axis=0)

        # make spectrum and normalize
        spectrum = cls(
            phase=Path(path).name, x=xs, y=ys, spectrum_settings=spectrum_settings, line_pattern=line_pattern
        )
        _ = spectrum.normalize()

        # output
        return spectrum

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame({"x": self.x, "y": self.y})

    def plot(self, path: Optional[str] = None) -> Union[list[pn.ggplot], None]:
        # make into a dataframe
        data = self.to_df()

        # make plot
        p = pn.ggplot(data=data, mapping=pn.aes("x", "y"))
        p = p + pn.geom_line() + pn.theme_gray(base_size=16) + pn.theme(figure_size=(6, 4))

        # save
        if path is not None:
            p.save(path)
        else:
            return p


class SpectrumCollection(Model):
    """
    TODO
    """

    spectra: list[Spectrum]
    spectrum_settings: SpectrumSettings
    reference_array: Optional[np.ndarray] = None

    @field_serializer("reference_array")
    def serialize_np_array(self, np_array: np.ndarray):
        return np_array.tolist()

    @field_validator("reference_array", mode="before")
    @classmethod
    def convert_to_np_array(cls, value: list) -> np.ndarray:
        if isinstance(value, list):
            value = np.array(value)
        return value

    @property
    def phases(self) -> list[str]:
        return [spectrum.phase for spectrum in self.spectra]

    @classmethod
    def from_reference_dir(cls, reference_dir: str, spectrum_settings: SpectrumSettings) -> "SpectrumCollection":
        spectra = cls.load_spectra(reference_dir, spectrum_settings)
        return cls(spectra=spectra, spectrum_settings=spectrum_settings)

    @staticmethod
    def load_spectra(reference_dir: str, spectrum_setting: SpectrumSettings) -> list[Spectrum]:
        """
        TODO
        """
        cif_files = list(Path(reference_dir).glob("*.cif"))
        if cif_files == []:
            raise XsamException("No cif files found", {"reference_dir": reference_dir})
        spectra = []
        logger.info("Loading spectra", reference_dir=reference_dir)
        for cif_file in tqdm(cif_files):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                spectrum = Spectrum.from_cif(cif_file, spectrum_setting)
            spectra.append(spectrum)

        return spectra

    @model_validator(mode="after")
    def set_reference_array(self) -> "SpectrumCollection":
        reference_array = np.array([spectrum.y for spectrum in self.spectra])
        self.reference_array = reference_array
        return self

    def get_spectrum_by_name(self, name: str) -> Spectrum:
        return self.spectra[self.phases.index(name)]
