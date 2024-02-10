import shutil
import unittest
from dataclasses import dataclass
from pathlib import Path

import pytest
import xsam.tests.constants as constants
from xsam.match import Match
from xsam.operations import SpectrumAlignment, SpectrumSubtraction
from xsam.settings import SpectrumSettings
from xsam.spectrum import Spectrum


@pytest.mark.usefixtures("plots_flag")  # see conftest.py
class TestOperations(unittest.TestCase):
    def setUp(self):
        self.plot_dir = Path("plots")
        self.plot_dir.mkdir(exist_ok=True)

    def tearDown(self):
        if not self.plots_flag:
            shutil.rmtree(self.plot_dir, ignore_errors=True)

    def test_alignment(self):
        # arrange
        @dataclass
        class TestCase:
            name: str
            spectrum_alignment: SpectrumAlignment
            last_normalization: float
            downsampling_resolution: float
            allow_shifts: float
            query_threshold: float

        # define test cases
        spectrum_settings = SpectrumSettings(min_angle=10, max_angle=100)
        input_spectrum = Spectrum.from_xy(constants.TEST_FILE_DIR.joinpath("Li2MnO3+MnO+TiO2.xy"), spectrum_settings)
        phase_spectrum = Spectrum.from_cif(constants.TEST_FILE_DIR.joinpath("Li2MnO3_12.cif"), spectrum_settings)
        test_cases = [
            TestCase(
                name="simple_alignment",
                spectrum_alignment=SpectrumAlignment(target=input_spectrum, query=phase_spectrum),
                last_normalization=1.0,
                downsampling_resolution=1.0,
                allow_shifts=0.75,
                query_threshold=0.01,
            ),
        ]

        for case in test_cases:
            # act
            aligned_spectrum = case.spectrum_alignment.apply(
                downsampling_resolution=case.downsampling_resolution,
                allow_shifts=case.allow_shifts,
                query_threshold=case.query_threshold,
            )

            # assert
            self.assertLess(aligned_spectrum.y.sum(), input_spectrum.y.sum())

            # plot check
            match = Match(
                phase="test",
                kernel=1.0,
                input_spectrum=input_spectrum,
                phase_spectrum=phase_spectrum,
                aligned_spectrum=aligned_spectrum,
            )
            match.plot_diagnostic(path=self.plot_dir.joinpath(f"{case.name}.pdf"))
