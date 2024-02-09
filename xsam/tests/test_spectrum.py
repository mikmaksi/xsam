import shutil
import unittest
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xsam.tests.constants as constants
from xsam.exceptions import XsamException
from xsam.settings import SpectrumSettings
from xsam.spectrum import LinePattern, Spectrum, SpectrumCollection


class TestLinePattern(unittest.TestCase):
    def test_init(self):
        # arrange
        @dataclass
        class TestCase:
            name: str
            path: str
            spectrum_settings: SpectrumSettings
            max_intensity: float

        # define test cases
        test_cases = [
            TestCase(
                name="basic",
                path=constants.TEST_FILE_DIR.joinpath("LiO2_12.cif"),
                spectrum_settings=SpectrumSettings(min_angle=40, max_angle=60),
                max_intensity=1.0,
            ),
            TestCase(
                name="scale to 100",
                path=constants.TEST_FILE_DIR.joinpath("LiO2_12.cif"),
                spectrum_settings=SpectrumSettings(min_angle=40, max_angle=60),
                max_intensity=100.0,
            ),
        ]

        for case in test_cases:
            with self.subTest(case.name):
                # act
                line_pattern = LinePattern.from_cif(case.path, case.spectrum_settings, case.max_intensity)

                # assert
                self.assertGreaterEqual(line_pattern.x.min(), 40.0)
                self.assertLessEqual(line_pattern.x.max(), 60.0)
                self.assertEqual(line_pattern.y.max(), case.max_intensity)


@pytest.mark.usefixtures("plots_flag")  # see conftest.py
class TestSpectrum(unittest.TestCase):
    def setUp(self):
        self.plot_dir = Path("plots")
        self.plot_dir.mkdir(exist_ok=True)

    def tearDown(self):
        if not self.plots_flag:
            shutil.rmtree(self.plot_dir, ignore_errors=True)

    def test_from_cif(self):
        # arrange
        @dataclass
        class TestCase:
            name: str
            path: str
            spectrum_settings: SpectrumSettings

        # define test cases
        test_cases = [
            TestCase(
                name="basic",
                path=constants.TEST_FILE_DIR.joinpath("LiO2_12.cif"),
                spectrum_settings=SpectrumSettings(min_angle=40, max_angle=60),
            ),
        ]

        for case in test_cases:
            with self.subTest(case.name):
                # act
                spectrum = Spectrum.from_cif(case.path, case.spectrum_settings)

                # assert
                self.assertGreaterEqual(spectrum.x.min(), 40.0)
                self.assertLessEqual(spectrum.x.max(), 60.0)
                self.assertTrue(np.isclose(spectrum.y.max(), 1.0))

    def test_from_xy(self):
        # arrange
        @dataclass
        class TestCase:
            name: str
            path: str
            spectrum_settings: SpectrumSettings

        # define test cases
        test_cases = [
            TestCase(
                name="basic",
                path=constants.TEST_FILE_DIR.joinpath("LiMnO2.xy"),
                spectrum_settings=SpectrumSettings(min_angle=10, max_angle=100),
            ),
        ]

        # assert
        for case in test_cases:
            with self.subTest(case.name):
                spectrum = Spectrum.from_xy(case.path, case.spectrum_settings)
                self.assertGreaterEqual(spectrum.x.min(), 10.0)
                self.assertLessEqual(spectrum.x.max(), 100.0)
                self.assertTrue(np.isclose(spectrum.y.max(), 1.0))

    def test_from_xy_exception(self):
        # arrange
        @dataclass
        class TestCase:
            name: str
            path: str
            spectrum_settings: SpectrumSettings
            expected_exception_type: Exception
            expected_exception_message: str

        # define test cases
        test_cases = [
            TestCase(
                name="wrong minimum",
                path=constants.TEST_FILE_DIR.joinpath("LiMnO2.xy"),
                spectrum_settings=SpectrumSettings(min_angle=40, max_angle=100),
                expected_exception_type=XsamException,
                expected_exception_message="Spectrum outside of specified angle range (min).",
            ),
            TestCase(
                name="wrong maximum",
                path=constants.TEST_FILE_DIR.joinpath("LiMnO2.xy"),
                spectrum_settings=SpectrumSettings(min_angle=10, max_angle=90),
                expected_exception_type=XsamException,
                expected_exception_message="Spectrum outside of specified angle range (max).",
            ),
        ]

        # assert
        for case in test_cases:
            with self.subTest(case.name):
                with self.assertRaises(Exception) as err:
                    _ = Spectrum.from_xy(case.path, case.spectrum_settings)
                self.assertEqual(type(err.exception), case.expected_exception_type)
                self.assertIn(case.expected_exception_message, err.exception.args[0])

    def test_resample(self):
        # arrange
        @dataclass
        class TestCase:
            name: str
            path: str
            spectrum_settings: SpectrumSettings
            down_n_points: int

        # define test cases
        test_cases = [
            TestCase(
                name="downsample",
                path=constants.TEST_FILE_DIR.joinpath("LiMnO2.xy"),
                spectrum_settings=SpectrumSettings(min_angle=10, max_angle=100),
                down_n_points=1000,
            ),
            TestCase(
                name="upsample",
                path=constants.TEST_FILE_DIR.joinpath("LiMnO2.xy"),
                spectrum_settings=SpectrumSettings(min_angle=10, max_angle=100),
                down_n_points=10000,
            ),
        ]

        # assert
        for case in test_cases:
            original_spectrum = Spectrum.from_xy(case.path, case.spectrum_settings)
            spectrum = deepcopy(original_spectrum)
            spectrum.resample(case.down_n_points)
            self.assertEqual(len(spectrum.x), len(spectrum.y))
            self.assertEqual(len(spectrum.x), case.down_n_points)
            self.assertAlmostEqual(spectrum.y.mean(), original_spectrum.y.mean())

            # plot
            plt.figure()
            plt.plot(
                original_spectrum.x, original_spectrum.y, "o-", spectrum.x, spectrum.y, "o-", markersize=2, linewidth=1
            )
            plt.legend(["data", "resampled"], loc="best")
            plt.savefig(self.plot_dir.joinpath(f"{case.name}.pdf"))

    def test_smooth(self):
        # arrange
        @dataclass
        class TestCase:
            name: str
            path: str
            spectrum_settings: SpectrumSettings
            smoothing_a: float
            smoothing_n: int

        # define test cases
        test_cases = [
            TestCase(
                name="smoothed",
                path=constants.TEST_FILE_DIR.joinpath("LiMnO2.xy"),
                spectrum_settings=SpectrumSettings(min_angle=10, max_angle=100),
                smoothing_a=1.5,
                smoothing_n=20,
            ),
        ]

        # assert
        for case in test_cases:
            original_spectrum = Spectrum.from_xy(case.path, case.spectrum_settings)
            spectrum = deepcopy(original_spectrum)
            spectrum.smooth(case.smoothing_a, case.smoothing_n)
            self.assertLess(spectrum.y.std(), original_spectrum.y.std())

            # plot
            plt.figure()
            plt.plot(
                original_spectrum.x, original_spectrum.y, "o-", spectrum.x, spectrum.y, "o-", markersize=2, linewidth=1
            )
            plt.legend(["data", "smoothed"], loc="best")
            plt.savefig(self.plot_dir.joinpath(f"{case.name}.pdf"))

    def test_normalize(self):
        # arrange
        @dataclass
        class TestCase:
            name: str
            spectrum: Spectrum
            max_intensity: float
            min_intensity: Optional[float] = None

        # define test cases
        spectrum_settings = SpectrumSettings(min_angle=10, max_angle=100)
        original_spectrum = Spectrum.from_xy(constants.TEST_FILE_DIR.joinpath("LiMnO2.xy"), spectrum_settings)
        original_spectrum.y += 0.1
        test_cases = [
            TestCase(name="normalize_max", spectrum=original_spectrum, max_intensity=2.0, min_intensity=None),
            TestCase(
                name="normalize_max_min_1",
                spectrum=original_spectrum,
                max_intensity=1.0,
                min_intensity=original_spectrum.y.min(),
            ),
            TestCase(
                name="normalize_max_min_3",
                spectrum=original_spectrum,
                max_intensity=3.0,
                min_intensity=original_spectrum.y.min(),
            ),
        ]

        # assert
        for case in test_cases:
            spectrum = deepcopy(case.spectrum)
            spectrum.normalize(max_intensity=case.max_intensity, min_intensity=case.min_intensity)
            self.assertEqual(spectrum.y.max(), case.max_intensity)
            if case.min_intensity is None:
                self.assertGreater(spectrum.y.min(), 0)
            else:
                self.assertEqual(spectrum.y.min(), 0)

            # plot
            plt.figure()
            plt.plot(
                original_spectrum.x, original_spectrum.y, "o-", spectrum.x, spectrum.y, "o-", markersize=2, linewidth=1
            )
            plt.legend(["data", "normalized"], loc="best")
            plt.savefig(self.plot_dir.joinpath(f"{case.name}.pdf"))

    def test_background_subtract(self):
        # arrange
        @dataclass
        class TestCase:
            name: str
            spectrum: Spectrum
            background_constant: float
            max_intensity: float
            rolling_ball_radius: float

        # define test cases
        spectrum_settings = SpectrumSettings(min_angle=10, max_angle=100)
        original_spectrum = Spectrum.from_xy(constants.TEST_FILE_DIR.joinpath("LiMnO2.xy"), spectrum_settings)
        test_cases = [
            TestCase(
                name="background_subtract",
                spectrum=original_spectrum,
                background_constant=2.0,
                max_intensity=255.0,
                rolling_ball_radius=800.0,
            ),
        ]

        # assert
        for case in test_cases:
            spectrum_w_background = deepcopy(case.spectrum)

            # add background
            background = case.background_constant / spectrum_w_background.x  # 1/X
            spectrum_w_background.y += background

            # subtract backgroun
            spectrum_wo_background = deepcopy(spectrum_w_background)
            spectrum_wo_background.normalize(
                max_intensity=case.max_intensity, min_intensity=spectrum_wo_background.y.min()
            )
            spectrum_wo_background.background_subtract(case.rolling_ball_radius)
            spectrum_wo_background.normalize(max_intensity=1.0, min_intensity=spectrum_wo_background.y.min())

            # plot
            plt.figure()
            plt.plot(original_spectrum.x, original_spectrum.y, "-", alpha=0.5)
            plt.plot(spectrum_w_background.x, spectrum_w_background.y, "-", alpha=0.5)
            plt.plot(spectrum_wo_background.x, spectrum_wo_background.y, "-", alpha=0.5)
            plt.legend(["data", "w/ background", "w/o background"], loc="best")
            plt.savefig(self.plot_dir.joinpath(f"{case.name}.pdf"))

    def test__eq__(self):
        @dataclass
        class TestCase:
            name: str
            spectrum_a: Spectrum
            spectrum_b: Spectrum
            assert_function: callable

        # define test cases
        spectrum_settings = SpectrumSettings(min_angle=10, max_angle=100)
        spectrum_a = Spectrum.from_cif(constants.TEST_FILE_DIR.joinpath("MnO_225.cif"), spectrum_settings)
        spectrum_b = Spectrum.from_cif(constants.TEST_FILE_DIR.joinpath("TiO2_136.cif"), spectrum_settings)
        test_cases = [
            TestCase(
                name="background_subtract",
                spectrum_a=spectrum_a,
                spectrum_b=spectrum_a,
                assert_function=self.assertTrue,
            ),
            TestCase(
                name="background_subtract",
                spectrum_a=spectrum_a,
                spectrum_b=spectrum_b,
                assert_function=self.assertFalse,
            ),
        ]

        # assert
        for case in test_cases:
            case.assert_function(case.spectrum_a == case.spectrum_b)


class TestSpectrumCollection(unittest.TestCase):
    def test_from_reference_dir(self):
        # arrange
        @dataclass
        class TestCase:
            name: str
            reference_dir: str
            spectrum_settings: SpectrumSettings
            expected_num_spectra: int
            expected_phases: list[str]

        # define test cases
        test_cases = [
            TestCase(
                name="basic",
                reference_dir=constants.TEST_FILE_DIR,
                spectrum_settings=SpectrumSettings(min_angle=10, max_angle=100),
                expected_num_spectra=7,
                expected_phases=[
                    "MnO_225.cif",
                    "MnO_186.cif",
                    "Li_213.cif",
                    "Li2MnO3_12.cif",
                    "TiO2_136.cif",
                    "LiO2_12.cif",
                    "MnF2_111.cif",
                ],
            ),
        ]

        for case in test_cases:
            with self.subTest(case.name):
                # act
                spectrum_collection = SpectrumCollection.from_reference_dir(case.reference_dir, case.spectrum_settings)
                self.assertIsInstance(spectrum_collection.reference_array, np.ndarray)
                self.assertEqual(spectrum_collection.reference_array.shape[0], case.expected_num_spectra)
                self.assertEqual(len(spectrum_collection.spectra), case.expected_num_spectra)
                self.assertEqual(spectrum_collection.phases, case.expected_phases)
