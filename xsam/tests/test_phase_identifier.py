import shutil
import unittest
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
import xsam.tests.constants as constants
from pandas.testing import assert_frame_equal
from xsam.constants import SIGNAL_TYPE, SIMILARITY_FUNCTION, TERMINATION_CONDITION
from xsam.match import Match, MatchSequence
from xsam.phase_identifier import PhaseIdentifier
from xsam.settings import SearchMatchSettings, SpectrumSettings
from xsam.spectrum import Spectrum, SpectrumCollection


@pytest.mark.usefixtures("plots_flag")  # see conftest.py
class TestPhaseIdentifier(unittest.TestCase):
    def setUp(self):
        self.plot_dir = Path("plots")
        self.plot_dir.mkdir(exist_ok=True)

    def tearDown(self):
        if not self.plots_flag:
            shutil.rmtree(self.plot_dir, ignore_errors=True)

    def test_identify_phases(self):
        # arrange
        @dataclass
        class TestCase:
            name: str
            spectrum_path: str
            reference_dir: str
            spectrum_settings: SpectrumSettings
            max_phases: int
            signal_cutoff: float
            signal_type: SIGNAL_TYPE
            min_kernel: float
            similarity_function: SIMILARITY_FUNCTION
            expected_ensemble_edges_path: str
            expected_ensemble_paths_path: str
            expected_top_match_sequence: MatchSequence

        # load spectra
        spectrum_settings = SpectrumSettings(min_angle=10, max_angle=100)
        scale_factor_a = 1.0
        scale_factor_b = 0.35324531197407727
        scale_factor_c = 0.23564029472259246
        spectrum_a = Spectrum.from_cif(
            path=constants.TEST_FILE_DIR.joinpath("MnO_225.cif"), spectrum_settings=spectrum_settings
        )
        spectrum_a.y = spectrum_a.y * scale_factor_a
        spectrum_b = Spectrum.from_cif(
            path=constants.TEST_FILE_DIR.joinpath("Li2MnO3_12.cif"), spectrum_settings=spectrum_settings
        )
        spectrum_b.y = spectrum_b.y * scale_factor_b
        spectrum_c = Spectrum.from_cif(
            path=constants.TEST_FILE_DIR.joinpath("TiO2_136.cif"), spectrum_settings=spectrum_settings
        )
        spectrum_c.y = spectrum_c.y * scale_factor_c

        # set up test case
        test_cases = [
            TestCase(
                name="simple_subtraction",
                spectrum_path=constants.TEST_FILE_DIR.joinpath("Li2MnO3+MnO+TiO2.xy"),
                reference_dir=constants.TEST_FILE_DIR,
                spectrum_settings=spectrum_settings,
                max_phases=3,
                signal_cutoff=0.05,
                signal_type=SIGNAL_TYPE.MAX_INTENSITY,
                min_kernel=0.1,
                similarity_function=SIMILARITY_FUNCTION.COSINE,
                # outputs
                expected_ensemble_edges_path=constants.TEST_FILE_DIR.joinpath("match_ensemble_edges.csv"),
                expected_ensemble_paths_path=constants.TEST_FILE_DIR.joinpath("match_ensemble_paths.csv"),
                expected_top_match_sequence=MatchSequence(
                    matches=[
                        Match(
                            phase="MnO_225.cif",
                            phase_spectrum=spectrum_a,
                            kernel=0.8373586602880208,
                            scale_factor=scale_factor_a,
                        ),
                        Match(
                            phase="Li2MnO3_12.cif",
                            phase_spectrum=spectrum_b,
                            kernel=0.6358769621203686,
                            scale_factor=scale_factor_b,
                        ),
                        Match(
                            phase="TiO2_136.cif",
                            phase_spectrum=spectrum_c,
                            kernel=0.13202963847033367,
                            scale_factor=scale_factor_c,
                        ),
                    ],
                    termination_condition=TERMINATION_CONDITION.MAX_PHASES,
                    is_complete=True,
                    input_signal=1.0,
                ),
            ),
        ]
        for case in test_cases:
            # act
            search_match_settings = SearchMatchSettings(
                max_phases=case.max_phases,
                signal_cutoff=case.signal_cutoff,
                signal_type=case.signal_type,
                min_kernel=case.min_kernel,
                spectrum_settings=spectrum_settings,
                similarity_function=case.similarity_function,
            )
            spectrum_collection = SpectrumCollection.from_reference_dir(case.reference_dir, case.spectrum_settings)
            spectrum = Spectrum.from_xy(path=case.spectrum_path, spectrum_settings=case.spectrum_settings)
            phase_identifier = PhaseIdentifier(
                spectrum=spectrum, reference_collection=spectrum_collection, search_match_settings=search_match_settings
            )
            match_ensemble = phase_identifier.identify_phases()

            # check the edges
            expected_match_ensemble_edges = pd.read_csv(case.expected_ensemble_edges_path)
            assert_frame_equal(
                match_ensemble.get_edges_summary()[["from_name", "to_name"]],
                expected_match_ensemble_edges[["from_name", "to_name"]],
            )

            # check the paths
            expected_match_ensemble_paths = pd.read_csv(case.expected_ensemble_paths_path)
            assert_frame_equal(match_ensemble.get_paths_summary(), expected_match_ensemble_paths)

            # check the top match
            self.assertEqual(case.expected_top_match_sequence, match_ensemble.top_match_sequence)
            match_ensemble.top_match_sequence.plot(self.plot_dir.joinpath("identified_phases.pdf"))

            # plot the explore match sequences
            match_ensemble.plot_explored_paths(self.plot_dir.joinpath("ensemble_summary.pdf"))
