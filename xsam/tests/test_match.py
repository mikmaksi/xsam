import unittest
from dataclasses import dataclass, field
from typing import Optional, Union

import xsam.tests.constants as constants
from xsam.constants import TERMINATION_CONDITION
from xsam.match import Match, MatchEnsemble, MatchingResult, MatchSequence
from xsam.settings import SpectrumSettings
from xsam.spectrum import Spectrum


class TestMatch(unittest.TestCase):
    def test_match_sequence_ensemble(self):
        # arrange
        @dataclass
        class ExpectedResult:
            expected_phases: Optional[list[str]] = field(default_factory=list)
            expected_scale_factors: Optional[list[float]] = field(default_factory=list)
            expected_kernels: Optional[list[float]] = field(default_factory=list)
            expected_average_kernel: Optional[float] = None

        @dataclass
        class TestCase:
            name: str
            match_ensemble: MatchEnsemble
            expected_top_match_sequence: Union[MatchSequence, None]
            expected_num_sequences: int
            expected_num_complete_sequences: int
            expected_match_found: bool
            expected_results: list[ExpectedResult]

        spectrum_settings = SpectrumSettings()
        matches_one = [
            Match(
                phase="LiO2",
                kernel=0.8,
                phase_spectrum=Spectrum.from_cif(constants.TEST_FILE_DIR.joinpath("LiO2_12.cif"), spectrum_settings),
                scale_factor=0.9,
            )
        ]
        matches_two = [
            Match(
                phase="LiO2",
                kernel=0.8,
                phase_spectrum=Spectrum.from_cif(constants.TEST_FILE_DIR.joinpath("LiO2_12.cif"), spectrum_settings),
                scale_factor=0.9,
            ),
            Match(
                phase="MnF2",
                kernel=0.7,
                phase_spectrum=Spectrum.from_cif(constants.TEST_FILE_DIR.joinpath("MnF2_111.cif"), spectrum_settings),
                scale_factor=0.66,
            ),
        ]
        match_sequence_a = MatchSequence(matches=matches_one, is_complete=True)
        match_sequence_b = MatchSequence(matches=matches_two, is_complete=True)
        match_sequence_none = MatchSequence(
            matches=[], is_complete=True, termination_condition=TERMINATION_CONDITION.NO_MATCHES
        )
        match_sequence_incomplete = MatchSequence(matches=matches_one, is_complete=False)

        # define test cases
        test_cases = [
            TestCase(
                name="no matches",
                match_ensemble=MatchEnsemble(match_sequences=[match_sequence_none]),
                expected_top_match_sequence=None,
                expected_num_sequences=1,
                expected_num_complete_sequences=0,
                expected_match_found=False,
                expected_results=[ExpectedResult()],
            ),
            TestCase(
                name="one sequence, one match",
                match_ensemble=MatchEnsemble(match_sequences=[match_sequence_a]),
                expected_top_match_sequence=match_sequence_a,
                expected_num_sequences=1,
                expected_num_complete_sequences=1,
                expected_match_found=True,
                expected_results=[
                    ExpectedResult(
                        expected_phases=["LiO2"],
                        expected_scale_factors=[0.9],
                        expected_kernels=[0.8],
                        expected_average_kernel=0.8,
                    )
                ],
            ),
            TestCase(
                name="one sequence, multiple matches",
                match_ensemble=MatchEnsemble(match_sequences=[match_sequence_b]),
                expected_top_match_sequence=match_sequence_b,
                expected_num_sequences=1,
                expected_num_complete_sequences=1,
                expected_match_found=True,
                expected_results=[
                    ExpectedResult(
                        expected_phases=["LiO2", "MnF2"],
                        expected_scale_factors=[0.9, 0.66],
                        expected_kernels=[0.8, 0.7],
                        expected_average_kernel=0.75,
                    )
                ],
            ),
            TestCase(
                name="two sequences",
                match_ensemble=MatchEnsemble(match_sequences=[match_sequence_a, match_sequence_b]),
                expected_top_match_sequence=match_sequence_a,
                expected_num_sequences=2,
                expected_num_complete_sequences=2,
                expected_match_found=True,
                expected_results=[
                    ExpectedResult(
                        expected_phases=["LiO2"],
                        expected_scale_factors=[0.9],
                        expected_kernels=[0.8],
                        expected_average_kernel=0.8,
                    ),
                    ExpectedResult(
                        expected_phases=["LiO2", "MnF2"],
                        expected_scale_factors=[0.9, 0.66],
                        expected_kernels=[0.8, 0.7],
                        expected_average_kernel=0.75,
                    ),
                ],
            ),
            TestCase(
                name="two sequences one incomplete",
                match_ensemble=MatchEnsemble(match_sequences=[match_sequence_b, match_sequence_incomplete]),
                expected_top_match_sequence=match_sequence_b,
                expected_num_sequences=2,
                expected_num_complete_sequences=1,
                expected_match_found=True,
                expected_results=[
                    ExpectedResult(
                        expected_phases=["LiO2", "MnF2"],
                        expected_scale_factors=[0.9, 0.66],
                        expected_kernels=[0.8, 0.7],
                        expected_average_kernel=0.75,
                    ),
                ],
            ),
            TestCase(
                name="no complete",
                match_ensemble=MatchEnsemble(match_sequences=[match_sequence_incomplete, match_sequence_incomplete]),
                expected_top_match_sequence=None,
                expected_num_sequences=2,
                expected_num_complete_sequences=0,
                expected_match_found=False,
                expected_results=[
                    ExpectedResult(
                        expected_phases=["LiO2"],
                        expected_scale_factors=[0.9],
                        expected_kernels=[0.8],
                        expected_average_kernel=0.8,
                    ),
                ],
            ),
        ]

        # assert
        for case in test_cases:
            with self.subTest(case.name):
                # test the ensemble
                self.assertEqual(case.match_ensemble.top_match_sequence, case.expected_top_match_sequence)
                self.assertEqual(case.match_ensemble.get_num_sequences(), case.expected_num_sequences)
                self.assertEqual(case.match_ensemble.get_num_sequences(True), case.expected_num_complete_sequences)
                self.assertEqual(case.match_ensemble.match_found, case.expected_match_found)

                # test each sequence
                for match_sequence, expected_result in zip(case.match_ensemble.match_sequences, case.expected_results):
                    self.assertEqual(match_sequence.phases, expected_result.expected_phases)
                    self.assertEqual(match_sequence.scale_factors, expected_result.expected_scale_factors)
                    self.assertEqual(match_sequence.kernels, expected_result.expected_kernels)
                    self.assertEqual(match_sequence.average_kernel, expected_result.expected_average_kernel)

    def test_matching_results(self):
        # arrange
        @dataclass
        class TestCase:
            name: str
            matching_result: list[MatchingResult]
            cutoff: float
            expected_sorted: list[str]
            expected_top_phase: tuple[str, float]
            expected_num_matches: int

        # define test cases
        test_cases = [
            TestCase(
                name="basic",
                matching_result=MatchingResult(phases=["A", "B", "C"], kernels=[0.3, 0.88, 0.1]),
                cutoff=0,
                expected_sorted=["B", "A", "C"],
                expected_top_phase=("B", 0.88),
                expected_num_matches=3,
            ),
            TestCase(
                name="with cutoff",
                matching_result=MatchingResult(phases=["D", "G", "F"], kernels=[0.3, 0.15, 0.95]),
                cutoff=0.3,
                expected_sorted=["F", "D"],
                expected_top_phase=("F", 0.95),
                expected_num_matches=2,
            ),
        ]

        # act
        for case in test_cases:
            with self.subTest(case.name):
                case.matching_result.filter_by_cutoff(case.cutoff)
                case.matching_result.sort()
                self.assertEqual(case.matching_result.phases, case.expected_sorted)
                self.assertEqual(case.matching_result.top_phase, case.expected_top_phase)
                self.assertEqual(case.matching_result.num_matches, case.expected_num_matches)
                self.assertTrue(case.matching_result.is_sorted)
