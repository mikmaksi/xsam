from copy import deepcopy
from typing import Optional

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance

from xsam import logger
from xsam.constants import TERMINATION_CONDITION
from xsam.match import Match, MatchEnsemble, MatchingResult, MatchSequence
from xsam.operations import SpectrumSubtraction
from xsam.settings import SearchMatchSettings
from xsam.spectrum import Spectrum, SpectrumCollection


class PhaseIdentifier:
    def __init__(
        self, spectrum: Spectrum, reference_collection: SpectrumCollection, search_match_settings: SearchMatchSettings
    ):
        """
        TODO
        """
        self.spectrum = spectrum
        self.reference_collection = reference_collection
        self.search_match_settings = search_match_settings

    def identify_phases(self) -> MatchEnsemble:
        """
        TODO:
        """

        # identify possible phases iteratively and return an ensemble
        logger.info("Starting route enumeration")
        match_ensemble: MatchEnsemble = self.enumerate_routes(input_spectrum=self.spectrum)

        return match_ensemble

    def enumerate_routes(
        self,
        input_spectrum: Spectrum,
        match_sequence: Optional[MatchSequence] = None,
        match_ensemble: Optional[MatchEnsemble] = None,
        is_first: bool = True,
        branch_depth: int = 0,
    ) -> MatchEnsemble:
        """
        TODO
        """

        # search for possbile matches
        matching_result = self.find_matches(input_spectrum)

        if is_first:
            # start a new match sequence and add it to the ensemble
            match_ensemble = MatchEnsemble()
            match_sequence = MatchSequence(input_signal=getattr(input_spectrum, self.search_match_settings.signal_type))
            match_ensemble.match_sequences.append(match_sequence)

        # remove existing matches, so we don't match the same phase twice
        matching_result.filter_by_name(match_sequence.phases)

        # debugging
        # logger.info("Matching result", matching_result_phases=matching_result.phases)

        # check if there are no matches
        if matching_result.num_matches == 0:
            match_sequence.is_complete = True
            match_sequence.termination_condition = TERMINATION_CONDITION.NO_MATCHES
            return match_ensemble

        # explore matches recursively
        for i, (phase, kernel) in enumerate(zip(matching_result.phases, matching_result.kernels)):
            # start a new sequence from the previous sequence
            match_sequence = MatchSequence.from_match_sequence(match_sequence, branch_depth)
            match_ensemble.match_sequences.append(match_sequence)

            # skip a matched phase if it has already been identified
            if phase in match_sequence.phases:
                continue

            # scale the phase spectrum to max of the input spectrum
            phase_spectrum = deepcopy(self.reference_collection.get_spectrum_by_name(phase))
            phase_spectrum.normalize(max_intensity=input_spectrum.y.max())

            # add the matched phase to the sequence
            match = Match(
                phase=phase,
                kernel=kernel,
                phase_spectrum=phase_spectrum,
                input_spectrum=input_spectrum,
                scale_factor=input_spectrum.y.max(),
            )
            match_sequence.matches.append(match)

            # subtract the matched phase from the input spectrum
            spectrum_subtraction = SpectrumSubtraction([input_spectrum, match.phase_spectrum])
            remain_spectrum, aligned_spectrum = spectrum_subtraction.apply(downsampling_resolution=0.1)
            match.aligned_spectrum = aligned_spectrum
            match.remain_spectrum = remain_spectrum

            # return if the maximum number of phases has been identified
            if len(match_sequence.matches) >= self.search_match_settings.max_phases:
                match_sequence.is_complete = True
                match_sequence.termination_condition = TERMINATION_CONDITION.MAX_PHASES

            # return if the residual drops below a cutoff signal
            remain_signal = getattr(remain_spectrum, self.search_match_settings.signal_type)
            signal_ratio = remain_signal / match_sequence.input_signal
            if signal_ratio < self.search_match_settings.signal_cutoff:
                match_sequence.is_complete = True
                match_sequence.termination_condition = TERMINATION_CONDITION.SIGNAL_CUTOFF

            # user info
            if match_sequence.is_complete:
                logger.info(
                    "Match sequence",
                    phases=match_sequence.phases,
                    is_complete=match_sequence.is_complete,
                    termination_condition=match_sequence.termination_condition,
                    average_kernel=f"{match_sequence.average_kernel:0.3f}",
                    signal_ratio=f"{signal_ratio:0.3f}",
                )
                continue

            # explore additional matches
            match_ensemble = self.enumerate_routes(
                input_spectrum=remain_spectrum,
                match_sequence=match_sequence,
                match_ensemble=match_ensemble,
                is_first=False,
                branch_depth=branch_depth + 1,
            )
        return match_ensemble

    def find_matches(self, spectrum: Spectrum) -> MatchingResult:
        """
        TODO:
        """
        # append the current spectrum to the reference array
        intensity_array = np.vstack([spectrum.y, self.reference_collection.reference_array])
        if self.search_match_settings.similarity_function == "earth-mover-distance":
            kernel_matrix = 1 - squareform(pdist(intensity_array, lambda u, v: wasserstein_distance(u, v).sum()))
        else:
            kernel_matrix = 1 - squareform(
                pdist(intensity_array, metric=self.search_match_settings.similarity_function)
            )

        # create a matching result
        matching_result = MatchingResult(phases=self.reference_collection.phases, kernels=kernel_matrix[1:, 0])
        matching_result.filter_by_cutoff(self.search_match_settings.min_kernel)
        matching_result.sort()

        return matching_result
