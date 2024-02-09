from abc import ABC
from copy import deepcopy

import numpy as np
import pandas as pd
from pyts import metrics

from xsam.spectrum import Spectrum
from xsam.exceptions import XsamException


class SpectrumOperation(ABC):
    def __init__(self, spectra: list[Spectrum]):
        self.spectra = spectra

    def vaildate_spectra(self):
        """
        Check that all spectra share the same config.
        """

        spectrum = self.spectrum[0]
        for i, next_spectrum in enumerate(self.spectrum[1:]):
            if spectrum.spectrum_settings != next_spectrum.spectrum_settings:
                raise XsamException(
                    "Spectrum configuration mismatch.", {"spectrum_index": i - 1, "next_spectrum_index": i}
                )
            spectrum = next_spectrum


class SpectrumSubtraction(SpectrumOperation):
    @staticmethod
    def subtract_patterns(
        first_spectrum: Spectrum, second_spectrum: Spectrum, censor_negative: bool = True
    ) -> Spectrum:
        remain_spectrum = deepcopy(first_spectrum)
        remain_spectrum.y = remain_spectrum.y - second_spectrum.y
        if censor_negative:
            remain_spectrum.y[remain_spectrum.y < 0] = 0
        return remain_spectrum

    def apply(
        self,
        last_normalization: float = 1.0,
        downsampling_resolution: float = 1.0,
        allow_shifts: float = 0.75,
        query_threshold: float = 0.01,
    ) -> tuple[Spectrum, Spectrum]:
        """
        1. Align the query with the target spectrum to account for peak shifting, i.e. shift query pattern such that
        query peaks align with the target peaks.
        2. Create a new "aligned" query pattern that has only peaks from the target at angle positions where peaks are
        present in the query after alignment.
            - There may be large differences in peak intensities from strain, texture, etc., so simply aligning query
            to target is not enough.
        3. Subtract the aligned query from the target .
        """
        # get the spectra
        target_spectrum: Spectrum = deepcopy(self.spectra[0])
        query_spectrum: Spectrum = deepcopy(self.spectra[1])
        spectrum_settings = target_spectrum.spectrum_settings  # already validated that this is identical b/w spectra

        # prepare for dynamic time-warping
        angle_range = spectrum_settings.max_angle - spectrum_settings.min_angle
        down_n_points = int(spectrum_settings.n_points * downsampling_resolution)
        window_size = int(allow_shifts * down_n_points / angle_range)
        target_spectrum.resample(down_n_points)
        query_spectrum.resample(down_n_points)

        # align the query spectrum to the target
        distance, path = metrics.dtw(
            target_spectrum.y,
            query_spectrum.y,
            method="sakoechiba",
            options={"window_size": window_size},
            return_path=True,
        )
        index_pairs = path.transpose()
        aligned_spectrum = deepcopy(query_spectrum)
        aligned_spectrum.y = np.zeros_like(query_spectrum.y)

        # put the path indexes into a DataFrame
        aligned_df = pd.DataFrame(index_pairs).rename(columns={0: "target_idx", 1: "query_idx"})

        # add the y values
        aligned_df["query_y"] = query_spectrum.y[aligned_df["query_idx"]]
        aligned_df["target_y"] = target_spectrum.y[aligned_df["target_idx"]]

        # find the average y values at each target index
        aligned_df = aligned_df.groupby("target_idx").mean().sort_index().reset_index()

        # zero out any values where the query is low
        aligned_df.loc[aligned_df["query_y"] < query_threshold, "target_y"] = 0

        # take intensity from the target
        aligned_spectrum.y = aligned_df["target_y"].to_numpy()

        # upsample the spectra back
        target_spectrum.resample(spectrum_settings.n_points)
        query_spectrum.resample(spectrum_settings.n_points)

        # subtract aligned spectrum from the original
        remain_spectrum = self.subtract_patterns(target_spectrum, query_spectrum)
        remain_spectrum.smooth()

        return remain_spectrum, aligned_spectrum
