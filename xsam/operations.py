from abc import ABC
from copy import deepcopy

import numpy as np
import pandas as pd
from pyts import metrics

from xsam.exceptions import XsamException
from xsam.spectrum import Spectrum


class SpectrumOperation(ABC):
    def __init__(self, target: Spectrum, query: Spectrum) -> None:
        self.target = target
        self.query = query
        self.validate_spectra()

    def validate_spectra(self):
        """
        Check that all spectra share the same config.
        """

        if self.query.spectrum_settings != self.target.spectrum_settings:
            raise XsamException("Spectrum configuration mismatch.")


class SpectrumAlignment(SpectrumOperation):
    def apply(
        self,
        last_normalization: float = 1.0,
        downsampling_resolution: float = 1.0,
        allow_shifts: float = 0.75,
        query_threshold: float = 0.01,
    ) -> Spectrum:
        """
        1. Align the query with the target spectrum to account for peak shifting, i.e. shift query pattern such that
        query peaks align with the target peaks.
        2. Create a new "aligned" query pattern that has only peaks from the target at angle positions where peaks are
        present in the query after alignment.
            - There may be large differences in peak intensities from strain, texture, etc., so simply aligning query
            to target is not enough.
        3. Subtract the aligned query from the target .
        """

        # copy objects
        target = deepcopy(self.target)
        query = deepcopy(self.query)

        # prepare for dynamic time-warping
        spectrum_settings = target.spectrum_settings
        angle_range = spectrum_settings.max_angle - spectrum_settings.min_angle
        down_n_points = int(spectrum_settings.n_points * downsampling_resolution)
        window_size = int(allow_shifts * down_n_points / angle_range)

        # downsample the spectra
        target.resample(down_n_points)
        query.resample(down_n_points)

        # align the query spectrum to the target
        distance, path = metrics.dtw(
            target.y, query.y, method="sakoechiba", options={"window_size": window_size}, return_path=True
        )
        index_pairs = path.transpose()
        aligned_spectrum = deepcopy(query)
        aligned_spectrum.y = np.zeros_like(query.y)

        # put the path indexes into a DataFrame
        aligned_df = pd.DataFrame(index_pairs).rename(columns={0: "target_idx", 1: "query_idx"})

        # add the y values
        aligned_df["query_y"] = query.y[aligned_df["query_idx"]]
        aligned_df["target_y"] = target.y[aligned_df["target_idx"]]

        # find the average y values at each target index
        aligned_df = aligned_df.groupby("target_idx").mean().sort_index().reset_index()

        # zero out any values where the query is low
        aligned_df.loc[aligned_df["query_y"] < query_threshold, "target_y"] = 0

        # take intensity from the target
        aligned_spectrum.y = aligned_df["target_y"].to_numpy()

        # upsample the spectra back
        query.resample(spectrum_settings.n_points)

        return query


class SpectrumSubtraction(SpectrumOperation):
    def apply(self, censor_negative: bool = True) -> Spectrum:
        # TODO: avoid using deepcopy here
        remain_spectrum = deepcopy(self.target)
        remain_spectrum.y = remain_spectrum.y - self.query.y
        if censor_negative:
            remain_spectrum.y[remain_spectrum.y < 0] = 0
        remain_spectrum.smooth()
        return remain_spectrum
