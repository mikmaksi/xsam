import unittest
from tempfile import NamedTemporaryFile

import xsam.tests.constants as constants
from xsam.constants import SIGNAL_TYPE
from xsam.settings import SearchMatchSettings, SpectrumSettings


class TestSettings(unittest.TestCase):
    def test_init(self):
        # arrange
        spectrum_settings = SpectrumSettings(
            min_angle=10.0, max_angle=90.0, domain_size=30.0, n_points=4501, wavelength="CuKa"
        )
        search_match_settings = SearchMatchSettings(
            max_phases=2,
            signal_cutoff=0.10,
            signal_type=SIGNAL_TYPE.MAX_INTENSITY,
            min_kernel=0.50,
            spectrum_settings=spectrum_settings,
        )
        expected_search_match_settings = SearchMatchSettings.from_json(
            constants.TEST_FILE_DIR.joinpath("search_match_settings.json")
        )

        # act
        with NamedTemporaryFile(suffix=".json", dir=".") as temp_file:
            search_match_settings.to_json(temp_file.name)
            search_match_settings_reloaded = SearchMatchSettings.from_json(temp_file.name)

        # assert
        self.assertEqual(search_match_settings_reloaded, expected_search_match_settings)
