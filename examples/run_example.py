# %%
# import
from pathlib import Path

from click.testing import CliRunner
from xsam.cli import search
from xsam.constants import SIGNAL_TYPE
from xsam.settings import SearchMatchSettings, SpectrumSettings

# %%
# config
reference_dir = Path("references")
spectra_dir = Path("spectra")

# %%
# search/match settings
spectrum_settings = SpectrumSettings(min_angle=10.0, max_angle=100.0)
search_match_settings = SearchMatchSettings(
    max_phases=3,
    signal_cutoff=0.1,
    signal_type=SIGNAL_TYPE.MAX_INTENSITY,
    min_kernel=0.4,
    spectrum_settings=spectrum_settings,
)
settings_path = "search_match_setings.json"
search_match_settings.to_json(settings_path)

# %%
# set up the cli Runner
runner = CliRunner()

# %%
spectrum_path = spectra_dir.joinpath("Li2MnO3+MnO+TiO2.xy")
result = runner.invoke(search, [str(spectrum_path), str(reference_dir), settings_path])

# %%
print(result.stdout)
