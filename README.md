# XRD search-and-match (XSAM)

A command line tool for identifying phases that may be present in an XRD diffractogram.

A number of tools have recently been developed for identifying phases that may be present in a crystalline sample from an experimentally observed powder X-ray diffraction pattern. Many of these methods use advanced machine learning techniques to predict phases from the peak fingerprint (e.g. CNN). This builds and often improves upon the traditional pattern search-and-match approach that uses similarity metrics to compare the target and query phase across the full pattern profile. However, using traditional search-and-match methods may be advantageous in some case such as better interpretability or avoiding training a custom model in a chemical space of interest.

Tradition phase finding routines are implemented in many of the most popular software tools for XRD analaysis (Bruker EVA, JADE etc.), but often lack a command line interface or API. This makes these tools difficult to use in a high-throughput manner, integrate with other data workflows or extend. This project was motivated by the lack of lightweight, modular, extensible tools for traditional phase identification from XRD patterns and aims to provide a simple interface for phase identification that can run in a cloud environment or be integrated with other chemi-informatics tools.

## Method

Possible phases are identified in the following way.

1. Find the top match to the input spectrum using traditional full pattern search/match (similarity score).
2. Subtract the top match from the input pattern.
3. Repeat until the maximum allowed number of phases has been identified or pattern intensity falls below a certain cutoff.

Since the top match at any given step may not in fact correspond to a true phase present in the pattern, all possible matching sequence branches are enumerated and the branch with the highest average similarity is picked. This approach was adapted from [XRD-AutoAnalyzer](https://github.com/njszym/XRD-AutoAnalyzer), which uses a deep learing model to identify potential phase matches instead of a traditional similarity score.

## Installation

Standard installation vis setuptools: `pip intall .`

## Usage

The `xsam` command should available after installation. The cli was built using `click` and follows its `command subcommand` interface. 

- `xsam --help`: get the list of available subcommands supported by `xsam`.
- `xsam search --help`: get the list of arguments and options for the search subcommand.
- `xsam search spectrum.xy reference_dir settings.json`: run pattern search-and-match on XRD spectrum data in `.xy` format against a reference set of `.cif` structure files located in `reference_dir` (more details below).

Additional options:

- `--cache`: save the reference patterns to disk and/or reuse the cache.
- `--out`: directory to use for writing output files [default: `out`].

`xsam` methods from this package can also be imported as a module in the traditional way or launched via a python script using the `CliRunner` from `click.testing`. An example of how to do this is given in `examples/run_example.py`.

## `search` sub-command

Sequences of matches are printed out as they are identified using structured logging. Installing the python [rich](https://github.com/Textualize/rich) package is recommended to enhance the logger output.

```bash
Match sequence                 phases=['MnO_225.cif', 'Li2MnO3_12.cif', 'TiO2_141.cif'] is_complete=True termination_condition=max_phases average_kernel=0.673 remain_intensity=0.135
Match sequence                 phases=['MnO_225.cif', 'Li2MnO3_12.cif', 'Ti6O11_12.cif'] is_complete=True termination_condition=max_phases average_kernel=0.671 remain_intensity=0.142
Match sequence                 phases=['MnO_225.cif', 'Li2MnO3_12.cif', 'TiO2_1.cif'] is_complete=True termination_condition=max_phases average_kernel=0.660 remain_intensity=0.150
```

A results summary is also printed to a JSON file (e.g. `examples/out/ensemble_summary.json`) that contains information about the top sequence of matches and the paths of sequence matches that were explored. This summary follows the model defined by the `settings.SearchMatchSettings` class.

A visualization of the explored match sequences is saved as a network plothe explored match sequences is saved as a network plot.

![ensemble_paths](/examples/out/ensemble_paths.png)

Finally, an overlay of aligned matches phases and the input pattern is also saved as a series of images for each step in the sequence (e.g. `examples/out/identified_phases/*.png`). Each step is an overlay of the input pattern at that step ("input"), the best maching phase ("phase") and the matching phase after dynamic time warping alignment to the input pattern.

![step_0](/examples/out/identified_phases/step_0.png)
![step_1](/examples/out/identified_phases/step_1.png)
![step_2](/examples/out/identified_phases/step_2.png)

## Input spectrum files

Currently a simple `.xy` tab-separated format is supported for inputting experimental XRD data. Examples can be found in `examples/spectra`.

## Reference structure files

A reference set of diffraction patterns is needed. Currently, the tool is configured to use `.cif` crystal structure files as a reference set and generates idealized XRD patterns from the structure definition using the `pymatgen` package.

## SearchMatchSettings

To avoid having to pass search options through the command line, `xsam` uses a configuration file in `.json` format instead. The configuration file schema is defined in by the `settings.SearchMatchSettings` class can can be written to a file using the `settings.SearchMatchSettings.to_json` method.

```python
from xrd_search.settings import SearchMatchSettings

search_and_match_settings = SearchMatchSettings()
search_and_match_settings.to_json("settings.json")
```

Defaults have been built into `SearchMatchSettings`, but can be modified to suit ones use-case.

1. `max_phases`: maximum number of phases that can be searched for iteratively in a single input pattern.
2. `signal_cutoff`: terminate searching for additional phases when the proportion of signal falls below this value [0, 1].
3. `signal_type`: signal type to use to determine early termination of search (see. `constants.SIGNAL_TYPE`). Currently `max_intensity` and `auc` are supported.
3. `min_kernel`: minimum similarity score for a phase to be considered a match to the input pattern at a given step.
4. `spectrum_settings`: setting defining how patterns should be read from `.xy` files and generate from `.cif` structure files.
5. `similarity_function`: similarity metric to use to compare input pattern and potential matches (refer to `constants.SIMILARITY_FUNCTION` Enum).

## Testing

This package uses the unittest library for testing and `pytest` is recommend for running the tests. A number of test can optionally generate a plot to visualize the test result then the `--plots` flag is enable: `pytest --plots xrd_search/tests/test_spectrum.py`. 

## Disclaimer

This package is provided as a personal side-project and is not intended for production use. The author makes no warranties, either express or implied, about the correctness or suitability of this software for any purpose. Users are encouraged to review and modify the code to meet their specific needs and are solely responsible for how they choose to use it. By using this package, you agree that the author and contributors are not liable for any damages or consequences resulting from its use.
