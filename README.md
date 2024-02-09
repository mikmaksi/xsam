# XRD search-and-match (XSAM)

A command line tool for identifying phases that may be present in an XRD diffractogram. 

A number of tools have recently been developed for identifying phases that may be present in a crystalline sample from an experimentally observed powder X-ray diffraction pattern. Many of these methods use advanced machine learning techniques to predict phases from the peak fingerprint (e.g. CNN). This builds and often improves upon the traditional pattern search-and-match approach that uses similarity metrics to compare the target and query phase across the full pattern profile. However, using traditional search-and-match methods may be advantageous in some case such as better interpretability or avoiding training a custom model in a chemical space of interest.

Tradition phase finding routines are implemented in many of the most popular software tools for XRD analaysis (Bruker EVA, JADE etc.), but often lack a command line interface or API. This makes these tools difficult to use in a high-throughput manner, integrate with other data workflows or extend. This project was motivated by the lack of lightweight, modular, extensible tools for traditional phase identification from XRD patterns and aims to provide a simple interface for phase identification that can run in a cloud environment or be integrated with other chemi-informatics tools. 

# Method 

Possible phases are identified in the following way.

1. Find the top match to the input spectrum using traditional full pattern search/match (similarity score).
2. Subtract the top match from the input pattern.
3. Repeat until the maximum allowed number of phases has been identified or pattern intensity falls below a certain cutoff.

Since the top match at any given step may not in fact correspond to a true phase present in the pattern, all possible matching sequence branches are enumerated and the branch with the highest average similarity is picked. This approach was adapted from [XRD-AutoAnalyzer](https://github.com/njszym/XRD-AutoAnalyzer), which uses a deep learing model to identify potential phase matches instead of a traditional similarity score. 

## Installation

Typical installation via setuptools is supported: `pip intall .`. 

## Usage

The `xsam` command should available after installation. The cli was built using `click` and follows its `command subcommand` interface. 

- `xsam --help`: get the list of available subcommands supported by `xsam`
- `xsam search --help`: get the list of arguments and options for the search-and-match routing
- `xsam search spectrum.xy reference_dir settings.json`: run pattern search-and-match on XRD spectrm data in `.xy` format against a reference set of `.cif` structure files located in `reference_dir` (more details below).

`xrd-search-and-match` methods from this package can also be imported as a module in the traditional way or launched via a python script using the `CliRunner` from `click.testing`. An example of this is given in `examples/run_example.py`.

## Example output

This package uses structured logging and the [rich python library](https://github.com/Textualize/rich) is recommended to be installed to enhance the logger output. An example can be found in `examples/run_example.py`. Phases are identified from a pattern known to be a mixture of Li2MnO3, MnO and TiO2. In addition to the

## Tree visualization

TODO:

## Input spectrum files

Currently a simple `.xy` tab-separated format is supported for inputting experimental XRD data.

## Reference structure files

A reference set of diffraction patterns is needed. Currently, the tool is configured to use `.cif` crystal structure files as a reference set and generates idealized XRD patterns from the structure definition using the `pymatgen` package. 


## SearchMatchSettings

To avoid having to specify all search-and-match options on the command line, `xrd-search-and-match` uses a configuration file in `.json` format instead. The configuration file can be easily made using the helper `settings.SearchMatchSettings` class in the following way.

```
from xrd_search.settings import SearchMatchSettings

search_and_match_settings = SearchMatchSettings()
search_and_match_settings.to_json("settings.json")
```

Defaults have been built into `SearchMatchSettings`, but can be modified to suit ones use-case.

## Testing

This package uses the unittest library for testing and `pytest` is recommend for running the tests. A number of test can optionally generate a plot to visualize the test result then the `--plots` flag is enable: `pytest --plots xrd_search/tests/test_spectrum.py`. 

## Disclaimer

This package is provided as a personal side-project and is not intended for production use. The author makes no warranties, either express or implied, about the correctness or suitability of this software for any purpose. Users are encouraged to review and modify the code to meet their specific needs and are solely responsible for how they choose to use it. By using this package, you agree that the author and contributors are not liable for any damages or consequences resulting from its use.
