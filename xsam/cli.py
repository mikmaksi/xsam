from pathlib import Path

import click
from monty.serialization import dumpfn, loadfn

from xsam import logger
from xsam.constants import TERMINATION_CONDITION
from xsam.match import MatchEnsembleSummary
from xsam.phase_identifier import PhaseIdentifier
from xsam.settings import SearchMatchSettings
from xsam.spectrum import Spectrum, SpectrumCollection


@click.group()
def cli():
    """cli entry point.
    """
    pass


@cli.command()
@click.argument("spectrum_path", type=click.Path(exists=True))
@click.argument("reference_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("settings_path", type=click.Path(exists=True))
@click.option("--cache", type=bool, default=False, is_flag=True)
@click.option("--out", type=click.Path(file_okay=False, dir_okay=True), default="out")
def search(spectrum_path: str, reference_dir: str, settings_path: str, cache: bool, out: str):
    """search for phases in a spectrum.

    Args:
        spectrum_path (str): path to the spectrum file in .xy format.
        reference_dir (str): directory containing .cif xtal structure files.
        settings_path (str): path to the SearchMatchSettings object serialized in JSON format.
        cache (bool): save/reload reference structures as a cache file.
        out (str): output directory to write result output files.
    """
    # read the settings
    search_match_settings = SearchMatchSettings.from_json(settings_path)

    # load the spectrum
    spectrum = Spectrum.from_xy(path=spectrum_path, spectrum_settings=search_match_settings.spectrum_settings)

    # load the reference spectra
    if cache:
        reference_spectra = SpectrumCollection.model_validate(loadfn("reference_spectra_cache.json"))
    else:
        reference_spectra = SpectrumCollection.from_reference_dir(
            reference_dir, search_match_settings.spectrum_settings
        )
        dumpfn(reference_spectra.model_dump(), "reference_spectra_cache.json")

    # run the identifier
    phase_identifier = PhaseIdentifier(
        spectrum=spectrum, reference_collection=reference_spectra, search_match_settings=search_match_settings
    )
    match_ensemble = phase_identifier.identify_phases()

    # create an output directory if necessary
    out = Path(out)
    if not out.exists():
        out.mkdir()

    # save a summary of the ensemble
    match_ensemble_summary = MatchEnsembleSummary.from_match_ensemble(match_ensemble)
    dumpfn(match_ensemble_summary.model_dump(), out.joinpath("ensemble_summary.json"), indent=4)

    if match_ensemble.match_found:
        # visualize the top match sequence
        match_ensemble.top_match_sequence.plot(out.joinpath("identified_phases.png"))

        # visualize the the explored paths through the network
        match_ensemble.plot_explored_paths(out.joinpath("ensemble_paths.png"))
    else:
        logger.info("No matches found")
