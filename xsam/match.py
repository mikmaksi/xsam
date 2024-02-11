from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotnine as pn
from pydantic import Field, field_serializer, model_validator

from xsam import logger
from xsam.constants import PLOT_FORMAT, TERMINATION_CONDITION
from xsam.exceptions import XsamException
from xsam.pydantic_config import Model
from xsam.settings import SearchMatchSettings
from xsam.spectrum import Spectrum


class MatchingResult(Model):
    phases: list[str]
    kernels: list[float]
    is_sorted: bool = False
    phase_kernel_dict: Optional[dict] = None

    @model_validator(mode="after")
    def check_phases_kernels_length(self) -> "MatchingResult":
        if len(self.phases) != len(self.kernels):
            raise XsamException(
                "Number of phases and kernel scores should be equal",
                {"num_phases": len(self.phases), "num_kernels": len(self.kernels)},
            )
        return self

    def __init__(self, **data) -> None:
        super().__init__(**data)
        self.phase_kernel_dict = dict(zip(self.phases, self.kernels))

    def reset_phases_kernels(self) -> None:
        self.phases = list(self.phase_kernel_dict.keys())
        self.kernels = list(self.phase_kernel_dict.values())

    def filter_by_cutoff(self, min_kernel: float) -> None:
        self.phase_kernel_dict = {
            phase: kernel for phase, kernel in self.phase_kernel_dict.items() if kernel >= min_kernel
        }
        self.reset_phases_kernels()
        self.sort()

    def filter_by_name(self, phases: list[str]) -> None:
        """
        TODO
        """
        self.phase_kernel_dict = {
            phase: kernel for phase, kernel in self.phase_kernel_dict.items() if phase not in phases
        }
        self.reset_phases_kernels()
        self.sort()

    def sort(self) -> None:
        """
        TODO:
        """
        self.phase_kernel_dict = dict(sorted(self.phase_kernel_dict.items(), key=lambda x: x[1], reverse=True))
        self.reset_phases_kernels()
        self.is_sorted = True

    @property
    def top_phase(self) -> list[str, float]:
        if not self.is_sorted:
            self.sort()
        return self.phases[0], self.kernels[0]

    @property
    def num_matches(self) -> int:
        return len(self.phases)


class Match(Model):
    phase: str
    kernel: float
    phase_spectrum: Spectrum = Field(repr=False)
    input_spectrum: Optional[Spectrum] = Field(default=None, repr=False)  # input to the phase matching step
    # the pattern of the phase aligned to the input
    aligned_spectrum: Optional[Spectrum] = Field(default=None, repr=False)
    # remainder spectrum after aligned phase subtraction
    remain_spectrum: Optional[Spectrum] = Field(default=None, repr=False)
    scale_factor: Optional[float] = None

    def spectra_to_df(self) -> pd.DataFrame:
        data_list = []
        for spectrum, label in zip(
            [self.input_spectrum, self.phase_spectrum, self.aligned_spectrum, self.remain_spectrum],
            ["input", "phase", "aligned", "remain"],
        ):
            if spectrum is not None:
                data_list.append(spectrum.to_df().assign(label=label))
        data = pd.concat(data_list)

        return data

    def __eq__(self, other: "Match") -> bool:
        # When comparing instances of generic types for equality, as long as all field values are equal,
        # only require their generic origin types to be equal, rather than exact type equality.
        # This prevents headaches like MyGeneric(x=1) != MyGeneric[Any](x=1).
        self_type = self.__pydantic_generic_metadata__["origin"] or self.__class__
        other_type = other.__pydantic_generic_metadata__["origin"] or other.__class__

        if self_type != other_type:
            return False

        required_fields = [
            field_name for field_name, field_info in self.model_fields.items() if field_info.is_required()
        ]
        return {k: v for k, v in self.__dict__.items() if k in required_fields} == {
            k: v for k, v in other.__dict__.items() if k in required_fields
        }

    def plot_overlay(self, path: Optional[str] = None, labels: Optional[list[str]] = None) -> Union[pn.ggplot, None]:
        """
        TODO
        """

        # create a DataFrame with each of the available spectra
        data = self.spectra_to_df()

        # filter by label
        if labels is not None:
            data = data[data.label.isin(labels)]
            categories = labels
        else:
            categories = ["input", "phase", "aligned", "remain"]

        # make plot
        p = pn.ggplot(
            data=data.assign(label=lambda df: pd.Categorical(df["label"], categories=categories)),
            mapping=pn.aes("x", "y", color="label", linetype="label"),
        )
        p = p + pn.geom_line() + pn.theme_gray(base_size=16) + pn.theme(figure_size=(6, 4))

        # save
        if path is not None:
            p.save(path)
        else:
            return p

    def plot_diagnostic(self, path: Optional[str] = None) -> Union[list[pn.ggplot], None]:
        """
        TODO
        """

        # create a DataFrame with each of the available spectra
        data = self.spectra_to_df()

        # all traces
        p1 = pn.ggplot(
            data=data.assign(
                label=lambda df: pd.Categorical(df["label"], categories=["input", "phase", "aligned", "remain"])
            ),
            mapping=pn.aes("x", "y", color="label"),
        )
        p1 = (
            p1
            + pn.geom_line()
            + pn.facet_wrap("label", ncol=1)
            + pn.theme_gray(base_size=16)
            + pn.theme(figure_size=(7, 8))
        )

        # input vs phase
        p2 = pn.ggplot(
            data=data.query("~label.isin(['aligned', 'remain'])").assign(
                label=lambda df: pd.Categorical(df["label"], categories=["input", "phase"])
            ),
            mapping=pn.aes("x", "y", color="label"),
        )
        p2 = p2 + pn.geom_line() + pn.theme_gray(base_size=16)

        # input vs aligned
        p3 = pn.ggplot(
            data=data.query("~label.isin(['phase', 'remain'])").assign(
                label=lambda df: pd.Categorical(df["label"], categories=["input", "aligned"])
            ),
            mapping=pn.aes("x", "y", color="label"),
        )
        p3 = p3 + pn.geom_line() + pn.theme_gray(base_size=16)

        # save plot
        if path is not None:
            pn.save_as_pdf_pages([p1, p2, p3], path, verbose=False)
        else:
            return [p1, p2, p3]


class MatchSequence(Model):
    matches: list[Match] = []
    termination_condition: Optional[TERMINATION_CONDITION] = None
    is_complete: bool = False
    input_signal: Optional[float] = None

    @property
    def phases(self) -> list[str]:
        return [match.phase for match in self.matches]

    @property
    def scale_factors(self) -> list[float]:
        return [match.scale_factor for match in self.matches]

    @property
    def kernels(self) -> list[float]:
        return [match.kernel for match in self.matches]

    @property
    def average_kernel(self) -> Union[float, None]:
        if self.kernels == []:
            return None
        return np.mean(self.kernels)

    @classmethod
    def from_match_sequence(cls, match_sequence: "MatchSequence", last_step: int) -> "MatchSequence":
        return MatchSequence(
            matches=match_sequence.matches[:last_step],
            termination_condition=match_sequence.termination_condition,
            is_complete=match_sequence.is_complete,
            input_signal=match_sequence.input_signal,
        )

    def plot(
        self, path_wo_suffix: Optional[str] = None, plot_format: PLOT_FORMAT = PLOT_FORMAT.PNG
    ) -> Union[list[pn.ggplot], None]:
        plot_list = [
            match.plot_overlay(labels=["input", "phase", "aligned"]) + pn.labs(title=f"Step {i}: {match.phase}")
            for i, match in enumerate(self.matches)
        ]
        if path_wo_suffix is not None:
            # check that the path does not have a suffix
            path_wo_suffix = Path(path_wo_suffix)
            if path_wo_suffix.suffix != "":
                raise XsamException("Path should be w/o suffix", {"path_wo_suffix": path_wo_suffix})

            # plot according to the format
            if plot_format == PLOT_FORMAT.PDF:
                pn.save_as_pdf_pages(plot_list, f"{path_wo_suffix}.{plot_format.value}", verbose=False)
            elif plot_format == PLOT_FORMAT.PNG:
                # save plots separately
                path_wo_suffix.mkdir(exist_ok=True)
                for i, p in enumerate(plot_list):
                    p.save(path_wo_suffix.joinpath(f"step_{i}.{plot_format.value}"), verbose=False)
        else:
            return plot_list


class MatchEnsemble(Model):
    match_sequences: list[MatchSequence] = []

    @property
    def top_match_sequence(self) -> Union[MatchSequence, None]:
        kernels = [
            match_sequence.average_kernel for match_sequence in self.complete_sequences if match_sequence.matches != []
        ]
        if kernels != []:
            return self.complete_sequences[np.argmax(kernels)]

    def get_num_sequences(self, complete_only: bool = False) -> int:
        if complete_only:
            return len(self.complete_sequences)
        return len(self.match_sequences)

    @property
    def complete_sequences(self) -> list[MatchSequence]:
        """return complete sequences that have at least one match."""
        # keep complete match sequences
        match_sequences = [match_sequence for match_sequence in self.match_sequences if match_sequence.is_complete]

        # keep match sequences with at least one match
        match_sequences = [match_sequence for match_sequence in match_sequences if match_sequence.matches != []]
        return match_sequences

    @property
    def match_found(self) -> bool:
        return self.complete_sequences != []

    def get_edges_summary(self) -> pd.DataFrame:
        # many ids to one phase
        id_uid_lookup = {}
        for match_sequence in self.complete_sequences:
            for match in match_sequence.matches:
                id_uid_lookup.setdefault(match.phase, []).append(id(match))
        id_uid_lookup = {
            phase: {uid: i for i, uid in enumerate(sorted(list(set(id_list))))}
            for phase, id_list in id_uid_lookup.items()
        }

        @dataclass
        class PathElement:
            phase: str
            uid: int

        # create a list of paths
        list_of_paths = [
            [
                PathElement(phase=match.phase.replace(".cif", ""), uid=id_uid_lookup[match.phase][id(match)])
                for match in match_sequence.matches
            ]
            for match_sequence in self.complete_sequences
        ]

        # converted to edges in a network
        edges = []
        for i, path in enumerate(list_of_paths):
            for j in range(len(path) - 1):
                edges.append(
                    {
                        "from_name": path[j].phase,
                        "from_id": path[j].uid,
                        "to_name": path[j + 1].phase,
                        "to_id": path[j + 1].uid,
                    }
                )
        edges = pd.DataFrame(edges)
        return edges

    def get_paths_summary(self) -> pd.DataFrame:
        data_list = []
        for match_sequence in self.complete_sequences:
            data_list.append(
                {"phases": ", ".join(match_sequence.phases), "avg_similarity": match_sequence.average_kernel}
            )
        paths_summary = pd.DataFrame(data_list)
        return paths_summary

    def plot_explored_paths(
        self, path_wo_suffix: Optional[str] = None, plot_format: PLOT_FORMAT = PLOT_FORMAT.PNG
    ) -> Union[list[pn.ggplot], None]:
        # create a DataFrame of edges
        edges = self.get_edges_summary()

        fig, ax = plt.subplots()
        if edges.empty:
            logger.info("No edges when plotting ensemble.")
        else:
            # combine name and id
            edges["from"] = edges.apply(lambda x: f"{x.from_name}-{x.from_id}", axis="columns")
            edges["to"] = edges.apply(lambda x: f"{x.to_name}-{x.to_id}", axis="columns")

            # create a graph
            G = nx.DiGraph()
            for i, (match_from, match_to) in edges[["from", "to"]].iterrows():
                G.add_edge(match_from, match_to)

            # create the layout
            pos = nx.nx_pydot.graphviz_layout(G, prog="dot")

            # create figure
            nx.draw(
                G,
                pos=pos,
                with_labels=True,
                node_color="none",
                bbox=dict(facecolor="skyblue", edgecolor="black", boxstyle="round,pad=0.2"),
            )

        # save
        if path_wo_suffix is not None:
            # check that the path does not have a suffix
            path_wo_suffix = Path(path_wo_suffix)
            if path_wo_suffix.suffix != "":
                raise XsamException("Path should be w/o suffix", {"path_wo_suffix": path_wo_suffix})

            # save figure
            fig.savefig(f"{path_wo_suffix}.{plot_format.value}")
        else:
            return fig


class MatchSequenceSummary(Model):
    phases: list[str] = Field(default_factory=list)
    kernels: list[float] = Field(default_factory=list)
    scale_factors: list[float] = Field(default_factory=list)
    termination_condition: Optional[TERMINATION_CONDITION] = None

    @classmethod
    def from_match_sequence(cls, match_sequence: MatchSequence) -> "MatchSequenceSummary":
        return cls(
            phases=match_sequence.phases,
            kernels=match_sequence.kernels,
            scale_factors=match_sequence.scale_factors,
            termination_condition=match_sequence.termination_condition,
        )


class MatchEnsembleSummary(Model):
    spectrum_path: str
    reference_dir: str
    match_found: bool
    top_match: MatchSequenceSummary
    edges: pd.DataFrame
    paths: pd.DataFrame
    search_match_settings: SearchMatchSettings

    @field_serializer("edges", "paths")
    def serialize_df(self, data_frame: pd.DataFrame):
        return data_frame.to_json()

    @classmethod
    def from_match_ensemble(
        cls,
        spectrum_path: str,
        reference_dir: str,
        search_match_settings: SearchMatchSettings,
        match_ensemble: MatchEnsemble,
    ) -> "MatchEnsembleSummary":
        if match_ensemble.match_found:
            top_match_sequence_summary = MatchSequenceSummary.from_match_sequence(match_ensemble.top_match_sequence)
            match_found = True
        else:
            top_match_sequence_summary = MatchSequenceSummary(termination_condition=TERMINATION_CONDITION.NO_MATCHES)
            match_found = False
        return cls(
            spectrum_path=spectrum_path,
            reference_dir=reference_dir,
            search_match_settings=search_match_settings,
            match_found=match_found,
            top_match=top_match_sequence_summary,
            edges=match_ensemble.get_edges_summary(),
            paths=match_ensemble.get_paths_summary(),
        )
