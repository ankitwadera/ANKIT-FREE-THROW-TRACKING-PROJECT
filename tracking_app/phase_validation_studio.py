from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "phase_biomechanics_validation.csv"
)

DEFAULT_OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "phase_validation_studio"
)


SEGMENTS = (
    "knee",
    "hip",
    "pelvis",
    "shoulder",
    "elbow",
    "wrist",
    "ball",
)


@dataclass(frozen=True)
class SummaryRow:
    metric: str
    group: str
    count: int
    mean: float | None
    median: float | None
    standard_deviation: float | None
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True)
class ComparisonRow:
    metric: str
    made_count: int
    made_mean: float | None
    missed_count: int
    missed_mean: float | None
    difference_made_minus_missed: float | None


class PhaseValidationStudio:
    """
    Summarize and visualize refined phase-validation results.

    The studio reads the balanced validation CSV and creates:
    - descriptive statistics;
    - made-versus-missed comparisons;
    - propulsion-onset sequence frequencies;
    - peak sequence frequencies;
    - a four-panel dashboard.
    """

    def __init__(
        self,
        input_file: Path = DEFAULT_INPUT_FILE,
        output_folder: Path = DEFAULT_OUTPUT_FOLDER,
    ) -> None:
        self.input_file = Path(input_file)
        self.output_folder = Path(output_folder)

        self.summary_file = (
            self.output_folder
            / "phase_descriptive_statistics.csv"
        )

        self.comparison_file = (
            self.output_folder
            / "phase_make_miss_comparison.csv"
        )

        self.propulsion_sequence_file = (
            self.output_folder
            / "propulsion_sequence_frequency.csv"
        )

        self.peak_sequence_file = (
            self.output_folder
            / "peak_sequence_frequency.csv"
        )

        self.dashboard_file = (
            self.output_folder
            / "phase_validation_dashboard.png"
        )

    # =====================================================
    # PUBLIC WORKFLOW
    # =====================================================

    def run(self) -> list[dict[str, str]]:
        """
        Load validation data and create all outputs.
        """

        records = self.load_records()

        successful = [
            record
            for record in records
            if record.get("analysis_status") == "success"
        ]

        if not successful:
            raise RuntimeError(
                "No successful phase-validation records were found."
            )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        summaries = self.build_summaries(
            successful
        )

        comparisons = self.build_comparisons(
            successful
        )

        propulsion_sequences = Counter(
            record.get(
                "propulsion_onset_sequence",
                "",
            )
            for record in successful
            if record.get(
                "propulsion_onset_sequence",
                "",
            )
        )

        peak_sequences = Counter(
            record.get(
                "peak_sequence",
                "",
            )
            for record in successful
            if record.get(
                "peak_sequence",
                "",
            )
        )

        self.export_dataclass_rows(
            self.summary_file,
            summaries,
        )

        self.export_dataclass_rows(
            self.comparison_file,
            comparisons,
        )

        self.export_sequence_frequency(
            self.propulsion_sequence_file,
            propulsion_sequences,
        )

        self.export_sequence_frequency(
            self.peak_sequence_file,
            peak_sequences,
        )

        self.create_dashboard(
            records=successful,
            comparisons=comparisons,
            propulsion_sequences=propulsion_sequences,
            peak_sequences=peak_sequences,
        )

        return successful

    # =====================================================
    # LOAD DATA
    # =====================================================

    def load_records(
        self,
    ) -> list[dict[str, str]]:
        """
        Read the phase-validation CSV.
        """

        if not self.input_file.exists():
            raise FileNotFoundError(
                "Phase-validation CSV was not found: "
                f"{self.input_file}"
            )

        with self.input_file.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            return list(
                csv.DictReader(file)
            )

    # =====================================================
    # METRICS
    # =====================================================

    def metric_fields(
        self,
    ) -> dict[str, str]:
        """
        Return metrics suitable for group comparison.
        """

        metrics: dict[str, str] = {}

        for segment in SEGMENTS:
            display = segment.title()

            metrics[
                f"{display} propulsion onset before release"
            ] = f"{segment}_propulsion_onset"

            metrics[
                f"{display} peak velocity before release"
            ] = f"{segment}_peak_velocity"

            metrics[
                f"{display} peak acceleration before release"
            ] = f"{segment}_peak_acceleration"

            metrics[
                f"{display} propulsion duration"
            ] = f"{segment}_propulsion_duration_ms"

        return metrics

    def build_summaries(
        self,
        records: list[dict[str, str]],
    ) -> list[SummaryRow]:
        """
        Build descriptive statistics for all, made, and missed shots.
        """

        output: list[SummaryRow] = []

        groups = {
            "All": records,
            "Made": [
                record
                for record in records
                if record.get("result") == "made"
            ],
            "Missed": [
                record
                for record in records
                if record.get("result") == "missed"
            ],
        }

        for group_name, group_records in groups.items():
            for metric_name, field_name in (
                self.metric_fields().items()
            ):
                values = self.metric_values(
                    records=group_records,
                    field_name=field_name,
                    relative_to_release=(
                        field_name.endswith(
                            (
                                "_propulsion_onset",
                                "_peak_velocity",
                                "_peak_acceleration",
                            )
                        )
                    ),
                )

                output.append(
                    self.summarize_values(
                        metric=metric_name,
                        group=group_name,
                        values=values,
                    )
                )

        return output

    def build_comparisons(
        self,
        records: list[dict[str, str]],
    ) -> list[ComparisonRow]:
        """
        Compare made and missed shots for every phase metric.
        """

        made = [
            record
            for record in records
            if record.get("result") == "made"
        ]

        missed = [
            record
            for record in records
            if record.get("result") == "missed"
        ]

        output: list[ComparisonRow] = []

        for metric_name, field_name in (
            self.metric_fields().items()
        ):
            relative = field_name.endswith(
                (
                    "_propulsion_onset",
                    "_peak_velocity",
                    "_peak_acceleration",
                )
            )

            made_values = self.metric_values(
                made,
                field_name,
                relative,
            )

            missed_values = self.metric_values(
                missed,
                field_name,
                relative,
            )

            made_mean = (
                float(mean(made_values))
                if made_values
                else None
            )

            missed_mean = (
                float(mean(missed_values))
                if missed_values
                else None
            )

            difference = None

            if (
                made_mean is not None
                and missed_mean is not None
            ):
                difference = (
                    made_mean
                    - missed_mean
                )

            output.append(
                ComparisonRow(
                    metric=metric_name,
                    made_count=len(made_values),
                    made_mean=made_mean,
                    missed_count=len(missed_values),
                    missed_mean=missed_mean,
                    difference_made_minus_missed=difference,
                )
            )

        return output

    @staticmethod
    def summarize_values(
        metric: str,
        group: str,
        values: list[float],
    ) -> SummaryRow:
        """
        Summarize one numeric series.
        """

        if not values:
            return SummaryRow(
                metric=metric,
                group=group,
                count=0,
                mean=None,
                median=None,
                standard_deviation=None,
                minimum=None,
                maximum=None,
            )

        return SummaryRow(
            metric=metric,
            group=group,
            count=len(values),
            mean=float(mean(values)),
            median=float(median(values)),
            standard_deviation=(
                float(pstdev(values))
                if len(values) > 1
                else 0.0
            ),
            minimum=float(min(values)),
            maximum=float(max(values)),
        )

    @staticmethod
    def metric_values(
        records: list[dict[str, str]],
        field_name: str,
        relative_to_release: bool,
    ) -> list[float]:
        """
        Extract numeric metric values.

        Event-frame metrics are converted to milliseconds before release.
        Duration metrics are already stored in milliseconds.
        """

        output: list[float] = []

        for record in records:
            raw_value = record.get(
                field_name,
                "",
            )

            if raw_value in {
                "",
                None,
            }:
                continue

            try:
                value = float(raw_value)
            except (
                TypeError,
                ValueError,
            ):
                continue

            if relative_to_release:
                try:
                    release_frame = float(
                        record["release_frame"]
                    )

                    sampling_rate = float(
                        record["sampling_rate"]
                    )
                except (
                    KeyError,
                    TypeError,
                    ValueError,
                ):
                    continue

                if sampling_rate <= 0:
                    continue

                value = (
                    (
                        release_frame
                        - value
                    )
                    / sampling_rate
                    * 1000.0
                )

            if np.isnan(value):
                continue

            output.append(value)

        return output

    # =====================================================
    # DASHBOARD
    # =====================================================

    def create_dashboard(
        self,
        records: list[dict[str, str]],
        comparisons: list[ComparisonRow],
        propulsion_sequences: Counter[str],
        peak_sequences: Counter[str],
    ) -> Path:
        """
        Create the phase-validation dashboard.
        """

        figure = plt.figure(
            figsize=(16, 12),
            constrained_layout=True,
        )

        grid = figure.add_gridspec(
            nrows=2,
            ncols=2,
        )

        axis_onsets = figure.add_subplot(
            grid[0, 0]
        )

        axis_durations = figure.add_subplot(
            grid[0, 1]
        )

        axis_propulsion = figure.add_subplot(
            grid[1, 0]
        )

        axis_peaks = figure.add_subplot(
            grid[1, 1]
        )

        self.plot_onset_timing(
            axis_onsets,
            records,
        )

        self.plot_duration_differences(
            axis_durations,
            comparisons,
        )

        self.plot_sequence_frequency(
            axis_propulsion,
            propulsion_sequences,
            "Most Common Propulsion-Onset Sequences",
        )

        self.plot_sequence_frequency(
            axis_peaks,
            peak_sequences,
            "Most Common Peak Sequences",
        )

        figure.suptitle(
            "REFINED PHASE BIOMECHANICS VALIDATION",
            fontsize=16,
            fontweight="bold",
        )

        figure.savefig(
            self.dashboard_file,
            dpi=170,
            bbox_inches="tight",
        )

        plt.close(figure)

        return self.dashboard_file

    def plot_onset_timing(
        self,
        axis,
        records: list[dict[str, str]],
    ) -> None:
        """
        Plot average propulsion onset timing for each segment.
        """

        labels: list[str] = []
        made_means: list[float] = []
        missed_means: list[float] = []

        for segment in SEGMENTS:
            field_name = (
                f"{segment}_propulsion_onset"
            )

            made_values = self.metric_values(
                [
                    record
                    for record in records
                    if record.get("result") == "made"
                ],
                field_name,
                True,
            )

            missed_values = self.metric_values(
                [
                    record
                    for record in records
                    if record.get("result") == "missed"
                ],
                field_name,
                True,
            )

            if (
                not made_values
                or not missed_values
            ):
                continue

            labels.append(
                segment.title()
            )

            made_means.append(
                float(mean(made_values))
            )

            missed_means.append(
                float(mean(missed_values))
            )

        positions = np.arange(
            len(labels)
        )

        width = 0.38

        axis.bar(
            positions - width / 2,
            made_means,
            width,
            label="Made",
        )

        axis.bar(
            positions + width / 2,
            missed_means,
            width,
            label="Missed",
        )

        axis.set_xticks(
            positions,
            labels=labels,
            rotation=30,
        )

        axis.set_ylabel(
            "Milliseconds before release"
        )

        axis.set_title(
            "Average Propulsion-Onset Timing"
        )

        axis.legend()

        axis.grid(
            visible=True,
            axis="y",
            alpha=0.25,
        )

    @staticmethod
    def plot_duration_differences(
        axis,
        comparisons: list[ComparisonRow],
    ) -> None:
        """
        Plot made-minus-missed propulsion-duration differences.
        """

        duration_rows = [
            row
            for row in comparisons
            if row.metric.endswith(
                "propulsion duration"
            )
            and row.difference_made_minus_missed
            is not None
        ]

        labels = [
            row.metric.replace(
                " propulsion duration",
                "",
            )
            for row in duration_rows
        ]

        differences = [
            float(
                row.difference_made_minus_missed
            )
            for row in duration_rows
        ]

        positions = np.arange(
            len(labels)
        )

        axis.bar(
            positions,
            differences,
        )

        axis.axhline(
            y=0,
            linewidth=1,
            linestyle="--",
        )

        axis.set_xticks(
            positions,
            labels=labels,
            rotation=30,
        )

        axis.set_ylabel(
            "Made minus missed (ms)"
        )

        axis.set_title(
            "Propulsion-Duration Differences"
        )

        axis.grid(
            visible=True,
            axis="y",
            alpha=0.25,
        )

    @staticmethod
    def plot_sequence_frequency(
        axis,
        sequences: Counter[str],
        title: str,
    ) -> None:
        """
        Plot the eight most common sequences.
        """

        top_sequences = sequences.most_common(
            8
        )

        if not top_sequences:
            axis.text(
                0.5,
                0.5,
                "No sequences detected",
                ha="center",
                va="center",
            )

            axis.set_axis_off()
            return

        labels = [
            sequence.replace(
                " -> ",
                "\n",
            )
            for sequence, _ in top_sequences
        ]

        counts = [
            count
            for _, count in top_sequences
        ]

        positions = np.arange(
            len(labels)
        )

        axis.barh(
            positions,
            counts,
        )

        axis.set_yticks(
            positions,
            labels=labels,
        )

        axis.invert_yaxis()

        axis.set_xlabel(
            "Shot count"
        )

        axis.set_title(title)

        axis.tick_params(
            axis="y",
            labelsize=7,
        )

        axis.grid(
            visible=True,
            axis="x",
            alpha=0.25,
        )

    # =====================================================
    # EXPORTS
    # =====================================================

    @staticmethod
    def export_dataclass_rows(
        output_file: Path,
        rows: list[object],
    ) -> Path:
        """
        Export dataclass records to CSV.
        """

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not rows:
            output_file.write_text(
                "",
                encoding="utf-8",
            )

            return output_file

        dictionaries = [
            row.__dict__
            for row in rows
        ]

        fieldnames = list(
            dictionaries[0]
        )

        with output_file.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(
                dictionaries
            )

        return output_file

    @staticmethod
    def export_sequence_frequency(
        output_file: Path,
        sequences: Counter[str],
    ) -> Path:
        """
        Export sequence counts and percentages.
        """

        total = sum(
            sequences.values()
        )

        with output_file.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            fieldnames = [
                "rank",
                "sequence",
                "shot_count",
                "percentage",
            ]

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for rank, (
                sequence,
                count,
            ) in enumerate(
                sequences.most_common(),
                start=1,
            ):
                writer.writerow(
                    {
                        "rank": rank,
                        "sequence": sequence,
                        "shot_count": count,
                        "percentage": (
                            count
                            / total
                            * 100.0
                            if total > 0
                            else 0.0
                        ),
                    }
                )

        return output_file


def _run_phase_validation_studio_test() -> None:
    """
    Run the complete phase-validation studio.
    """

    studio = PhaseValidationStudio()

    records = studio.run()

    made_count = sum(
        record.get("result") == "made"
        for record in records
    )

    missed_count = sum(
        record.get("result") == "missed"
        for record in records
    )

    assert studio.summary_file.exists()
    assert studio.comparison_file.exists()
    assert studio.propulsion_sequence_file.exists()
    assert studio.peak_sequence_file.exists()
    assert studio.dashboard_file.exists()

    print(
        "PHASE VALIDATION STUDIO TEST PASSED"
    )

    print(
        f"Successful records: {len(records)}"
    )

    print(
        f"Made shots: {made_count}"
    )

    print(
        f"Missed shots: {missed_count}"
    )

    print()

    print(
        "Descriptive statistics: "
        f"{studio.summary_file.resolve()}"
    )

    print(
        "Make/miss comparison: "
        f"{studio.comparison_file.resolve()}"
    )

    print(
        "Propulsion sequences: "
        f"{studio.propulsion_sequence_file.resolve()}"
    )

    print(
        "Peak sequences: "
        f"{studio.peak_sequence_file.resolve()}"
    )

    print(
        "Dashboard PNG: "
        f"{studio.dashboard_file.resolve()}"
    )


if __name__ == "__main__":
    _run_phase_validation_studio_test()