from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, pstdev

import matplotlib.pyplot as plt
import numpy as np

from tracking_app.kinetic_chain_validation import (
    KineticChainValidator,
    ValidationRecord,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "kinetic_chain_studio"
)


@dataclass(frozen=True)
class MetricSummary:
    """
    Descriptive statistics for one numeric validation metric.
    """

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
    """
    Made-versus-missed comparison for one metric.
    """

    metric: str
    made_count: int
    made_mean: float | None
    missed_count: int
    missed_mean: float | None
    difference_made_minus_missed: float | None


class BiomechanicsValidationStudio:
    """
    Run a larger kinetic-chain sample and create validation outputs.

    Outputs
    -------
    1. Full 60-shot validation CSV
    2. Descriptive statistics CSV
    3. Made-versus-missed comparison CSV
    4. Sequence-frequency CSV
    5. Validation dashboard PNG

    The studio is designed for calibration. It summarizes what the
    current detector finds without assuming the current sequence model
    is already biomechanically correct.
    """

    NUMERIC_METRICS = {
        "Sequence score": "sequence_score",
        "Knee peak timing": "knee_peak_frame",
        "Hip peak timing": "hip_peak_frame",
        "Pelvis peak timing": "pelvis_peak_frame",
        "Shoulder peak timing": "shoulder_peak_frame",
        "Elbow peak timing": "elbow_peak_frame",
        "Wrist peak timing": "wrist_peak_frame",
        "Knee-to-hip gap": "knee_to_hip_ms",
        "Hip-to-pelvis gap": "hip_to_pelvis_ms",
        "Pelvis-to-shoulder gap": "pelvis_to_shoulder_ms",
        "Shoulder-to-elbow gap": "shoulder_to_elbow_ms",
        "Elbow-to-wrist gap": "elbow_to_wrist_ms",
        "Wrist-to-release": "wrist_to_release_ms",
        "Knee peak value": "knee_peak_value",
        "Hip peak value": "hip_peak_value",
        "Pelvis peak value": "pelvis_peak_value",
        "Shoulder peak value": "shoulder_peak_value",
        "Elbow peak value": "elbow_peak_value",
        "Wrist peak value": "wrist_peak_value",
    }

    TIMING_FIELDS = {
        "Knee": "knee_peak_frame",
        "Hip": "hip_peak_frame",
        "Pelvis": "pelvis_peak_frame",
        "Shoulder": "shoulder_peak_frame",
        "Elbow": "elbow_peak_frame",
        "Wrist": "wrist_peak_frame",
    }

    def __init__(
        self,
        sample_size: int = 60,
        output_folder: Path = DEFAULT_OUTPUT_FOLDER,
        shooting_side: str = "RIGHT",
    ) -> None:
        self.sample_size = max(
            1,
            int(sample_size),
        )

        self.output_folder = Path(
            output_folder
        )

        self.shooting_side = (
            shooting_side
            .strip()
            .upper()
        )

        if self.shooting_side not in {
            "RIGHT",
            "LEFT",
        }:
            raise ValueError(
                "shooting_side must be RIGHT or LEFT."
            )

        self.full_validation_file = (
            self.output_folder
            / "validation_60_shots.csv"
        )

        self.summary_file = (
            self.output_folder
            / "descriptive_statistics.csv"
        )

        self.comparison_file = (
            self.output_folder
            / "make_miss_comparison.csv"
        )

        self.sequence_file = (
            self.output_folder
            / "sequence_frequency.csv"
        )

        self.dashboard_file = (
            self.output_folder
            / "validation_dashboard.png"
        )

    # =====================================================
    # PUBLIC WORKFLOW
    # =====================================================

    def run(
        self,
    ) -> list[ValidationRecord]:
        """
        Run validation and create all analysis outputs.
        """

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        validator = KineticChainValidator(
            output_file=self.full_validation_file,
            sample_size=self.sample_size,
            shooting_side=self.shooting_side,
        )

        records = validator.run()

        successful_records = [
            record
            for record in records
            if record.analysis_status == "success"
        ]

        if not successful_records:
            raise RuntimeError(
                "No successful validation records were produced."
            )

        summaries = self.build_descriptive_statistics(
            successful_records
        )

        comparisons = self.build_make_miss_comparisons(
            successful_records
        )

        sequence_counts = self.build_sequence_frequency(
            successful_records
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
            sequence_counts
        )

        self.create_dashboard(
            successful_records,
            comparisons,
            sequence_counts,
        )

        return records

    # =====================================================
    # DESCRIPTIVE STATISTICS
    # =====================================================

    def build_descriptive_statistics(
        self,
        records: list[ValidationRecord],
    ) -> list[MetricSummary]:
        """
        Summarize every numeric metric for all, made, and missed shots.
        """

        output: list[MetricSummary] = []

        groups = {
            "All": records,
            "Made": [
                record
                for record in records
                if record.result == "made"
            ],
            "Missed": [
                record
                for record in records
                if record.result == "missed"
            ],
        }

        for group_name, group_records in groups.items():
            for metric_name, field_name in (
                self.NUMERIC_METRICS.items()
            ):
                values = self._numeric_values(
                    group_records,
                    field_name,
                )

                output.append(
                    self._summarize_values(
                        metric=metric_name,
                        group=group_name,
                        values=values,
                    )
                )

        return output

    @staticmethod
    def _summarize_values(
        metric: str,
        group: str,
        values: list[float],
    ) -> MetricSummary:
        """
        Build one descriptive-statistics row.
        """

        if not values:
            return MetricSummary(
                metric=metric,
                group=group,
                count=0,
                mean=None,
                median=None,
                standard_deviation=None,
                minimum=None,
                maximum=None,
            )

        return MetricSummary(
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

    # =====================================================
    # MAKE / MISS COMPARISON
    # =====================================================

    def build_make_miss_comparisons(
        self,
        records: list[ValidationRecord],
    ) -> list[ComparisonRow]:
        """
        Compare made and missed shots for every numeric metric.
        """

        made_records = [
            record
            for record in records
            if record.result == "made"
        ]

        missed_records = [
            record
            for record in records
            if record.result == "missed"
        ]

        output: list[ComparisonRow] = []

        for metric_name, field_name in (
            self.NUMERIC_METRICS.items()
        ):
            made_values = self._numeric_values(
                made_records,
                field_name,
            )

            missed_values = self._numeric_values(
                missed_records,
                field_name,
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

    # =====================================================
    # SEQUENCE FREQUENCY
    # =====================================================

    @staticmethod
    def build_sequence_frequency(
        records: list[ValidationRecord],
    ) -> Counter[str]:
        """
        Count how often each detected sequence occurs.
        """

        return Counter(
            record.detected_sequence
            for record in records
            if record.detected_sequence
        )

    # =====================================================
    # DASHBOARD
    # =====================================================

    def create_dashboard(
        self,
        records: list[ValidationRecord],
        comparisons: list[ComparisonRow],
        sequence_counts: Counter[str],
    ) -> Path:
        """
        Create a four-panel validation dashboard.
        """

        figure = plt.figure(
            figsize=(15, 11),
            constrained_layout=True,
        )

        grid = figure.add_gridspec(
            nrows=2,
            ncols=2,
        )

        axis_score = figure.add_subplot(
            grid[0, 0]
        )

        axis_timing = figure.add_subplot(
            grid[0, 1]
        )

        axis_sequences = figure.add_subplot(
            grid[1, 0]
        )

        axis_gaps = figure.add_subplot(
            grid[1, 1]
        )

        self._plot_sequence_scores(
            axis_score,
            records,
        )

        self._plot_peak_timing(
            axis_timing,
            records,
        )

        self._plot_sequence_frequency(
            axis_sequences,
            sequence_counts,
        )

        self._plot_make_miss_gaps(
            axis_gaps,
            comparisons,
        )

        figure.suptitle(
            "KINETIC-CHAIN VALIDATION STUDIO",
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

    @staticmethod
    def _plot_sequence_scores(
        axis,
        records: list[ValidationRecord],
    ) -> None:
        made_scores = [
            record.sequence_score
            for record in records
            if record.result == "made"
        ]

        missed_scores = [
            record.sequence_score
            for record in records
            if record.result == "missed"
        ]

        # Matplotlib 3.9+ renamed 'labels' to 'tick_labels'.
        # Using tick_labels keeps compatibility with newer releases.
        axis.boxplot(
            [
                made_scores,
                missed_scores,
            ],
            tick_labels=[
                "Made",
                "Missed",
            ],
        )

        axis.set_title(
            "Sequence Score Distribution"
        )

        axis.set_ylabel(
            "Score (0-100)"
        )

        axis.grid(
            visible=True,
            alpha=0.25,
        )

    def _plot_peak_timing(
        self,
        axis,
        records: list[ValidationRecord],
    ) -> None:
        """
        Plot average milliseconds before release for each segment.
        """

        segment_names: list[str] = []
        average_timings: list[float] = []
        standard_deviations: list[float] = []

        for segment_name, field_name in (
            self.TIMING_FIELDS.items()
        ):
            values = []

            for record in records:
                frame = getattr(
                    record,
                    field_name,
                )

                release_frame = (
                    record.release_frame
                )

                if (
                    frame is None
                    or release_frame is None
                    or record.sampling_rate <= 0
                ):
                    continue

                timing = (
                    (
                        release_frame
                        - frame
                    )
                    / record.sampling_rate
                    * 1000.0
                )

                values.append(timing)

            if not values:
                continue

            segment_names.append(
                segment_name
            )

            average_timings.append(
                float(mean(values))
            )

            standard_deviations.append(
                float(pstdev(values))
                if len(values) > 1
                else 0.0
            )

        axis.errorbar(
            segment_names,
            average_timings,
            yerr=standard_deviations,
            marker="o",
            linestyle="-",
            capsize=4,
        )

        axis.axhline(
            y=0,
            linewidth=1,
            linestyle="--",
        )

        axis.set_title(
            "Peak Timing Relative to Release"
        )

        axis.set_ylabel(
            "Milliseconds before release"
        )

        axis.tick_params(
            axis="x",
            rotation=30,
        )

        axis.grid(
            visible=True,
            alpha=0.25,
        )

    @staticmethod
    def _plot_sequence_frequency(
        axis,
        sequence_counts: Counter[str],
    ) -> None:
        top_sequences = (
            sequence_counts
            .most_common(8)
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
            sequence
            .replace(
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

        axis.set_title(
            "Most Common Detected Sequences"
        )

        axis.set_xlabel(
            "Shot count"
        )

        axis.tick_params(
            axis="y",
            labelsize=7,
        )

        axis.grid(
            visible=True,
            axis="x",
            alpha=0.25,
        )

    @staticmethod
    def _plot_make_miss_gaps(
        axis,
        comparisons: list[ComparisonRow],
    ) -> None:
        desired_metrics = [
            "Knee-to-hip gap",
            "Hip-to-pelvis gap",
            "Pelvis-to-shoulder gap",
            "Shoulder-to-elbow gap",
            "Elbow-to-wrist gap",
            "Wrist-to-release",
        ]

        lookup = {
            row.metric: row
            for row in comparisons
        }

        labels: list[str] = []
        differences: list[float] = []

        for metric in desired_metrics:
            row = lookup.get(metric)

            if (
                row is None
                or row.difference_made_minus_missed
                is None
            ):
                continue

            labels.append(metric)

            differences.append(
                row.difference_made_minus_missed
            )

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
            rotation=35,
            ha="right",
        )

        axis.set_title(
            "Made Minus Missed Timing Differences"
        )

        axis.set_ylabel(
            "Difference (ms)"
        )

        axis.grid(
            visible=True,
            axis="y",
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
        Export dataclass rows to CSV.
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

        row_dicts = [
            asdict(row)
            for row in rows
        ]

        fieldnames = list(
            row_dicts[0]
        )

        with output_file.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(row_dicts)

        return output_file

    def export_sequence_frequency(
        self,
        sequence_counts: Counter[str],
    ) -> Path:
        """
        Export detected sequences and their frequencies.
        """

        total = sum(
            sequence_counts.values()
        )

        with self.sequence_file.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            fieldnames = [
                "rank",
                "detected_sequence",
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
                sequence_counts.most_common(),
                start=1,
            ):
                percentage = (
                    count / total * 100.0
                    if total > 0
                    else 0.0
                )

                writer.writerow(
                    {
                        "rank": rank,
                        "detected_sequence": sequence,
                        "shot_count": count,
                        "percentage": percentage,
                    }
                )

        return self.sequence_file

    # =====================================================
    # HELPERS
    # =====================================================

    @staticmethod
    def _numeric_values(
        records: list[ValidationRecord],
        field_name: str,
    ) -> list[float]:
        """
        Extract usable numeric values from validation records.
        """

        output: list[float] = []

        for record in records:
            value = getattr(
                record,
                field_name,
            )

            if value is None:
                continue

            try:
                numeric_value = float(
                    value
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if np.isnan(numeric_value):
                continue

            output.append(numeric_value)

        return output


def _run_validation_studio_test() -> None:
    """
    Run the full 60-shot validation studio.
    """

    studio = BiomechanicsValidationStudio(
        sample_size=60,
        shooting_side="RIGHT",
    )

    records = studio.run()

    successful = [
        record
        for record in records
        if record.analysis_status == "success"
    ]

    failed = [
        record
        for record in records
        if record.analysis_status == "failed"
    ]

    made_count = sum(
        record.result == "made"
        for record in successful
    )

    missed_count = sum(
        record.result == "missed"
        for record in successful
    )

    participants = {
        record.participant_id
        for record in successful
    }

    assert len(records) > 0
    assert studio.full_validation_file.exists()
    assert studio.summary_file.exists()
    assert studio.comparison_file.exists()
    assert studio.sequence_file.exists()
    assert studio.dashboard_file.exists()

    print(
        "BIOMECHANICS VALIDATION STUDIO TEST PASSED"
    )
    print(
        f"Trials analyzed: {len(records)}"
    )
    print(
        f"Successful: {len(successful)}"
    )
    print(
        f"Failed: {len(failed)}"
    )
    print(
        "Participants represented: "
        f"{len(participants)}"
    )
    print(
        f"Made shots: {made_count}"
    )
    print(
        f"Missed shots: {missed_count}"
    )
    print()
    print(
        "Validation CSV: "
        f"{studio.full_validation_file.resolve()}"
    )
    print(
        "Descriptive statistics: "
        f"{studio.summary_file.resolve()}"
    )
    print(
        "Make/miss comparison: "
        f"{studio.comparison_file.resolve()}"
    )
    print(
        "Sequence frequency: "
        f"{studio.sequence_file.resolve()}"
    )
    print(
        "Dashboard PNG: "
        f"{studio.dashboard_file.resolve()}"
    )


if __name__ == "__main__":
    _run_validation_studio_test()