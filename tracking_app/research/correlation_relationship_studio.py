from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "all_shot_features.csv"
)

DEFAULT_OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "correlations"
)


NON_NUMERIC_FIELDS = {
    "participant_id",
    "trial_id",
    "result",
    "trial_file",
    "propulsion_onset_sequence",
    "peak_velocity_sequence",
    "analysis_status",
    "analysis_error",
}


@dataclass(frozen=True)
class CorrelationPair:
    """
    One pairwise Pearson correlation result.
    """

    feature_a: str
    feature_b: str

    paired_count: int
    correlation: float | None
    absolute_correlation: float | None

    relationship_strength: str
    relationship_direction: str


@dataclass(frozen=True)
class RedundancyCandidate:
    """
    A highly correlated feature pair that may contain overlapping information.
    """

    feature_a: str
    feature_b: str
    paired_count: int
    correlation: float
    absolute_correlation: float
    recommendation: str


class CorrelationRelationshipStudio:
    """
    Analyze relationships among numeric free-throw features.

    Outputs
    -------
    correlation_matrix.csv
        Full square correlation matrix.

    correlation_pairs.csv
        Every unique pair, sorted by absolute correlation.

    redundancy_candidates.csv
        Feature pairs above the configured redundancy threshold.

    top_correlations.png
        Horizontal bar chart of the strongest non-identical feature pairs.

    correlation_heatmap.png
        Heatmap of the most complete, variable numeric features.

    correlation_report.txt
        Human-readable interpretation summary.

    Important
    ---------
    Correlation shows association, not causation.
    """

    def __init__(
        self,
        input_file: Path = DEFAULT_INPUT_FILE,
        output_folder: Path = DEFAULT_OUTPUT_FOLDER,
        minimum_paired_count: int = 30,
        redundancy_threshold: float = 0.90,
        heatmap_feature_limit: int = 30,
        top_pair_limit: int = 25,
    ) -> None:
        self.input_file = Path(input_file)
        self.output_folder = Path(output_folder)

        self.minimum_paired_count = max(
            3,
            int(minimum_paired_count),
        )

        self.redundancy_threshold = min(
            1.0,
            max(
                0.0,
                float(redundancy_threshold),
            ),
        )

        self.heatmap_feature_limit = max(
            5,
            int(heatmap_feature_limit),
        )

        self.top_pair_limit = max(
            5,
            int(top_pair_limit),
        )

        self.matrix_file = (
            self.output_folder
            / "correlation_matrix.csv"
        )

        self.pairs_file = (
            self.output_folder
            / "correlation_pairs.csv"
        )

        self.redundancy_file = (
            self.output_folder
            / "redundancy_candidates.csv"
        )

        self.top_chart_file = (
            self.output_folder
            / "top_correlations.png"
        )

        self.heatmap_file = (
            self.output_folder
            / "correlation_heatmap.png"
        )

        self.report_file = (
            self.output_folder
            / "correlation_report.txt"
        )

    # =====================================================
    # PUBLIC WORKFLOW
    # =====================================================

    def run(
        self,
    ) -> tuple[
        list[str],
        np.ndarray,
        list[CorrelationPair],
        list[RedundancyCandidate],
    ]:
        """
        Run all correlation analyses and export results.
        """

        records = self.load_records()

        successful = [
            record
            for record in records
            if self._normalize_text(
                record.get(
                    "analysis_status",
                    "",
                )
            )
            == "success"
        ]

        if not successful:
            raise RuntimeError(
                "No successful shot records were found."
            )

        numeric_features = self.identify_numeric_features(
            successful
        )

        if len(numeric_features) < 2:
            raise RuntimeError(
                "At least two numeric features are required."
            )

        matrix = self.build_correlation_matrix(
            successful,
            numeric_features,
        )

        pairs = self.build_correlation_pairs(
            successful,
            numeric_features,
        )

        redundancy_candidates = (
            self.build_redundancy_candidates(
                pairs
            )
        )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.export_matrix(
            numeric_features,
            matrix,
        )

        self.export_dataclass_rows(
            self.pairs_file,
            pairs,
        )

        self.export_dataclass_rows(
            self.redundancy_file,
            redundancy_candidates,
        )

        self.create_top_correlations_chart(
            pairs
        )

        selected_features = (
            self.select_heatmap_features(
                successful,
                numeric_features,
            )
        )

        self.create_heatmap(
            successful,
            selected_features,
        )

        self.export_text_report(
            successful_records=successful,
            numeric_features=numeric_features,
            pairs=pairs,
            redundancy_candidates=(
                redundancy_candidates
            ),
            heatmap_features=(
                selected_features
            ),
        )

        return (
            numeric_features,
            matrix,
            pairs,
            redundancy_candidates,
        )

    # =====================================================
    # LOAD DATA
    # =====================================================

    def load_records(
        self,
    ) -> list[dict[str, str]]:
        """
        Read the master feature dataset.
        """

        if not self.input_file.exists():
            raise FileNotFoundError(
                "Master feature CSV was not found: "
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
    # NUMERIC FEATURE DISCOVERY
    # =====================================================

    def identify_numeric_features(
        self,
        records: list[dict[str, str]],
    ) -> list[str]:
        """
        Identify features containing at least two distinct numeric values.
        """

        fieldnames = list(
            records[0]
        )

        numeric_features: list[
            str
        ] = []

        for field_name in fieldnames:
            if field_name in NON_NUMERIC_FIELDS:
                continue

            values = self._numeric_values(
                records,
                field_name,
            )

            if len(values) < 2:
                continue

            if len(set(values)) < 2:
                continue

            numeric_features.append(
                field_name
            )

        return numeric_features

    # =====================================================
    # MATRIX
    # =====================================================

    def build_correlation_matrix(
        self,
        records: list[dict[str, str]],
        features: list[str],
    ) -> np.ndarray:
        """
        Build a square pairwise Pearson-correlation matrix.
        """

        size = len(features)

        matrix = np.full(
            (
                size,
                size,
            ),
            np.nan,
            dtype=float,
        )

        for row_index, feature_a in enumerate(
            features
        ):
            for column_index, feature_b in enumerate(
                features
            ):
                if row_index == column_index:
                    matrix[
                        row_index,
                        column_index,
                    ] = 1.0
                    continue

                if column_index < row_index:
                    matrix[
                        row_index,
                        column_index,
                    ] = matrix[
                        column_index,
                        row_index,
                    ]
                    continue

                paired_a, paired_b = (
                    self._paired_numeric_values(
                        records,
                        feature_a,
                        feature_b,
                    )
                )

                correlation = self._pearson_correlation(
                    paired_a,
                    paired_b,
                )

                if correlation is not None:
                    matrix[
                        row_index,
                        column_index,
                    ] = correlation

        return matrix

    # =====================================================
    # UNIQUE PAIRS
    # =====================================================

    def build_correlation_pairs(
        self,
        records: list[dict[str, str]],
        features: list[str],
    ) -> list[CorrelationPair]:
        """
        Build one row per unique feature pair.
        """

        rows: list[
            CorrelationPair
        ] = []

        for first_index, feature_a in enumerate(
            features
        ):
            for feature_b in features[
                first_index + 1:
            ]:
                paired_a, paired_b = (
                    self._paired_numeric_values(
                        records,
                        feature_a,
                        feature_b,
                    )
                )

                paired_count = len(
                    paired_a
                )

                correlation = None

                if (
                    paired_count
                    >= self.minimum_paired_count
                ):
                    correlation = (
                        self._pearson_correlation(
                            paired_a,
                            paired_b,
                        )
                    )

                absolute_correlation = (
                    None
                    if correlation is None
                    else abs(correlation)
                )

                rows.append(
                    CorrelationPair(
                        feature_a=feature_a,
                        feature_b=feature_b,
                        paired_count=paired_count,
                        correlation=correlation,
                        absolute_correlation=(
                            absolute_correlation
                        ),
                        relationship_strength=(
                            self._strength_label(
                                absolute_correlation
                            )
                        ),
                        relationship_direction=(
                            self._direction_label(
                                correlation
                            )
                        ),
                    )
                )

        rows.sort(
            key=lambda row: (
                -row.absolute_correlation
                if row.absolute_correlation
                is not None
                else float("-inf"),
                row.feature_a,
                row.feature_b,
            )
        )

        return rows

    # =====================================================
    # REDUNDANCY
    # =====================================================

    def build_redundancy_candidates(
        self,
        pairs: list[CorrelationPair],
    ) -> list[RedundancyCandidate]:
        """
        Identify highly correlated pairs that may overlap conceptually.
        """

        candidates: list[
            RedundancyCandidate
        ] = []

        for pair in pairs:
            if (
                pair.correlation is None
                or pair.absolute_correlation is None
                or pair.paired_count
                < self.minimum_paired_count
                or pair.absolute_correlation
                < self.redundancy_threshold
            ):
                continue

            candidates.append(
                RedundancyCandidate(
                    feature_a=pair.feature_a,
                    feature_b=pair.feature_b,
                    paired_count=(
                        pair.paired_count
                    ),
                    correlation=(
                        pair.correlation
                    ),
                    absolute_correlation=(
                        pair.absolute_correlation
                    ),
                    recommendation=(
                        "Review whether both features are needed in "
                        "future models; retain both until biomechanical "
                        "meaning and measurement reliability are compared."
                    ),
                )
            )

        return candidates

    # =====================================================
    # HEATMAP FEATURE SELECTION
    # =====================================================

    def select_heatmap_features(
        self,
        records: list[dict[str, str]],
        features: list[str],
    ) -> list[str]:
        """
        Select complete and variable features for a readable heatmap.
        """

        ranked: list[
            tuple[
                float,
                float,
                str,
            ]
        ] = []

        total_rows = len(
            records
        )

        for feature in features:
            values = self._numeric_values(
                records,
                feature,
            )

            completeness = (
                len(values)
                / total_rows
                if total_rows > 0
                else 0.0
            )

            variability = (
                float(
                    np.std(
                        values
                    )
                )
                if len(values) > 1
                else 0.0
            )

            ranked.append(
                (
                    completeness,
                    variability,
                    feature,
                )
            )

        ranked.sort(
            key=lambda item: (
                -item[0],
                -item[1],
                item[2],
            )
        )

        return [
            feature
            for _, _, feature in ranked[
                :self.heatmap_feature_limit
            ]
        ]

    # =====================================================
    # VISUALS
    # =====================================================

    def create_top_correlations_chart(
        self,
        pairs: list[CorrelationPair],
    ) -> Path:
        """
        Create a chart of the strongest pairwise relationships.
        """

        usable_pairs = [
            pair
            for pair in pairs
            if (
                pair.correlation is not None
                and pair.paired_count
                >= self.minimum_paired_count
            )
        ][
            :self.top_pair_limit
        ]

        figure, axis = plt.subplots(
            figsize=(
                12,
                max(
                    7,
                    len(usable_pairs)
                    * 0.38,
                ),
            )
        )

        if not usable_pairs:
            axis.text(
                0.5,
                0.5,
                "No valid correlations available",
                ha="center",
                va="center",
            )

            axis.set_axis_off()

        else:
            labels = [
                (
                    f"{pair.feature_a}\n"
                    f"vs {pair.feature_b}"
                )
                for pair in reversed(
                    usable_pairs
                )
            ]

            correlations = [
                float(
                    pair.correlation
                )
                for pair in reversed(
                    usable_pairs
                )
            ]

            positions = np.arange(
                len(labels)
            )

            axis.barh(
                positions,
                correlations,
            )

            axis.set_yticks(
                positions,
                labels=labels,
            )

            axis.axvline(
                x=0,
                linewidth=1,
                linestyle="--",
            )

            axis.set_xlim(
                -1.0,
                1.0,
            )

            axis.set_xlabel(
                "Pearson correlation"
            )

            axis.set_title(
                "Strongest Feature Relationships"
            )

            axis.grid(
                visible=True,
                axis="x",
                alpha=0.25,
            )

        figure.tight_layout()

        figure.savefig(
            self.top_chart_file,
            dpi=170,
            bbox_inches="tight",
        )

        plt.close(figure)

        return self.top_chart_file

    def create_heatmap(
        self,
        records: list[dict[str, str]],
        features: list[str],
    ) -> Path:
        """
        Create a readable heatmap of selected numeric features.
        """

        matrix = self.build_correlation_matrix(
            records,
            features,
        )

        figure, axis = plt.subplots(
            figsize=(
                15,
                13,
            )
        )

        image = axis.imshow(
            matrix,
            vmin=-1.0,
            vmax=1.0,
            aspect="auto",
        )

        positions = np.arange(
            len(features)
        )

        axis.set_xticks(
            positions,
            labels=features,
            rotation=90,
            fontsize=7,
        )

        axis.set_yticks(
            positions,
            labels=features,
            fontsize=7,
        )

        axis.set_title(
            "Selected Feature Correlation Heatmap"
        )

        figure.colorbar(
            image,
            ax=axis,
            label="Pearson correlation",
        )

        figure.tight_layout()

        figure.savefig(
            self.heatmap_file,
            dpi=170,
            bbox_inches="tight",
        )

        plt.close(figure)

        return self.heatmap_file

    # =====================================================
    # REPORT
    # =====================================================

    def export_text_report(
        self,
        successful_records: list[
            dict[str, str]
        ],
        numeric_features: list[str],
        pairs: list[CorrelationPair],
        redundancy_candidates: list[
            RedundancyCandidate
        ],
        heatmap_features: list[str],
    ) -> Path:
        """
        Write a readable correlation summary.
        """

        valid_pairs = [
            pair
            for pair in pairs
            if pair.correlation is not None
        ]

        strong_pairs = [
            pair
            for pair in valid_pairs
            if (
                pair.absolute_correlation
                is not None
                and pair.absolute_correlation
                >= 0.70
            )
        ]

        moderate_pairs = [
            pair
            for pair in valid_pairs
            if (
                pair.absolute_correlation
                is not None
                and 0.40
                <= pair.absolute_correlation
                < 0.70
            )
        ]

        lines = [
            "=" * 80,
            "BASKETBALL BIOMECHANICS RESEARCH STUDIO",
            "CORRELATION & RELATIONSHIP REPORT",
            "=" * 80,
            "",
            "DATASET",
            "-" * 80,
            (
                "Successful shot records: "
                f"{len(successful_records)}"
            ),
            (
                "Numeric features included: "
                f"{len(numeric_features)}"
            ),
            (
                "Unique feature pairs evaluated: "
                f"{len(pairs)}"
            ),
            (
                "Valid pairwise correlations: "
                f"{len(valid_pairs)}"
            ),
            (
                "Minimum paired observations: "
                f"{self.minimum_paired_count}"
            ),
            "",
            "IMPORTANT INTERPRETATION",
            "-" * 80,
            (
                "Correlation describes how two measurements change together. "
                "It does not prove that one feature causes the other."
            ),
            (
                "High correlations may reflect true biomechanics, shared "
                "calculation inputs, duplicate information, or participant "
                "differences."
            ),
            "",
            "STRONGEST CORRELATIONS",
            "-" * 80,
        ]

        for pair in valid_pairs[
            :25
        ]:
            lines.append(
                (
                    f"{pair.feature_a} vs {pair.feature_b}: "
                    f"r={pair.correlation:.3f}, "
                    f"n={pair.paired_count}, "
                    f"{pair.relationship_strength} "
                    f"{pair.relationship_direction}"
                )
            )

        lines.extend(
            [
                "",
                "RELATIONSHIP COUNTS",
                "-" * 80,
                (
                    "Strong relationships "
                    f"(|r| >= 0.70): {len(strong_pairs)}"
                ),
                (
                    "Moderate relationships "
                    f"(0.40 <= |r| < 0.70): {len(moderate_pairs)}"
                ),
                (
                    "Redundancy candidates "
                    f"(|r| >= {self.redundancy_threshold:.2f}): "
                    f"{len(redundancy_candidates)}"
                ),
                "",
                "REDUNDANCY CANDIDATES",
                "-" * 80,
            ]
        )

        if redundancy_candidates:
            for candidate in redundancy_candidates[
                :30
            ]:
                lines.append(
                    (
                        f"{candidate.feature_a} vs "
                        f"{candidate.feature_b}: "
                        f"r={candidate.correlation:.3f}, "
                        f"n={candidate.paired_count}"
                    )
                )
        else:
            lines.append(
                "No feature pairs exceeded the redundancy threshold."
            )

        lines.extend(
            [
                "",
                "HEATMAP FEATURES",
                "-" * 80,
            ]
        )

        for feature in heatmap_features:
            lines.append(
                feature
            )

        self.report_file.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        return self.report_file

    # =====================================================
    # EXPORT HELPERS
    # =====================================================

    def export_matrix(
        self,
        features: list[str],
        matrix: np.ndarray,
    ) -> Path:
        """
        Export the full square correlation matrix.
        """

        with self.matrix_file.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.writer(
                file
            )

            writer.writerow(
                [
                    "feature",
                    *features,
                ]
            )

            for row_index, feature in enumerate(
                features
            ):
                row_values = []

                for value in matrix[
                    row_index
                ]:
                    row_values.append(
                        ""
                        if np.isnan(value)
                        else float(value)
                    )

                writer.writerow(
                    [
                        feature,
                        *row_values,
                    ]
                )

        return self.matrix_file

    @staticmethod
    def export_dataclass_rows(
        output_file: Path,
        rows: list[object],
    ) -> Path:
        """
        Export dataclass records to CSV.
        """

        if not rows:
            output_file.write_text(
                "",
                encoding="utf-8",
            )
            return output_file

        dictionaries = [
            asdict(row)
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

    # =====================================================
    # NUMERIC HELPERS
    # =====================================================

    @staticmethod
    def _paired_numeric_values(
        records: list[dict[str, str]],
        feature_a: str,
        feature_b: str,
    ) -> tuple[
        list[float],
        list[float],
    ]:
        """
        Extract only rows where both features contain finite numbers.
        """

        values_a: list[
            float
        ] = []

        values_b: list[
            float
        ] = []

        for record in records:
            value_a = (
                CorrelationRelationshipStudio
                ._to_finite_float(
                    record.get(
                        feature_a,
                        "",
                    )
                )
            )

            value_b = (
                CorrelationRelationshipStudio
                ._to_finite_float(
                    record.get(
                        feature_b,
                        "",
                    )
                )
            )

            if (
                value_a is None
                or value_b is None
            ):
                continue

            values_a.append(
                value_a
            )

            values_b.append(
                value_b
            )

        return (
            values_a,
            values_b,
        )

    @staticmethod
    def _numeric_values(
        records: list[dict[str, str]],
        feature: str,
    ) -> list[float]:
        """
        Extract finite numeric values from one feature.
        """

        values: list[
            float
        ] = []

        for record in records:
            numeric = (
                CorrelationRelationshipStudio
                ._to_finite_float(
                    record.get(
                        feature,
                        "",
                    )
                )
            )

            if numeric is not None:
                values.append(
                    numeric
                )

        return values

    @staticmethod
    def _pearson_correlation(
        values_a: list[float],
        values_b: list[float],
    ) -> float | None:
        """
        Calculate Pearson correlation safely.
        """

        if (
            len(values_a) < 2
            or len(values_b) < 2
            or len(values_a)
            != len(values_b)
        ):
            return None

        array_a = np.asarray(
            values_a,
            dtype=float,
        )

        array_b = np.asarray(
            values_b,
            dtype=float,
        )

        if (
            np.std(array_a) == 0
            or np.std(array_b) == 0
        ):
            return None

        correlation = float(
            np.corrcoef(
                array_a,
                array_b,
            )[0, 1]
        )

        if not np.isfinite(
            correlation
        ):
            return None

        return correlation

    @staticmethod
    def _to_finite_float(
        value: object,
    ) -> float | None:
        """
        Convert a value to a finite float.
        """

        if value is None:
            return None

        normalized = str(
            value
        ).strip().lower()

        if normalized in {
            "",
            "none",
            "nan",
            "null",
        }:
            return None

        try:
            numeric = float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            return None

        if not np.isfinite(
            numeric
        ):
            return None

        return numeric

    @staticmethod
    def _strength_label(
        absolute_correlation: float | None,
    ) -> str:
        """
        Assign a readable relationship-strength label.
        """

        if absolute_correlation is None:
            return "unavailable"

        if absolute_correlation >= 0.90:
            return "very strong"

        if absolute_correlation >= 0.70:
            return "strong"

        if absolute_correlation >= 0.40:
            return "moderate"

        if absolute_correlation >= 0.20:
            return "weak"

        return "very weak"

    @staticmethod
    def _direction_label(
        correlation: float | None,
    ) -> str:
        """
        Assign a direction label.
        """

        if correlation is None:
            return "unavailable"

        if correlation > 0:
            return "positive"

        if correlation < 0:
            return "negative"

        return "none"

    @staticmethod
    def _normalize_text(
        value: object,
    ) -> str:
        return str(
            value
        ).strip().lower()


def _run_correlation_studio_test() -> None:
    """
    Run the third Basketball Biomechanics Research Studio module.
    """

    studio = CorrelationRelationshipStudio()

    (
        numeric_features,
        matrix,
        pairs,
        redundancy_candidates,
    ) = studio.run()

    valid_pairs = [
        pair
        for pair in pairs
        if pair.correlation is not None
    ]

    assert len(
        numeric_features
    ) > 1

    assert matrix.shape == (
        len(numeric_features),
        len(numeric_features),
    )

    assert len(
        valid_pairs
    ) > 0

    assert studio.matrix_file.exists()
    assert studio.pairs_file.exists()
    assert studio.redundancy_file.exists()
    assert studio.top_chart_file.exists()
    assert studio.heatmap_file.exists()
    assert studio.report_file.exists()

    print(
        "CORRELATION & RELATIONSHIP STUDIO TEST PASSED"
    )

    print(
        "Numeric features analyzed: "
        f"{len(numeric_features)}"
    )

    print(
        "Unique feature pairs: "
        f"{len(pairs)}"
    )

    print(
        "Valid correlations: "
        f"{len(valid_pairs)}"
    )

    print(
        "Redundancy candidates: "
        f"{len(redundancy_candidates)}"
    )

    print()

    print(
        "Correlation matrix: "
        f"{studio.matrix_file.resolve()}"
    )

    print(
        "Correlation pairs: "
        f"{studio.pairs_file.resolve()}"
    )

    print(
        "Redundancy candidates: "
        f"{studio.redundancy_file.resolve()}"
    )

    print(
        "Top correlations chart: "
        f"{studio.top_chart_file.resolve()}"
    )

    print(
        "Correlation heatmap: "
        f"{studio.heatmap_file.resolve()}"
    )

    print(
        "Text report: "
        f"{studio.report_file.resolve()}"
    )

    print()

    print(
        "Top 10 absolute correlations:"
    )

    for pair in valid_pairs[
        :10
    ]:
        print(
            f"{pair.feature_a} vs "
            f"{pair.feature_b}: "
            f"{pair.correlation:.3f} "
            f"(n={pair.paired_count})"
        )


if __name__ == "__main__":
    _run_correlation_studio_test()