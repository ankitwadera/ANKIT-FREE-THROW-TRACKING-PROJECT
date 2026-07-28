from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, pstdev

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
class FeatureStatisticsRow:
    """
    Descriptive statistics and outlier information for one numeric feature.
    """

    feature: str
    group: str

    count: int
    missing_count: int
    completeness_percentage: float

    mean: float | None
    median: float | None
    standard_deviation: float | None

    minimum: float | None
    first_quartile: float | None
    third_quartile: float | None
    maximum: float | None
    interquartile_range: float | None

    lower_outlier_boundary: float | None
    upper_outlier_boundary: float | None
    low_outlier_count: int
    high_outlier_count: int
    total_outlier_count: int
    outlier_percentage: float

    coefficient_of_variation_percentage: float | None


@dataclass(frozen=True)
class MakeMissComparisonRow:
    """
    Made-versus-missed comparison for one numeric feature.
    """

    feature: str

    made_count: int
    made_mean: float | None
    made_median: float | None
    made_standard_deviation: float | None

    missed_count: int
    missed_mean: float | None
    missed_median: float | None
    missed_standard_deviation: float | None

    mean_difference_made_minus_missed: float | None
    median_difference_made_minus_missed: float | None
    pooled_standardized_difference: float | None


@dataclass(frozen=True)
class ParticipantFeatureStatisticsRow:
    """
    Per-participant statistics for one numeric feature.
    """

    participant_id: str
    feature: str

    count: int
    missing_count: int
    completeness_percentage: float

    mean: float | None
    median: float | None
    standard_deviation: float | None
    minimum: float | None
    maximum: float | None
    coefficient_of_variation_percentage: float | None


class FeatureStatisticsStudio:
    """
    Detailed numeric feature analysis for the research dataset.

    Outputs
    -------
    1. feature_statistics.csv
       Overall, made, and missed descriptive statistics.

    2. make_miss_feature_comparison.csv
       Side-by-side made/missed comparisons.

    3. participant_feature_statistics.csv
       Per-player means, medians, and consistency.

    4. feature_statistics_report.txt
       Human-readable research summary.

    The module describes the dataset. It does not declare that any value
    is ideal or mechanically correct.
    """

    def __init__(
        self,
        input_file: Path = DEFAULT_INPUT_FILE,
        output_folder: Path = DEFAULT_OUTPUT_FOLDER,
    ) -> None:
        self.input_file = Path(input_file)
        self.output_folder = Path(output_folder)

        self.statistics_file = (
            self.output_folder
            / "feature_statistics.csv"
        )

        self.make_miss_file = (
            self.output_folder
            / "make_miss_feature_comparison.csv"
        )

        self.participant_file = (
            self.output_folder
            / "participant_feature_statistics.csv"
        )

        self.report_file = (
            self.output_folder
            / "feature_statistics_report.txt"
        )

    # =====================================================
    # PUBLIC WORKFLOW
    # =====================================================

    def run(
        self,
    ) -> tuple[
        list[FeatureStatisticsRow],
        list[MakeMissComparisonRow],
        list[ParticipantFeatureStatisticsRow],
    ]:
        """
        Run all numeric feature analyses and export the outputs.
        """

        records = self.load_records()

        if not records:
            raise RuntimeError(
                "The master feature dataset contains no rows."
            )

        successful_records = [
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

        if not successful_records:
            raise RuntimeError(
                "No successful shot records were available."
            )

        numeric_features = self.identify_numeric_features(
            successful_records
        )

        if not numeric_features:
            raise RuntimeError(
                "No numeric features were detected."
            )

        statistics_rows = self.build_feature_statistics(
            successful_records,
            numeric_features,
        )

        comparison_rows = self.build_make_miss_comparisons(
            successful_records,
            numeric_features,
        )

        participant_rows = (
            self.build_participant_feature_statistics(
                successful_records,
                numeric_features,
            )
        )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.export_dataclass_rows(
            self.statistics_file,
            statistics_rows,
        )

        self.export_dataclass_rows(
            self.make_miss_file,
            comparison_rows,
        )

        self.export_dataclass_rows(
            self.participant_file,
            participant_rows,
        )

        self.export_text_report(
            successful_records=successful_records,
            numeric_features=numeric_features,
            statistics_rows=statistics_rows,
            comparison_rows=comparison_rows,
            participant_rows=participant_rows,
        )

        return (
            statistics_rows,
            comparison_rows,
            participant_rows,
        )

    # =====================================================
    # LOAD DATA
    # =====================================================

    def load_records(
        self,
    ) -> list[dict[str, str]]:
        """
        Read the master feature CSV.
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
        Detect fields that contain usable numeric values.
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

            if values:
                numeric_features.append(
                    field_name
                )

        return numeric_features

    # =====================================================
    # DESCRIPTIVE STATISTICS
    # =====================================================

    def build_feature_statistics(
        self,
        records: list[dict[str, str]],
        numeric_features: list[str],
    ) -> list[FeatureStatisticsRow]:
        """
        Build overall, made, and missed statistics for every feature.
        """

        groups = {
            "All": records,
            "Made": [
                record
                for record in records
                if self._normalize_result(
                    record.get(
                        "result",
                        "",
                    )
                )
                == "made"
            ],
            "Missed": [
                record
                for record in records
                if self._normalize_result(
                    record.get(
                        "result",
                        "",
                    )
                )
                == "missed"
            ],
        }

        rows: list[
            FeatureStatisticsRow
        ] = []

        for group_name, group_records in groups.items():
            for feature in numeric_features:
                values = self._numeric_values(
                    group_records,
                    feature,
                )

                rows.append(
                    self._summarize_feature(
                        feature=feature,
                        group=group_name,
                        total_rows=len(
                            group_records
                        ),
                        values=values,
                    )
                )

        rows.sort(
            key=lambda row: (
                row.feature,
                row.group,
            )
        )

        return rows

    def _summarize_feature(
        self,
        feature: str,
        group: str,
        total_rows: int,
        values: list[float],
    ) -> FeatureStatisticsRow:
        """
        Build one complete descriptive statistics row.
        """

        count = len(values)
        missing_count = (
            total_rows
            - count
        )

        completeness = (
            count
            / total_rows
            * 100.0
            if total_rows > 0
            else 0.0
        )

        if not values:
            return FeatureStatisticsRow(
                feature=feature,
                group=group,
                count=0,
                missing_count=missing_count,
                completeness_percentage=(
                    completeness
                ),
                mean=None,
                median=None,
                standard_deviation=None,
                minimum=None,
                first_quartile=None,
                third_quartile=None,
                maximum=None,
                interquartile_range=None,
                lower_outlier_boundary=None,
                upper_outlier_boundary=None,
                low_outlier_count=0,
                high_outlier_count=0,
                total_outlier_count=0,
                outlier_percentage=0.0,
                coefficient_of_variation_percentage=None,
            )

        array = np.asarray(
            values,
            dtype=float,
        )

        first_quartile = float(
            np.percentile(
                array,
                25,
            )
        )

        third_quartile = float(
            np.percentile(
                array,
                75,
            )
        )

        interquartile_range = (
            third_quartile
            - first_quartile
        )

        lower_boundary = (
            first_quartile
            - 1.5
            * interquartile_range
        )

        upper_boundary = (
            third_quartile
            + 1.5
            * interquartile_range
        )

        low_outliers = int(
            np.sum(
                array
                < lower_boundary
            )
        )

        high_outliers = int(
            np.sum(
                array
                > upper_boundary
            )
        )

        total_outliers = (
            low_outliers
            + high_outliers
        )

        outlier_percentage = (
            total_outliers
            / count
            * 100.0
            if count > 0
            else 0.0
        )

        average = float(
            mean(values)
        )

        standard_deviation = (
            float(
                pstdev(values)
            )
            if count > 1
            else 0.0
        )

        coefficient_of_variation = None

        if average != 0:
            coefficient_of_variation = (
                abs(
                    standard_deviation
                    / average
                )
                * 100.0
            )

        return FeatureStatisticsRow(
            feature=feature,
            group=group,
            count=count,
            missing_count=missing_count,
            completeness_percentage=(
                completeness
            ),
            mean=average,
            median=float(
                median(values)
            ),
            standard_deviation=(
                standard_deviation
            ),
            minimum=float(
                np.min(array)
            ),
            first_quartile=(
                first_quartile
            ),
            third_quartile=(
                third_quartile
            ),
            maximum=float(
                np.max(array)
            ),
            interquartile_range=(
                interquartile_range
            ),
            lower_outlier_boundary=(
                lower_boundary
            ),
            upper_outlier_boundary=(
                upper_boundary
            ),
            low_outlier_count=(
                low_outliers
            ),
            high_outlier_count=(
                high_outliers
            ),
            total_outlier_count=(
                total_outliers
            ),
            outlier_percentage=(
                outlier_percentage
            ),
            coefficient_of_variation_percentage=(
                coefficient_of_variation
            ),
        )

    # =====================================================
    # MAKE / MISS COMPARISONS
    # =====================================================

    def build_make_miss_comparisons(
        self,
        records: list[dict[str, str]],
        numeric_features: list[str],
    ) -> list[MakeMissComparisonRow]:
        """
        Compare made and missed shots for each numeric feature.
        """

        made_records = [
            record
            for record in records
            if self._normalize_result(
                record.get(
                    "result",
                    "",
                )
            )
            == "made"
        ]

        missed_records = [
            record
            for record in records
            if self._normalize_result(
                record.get(
                    "result",
                    "",
                )
            )
            == "missed"
        ]

        rows: list[
            MakeMissComparisonRow
        ] = []

        for feature in numeric_features:
            made_values = self._numeric_values(
                made_records,
                feature,
            )

            missed_values = self._numeric_values(
                missed_records,
                feature,
            )

            made_mean = self._safe_mean(
                made_values
            )

            missed_mean = self._safe_mean(
                missed_values
            )

            made_median = self._safe_median(
                made_values
            )

            missed_median = self._safe_median(
                missed_values
            )

            made_standard_deviation = (
                self._safe_pstdev(
                    made_values
                )
            )

            missed_standard_deviation = (
                self._safe_pstdev(
                    missed_values
                )
            )

            mean_difference = None
            median_difference = None

            if (
                made_mean is not None
                and missed_mean is not None
            ):
                mean_difference = (
                    made_mean
                    - missed_mean
                )

            if (
                made_median is not None
                and missed_median is not None
            ):
                median_difference = (
                    made_median
                    - missed_median
                )

            standardized_difference = (
                self._pooled_standardized_difference(
                    made_values,
                    missed_values,
                )
            )

            rows.append(
                MakeMissComparisonRow(
                    feature=feature,

                    made_count=len(
                        made_values
                    ),
                    made_mean=made_mean,
                    made_median=made_median,
                    made_standard_deviation=(
                        made_standard_deviation
                    ),

                    missed_count=len(
                        missed_values
                    ),
                    missed_mean=missed_mean,
                    missed_median=(
                        missed_median
                    ),
                    missed_standard_deviation=(
                        missed_standard_deviation
                    ),

                    mean_difference_made_minus_missed=(
                        mean_difference
                    ),
                    median_difference_made_minus_missed=(
                        median_difference
                    ),
                    pooled_standardized_difference=(
                        standardized_difference
                    ),
                )
            )

        rows.sort(
            key=lambda row: (
                -abs(
                    row.pooled_standardized_difference
                )
                if row.pooled_standardized_difference
                is not None
                else float("-inf"),
                row.feature,
            )
        )

        return rows

    # =====================================================
    # PARTICIPANT STATISTICS
    # =====================================================

    def build_participant_feature_statistics(
        self,
        records: list[dict[str, str]],
        numeric_features: list[str],
    ) -> list[
        ParticipantFeatureStatisticsRow
    ]:
        """
        Build one row per participant and numeric feature.
        """

        grouped: dict[
            str,
            list[dict[str, str]],
        ] = {}

        for record in records:
            participant_id = str(
                record.get(
                    "participant_id",
                    "Unknown",
                )
            ).strip()

            if not participant_id:
                participant_id = "Unknown"

            grouped.setdefault(
                participant_id,
                [],
            ).append(record)

        rows: list[
            ParticipantFeatureStatisticsRow
        ] = []

        for participant_id in sorted(
            grouped
        ):
            participant_records = grouped[
                participant_id
            ]

            total_rows = len(
                participant_records
            )

            for feature in numeric_features:
                values = self._numeric_values(
                    participant_records,
                    feature,
                )

                count = len(values)
                missing_count = (
                    total_rows
                    - count
                )

                completeness = (
                    count
                    / total_rows
                    * 100.0
                    if total_rows > 0
                    else 0.0
                )

                average = self._safe_mean(
                    values
                )

                standard_deviation = (
                    self._safe_pstdev(
                        values
                    )
                )

                coefficient_of_variation = None

                if (
                    average is not None
                    and standard_deviation
                    is not None
                    and average != 0
                ):
                    coefficient_of_variation = (
                        abs(
                            standard_deviation
                            / average
                        )
                        * 100.0
                    )

                rows.append(
                    ParticipantFeatureStatisticsRow(
                        participant_id=(
                            participant_id
                        ),
                        feature=feature,
                        count=count,
                        missing_count=(
                            missing_count
                        ),
                        completeness_percentage=(
                            completeness
                        ),
                        mean=average,
                        median=(
                            self._safe_median(
                                values
                            )
                        ),
                        standard_deviation=(
                            standard_deviation
                        ),
                        minimum=(
                            min(values)
                            if values
                            else None
                        ),
                        maximum=(
                            max(values)
                            if values
                            else None
                        ),
                        coefficient_of_variation_percentage=(
                            coefficient_of_variation
                        ),
                    )
                )

        rows.sort(
            key=lambda row: (
                row.participant_id,
                row.feature,
            )
        )

        return rows

    # =====================================================
    # TEXT REPORT
    # =====================================================

    def export_text_report(
        self,
        successful_records: list[
            dict[str, str]
        ],
        numeric_features: list[str],
        statistics_rows: list[
            FeatureStatisticsRow
        ],
        comparison_rows: list[
            MakeMissComparisonRow
        ],
        participant_rows: list[
            ParticipantFeatureStatisticsRow
        ],
    ) -> Path:
        """
        Write a readable research summary.
        """

        overall_rows = [
            row
            for row in statistics_rows
            if row.group == "All"
        ]

        highest_outlier_features = sorted(
            overall_rows,
            key=lambda row: (
                row.outlier_percentage
            ),
            reverse=True,
        )[:15]

        lowest_completeness_features = sorted(
            overall_rows,
            key=lambda row: (
                row.completeness_percentage,
                row.feature,
            ),
        )[:15]

        strongest_differences = [
            row
            for row in comparison_rows
            if row.pooled_standardized_difference
            is not None
        ][:20]

        participant_ids = sorted(
            {
                row.participant_id
                for row in participant_rows
            }
        )

        lines = [
            "=" * 78,
            "BASKETBALL BIOMECHANICS RESEARCH STUDIO",
            "FEATURE STATISTICS REPORT",
            "=" * 78,
            "",
            "DATASET USED",
            "-" * 78,
            (
                "Successful shot records: "
                f"{len(successful_records)}"
            ),
            (
                "Numeric features analyzed: "
                f"{len(numeric_features)}"
            ),
            (
                "Participants represented: "
                f"{len(participant_ids)}"
            ),
            "",
            "IMPORTANT INTERPRETATION",
            "-" * 78,
            (
                "These results describe differences and variability. "
                "They do not prove that a feature causes makes or misses."
            ),
            (
                "The standardized difference is a scale-free comparison. "
                "Larger absolute values indicate more separation between "
                "made and missed groups, but validation is still required."
            ),
            "",
            "LARGEST MADE-VERSUS-MISSED STANDARDIZED DIFFERENCES",
            "-" * 78,
        ]

        if strongest_differences:
            for row in strongest_differences:
                lines.append(
                    (
                        f"{row.feature}: "
                        f"standardized difference "
                        f"{row.pooled_standardized_difference:.3f}, "
                        f"made mean {self._format_optional(row.made_mean)}, "
                        f"missed mean {self._format_optional(row.missed_mean)}"
                    )
                )
        else:
            lines.append(
                "No comparable made-versus-missed features were available."
            )

        lines.extend(
            [
                "",
                "FEATURES WITH THE HIGHEST OUTLIER PERCENTAGES",
                "-" * 78,
            ]
        )

        for row in highest_outlier_features:
            lines.append(
                (
                    f"{row.feature}: "
                    f"{row.outlier_percentage:.2f}% "
                    f"({row.total_outlier_count} of {row.count})"
                )
            )

        lines.extend(
            [
                "",
                "LOWEST-COMPLETENESS NUMERIC FEATURES",
                "-" * 78,
            ]
        )

        for row in lowest_completeness_features:
            lines.append(
                (
                    f"{row.feature}: "
                    f"{row.completeness_percentage:.2f}% "
                    f"complete"
                )
            )

        lines.extend(
            [
                "",
                "PARTICIPANT COVERAGE",
                "-" * 78,
            ]
        )

        for participant_id in participant_ids:
            participant_feature_rows = [
                row
                for row in participant_rows
                if row.participant_id
                == participant_id
            ]

            average_completeness = (
                float(
                    mean(
                        row.completeness_percentage
                        for row in participant_feature_rows
                    )
                )
                if participant_feature_rows
                else 0.0
            )

            lines.append(
                (
                    f"{participant_id}: "
                    f"{len(participant_feature_rows)} numeric features, "
                    f"average feature completeness "
                    f"{average_completeness:.2f}%"
                )
            )

        self.report_file.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        return self.report_file

    # =====================================================
    # EXPORT HELPERS
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
    # STATISTICAL HELPERS
    # =====================================================

    @staticmethod
    def _numeric_values(
        records: list[dict[str, str]],
        feature: str,
    ) -> list[float]:
        """
        Extract valid finite numeric values for one feature.
        """

        values: list[
            float
        ] = []

        for record in records:
            raw_value = record.get(
                feature,
                "",
            )

            if not FeatureStatisticsStudio._is_populated(
                raw_value
            ):
                continue

            try:
                numeric = float(
                    raw_value
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if not np.isfinite(numeric):
                continue

            values.append(
                numeric
            )

        return values

    @staticmethod
    def _pooled_standardized_difference(
        made_values: list[float],
        missed_values: list[float],
    ) -> float | None:
        """
        Calculate a pooled standardized mean difference.

        Positive:
            Made-shot mean is higher.

        Negative:
            Missed-shot mean is higher.
        """

        made_count = len(
            made_values
        )

        missed_count = len(
            missed_values
        )

        if (
            made_count < 2
            or missed_count < 2
        ):
            return None

        made_mean = float(
            mean(made_values)
        )

        missed_mean = float(
            mean(missed_values)
        )

        made_variance = float(
            np.var(
                made_values,
                ddof=1,
            )
        )

        missed_variance = float(
            np.var(
                missed_values,
                ddof=1,
            )
        )

        degrees_of_freedom = (
            made_count
            + missed_count
            - 2
        )

        if degrees_of_freedom <= 0:
            return None

        pooled_variance = (
            (
                (
                    made_count - 1
                )
                * made_variance
            )
            + (
                (
                    missed_count - 1
                )
                * missed_variance
            )
        ) / degrees_of_freedom

        if pooled_variance <= 0:
            return None

        pooled_standard_deviation = float(
            np.sqrt(
                pooled_variance
            )
        )

        return (
            made_mean
            - missed_mean
        ) / pooled_standard_deviation

    @staticmethod
    def _safe_mean(
        values: list[float],
    ) -> float | None:
        if not values:
            return None

        return float(
            mean(values)
        )

    @staticmethod
    def _safe_median(
        values: list[float],
    ) -> float | None:
        if not values:
            return None

        return float(
            median(values)
        )

    @staticmethod
    def _safe_pstdev(
        values: list[float],
    ) -> float | None:
        if not values:
            return None

        if len(values) == 1:
            return 0.0

        return float(
            pstdev(values)
        )

    @staticmethod
    def _is_populated(
        value: object,
    ) -> bool:
        if value is None:
            return False

        normalized = str(
            value
        ).strip().lower()

        return normalized not in {
            "",
            "none",
            "nan",
            "null",
        }

    @staticmethod
    def _normalize_text(
        value: object,
    ) -> str:
        return str(
            value
        ).strip().lower()

    @staticmethod
    def _normalize_result(
        value: object,
    ) -> str:
        normalized = str(
            value
        ).strip().lower()

        if normalized in {
            "made",
            "make",
            "hit",
            "successful",
        }:
            return "made"

        if normalized in {
            "missed",
            "miss",
            "failed",
        }:
            return "missed"

        return "unknown"

    @staticmethod
    def _format_optional(
        value: float | None,
    ) -> str:
        if value is None:
            return "N/A"

        return f"{value:.3f}"


def _run_feature_statistics_test() -> None:
    """
    Run the second Basketball Biomechanics Research Studio module.
    """

    studio = FeatureStatisticsStudio()

    (
        statistics_rows,
        comparison_rows,
        participant_rows,
    ) = studio.run()

    assert len(
        statistics_rows
    ) > 0

    assert len(
        comparison_rows
    ) > 0

    assert len(
        participant_rows
    ) > 0

    assert studio.statistics_file.exists()
    assert studio.make_miss_file.exists()
    assert studio.participant_file.exists()
    assert studio.report_file.exists()

    overall_rows = [
        row
        for row in statistics_rows
        if row.group == "All"
    ]

    comparable_rows = [
        row
        for row in comparison_rows
        if row.pooled_standardized_difference
        is not None
    ]

    print(
        "FEATURE STATISTICS TEST PASSED"
    )

    print(
        "Numeric features analyzed: "
        f"{len(overall_rows)}"
    )

    print(
        "Made/missed comparisons: "
        f"{len(comparison_rows)}"
    )

    print(
        "Comparable standardized differences: "
        f"{len(comparable_rows)}"
    )

    print(
        "Participant-feature rows: "
        f"{len(participant_rows)}"
    )

    print()

    print(
        "Feature statistics: "
        f"{studio.statistics_file.resolve()}"
    )

    print(
        "Make/miss comparison: "
        f"{studio.make_miss_file.resolve()}"
    )

    print(
        "Participant statistics: "
        f"{studio.participant_file.resolve()}"
    )

    print(
        "Text report: "
        f"{studio.report_file.resolve()}"
    )

    print()

    print(
        "Top 10 absolute standardized differences:"
    )

    for row in comparable_rows[
        :10
    ]:
        print(
            f"{row.feature}: "
            f"{row.pooled_standardized_difference:.3f}"
        )


if __name__ == "__main__":
    _run_feature_statistics_test()