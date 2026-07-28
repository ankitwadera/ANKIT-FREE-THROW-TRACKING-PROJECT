from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
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


IDENTIFIER_FIELDS = {
    "participant_id",
    "trial_id",
    "result",
    "trial_file",
    "analysis_status",
    "analysis_error",
    "propulsion_onset_sequence",
    "peak_velocity_sequence",
}


@dataclass(frozen=True)
class DatasetSummary:
    """
    One-row overview of the complete feature dataset.
    """

    total_rows: int
    successful_rows: int
    failed_rows: int
    participant_count: int
    made_shots: int
    missed_shots: int
    unknown_results: int
    feature_count: int

    analysis_success_percentage: float
    overall_feature_completeness_percentage: float
    average_populated_fields_per_row: float
    missing_value_percentage: float

    duplicate_trial_count: int
    duplicate_row_count: int
    constant_feature_count: int
    never_populated_feature_count: int

    dataset_health_score: float
    dataset_health_status: str

    average_release_angle_deg: float | None
    average_release_ball_speed_ft_s: float | None
    average_release_height_ft: float | None
    average_motion_to_release_ms: float | None
    average_right_knee_rom_deg: float | None
    average_right_hip_rom_deg: float | None
    average_wrist_to_release_gap_ms: float | None
    average_hip_to_knee_gap_ms: float | None


@dataclass(frozen=True)
class ParticipantSummary:
    """
    Per-participant dataset and biomechanics overview.
    """

    participant_id: str
    total_shots: int
    successful_shots: int
    failed_shots: int
    made_shots: int
    missed_shots: int
    make_percentage: float | None

    average_release_angle_deg: float | None
    release_angle_standard_deviation_deg: float | None

    average_release_ball_speed_ft_s: float | None
    release_ball_speed_standard_deviation_ft_s: float | None

    average_release_height_ft: float | None
    average_motion_to_release_ms: float | None

    average_right_knee_rom_deg: float | None
    average_right_hip_rom_deg: float | None

    average_wrist_to_release_gap_ms: float | None
    average_hip_to_knee_gap_ms: float | None

    average_populated_fields: float
    feature_completeness_percentage: float


@dataclass(frozen=True)
class FeatureHealthRow:
    """
    Data-quality summary for one feature.
    """

    feature: str
    populated_count: int
    missing_count: int
    completeness_percentage: float

    unique_value_count: int
    is_constant: bool
    is_never_populated: bool

    numeric_count: int
    numeric_mean: float | None
    numeric_median: float | None
    numeric_standard_deviation: float | None
    numeric_minimum: float | None
    numeric_maximum: float | None


class DatasetOverview:
    """
    Front page of the Basketball Biomechanics Research Studio.

    The module reads the full shot-feature dataset and creates:

    - dataset_summary.csv
    - participant_summary.csv
    - feature_health.csv
    - dataset_overview.txt

    The dataset health score evaluates only data quality. It is not a
    basketball-performance score.
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
            / "dataset_summary.csv"
        )

        self.participant_file = (
            self.output_folder
            / "participant_summary.csv"
        )

        self.feature_health_file = (
            self.output_folder
            / "feature_health.csv"
        )

        self.text_report_file = (
            self.output_folder
            / "dataset_overview.txt"
        )

    # =====================================================
    # PUBLIC WORKFLOW
    # =====================================================

    def run(
        self,
    ) -> tuple[
        DatasetSummary,
        list[ParticipantSummary],
        list[FeatureHealthRow],
    ]:
        """
        Load the feature dataset and create all overview outputs.
        """

        records = self.load_records()

        if not records:
            raise RuntimeError(
                "The master feature dataset contains no rows."
            )

        fieldnames = list(records[0])

        feature_health = self.build_feature_health(
            records=records,
            fieldnames=fieldnames,
        )

        summary = self.build_dataset_summary(
            records=records,
            fieldnames=fieldnames,
            feature_health=feature_health,
        )

        participants = self.build_participant_summaries(
            records=records,
            fieldnames=fieldnames,
        )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.export_dataclass_rows(
            self.summary_file,
            [summary],
        )

        self.export_dataclass_rows(
            self.participant_file,
            participants,
        )

        self.export_dataclass_rows(
            self.feature_health_file,
            feature_health,
        )

        self.export_text_report(
            summary=summary,
            participants=participants,
            feature_health=feature_health,
        )

        return (
            summary,
            participants,
            feature_health,
        )

    # =====================================================
    # LOAD DATA
    # =====================================================

    def load_records(
        self,
    ) -> list[dict[str, str]]:
        """
        Read the full shot-feature CSV.
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
    # DATASET SUMMARY
    # =====================================================

    def build_dataset_summary(
        self,
        records: list[dict[str, str]],
        fieldnames: list[str],
        feature_health: list[FeatureHealthRow],
    ) -> DatasetSummary:
        """
        Calculate the top-level dataset overview.
        """

        total_rows = len(records)

        successful_rows = sum(
            self._normalize_text(
                record.get(
                    "analysis_status",
                    "",
                )
            )
            == "success"
            for record in records
        )

        failed_rows = (
            total_rows
            - successful_rows
        )

        participant_ids = {
            self._normalize_text(
                record.get(
                    "participant_id",
                    "",
                )
            )
            for record in records
            if self._normalize_text(
                record.get(
                    "participant_id",
                    "",
                )
            )
            not in {
                "",
                "unknown",
            }
        }

        made_shots = sum(
            self._normalize_result(
                record.get(
                    "result",
                    "",
                )
            )
            == "made"
            for record in records
        )

        missed_shots = sum(
            self._normalize_result(
                record.get(
                    "result",
                    "",
                )
            )
            == "missed"
            for record in records
        )

        unknown_results = (
            total_rows
            - made_shots
            - missed_shots
        )

        feature_count = len(
            fieldnames
        )

        populated_cells = sum(
            self._is_populated(
                record.get(
                    field_name,
                    "",
                )
            )
            for record in records
            for field_name in fieldnames
        )

        total_cells = (
            total_rows
            * feature_count
        )

        completeness = (
            populated_cells
            / total_cells
            * 100.0
            if total_cells > 0
            else 0.0
        )

        average_populated = (
            populated_cells
            / total_rows
            if total_rows > 0
            else 0.0
        )

        missing_percentage = (
            100.0
            - completeness
        )

        duplicate_trial_count = (
            self._count_duplicate_trials(
                records
            )
        )

        duplicate_row_count = (
            self._count_duplicate_rows(
                records
            )
        )

        constant_feature_count = sum(
            row.is_constant
            for row in feature_health
        )

        never_populated_count = sum(
            row.is_never_populated
            for row in feature_health
        )

        success_percentage = (
            successful_rows
            / total_rows
            * 100.0
            if total_rows > 0
            else 0.0
        )

        health_score = (
            self._calculate_health_score(
                analysis_success_percentage=(
                    success_percentage
                ),
                completeness_percentage=(
                    completeness
                ),
                duplicate_trial_count=(
                    duplicate_trial_count
                ),
                duplicate_row_count=(
                    duplicate_row_count
                ),
                constant_feature_count=(
                    constant_feature_count
                ),
                never_populated_feature_count=(
                    never_populated_count
                ),
                total_rows=total_rows,
                feature_count=feature_count,
            )
        )

        return DatasetSummary(
            total_rows=total_rows,
            successful_rows=successful_rows,
            failed_rows=failed_rows,
            participant_count=len(
                participant_ids
            ),
            made_shots=made_shots,
            missed_shots=missed_shots,
            unknown_results=unknown_results,
            feature_count=feature_count,

            analysis_success_percentage=(
                success_percentage
            ),
            overall_feature_completeness_percentage=(
                completeness
            ),
            average_populated_fields_per_row=(
                average_populated
            ),
            missing_value_percentage=(
                missing_percentage
            ),

            duplicate_trial_count=(
                duplicate_trial_count
            ),
            duplicate_row_count=(
                duplicate_row_count
            ),
            constant_feature_count=(
                constant_feature_count
            ),
            never_populated_feature_count=(
                never_populated_count
            ),

            dataset_health_score=(
                health_score
            ),
            dataset_health_status=(
                self._health_status(
                    health_score
                )
            ),

            average_release_angle_deg=(
                self._mean_field(
                    records,
                    "release_angle_deg",
                )
            ),
            average_release_ball_speed_ft_s=(
                self._mean_field(
                    records,
                    "release_ball_speed_ft_s",
                )
            ),
            average_release_height_ft=(
                self._mean_field(
                    records,
                    "release_height_ft",
                )
            ),
            average_motion_to_release_ms=(
                self._mean_field(
                    records,
                    "motion_to_release_ms",
                )
            ),
            average_right_knee_rom_deg=(
                self._mean_field(
                    records,
                    "right_knee_range_of_motion_deg",
                )
            ),
            average_right_hip_rom_deg=(
                self._mean_field(
                    records,
                    "right_hip_range_of_motion_deg",
                )
            ),
            average_wrist_to_release_gap_ms=(
                self._mean_field(
                    records,
                    "wrist_to_release_gap_ms",
                )
            ),
            average_hip_to_knee_gap_ms=(
                self._mean_field(
                    records,
                    "hip_to_knee_gap_ms",
                )
            ),
        )

    # =====================================================
    # PARTICIPANT SUMMARY
    # =====================================================

    def build_participant_summaries(
        self,
        records: list[dict[str, str]],
        fieldnames: list[str],
    ) -> list[ParticipantSummary]:
        """
        Create one summary row per participant.
        """

        grouped: dict[
            str,
            list[dict[str, str]],
        ] = {}

        for record in records:
            participant_id = (
                self._normalize_text(
                    record.get(
                        "participant_id",
                        "Unknown",
                    )
                )
                or "Unknown"
            )

            grouped.setdefault(
                participant_id,
                [],
            ).append(record)

        output: list[
            ParticipantSummary
        ] = []

        for participant_id in sorted(
            grouped
        ):
            participant_records = grouped[
                participant_id
            ]

            total_shots = len(
                participant_records
            )

            successful = sum(
                self._normalize_text(
                    record.get(
                        "analysis_status",
                        "",
                    )
                )
                == "success"
                for record in participant_records
            )

            failed = (
                total_shots
                - successful
            )

            made = sum(
                self._normalize_result(
                    record.get(
                        "result",
                        "",
                    )
                )
                == "made"
                for record in participant_records
            )

            missed = sum(
                self._normalize_result(
                    record.get(
                        "result",
                        "",
                    )
                )
                == "missed"
                for record in participant_records
            )

            known_attempts = (
                made
                + missed
            )

            make_percentage = (
                made
                / known_attempts
                * 100.0
                if known_attempts > 0
                else None
            )

            populated_cells = sum(
                self._is_populated(
                    record.get(
                        field_name,
                        "",
                    )
                )
                for record in participant_records
                for field_name in fieldnames
            )

            total_cells = (
                total_shots
                * len(fieldnames)
            )

            average_populated = (
                populated_cells
                / total_shots
                if total_shots > 0
                else 0.0
            )

            completeness = (
                populated_cells
                / total_cells
                * 100.0
                if total_cells > 0
                else 0.0
            )

            release_angles = (
                self._numeric_values(
                    participant_records,
                    "release_angle_deg",
                )
            )

            release_speeds = (
                self._numeric_values(
                    participant_records,
                    "release_ball_speed_ft_s",
                )
            )

            output.append(
                ParticipantSummary(
                    participant_id=(
                        participant_id
                    ),
                    total_shots=total_shots,
                    successful_shots=successful,
                    failed_shots=failed,
                    made_shots=made,
                    missed_shots=missed,
                    make_percentage=(
                        make_percentage
                    ),

                    average_release_angle_deg=(
                        self._safe_mean(
                            release_angles
                        )
                    ),
                    release_angle_standard_deviation_deg=(
                        self._safe_pstdev(
                            release_angles
                        )
                    ),

                    average_release_ball_speed_ft_s=(
                        self._safe_mean(
                            release_speeds
                        )
                    ),
                    release_ball_speed_standard_deviation_ft_s=(
                        self._safe_pstdev(
                            release_speeds
                        )
                    ),

                    average_release_height_ft=(
                        self._mean_field(
                            participant_records,
                            "release_height_ft",
                        )
                    ),
                    average_motion_to_release_ms=(
                        self._mean_field(
                            participant_records,
                            "motion_to_release_ms",
                        )
                    ),

                    average_right_knee_rom_deg=(
                        self._mean_field(
                            participant_records,
                            "right_knee_range_of_motion_deg",
                        )
                    ),
                    average_right_hip_rom_deg=(
                        self._mean_field(
                            participant_records,
                            "right_hip_range_of_motion_deg",
                        )
                    ),

                    average_wrist_to_release_gap_ms=(
                        self._mean_field(
                            participant_records,
                            "wrist_to_release_gap_ms",
                        )
                    ),
                    average_hip_to_knee_gap_ms=(
                        self._mean_field(
                            participant_records,
                            "hip_to_knee_gap_ms",
                        )
                    ),

                    average_populated_fields=(
                        average_populated
                    ),
                    feature_completeness_percentage=(
                        completeness
                    ),
                )
            )

        return output

    # =====================================================
    # FEATURE HEALTH
    # =====================================================

    def build_feature_health(
        self,
        records: list[dict[str, str]],
        fieldnames: list[str],
    ) -> list[FeatureHealthRow]:
        """
        Create one quality row per feature.
        """

        output: list[
            FeatureHealthRow
        ] = []

        total_rows = len(
            records
        )

        for field_name in fieldnames:
            raw_values = [
                record.get(
                    field_name,
                    "",
                )
                for record in records
            ]

            populated_values = [
                value
                for value in raw_values
                if self._is_populated(
                    value
                )
            ]

            populated_count = len(
                populated_values
            )

            missing_count = (
                total_rows
                - populated_count
            )

            completeness = (
                populated_count
                / total_rows
                * 100.0
                if total_rows > 0
                else 0.0
            )

            normalized_unique_values = {
                str(value).strip()
                for value in populated_values
            }

            unique_count = len(
                normalized_unique_values
            )

            never_populated = (
                populated_count == 0
            )

            constant = (
                populated_count > 0
                and unique_count == 1
            )

            numeric_values = (
                self._numeric_values(
                    records,
                    field_name,
                )
            )

            output.append(
                FeatureHealthRow(
                    feature=field_name,
                    populated_count=(
                        populated_count
                    ),
                    missing_count=(
                        missing_count
                    ),
                    completeness_percentage=(
                        completeness
                    ),

                    unique_value_count=(
                        unique_count
                    ),
                    is_constant=constant,
                    is_never_populated=(
                        never_populated
                    ),

                    numeric_count=len(
                        numeric_values
                    ),
                    numeric_mean=(
                        self._safe_mean(
                            numeric_values
                        )
                    ),
                    numeric_median=(
                        self._safe_median(
                            numeric_values
                        )
                    ),
                    numeric_standard_deviation=(
                        self._safe_pstdev(
                            numeric_values
                        )
                    ),
                    numeric_minimum=(
                        min(numeric_values)
                        if numeric_values
                        else None
                    ),
                    numeric_maximum=(
                        max(numeric_values)
                        if numeric_values
                        else None
                    ),
                )
            )

        output.sort(
            key=lambda row: (
                row.completeness_percentage,
                row.feature,
            )
        )

        return output

    # =====================================================
    # DATASET HEALTH SCORE
    # =====================================================

    @staticmethod
    def _calculate_health_score(
        analysis_success_percentage: float,
        completeness_percentage: float,
        duplicate_trial_count: int,
        duplicate_row_count: int,
        constant_feature_count: int,
        never_populated_feature_count: int,
        total_rows: int,
        feature_count: int,
    ) -> float:
        """
        Calculate a transparent data-quality score.

        Weighting
        ---------
        Analysis success: 45%
        Feature completeness: 45%
        Duplicate penalty: up to 5%
        Constant/empty feature penalty: up to 5%
        """

        base_score = (
            analysis_success_percentage
            * 0.45
            + completeness_percentage
            * 0.45
        )

        duplicate_total = (
            duplicate_trial_count
            + duplicate_row_count
        )

        duplicate_rate = (
            duplicate_total
            / total_rows
            if total_rows > 0
            else 0.0
        )

        duplicate_penalty = min(
            5.0,
            duplicate_rate
            * 100.0,
        )

        weak_feature_total = (
            constant_feature_count
            + never_populated_feature_count
        )

        weak_feature_rate = (
            weak_feature_total
            / feature_count
            if feature_count > 0
            else 0.0
        )

        weak_feature_penalty = min(
            5.0,
            weak_feature_rate
            * 100.0,
        )

        score = (
            base_score
            + 10.0
            - duplicate_penalty
            - weak_feature_penalty
        )

        return float(
            np.clip(
                score,
                0.0,
                100.0,
            )
        )

    @staticmethod
    def _health_status(
        score: float,
    ) -> str:
        if score >= 95:
            return "Excellent"

        if score >= 90:
            return "Very Good"

        if score >= 80:
            return "Good"

        if score >= 70:
            return "Needs Review"

        return "Not Ready"

    # =====================================================
    # TEXT REPORT
    # =====================================================

    def export_text_report(
        self,
        summary: DatasetSummary,
        participants: list[
            ParticipantSummary
        ],
        feature_health: list[
            FeatureHealthRow
        ],
    ) -> Path:
        """
        Write a readable permanent overview report.
        """

        complete_features = [
            row
            for row in feature_health
            if row.completeness_percentage
            == 100.0
        ]

        below_90 = [
            row
            for row in feature_health
            if row.completeness_percentage
            < 90.0
        ]

        below_50 = [
            row
            for row in feature_health
            if row.completeness_percentage
            < 50.0
        ]

        constant_features = [
            row
            for row in feature_health
            if row.is_constant
        ]

        never_populated = [
            row
            for row in feature_health
            if row.is_never_populated
        ]

        lines = [
            "=" * 72,
            "BASKETBALL BIOMECHANICS RESEARCH STUDIO",
            "DATASET OVERVIEW",
            "=" * 72,
            "",
            "DATASET HEALTH",
            "-" * 72,
            (
                "Overall score: "
                f"{summary.dataset_health_score:.1f} / 100"
            ),
            (
                "Status: "
                f"{summary.dataset_health_status}"
            ),
            (
                "Analysis success: "
                f"{summary.analysis_success_percentage:.2f}%"
            ),
            (
                "Feature completeness: "
                f"{summary.overall_feature_completeness_percentage:.2f}%"
            ),
            (
                "Missing values: "
                f"{summary.missing_value_percentage:.2f}%"
            ),
            "",
            "DATASET SIZE",
            "-" * 72,
            f"Rows: {summary.total_rows}",
            f"Successful analyses: {summary.successful_rows}",
            f"Failed analyses: {summary.failed_rows}",
            f"Participants: {summary.participant_count}",
            f"Made shots: {summary.made_shots}",
            f"Missed shots: {summary.missed_shots}",
            f"Unknown results: {summary.unknown_results}",
            f"Feature fields: {summary.feature_count}",
            (
                "Average populated fields: "
                f"{summary.average_populated_fields_per_row:.1f} "
                f"/ {summary.feature_count}"
            ),
            "",
            "QUALITY CHECKS",
            "-" * 72,
            (
                "Duplicate trial identifiers: "
                f"{summary.duplicate_trial_count}"
            ),
            (
                "Duplicate full rows: "
                f"{summary.duplicate_row_count}"
            ),
            (
                "Constant features: "
                f"{summary.constant_feature_count}"
            ),
            (
                "Never-populated features: "
                f"{summary.never_populated_feature_count}"
            ),
            (
                "Features with 100% completeness: "
                f"{len(complete_features)}"
            ),
            (
                "Features below 90% completeness: "
                f"{len(below_90)}"
            ),
            (
                "Features below 50% completeness: "
                f"{len(below_50)}"
            ),
            "",
            "SELECTED DATASET AVERAGES",
            "-" * 72,
            self._format_metric(
                "Release angle",
                summary.average_release_angle_deg,
                "degrees",
            ),
            self._format_metric(
                "Release ball speed",
                summary.average_release_ball_speed_ft_s,
                "ft/s",
            ),
            self._format_metric(
                "Release height",
                summary.average_release_height_ft,
                "ft",
            ),
            self._format_metric(
                "Motion to release",
                summary.average_motion_to_release_ms,
                "ms",
            ),
            self._format_metric(
                "Right knee range of motion",
                summary.average_right_knee_rom_deg,
                "degrees",
            ),
            self._format_metric(
                "Right hip range of motion",
                summary.average_right_hip_rom_deg,
                "degrees",
            ),
            self._format_metric(
                "Wrist to release gap",
                summary.average_wrist_to_release_gap_ms,
                "ms",
            ),
            self._format_metric(
                "Hip to knee gap",
                summary.average_hip_to_knee_gap_ms,
                "ms",
            ),
            "",
            "PARTICIPANT BREAKDOWN",
            "-" * 72,
        ]

        for participant in participants:
            lines.extend(
                [
                    (
                        f"{participant.participant_id}: "
                        f"{participant.total_shots} shots, "
                        f"{participant.made_shots} made, "
                        f"{participant.missed_shots} missed"
                    ),
                    self._format_metric(
                        "  Make percentage",
                        participant.make_percentage,
                        "%",
                    ),
                    self._format_metric(
                        "  Average release angle",
                        participant.average_release_angle_deg,
                        "degrees",
                    ),
                    self._format_metric(
                        "  Release-angle variability",
                        participant.release_angle_standard_deviation_deg,
                        "degrees SD",
                    ),
                    self._format_metric(
                        "  Average ball speed",
                        participant.average_release_ball_speed_ft_s,
                        "ft/s",
                    ),
                    self._format_metric(
                        "  Average motion duration",
                        participant.average_motion_to_release_ms,
                        "ms",
                    ),
                    "",
                ]
            )

        lines.extend(
            [
                "LOW-COMPLETENESS FEATURES",
                "-" * 72,
            ]
        )

        if below_90:
            for row in below_90:
                lines.append(
                    f"{row.feature}: "
                    f"{row.completeness_percentage:.2f}%"
                )
        else:
            lines.append(
                "No features are below 90% completeness."
            )

        lines.extend(
            [
                "",
                "CONSTANT FEATURES",
                "-" * 72,
            ]
        )

        if constant_features:
            for row in constant_features:
                lines.append(
                    row.feature
                )
        else:
            lines.append(
                "No constant features detected."
            )

        lines.extend(
            [
                "",
                "NEVER-POPULATED FEATURES",
                "-" * 72,
            ]
        )

        if never_populated:
            for row in never_populated:
                lines.append(
                    row.feature
                )
        else:
            lines.append(
                "No never-populated features detected."
            )

        self.text_report_file.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        return self.text_report_file

    # =====================================================
    # EXPORT HELPERS
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
    # DUPLICATE CHECKS
    # =====================================================

    @staticmethod
    def _count_duplicate_trials(
        records: list[dict[str, str]],
    ) -> int:
        """
        Count repeated participant/trial/file identifiers.
        """

        seen: set[
            tuple[str, str, str]
        ] = set()

        duplicates = 0

        for record in records:
            identifier = (
                str(
                    record.get(
                        "participant_id",
                        "",
                    )
                ).strip(),
                str(
                    record.get(
                        "trial_id",
                        "",
                    )
                ).strip(),
                str(
                    record.get(
                        "trial_file",
                        "",
                    )
                ).strip(),
            )

            if identifier in seen:
                duplicates += 1
            else:
                seen.add(identifier)

        return duplicates

    @staticmethod
    def _count_duplicate_rows(
        records: list[dict[str, str]],
    ) -> int:
        """
        Count fully identical rows.
        """

        seen: set[
            tuple[
                tuple[str, str],
                ...
            ]
        ] = set()

        duplicates = 0

        for record in records:
            row_key = tuple(
                sorted(
                    (
                        key,
                        str(value),
                    )
                    for key, value in (
                        record.items()
                    )
                )
            )

            if row_key in seen:
                duplicates += 1
            else:
                seen.add(row_key)

        return duplicates

    # =====================================================
    # NUMERIC HELPERS
    # =====================================================

    @classmethod
    def _mean_field(
        cls,
        records: list[dict[str, str]],
        field_name: str,
    ) -> float | None:
        return cls._safe_mean(
            cls._numeric_values(
                records,
                field_name,
            )
        )

    @staticmethod
    def _numeric_values(
        records: list[dict[str, str]],
        field_name: str,
    ) -> list[float]:
        """
        Extract valid numeric values from one field.
        """

        output: list[float] = []

        for record in records:
            raw_value = record.get(
                field_name,
                "",
            )

            if not DatasetOverview._is_populated(
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

            if np.isnan(numeric):
                continue

            output.append(numeric)

        return output

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
        """
        Treat blank strings, None, and text NaN values as missing.
        """

        if value is None:
            return False

        normalized = str(
            value
        ).strip()

        if normalized.lower() in {
            "",
            "none",
            "nan",
            "null",
        }:
            return False

        return True

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
    def _format_metric(
        label: str,
        value: float | None,
        suffix: str,
    ) -> str:
        if value is None:
            return f"{label}: N/A"

        return (
            f"{label}: "
            f"{value:.2f} {suffix}"
        )


def _run_dataset_overview_test() -> None:
    """
    Run the first Basketball Biomechanics Research Studio module.
    """

    overview = DatasetOverview()

    (
        summary,
        participants,
        feature_health,
    ) = overview.run()

    assert summary.total_rows > 0
    assert len(participants) > 0
    assert len(feature_health) > 0

    assert overview.summary_file.exists()
    assert overview.participant_file.exists()
    assert overview.feature_health_file.exists()
    assert overview.text_report_file.exists()

    print(
        "DATASET OVERVIEW TEST PASSED"
    )

    print(
        f"Rows analyzed: "
        f"{summary.total_rows}"
    )

    print(
        f"Successful analyses: "
        f"{summary.successful_rows}"
    )

    print(
        f"Failed analyses: "
        f"{summary.failed_rows}"
    )

    print(
        f"Participants: "
        f"{summary.participant_count}"
    )

    print(
        f"Made shots: "
        f"{summary.made_shots}"
    )

    print(
        f"Missed shots: "
        f"{summary.missed_shots}"
    )

    print(
        f"Feature fields: "
        f"{summary.feature_count}"
    )

    print(
        "Feature completeness: "
        f"{summary.overall_feature_completeness_percentage:.2f}%"
    )

    print(
        "Dataset health score: "
        f"{summary.dataset_health_score:.1f} / 100"
    )

    print(
        "Dataset health status: "
        f"{summary.dataset_health_status}"
    )

    print()

    print(
        "Dataset summary: "
        f"{overview.summary_file.resolve()}"
    )

    print(
        "Participant summary: "
        f"{overview.participant_file.resolve()}"
    )

    print(
        "Feature health: "
        f"{overview.feature_health_file.resolve()}"
    )

    print(
        "Text report: "
        f"{overview.text_report_file.resolve()}"
    )


if __name__ == "__main__":
    _run_dataset_overview_test()