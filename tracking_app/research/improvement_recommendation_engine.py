from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, pstdev

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FEATURE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "all_shot_features.csv"
)

DEFAULT_DIAGNOSTIC_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "personal_signal_diagnostics"
    / "personal_feature_diagnostics.csv"
)

DEFAULT_OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "improvement_recommendations"
)


NON_COACHING_FIELDS = {
    "participant_id",
    "trial_id",
    "result",
    "trial_file",
    "sampling_rate",
    "total_frames",
    "analysis_status",
    "analysis_error",
    "propulsion_onset_sequence",
    "peak_velocity_sequence",
    "authoritative_event_count",
    "average_event_confidence",
}


@dataclass(frozen=True)
class SuccessfulBaselineFeature:
    """
    Personal successful-shot baseline for one feature.
    """

    participant_id: str
    feature: str

    made_count: int
    made_mean: float
    made_median: float
    made_standard_deviation: float

    missed_count: int
    missed_mean: float | None
    missed_standard_deviation: float | None

    diagnostic_score: float
    diagnostic_rank: int
    signal_strength: str
    direction_stability_percentage: float | None


@dataclass(frozen=True)
class ShotDeviation:
    """
    Difference between one shot and the player's successful baseline.
    """

    participant_id: str
    trial_id: str
    trial_file: str
    actual_result: str

    feature: str
    shot_value: float

    successful_mean: float
    successful_median: float
    successful_standard_deviation: float

    signed_z_score: float | None
    absolute_z_score: float | None
    percentage_difference_from_successful_mean: float | None

    diagnostic_score: float
    diagnostic_rank: int
    signal_strength: str
    direction_stability_percentage: float | None

    deviation_priority_score: float
    deviation_level: str

    direction_text: str
    coaching_message: str
    evidence_statement: str


@dataclass(frozen=True)
class RecommendationReportSummary:
    """
    High-level summary for one analyzed shot.
    """

    participant_id: str
    trial_id: str
    trial_file: str
    actual_result: str

    baseline_made_shots: int
    baseline_missed_shots: int

    features_compared: int
    high_priority_deviations: int
    moderate_priority_deviations: int
    low_priority_deviations: int

    strongest_deviation_feature: str
    strongest_deviation_score: float

    overall_similarity_to_successful_baseline: float
    recommendation_confidence: str


class ImprovementRecommendationEngine:
    """
    Convert personal feature differences into cautious coaching feedback.

    This engine does not compare the shooter to a universal ideal.
    It compares one shot against that participant's own successful-shot
    baseline and prioritizes only features with personal diagnostic support.

    The recommendation process is:

        selected shot
            ↓
        compare with player's made-shot baseline
            ↓
        calculate standardized deviation
            ↓
        weight by personal signal strength and stability
            ↓
        create coach-readable observations

    A recommendation describes what changed. It does not prove that changing
    the feature will cause a make.
    """

    def __init__(
        self,
        feature_file: Path = DEFAULT_FEATURE_FILE,
        diagnostic_file: Path = DEFAULT_DIAGNOSTIC_FILE,
        output_folder: Path = DEFAULT_OUTPUT_FOLDER,
        minimum_diagnostic_score: float = 45.0,
        maximum_recommendations: int = 8,
    ) -> None:
        self.feature_file = Path(feature_file)
        self.diagnostic_file = Path(diagnostic_file)
        self.output_folder = Path(output_folder)

        self.minimum_diagnostic_score = float(
            minimum_diagnostic_score
        )

        self.maximum_recommendations = max(
            1,
            int(maximum_recommendations),
        )

        self.deviations_file = (
            self.output_folder
            / "shot_deviations.csv"
        )

        self.summary_file = (
            self.output_folder
            / "recommendation_summary.csv"
        )

        self.report_file = (
            self.output_folder
            / "improvement_recommendation_report.txt"
        )

        self.chart_file = (
            self.output_folder
            / "shot_deviation_chart.png"
        )

    # =====================================================
    # PUBLIC WORKFLOW
    # =====================================================

    def run(
        self,
        participant_id: str | None = None,
        trial_id: str | None = None,
    ) -> tuple[
        RecommendationReportSummary,
        list[ShotDeviation],
    ]:
        """
        Analyze one shot.

        When no participant or trial is supplied, the module uses the first
        successful row in the dataset so the test can run automatically.
        """

        feature_records = self.load_csv(
            self.feature_file
        )

        diagnostic_records = self.load_csv(
            self.diagnostic_file
        )

        successful_rows = [
            record
            for record in feature_records
            if self._normalize_text(
                record.get(
                    "analysis_status",
                    "",
                )
            )
            == "success"
        ]

        if not successful_rows:
            raise RuntimeError(
                "No successful feature rows were available."
            )

        selected_shot = self.select_shot(
            records=successful_rows,
            participant_id=participant_id,
            trial_id=trial_id,
        )

        selected_participant = str(
            selected_shot.get(
                "participant_id",
                "",
            )
        ).strip()

        participant_rows = [
            record
            for record in successful_rows
            if str(
                record.get(
                    "participant_id",
                    "",
                )
            ).strip()
            == selected_participant
        ]

        made_rows = [
            record
            for record in participant_rows
            if self._normalize_result(
                record.get(
                    "result",
                    "",
                )
            )
            == "made"
        ]

        missed_rows = [
            record
            for record in participant_rows
            if self._normalize_result(
                record.get(
                    "result",
                    "",
                )
            )
            == "missed"
        ]

        if len(
            made_rows
        ) < 10:
            raise RuntimeError(
                f"{selected_participant} does not have at least "
                "10 successful shots for a personal baseline."
            )

        personal_diagnostics = [
            record
            for record in diagnostic_records
            if str(
                record.get(
                    "participant_id",
                    "",
                )
            ).strip()
            == selected_participant
        ]

        baselines = self.build_successful_baselines(
            participant_id=selected_participant,
            made_rows=made_rows,
            missed_rows=missed_rows,
            diagnostic_rows=personal_diagnostics,
        )

        deviations = self.compare_shot_to_baseline(
            selected_shot=selected_shot,
            baselines=baselines,
        )

        summary = self.build_summary(
            selected_shot=selected_shot,
            made_count=len(
                made_rows
            ),
            missed_count=len(
                missed_rows
            ),
            deviations=deviations,
        )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.export_dataclass_rows(
            self.deviations_file,
            deviations,
        )

        self.export_dataclass_rows(
            self.summary_file,
            [
                summary
            ],
        )

        self.export_text_report(
            summary=summary,
            deviations=deviations,
        )

        self.create_deviation_chart(
            deviations
        )

        return (
            summary,
            deviations,
        )

    # =====================================================
    # LOAD AND SELECT
    # =====================================================

    @staticmethod
    def load_csv(
        path: Path,
    ) -> list[dict[str, str]]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required CSV was not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            return list(
                csv.DictReader(
                    file
                )
            )

    def select_shot(
        self,
        records: list[dict[str, str]],
        participant_id: str | None,
        trial_id: str | None,
    ) -> dict[str, str]:
        """
        Select one shot by participant and/or trial.
        """

        candidates = records

        if participant_id is not None:
            candidates = [
                record
                for record in candidates
                if str(
                    record.get(
                        "participant_id",
                        "",
                    )
                ).strip()
                == participant_id
            ]

        if trial_id is not None:
            candidates = [
                record
                for record in candidates
                if str(
                    record.get(
                        "trial_id",
                        "",
                    )
                ).strip()
                == trial_id
            ]

        if not candidates:
            raise ValueError(
                "No shot matched the requested participant/trial."
            )

        # Prefer a missed shot for the automatic demonstration because
        # that makes the generated feedback easier to inspect.
        missed_candidates = [
            record
            for record in candidates
            if self._normalize_result(
                record.get(
                    "result",
                    "",
                )
            )
            == "missed"
        ]

        if missed_candidates:
            return missed_candidates[0]

        return candidates[0]

    # =====================================================
    # BASELINE
    # =====================================================

    def build_successful_baselines(
        self,
        participant_id: str,
        made_rows: list[dict[str, str]],
        missed_rows: list[dict[str, str]],
        diagnostic_rows: list[dict[str, str]],
    ) -> list[SuccessfulBaselineFeature]:
        """
        Build a baseline only for personally supported coaching features.
        """

        baselines: list[
            SuccessfulBaselineFeature
        ] = []

        for diagnostic in diagnostic_rows:
            feature = str(
                diagnostic.get(
                    "feature",
                    "",
                )
            ).strip()

            if (
                not feature
                or feature
                in NON_COACHING_FIELDS
            ):
                continue

            diagnostic_score = self._to_finite_float(
                diagnostic.get(
                    "diagnostic_score",
                    "",
                )
            )

            if (
                diagnostic_score is None
                or diagnostic_score
                < self.minimum_diagnostic_score
            ):
                continue

            made_values = self._numeric_values(
                made_rows,
                feature,
            )

            missed_values = self._numeric_values(
                missed_rows,
                feature,
            )

            if len(
                made_values
            ) < 10:
                continue

            made_average = float(
                mean(
                    made_values
                )
            )

            made_median = float(
                median(
                    made_values
                )
            )

            made_sd = (
                float(
                    pstdev(
                        made_values
                    )
                )
                if len(
                    made_values
                )
                > 1
                else 0.0
            )

            missed_mean = (
                float(
                    mean(
                        missed_values
                    )
                )
                if missed_values
                else None
            )

            missed_sd = (
                float(
                    pstdev(
                        missed_values
                    )
                )
                if len(
                    missed_values
                )
                > 1
                else (
                    0.0
                    if missed_values
                    else None
                )
            )

            stability = self._to_finite_float(
                diagnostic.get(
                    "bootstrap_direction_stability_percentage",
                    "",
                )
            )

            rank = self._to_int(
                diagnostic.get(
                    "diagnostic_rank",
                    "",
                ),
                fallback=9999,
            )

            baselines.append(
                SuccessfulBaselineFeature(
                    participant_id=participant_id,
                    feature=feature,

                    made_count=len(
                        made_values
                    ),
                    made_mean=made_average,
                    made_median=made_median,
                    made_standard_deviation=(
                        made_sd
                    ),

                    missed_count=len(
                        missed_values
                    ),
                    missed_mean=missed_mean,
                    missed_standard_deviation=(
                        missed_sd
                    ),

                    diagnostic_score=float(
                        diagnostic_score
                    ),
                    diagnostic_rank=rank,
                    signal_strength=str(
                        diagnostic.get(
                            "signal_strength",
                            "unknown",
                        )
                    ),
                    direction_stability_percentage=(
                        stability
                    ),
                )
            )

        baselines.sort(
            key=lambda row: (
                row.diagnostic_rank,
                -row.diagnostic_score,
                row.feature,
            )
        )

        return baselines

    # =====================================================
    # SHOT COMPARISON
    # =====================================================

    def compare_shot_to_baseline(
        self,
        selected_shot: dict[str, str],
        baselines: list[SuccessfulBaselineFeature],
    ) -> list[ShotDeviation]:
        """
        Compare the selected shot to each supported successful baseline.
        """

        rows: list[
            ShotDeviation
        ] = []

        participant_id = str(
            selected_shot.get(
                "participant_id",
                "",
            )
        )

        trial_id = str(
            selected_shot.get(
                "trial_id",
                "",
            )
        )

        trial_file = str(
            selected_shot.get(
                "trial_file",
                "",
            )
        )

        actual_result = self._normalize_result(
            selected_shot.get(
                "result",
                "",
            )
        )

        for baseline in baselines:
            shot_value = self._to_finite_float(
                selected_shot.get(
                    baseline.feature,
                    "",
                )
            )

            if shot_value is None:
                continue

            z_score = None

            if (
                baseline.made_standard_deviation
                > 1e-9
            ):
                z_score = (
                    shot_value
                    - baseline.made_mean
                ) / baseline.made_standard_deviation

            percentage_difference = None

            if abs(
                baseline.made_mean
            ) > 1e-9:
                percentage_difference = (
                    (
                        shot_value
                        - baseline.made_mean
                    )
                    / abs(
                        baseline.made_mean
                    )
                    * 100.0
                )

            absolute_z = (
                None
                if z_score is None
                else abs(
                    z_score
                )
            )

            priority_score = self._priority_score(
                absolute_z_score=absolute_z,
                diagnostic_score=(
                    baseline.diagnostic_score
                ),
                direction_stability=(
                    baseline.direction_stability_percentage
                ),
            )

            deviation_level = self._deviation_level(
                priority_score
            )

            direction_text = self._direction_text(
                shot_value=shot_value,
                baseline_mean=(
                    baseline.made_mean
                ),
            )

            coaching_message = self._coaching_message(
                feature=baseline.feature,
                direction_text=direction_text,
                shot_value=shot_value,
                baseline_mean=(
                    baseline.made_mean
                ),
                z_score=z_score,
            )

            evidence_statement = (
                self._evidence_statement(
                    baseline=baseline
                )
            )

            rows.append(
                ShotDeviation(
                    participant_id=participant_id,
                    trial_id=trial_id,
                    trial_file=trial_file,
                    actual_result=actual_result,

                    feature=baseline.feature,
                    shot_value=float(
                        shot_value
                    ),

                    successful_mean=(
                        baseline.made_mean
                    ),
                    successful_median=(
                        baseline.made_median
                    ),
                    successful_standard_deviation=(
                        baseline.made_standard_deviation
                    ),

                    signed_z_score=(
                        None
                        if z_score is None
                        else float(
                            z_score
                        )
                    ),
                    absolute_z_score=(
                        None
                        if absolute_z is None
                        else float(
                            absolute_z
                        )
                    ),
                    percentage_difference_from_successful_mean=(
                        None
                        if percentage_difference is None
                        else float(
                            percentage_difference
                        )
                    ),

                    diagnostic_score=(
                        baseline.diagnostic_score
                    ),
                    diagnostic_rank=(
                        baseline.diagnostic_rank
                    ),
                    signal_strength=(
                        baseline.signal_strength
                    ),
                    direction_stability_percentage=(
                        baseline.direction_stability_percentage
                    ),

                    deviation_priority_score=(
                        priority_score
                    ),
                    deviation_level=(
                        deviation_level
                    ),

                    direction_text=(
                        direction_text
                    ),
                    coaching_message=(
                        coaching_message
                    ),
                    evidence_statement=(
                        evidence_statement
                    ),
                )
            )

        rows.sort(
            key=lambda row: (
                -row.deviation_priority_score,
                row.diagnostic_rank,
                row.feature,
            )
        )

        return rows

    # =====================================================
    # SCORING AND LANGUAGE
    # =====================================================

    @staticmethod
    def _priority_score(
        absolute_z_score: float | None,
        diagnostic_score: float,
        direction_stability: float | None,
    ) -> float:
        """
        Combine size of deviation with personal evidence strength.
        """

        deviation_component = (
            0.0
            if absolute_z_score is None
            else min(
                1.0,
                absolute_z_score
                / 2.5,
            )
        )

        diagnostic_component = (
            min(
                100.0,
                max(
                    0.0,
                    diagnostic_score,
                ),
            )
            / 100.0
        )

        stability_component = (
            0.50
            if direction_stability is None
            else min(
                100.0,
                max(
                    0.0,
                    direction_stability,
                ),
            )
            / 100.0
        )

        score = (
            deviation_component
            * 0.50
            + diagnostic_component
            * 0.30
            + stability_component
            * 0.20
        ) * 100.0

        return float(
            np.clip(
                score,
                0.0,
                100.0,
            )
        )

    @staticmethod
    def _deviation_level(
        score: float,
    ) -> str:
        if score >= 70:
            return "high"

        if score >= 50:
            return "moderate"

        return "low"

    @staticmethod
    def _direction_text(
        shot_value: float,
        baseline_mean: float,
    ) -> str:
        if shot_value > baseline_mean:
            return "higher"

        if shot_value < baseline_mean:
            return "lower"

        return "similar"

    @staticmethod
    def _friendly_feature_name(
        feature: str,
    ) -> str:
        """
        Convert technical field names to coach-readable labels.
        """

        replacements = {
            "release_right_elbow_angle_deg":
                "right elbow angle at release",
            "release_left_elbow_angle_deg":
                "left elbow angle at release",
            "release_right_knee_angle_deg":
                "right knee angle at release",
            "release_left_knee_angle_deg":
                "left knee angle at release",
            "release_right_hip_angle_deg":
                "right hip angle at release",
            "release_left_hip_angle_deg":
                "left hip angle at release",
            "release_angle_deg":
                "ball release angle",
            "release_ball_speed_ft_s":
                "ball speed at release",
            "release_vertical_speed_ft_s":
                "vertical ball speed at release",
            "peak_right_wrist_speed_ft_s":
                "peak right wrist speed",
            "peak_left_wrist_speed_ft_s":
                "peak left wrist speed",
            "peak_ball_speed_ft_s":
                "peak ball speed",
            "knee_to_elbow_gap_ms":
                "knee-to-elbow timing gap",
            "elbow_to_release_gap_ms":
                "elbow-to-release timing gap",
            "wrist_to_release_gap_ms":
                "wrist-to-release timing gap",
            "hip_to_knee_gap_ms":
                "hip-to-knee timing gap",
            "pelvis_vertical_displacement_ft":
                "pelvis vertical displacement",
            "release_to_landing_ms":
                "release-to-landing time",
        }

        if feature in replacements:
            return replacements[
                feature
            ]

        return feature.replace(
            "_",
            " ",
        )

    def _coaching_message(
        self,
        feature: str,
        direction_text: str,
        shot_value: float,
        baseline_mean: float,
        z_score: float | None,
    ) -> str:
        friendly_name = self._friendly_feature_name(
            feature
        )

        if direction_text == "similar":
            return (
                f"Your {friendly_name} was close to your "
                "successful-shot average."
            )

        magnitude_text = (
            ""
            if z_score is None
            else (
                f" It was {abs(z_score):.2f} personal standard "
                "deviations from that baseline."
            )
        )

        return (
            f"Your {friendly_name} was {direction_text} than in "
            f"your successful shots "
            f"({shot_value:.2f} versus a made-shot average of "
            f"{baseline_mean:.2f})."
            f"{magnitude_text}"
        )

    @staticmethod
    def _evidence_statement(
        baseline: SuccessfulBaselineFeature,
    ) -> str:
        stability_text = (
            "stability unavailable"
            if baseline.direction_stability_percentage
            is None
            else (
                f"{baseline.direction_stability_percentage:.1f}% "
                "bootstrap direction stability"
            )
        )

        return (
            f"Personal diagnostic score "
            f"{baseline.diagnostic_score:.1f}; "
            f"{stability_text}; "
            f"baseline from {baseline.made_count} made shots."
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    def build_summary(
        self,
        selected_shot: dict[str, str],
        made_count: int,
        missed_count: int,
        deviations: list[ShotDeviation],
    ) -> RecommendationReportSummary:
        high_count = sum(
            row.deviation_level
            == "high"
            for row in deviations
        )

        moderate_count = sum(
            row.deviation_level
            == "moderate"
            for row in deviations
        )

        low_count = sum(
            row.deviation_level
            == "low"
            for row in deviations
        )

        strongest_feature = (
            deviations[0].feature
            if deviations
            else "N/A"
        )

        strongest_score = (
            deviations[0].deviation_priority_score
            if deviations
            else 0.0
        )

        z_values = [
            row.absolute_z_score
            for row in deviations
            if row.absolute_z_score
            is not None
        ]

        if z_values:
            average_clipped_z = float(
                mean(
                    min(
                        2.5,
                        value,
                    )
                    for value in z_values
                )
            )

            similarity = (
                1.0
                - average_clipped_z
                / 2.5
            ) * 100.0

        else:
            similarity = 0.0

        confidence = self._recommendation_confidence(
            deviations=deviations,
            made_count=made_count,
        )

        return RecommendationReportSummary(
            participant_id=str(
                selected_shot.get(
                    "participant_id",
                    "",
                )
            ),
            trial_id=str(
                selected_shot.get(
                    "trial_id",
                    "",
                )
            ),
            trial_file=str(
                selected_shot.get(
                    "trial_file",
                    "",
                )
            ),
            actual_result=(
                self._normalize_result(
                    selected_shot.get(
                        "result",
                        "",
                    )
                )
            ),

            baseline_made_shots=made_count,
            baseline_missed_shots=(
                missed_count
            ),

            features_compared=len(
                deviations
            ),
            high_priority_deviations=(
                high_count
            ),
            moderate_priority_deviations=(
                moderate_count
            ),
            low_priority_deviations=(
                low_count
            ),

            strongest_deviation_feature=(
                strongest_feature
            ),
            strongest_deviation_score=(
                strongest_score
            ),

            overall_similarity_to_successful_baseline=float(
                np.clip(
                    similarity,
                    0.0,
                    100.0,
                )
            ),
            recommendation_confidence=(
                confidence
            ),
        )

    @staticmethod
    def _recommendation_confidence(
        deviations: list[ShotDeviation],
        made_count: int,
    ) -> str:
        strong_rows = [
            row
            for row in deviations
            if (
                row.diagnostic_score
                >= 65
                and (
                    row.direction_stability_percentage
                    or 0.0
                )
                >= 75
            )
        ]

        if (
            made_count >= 50
            and len(
                strong_rows
            )
            >= 3
        ):
            return "moderate"

        if (
            made_count >= 25
            and len(
                strong_rows
            )
            >= 1
        ):
            return "limited"

        return "exploratory"

    # =====================================================
    # REPORT
    # =====================================================

    def export_text_report(
        self,
        summary: RecommendationReportSummary,
        deviations: list[ShotDeviation],
    ) -> Path:
        lines = [
            "=" * 88,
            "PERSONAL FREE-THROW IMPROVEMENT REPORT",
            "=" * 88,
            "",
            "SHOT",
            "-" * 88,
            f"Participant: {summary.participant_id}",
            f"Trial: {summary.trial_id}",
            f"File: {summary.trial_file}",
            f"Recorded result: {summary.actual_result}",
            "",
            "PERSONAL BASELINE",
            "-" * 88,
            (
                "Made shots used: "
                f"{summary.baseline_made_shots}"
            ),
            (
                "Missed shots available: "
                f"{summary.baseline_missed_shots}"
            ),
            (
                "Features compared: "
                f"{summary.features_compared}"
            ),
            (
                "Similarity to successful baseline: "
                f"{summary.overall_similarity_to_successful_baseline:.1f}%"
            ),
            (
                "Recommendation confidence: "
                f"{summary.recommendation_confidence}"
            ),
            "",
            "PRIORITY OBSERVATIONS",
            "-" * 88,
        ]

        priority_rows = [
            row
            for row in deviations
            if row.deviation_level
            in {
                "high",
                "moderate",
            }
        ][
            :self.maximum_recommendations
        ]

        if not priority_rows:
            lines.append(
                "No moderate- or high-priority deviations were detected."
            )

        for index, row in enumerate(
            priority_rows,
            start=1,
        ):
            lines.extend(
                [
                    (
                        f"{index}. "
                        f"{self._friendly_feature_name(row.feature)}"
                    ),
                    (
                        f"   Priority: {row.deviation_level} "
                        f"({row.deviation_priority_score:.1f}/100)"
                    ),
                    (
                        f"   Observation: {row.coaching_message}"
                    ),
                    (
                        f"   Evidence: {row.evidence_statement}"
                    ),
                    "",
                ]
            )

        lines.extend(
            [
                "IMPORTANT LIMITATION",
                "-" * 88,
                (
                    "These observations compare this shot with the player's "
                    "own successful-shot history. They identify deviations, "
                    "not proven causes of the make or miss."
                ),
                (
                    "A coach should combine this report with video review, "
                    "ball-flight information, and the player's physical and "
                    "technical context."
                ),
            ]
        )

        self.report_file.write_text(
            "\n".join(
                lines
            ),
            encoding="utf-8",
        )

        return self.report_file

    # =====================================================
    # CHART
    # =====================================================

    def create_deviation_chart(
        self,
        deviations: list[ShotDeviation],
    ) -> Path:
        top_rows = deviations[
            :15
        ]

        labels = [
            self._friendly_feature_name(
                row.feature
            )
            for row in reversed(
                top_rows
            )
        ]

        values = [
            row.deviation_priority_score
            for row in reversed(
                top_rows
            )
        ]

        figure, axis = plt.subplots(
            figsize=(
                13,
                max(
                    7,
                    len(labels)
                    * 0.45,
                ),
            )
        )

        positions = np.arange(
            len(labels)
        )

        axis.barh(
            positions,
            values,
        )

        axis.set_yticks(
            positions,
            labels=labels,
        )

        axis.set_xlim(
            0,
            100,
        )

        axis.set_xlabel(
            "Deviation priority score"
        )

        axis.set_title(
            "Largest Deviations From Personal Successful Baseline"
        )

        axis.grid(
            visible=True,
            axis="x",
            alpha=0.25,
        )

        figure.tight_layout()

        figure.savefig(
            self.chart_file,
            dpi=170,
            bbox_inches="tight",
        )

        plt.close(
            figure
        )

        return self.chart_file

    # =====================================================
    # EXPORT
    # =====================================================

    @staticmethod
    def export_dataclass_rows(
        output_file: Path,
        rows: list[object],
    ) -> Path:
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
            asdict(
                row
            )
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
    # HELPERS
    # =====================================================

    @staticmethod
    def _numeric_values(
        records: list[dict[str, str]],
        feature: str,
    ) -> list[float]:
        values: list[
            float
        ] = []

        for record in records:
            value = ImprovementRecommendationEngine._to_finite_float(
                record.get(
                    feature,
                    "",
                )
            )

            if value is not None:
                values.append(
                    value
                )

        return values

    @staticmethod
    def _to_finite_float(
        value: object,
    ) -> float | None:
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
    def _to_int(
        value: object,
        fallback: int,
    ) -> int:
        try:
            return int(
                float(
                    value
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            return fallback

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


def _run_improvement_recommendation_test() -> None:
    """
    Generate a recommendation report for the first available missed shot.
    """

    engine = ImprovementRecommendationEngine()

    summary, deviations = engine.run()

    assert summary.features_compared > 0
    assert len(
        deviations
    ) > 0

    assert engine.deviations_file.exists()
    assert engine.summary_file.exists()
    assert engine.report_file.exists()
    assert engine.chart_file.exists()

    print(
        "IMPROVEMENT RECOMMENDATION ENGINE TEST PASSED"
    )

    print(
        f"Participant: {summary.participant_id}"
    )

    print(
        f"Trial: {summary.trial_id}"
    )

    print(
        f"Recorded result: {summary.actual_result}"
    )

    print(
        "Successful baseline shots: "
        f"{summary.baseline_made_shots}"
    )

    print(
        "Features compared: "
        f"{summary.features_compared}"
    )

    print(
        "Similarity to successful baseline: "
        f"{summary.overall_similarity_to_successful_baseline:.1f}%"
    )

    print(
        "Recommendation confidence: "
        f"{summary.recommendation_confidence}"
    )

    print()

    print(
        "Shot deviations: "
        f"{engine.deviations_file.resolve()}"
    )

    print(
        "Summary: "
        f"{engine.summary_file.resolve()}"
    )

    print(
        "Text report: "
        f"{engine.report_file.resolve()}"
    )

    print(
        "Deviation chart: "
        f"{engine.chart_file.resolve()}"
    )

    print()
    print(
        "Top recommendations:"
    )

    for row in deviations[
        :5
    ]:
        print(
            f"- {row.feature}: "
            f"{row.deviation_level}, "
            f"priority "
            f"{row.deviation_priority_score:.1f}"
        )

        print(
            f"  {row.coaching_message}"
        )


if __name__ == "__main__":
    _run_improvement_recommendation_test()