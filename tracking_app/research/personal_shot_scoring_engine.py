from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, pstdev

import matplotlib.pyplot as plt
import numpy as np

from tracking_app.research.feature_metadata import (
    describe_direction,
    get_feature_metadata,
)


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
    / "shot_scoring"
)


NON_SCORING_FIELDS = {
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
class ScoredFeature:
    participant_id: str
    trial_id: str
    trial_file: str
    actual_result: str

    feature: str
    category: str

    shot_value: float
    successful_mean: float
    successful_median: float
    successful_standard_deviation: float

    signed_z_score: float | None
    absolute_z_score: float | None

    diagnostic_score: float
    direction_stability_percentage: float | None

    evidence_weight: float
    feature_similarity_score: float
    weighted_similarity_points: float

    deviation_level: str
    coach_label: str
    coach_observation: str


@dataclass(frozen=True)
class CategoryScore:
    category: str

    feature_count: int
    total_evidence_weight: float

    category_similarity_score: float
    strongest_deviation_feature: str
    strongest_deviation_z_score: float | None

    category_status: str


@dataclass(frozen=True)
class ShotScoreSummary:
    participant_id: str
    trial_id: str
    trial_file: str
    actual_result: str

    baseline_made_shots: int
    baseline_missed_shots: int

    scored_feature_count: int
    overall_similarity_score: float
    confidence_adjusted_score: float

    lower_body_score: float | None
    upper_body_score: float | None
    ball_release_score: float | None
    timing_coordination_score: float | None
    other_score: float | None

    strongest_deviation_feature: str
    strongest_deviation_category: str
    strongest_deviation_z_score: float | None

    high_deviation_count: int
    moderate_deviation_count: int
    low_deviation_count: int

    score_confidence: str
    overall_assessment: str


class PersonalShotScoringEngine:
    """
    Score how closely one shot resembles the player's own successful baseline.

    Important:
    ----------
    This is not a universal free-throw quality score.

    A score of 90 means:
        "This shot closely matched this player's previous made-shot pattern."

    A score of 50 means:
        "This shot differed substantially from this player's made-shot pattern."

    The score does not prove why a shot was made or missed.
    """

    def __init__(
        self,
        feature_file: Path = DEFAULT_FEATURE_FILE,
        diagnostic_file: Path = DEFAULT_DIAGNOSTIC_FILE,
        output_folder: Path = DEFAULT_OUTPUT_FOLDER,
        minimum_diagnostic_score: float = 45.0,
        maximum_absolute_z_score: float = 3.0,
    ) -> None:
        self.feature_file = Path(feature_file)
        self.diagnostic_file = Path(diagnostic_file)
        self.output_folder = Path(output_folder)

        self.minimum_diagnostic_score = float(
            minimum_diagnostic_score
        )

        self.maximum_absolute_z_score = max(
            1.0,
            float(maximum_absolute_z_score),
        )

        self.feature_scores_file = (
            self.output_folder
            / "scored_features.csv"
        )

        self.category_scores_file = (
            self.output_folder
            / "category_scores.csv"
        )

        self.summary_file = (
            self.output_folder
            / "shot_score_summary.csv"
        )

        self.report_file = (
            self.output_folder
            / "shot_score_report.txt"
        )

        self.chart_file = (
            self.output_folder
            / "shot_score_dashboard.png"
        )

    # =====================================================
    # PUBLIC WORKFLOW
    # =====================================================

    def run(
        self,
        participant_id: str | None = None,
        trial_id: str | None = None,
    ) -> tuple[
        ShotScoreSummary,
        list[ScoredFeature],
        list[CategoryScore],
    ]:
        feature_records = self.load_csv(
            self.feature_file
        )

        diagnostic_records = self.load_csv(
            self.diagnostic_file
        )

        successful_records = [
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

        if not successful_records:
            raise RuntimeError(
                "No successful shot-feature rows were available."
            )

        selected_shot = self.select_shot(
            records=successful_records,
            participant_id=participant_id,
            trial_id=trial_id,
        )

        selected_participant = str(
            selected_shot.get(
                "participant_id",
                "",
            )
        ).strip()

        participant_records = [
            record
            for record in successful_records
            if str(
                record.get(
                    "participant_id",
                    "",
                )
            ).strip()
            == selected_participant
        ]

        made_records = [
            record
            for record in participant_records
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
            for record in participant_records
            if self._normalize_result(
                record.get(
                    "result",
                    "",
                )
            )
            == "missed"
        ]

        if len(made_records) < 10:
            raise RuntimeError(
                f"{selected_participant} does not have enough made shots "
                "to build a reliable personal baseline."
            )

        participant_diagnostics = [
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

        scored_features = self.score_features(
            selected_shot=selected_shot,
            made_records=made_records,
            diagnostic_records=participant_diagnostics,
        )

        if not scored_features:
            raise RuntimeError(
                "No personally supported features were available for scoring."
            )

        category_scores = self.build_category_scores(
            scored_features
        )

        summary = self.build_summary(
            selected_shot=selected_shot,
            made_count=len(
                made_records
            ),
            missed_count=len(
                missed_records
            ),
            scored_features=scored_features,
            category_scores=category_scores,
        )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.export_dataclass_rows(
            self.feature_scores_file,
            scored_features,
        )

        self.export_dataclass_rows(
            self.category_scores_file,
            category_scores,
        )

        self.export_dataclass_rows(
            self.summary_file,
            [
                summary
            ],
        )

        self.export_text_report(
            summary=summary,
            scored_features=scored_features,
            category_scores=category_scores,
        )

        self.create_dashboard(
            summary=summary,
            category_scores=category_scores,
            scored_features=scored_features,
        )

        return (
            summary,
            scored_features,
            category_scores,
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
    # FEATURE SCORING
    # =====================================================

    def score_features(
        self,
        selected_shot: dict[str, str],
        made_records: list[dict[str, str]],
        diagnostic_records: list[dict[str, str]],
    ) -> list[ScoredFeature]:
        rows: list[
            ScoredFeature
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

        for diagnostic in diagnostic_records:
            feature = str(
                diagnostic.get(
                    "feature",
                    "",
                )
            ).strip()

            if (
                not feature
                or feature
                in NON_SCORING_FIELDS
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

            shot_value = self._to_finite_float(
                selected_shot.get(
                    feature,
                    "",
                )
            )

            if shot_value is None:
                continue

            made_values = self._numeric_values(
                made_records,
                feature,
            )

            if len(
                made_values
            ) < 10:
                continue

            successful_mean = float(
                mean(
                    made_values
                )
            )

            successful_median = float(
                median(
                    made_values
                )
            )

            successful_sd = (
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

            z_score = None

            if successful_sd > 1e-9:
                z_score = (
                    shot_value
                    - successful_mean
                ) / successful_sd

            absolute_z = (
                None
                if z_score is None
                else abs(
                    z_score
                )
            )

            stability = self._to_finite_float(
                diagnostic.get(
                    "bootstrap_direction_stability_percentage",
                    "",
                )
            )

            evidence_weight = self._evidence_weight(
                diagnostic_score=diagnostic_score,
                direction_stability=stability,
                made_count=len(
                    made_values
                ),
            )

            feature_similarity = self._feature_similarity_score(
                absolute_z_score=absolute_z
            )

            weighted_points = (
                feature_similarity
                * evidence_weight
            )

            deviation_level = self._deviation_level(
                feature_similarity
            )

            coach_label = self._friendly_feature_name(
                feature
            )

            coach_observation = self._coach_observation(
                feature=feature,
                shot_value=shot_value,
                baseline_mean=successful_mean,
                z_score=z_score,
            )

            rows.append(
                ScoredFeature(
                    participant_id=participant_id,
                    trial_id=trial_id,
                    trial_file=trial_file,
                    actual_result=actual_result,

                    feature=feature,
                    category=self._category_for_feature(
                        feature
                    ),

                    shot_value=float(
                        shot_value
                    ),
                    successful_mean=successful_mean,
                    successful_median=successful_median,
                    successful_standard_deviation=successful_sd,

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

                    diagnostic_score=float(
                        diagnostic_score
                    ),
                    direction_stability_percentage=(
                        stability
                    ),

                    evidence_weight=evidence_weight,
                    feature_similarity_score=(
                        feature_similarity
                    ),
                    weighted_similarity_points=(
                        weighted_points
                    ),

                    deviation_level=deviation_level,
                    coach_label=coach_label,
                    coach_observation=coach_observation,
                )
            )

        rows.sort(
            key=lambda row: (
                row.feature_similarity_score,
                -row.evidence_weight,
                row.feature,
            )
        )

        return rows

    def _feature_similarity_score(
        self,
        absolute_z_score: float | None,
    ) -> float:
        """
        Convert distance from personal made-shot average into 0-100 similarity.

        0 SD difference  -> 100
        1 SD difference  -> about 67
        2 SD difference  -> about 33
        3+ SD difference -> 0
        """

        if absolute_z_score is None:
            return 50.0

        clipped = min(
            self.maximum_absolute_z_score,
            max(
                0.0,
                absolute_z_score,
            ),
        )

        score = (
            1.0
            - clipped
            / self.maximum_absolute_z_score
        ) * 100.0

        return float(
            np.clip(
                score,
                0.0,
                100.0,
            )
        )

    @staticmethod
    def _evidence_weight(
        diagnostic_score: float,
        direction_stability: float | None,
        made_count: int,
    ) -> float:
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

        sample_component = min(
            1.0,
            made_count
            / 100.0,
        )

        return float(
            diagnostic_component
            * 0.50
            + stability_component
            * 0.30
            + sample_component
            * 0.20
        )

    @staticmethod
    def _deviation_level(
        similarity_score: float,
    ) -> str:
        if similarity_score < 45:
            return "high"

        if similarity_score < 70:
            return "moderate"

        return "low"

    # =====================================================
    # CATEGORY SCORES
    # =====================================================

    def build_category_scores(
        self,
        scored_features: list[ScoredFeature],
    ) -> list[CategoryScore]:
        categories = sorted(
            {
                row.category
                for row in scored_features
            }
        )

        output: list[
            CategoryScore
        ] = []

        for category in categories:
            category_rows = [
                row
                for row in scored_features
                if row.category
                == category
            ]

            total_weight = sum(
                row.evidence_weight
                for row in category_rows
            )

            if total_weight > 0:
                category_score = (
                    sum(
                        row.weighted_similarity_points
                        for row in category_rows
                    )
                    / total_weight
                )
            else:
                category_score = 0.0

            strongest = min(
                category_rows,
                key=lambda row: (
                    row.feature_similarity_score,
                    -row.evidence_weight,
                ),
            )

            output.append(
                CategoryScore(
                    category=category,

                    feature_count=len(
                        category_rows
                    ),
                    total_evidence_weight=float(
                        total_weight
                    ),

                    category_similarity_score=float(
                        category_score
                    ),
                    strongest_deviation_feature=(
                        strongest.feature
                    ),
                    strongest_deviation_z_score=(
                        strongest.absolute_z_score
                    ),

                    category_status=self._category_status(
                        category_score
                    ),
                )
            )

        output.sort(
            key=lambda row: (
                row.category
            )
        )

        return output

    @staticmethod
    def _category_status(
        score: float,
    ) -> str:
        if score >= 85:
            return "closely matched baseline"

        if score >= 70:
            return "generally matched baseline"

        if score >= 50:
            return "meaningful differences detected"

        return "large differences detected"

    # =====================================================
    # SUMMARY
    # =====================================================

    def build_summary(
        self,
        selected_shot: dict[str, str],
        made_count: int,
        missed_count: int,
        scored_features: list[ScoredFeature],
        category_scores: list[CategoryScore],
    ) -> ShotScoreSummary:
        total_weight = sum(
            row.evidence_weight
            for row in scored_features
        )

        overall_score = (
            sum(
                row.weighted_similarity_points
                for row in scored_features
            )
            / total_weight
            if total_weight > 0
            else 0.0
        )

        confidence_factor = self._confidence_factor(
            made_count=made_count,
            scored_features=scored_features,
        )

        confidence_adjusted_score = (
            overall_score
            * confidence_factor
            + 50.0
            * (
                1.0
                - confidence_factor
            )
        )

        category_lookup = {
            row.category:
            row.category_similarity_score
            for row in category_scores
        }

        strongest = min(
            scored_features,
            key=lambda row: (
                row.feature_similarity_score,
                -row.evidence_weight,
            ),
        )

        high_count = sum(
            row.deviation_level
            == "high"
            for row in scored_features
        )

        moderate_count = sum(
            row.deviation_level
            == "moderate"
            for row in scored_features
        )

        low_count = sum(
            row.deviation_level
            == "low"
            for row in scored_features
        )

        score_confidence = self._score_confidence(
            made_count=made_count,
            scored_features=scored_features,
        )

        assessment = self._overall_assessment(
            overall_score=overall_score,
            strongest_category=(
                strongest.category
            ),
            high_count=high_count,
            moderate_count=moderate_count,
        )

        return ShotScoreSummary(
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
            actual_result=self._normalize_result(
                selected_shot.get(
                    "result",
                    "",
                )
            ),

            baseline_made_shots=made_count,
            baseline_missed_shots=missed_count,

            scored_feature_count=len(
                scored_features
            ),
            overall_similarity_score=float(
                overall_score
            ),
            confidence_adjusted_score=float(
                confidence_adjusted_score
            ),

            lower_body_score=category_lookup.get(
                "Lower Body"
            ),
            upper_body_score=category_lookup.get(
                "Upper Body"
            ),
            ball_release_score=category_lookup.get(
                "Ball & Release"
            ),
            timing_coordination_score=category_lookup.get(
                "Timing & Coordination"
            ),
            other_score=category_lookup.get(
                "Other"
            ),

            strongest_deviation_feature=(
                strongest.feature
            ),
            strongest_deviation_category=(
                strongest.category
            ),
            strongest_deviation_z_score=(
                strongest.absolute_z_score
            ),

            high_deviation_count=high_count,
            moderate_deviation_count=(
                moderate_count
            ),
            low_deviation_count=low_count,

            score_confidence=score_confidence,
            overall_assessment=assessment,
        )

    @staticmethod
    def _confidence_factor(
        made_count: int,
        scored_features: list[ScoredFeature],
    ) -> float:
        sample_factor = min(
            1.0,
            made_count
            / 100.0,
        )

        average_evidence = (
            float(
                mean(
                    row.evidence_weight
                    for row in scored_features
                )
            )
            if scored_features
            else 0.0
        )

        return float(
            np.clip(
                sample_factor
                * 0.40
                + average_evidence
                * 0.60,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _score_confidence(
        made_count: int,
        scored_features: list[ScoredFeature],
    ) -> str:
        strong_features = sum(
            row.diagnostic_score
            >= 65
            and (
                row.direction_stability_percentage
                or 0.0
            )
            >= 75
            for row in scored_features
        )

        if (
            made_count >= 75
            and strong_features >= 5
        ):
            return "moderate"

        if (
            made_count >= 30
            and strong_features >= 2
        ):
            return "limited"

        return "exploratory"

    @staticmethod
    def _overall_assessment(
        overall_score: float,
        strongest_category: str,
        high_count: int,
        moderate_count: int,
    ) -> str:
        if overall_score >= 85:
            similarity_text = (
                "This shot closely matched the player's successful "
                "biomechanical baseline."
            )
        elif overall_score >= 70:
            similarity_text = (
                "This shot generally matched the player's successful "
                "baseline, with several measurable differences."
            )
        elif overall_score >= 50:
            similarity_text = (
                "This shot showed meaningful differences from the player's "
                "successful baseline."
            )
        else:
            similarity_text = (
                "This shot differed substantially from the player's "
                "successful baseline."
            )

        deviation_text = (
            f" The largest differences were concentrated in "
            f"{strongest_category.lower()}. "
            f"{high_count} high-priority and {moderate_count} "
            f"moderate-priority deviations were detected."
        )

        return (
            similarity_text
            + deviation_text
        )

    # =====================================================
    # LANGUAGE
    # =====================================================

    @staticmethod
    def _category_for_feature(
        feature: str,
    ) -> str:
        """
        Read the category from the central feature dictionary.

        This prevents timing variables such as takeoff_to_release_ms
        from being incorrectly classified as lower-body features.
        """

        return get_feature_metadata(
            feature
        ).category

    @staticmethod
    def _friendly_feature_name(
        feature: str,
    ) -> str:
        """
        Read the coach-facing display name from one central source.
        """

        return get_feature_metadata(
            feature
        ).display_name

    @staticmethod
    def _format_feature_value(
        value: float,
        unit: str,
    ) -> str:
        """
        Format a value with the correct unit for coach-facing language.
        """

        if unit == "degrees":
            return f"{value:.1f}°"

        if unit == "ms":
            return f"{value:.0f} ms"

        if unit == "ft/s":
            return f"{value:.2f} ft/s"

        if unit == "ft":
            return f"{value:.2f} ft"

        if unit:
            return f"{value:.2f} {unit}"

        return f"{value:.2f}"

    @staticmethod
    def _absolute_difference_text(
        current_value: float,
        baseline_value: float,
        unit: str,
    ) -> str:
        """
        Express the difference in natural basketball units first.
        """

        difference = abs(
            current_value
            - baseline_value
        )

        if unit == "degrees":
            return f"{difference:.1f}°"

        if unit == "ms":
            return f"{difference:.0f} ms"

        if unit == "ft/s":
            return f"{difference:.2f} ft/s"

        if unit == "ft":
            return f"{difference:.2f} ft"

        if unit:
            return f"{difference:.2f} {unit}"

        return f"{difference:.2f}"

    @classmethod
    def _coach_observation(
        cls,
        feature: str,
        shot_value: float,
        baseline_mean: float,
        z_score: float | None,
    ) -> str:
        """
        Translate a numeric difference into coach-readable basketball language.

        Natural units are presented before statistical standard deviations.
        """

        metadata = get_feature_metadata(
            feature
        )

        current_text = cls._format_feature_value(
            shot_value,
            metadata.unit,
        )

        baseline_text = cls._format_feature_value(
            baseline_mean,
            metadata.unit,
        )

        difference_text = cls._absolute_difference_text(
            shot_value,
            baseline_mean,
            metadata.unit,
        )

        if abs(
            shot_value
            - baseline_mean
        ) <= 1e-9:
            return (
                f"The {metadata.display_name.lower()} closely matched "
                f"the player's successful-shot average of {baseline_text}."
            )

        basketball_direction = describe_direction(
            feature_name=feature,
            current_value=shot_value,
            baseline_value=baseline_mean,
        )

        statistical_text = (
            ""
            if z_score is None
            else (
                f" Statistically, this was {abs(z_score):.2f} personal "
                "standard deviations from the made-shot baseline."
            )
        )

        return (
            f"The shot showed {basketball_direction}: "
            f"{current_text} compared with a successful-shot average of "
            f"{baseline_text}, a difference of {difference_text}."
            f"{statistical_text}"
        )

    # =====================================================
    # REPORT
    # =====================================================

    def export_text_report(
        self,
        summary: ShotScoreSummary,
        scored_features: list[ScoredFeature],
        category_scores: list[CategoryScore],
    ) -> Path:
        lines = [
            "=" * 90,
            "PERSONAL FREE-THROW SHOT SCORE",
            "=" * 90,
            "",
            "SHOT",
            "-" * 90,
            f"Participant: {summary.participant_id}",
            f"Trial: {summary.trial_id}",
            f"File: {summary.trial_file}",
            f"Recorded result: {summary.actual_result}",
            "",
            "PERSONAL SIMILARITY SCORE",
            "-" * 90,
            (
                "Overall similarity to successful baseline: "
                f"{summary.overall_similarity_score:.1f} / 100"
            ),
            (
                "Confidence-adjusted score: "
                f"{summary.confidence_adjusted_score:.1f} / 100"
            ),
            (
                "Score confidence: "
                f"{summary.score_confidence}"
            ),
            (
                "Made shots in personal baseline: "
                f"{summary.baseline_made_shots}"
            ),
            (
                "Personally supported features scored: "
                f"{summary.scored_feature_count}"
            ),
            "",
            "CATEGORY SCORES",
            "-" * 90,
        ]

        for category in category_scores:
            lines.append(
                (
                    f"{category.category}: "
                    f"{category.category_similarity_score:.1f} / 100 "
                    f"({category.category_status})"
                )
            )

        lines.extend(
            [
                "",
                "OVERALL ASSESSMENT",
                "-" * 90,
                summary.overall_assessment,
                "",
                "LARGEST DIFFERENCES",
                "-" * 90,
            ]
        )

        priority_features = scored_features[
            :8
        ]

        for index, row in enumerate(
            priority_features,
            start=1,
        ):
            lines.extend(
                [
                    (
                        f"{index}. {row.coach_label}"
                    ),
                    (
                        f"   Feature similarity: "
                        f"{row.feature_similarity_score:.1f} / 100"
                    ),
                    (
                        f"   Deviation level: "
                        f"{row.deviation_level}"
                    ),
                    (
                        f"   Observation: "
                        f"{row.coach_observation}"
                    ),
                    (
                        f"   Personal evidence: "
                        f"diagnostic score {row.diagnostic_score:.1f}, "
                        f"evidence weight {row.evidence_weight:.3f}"
                    ),
                    "",
                ]
            )

        lines.extend(
            [
                "IMPORTANT INTERPRETATION",
                "-" * 90,
                (
                    "This score measures similarity to this player's own "
                    "successful-shot history. It is not a universal mechanics "
                    "grade and does not prove why the shot was made or missed."
                ),
                (
                    "Use the score to organize video review and identify where "
                    "the shot differed from the player's normal successful "
                    "pattern."
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
    # DASHBOARD
    # =====================================================

    def create_dashboard(
        self,
        summary: ShotScoreSummary,
        category_scores: list[CategoryScore],
        scored_features: list[ScoredFeature],
    ) -> Path:
        figure = plt.figure(
            figsize=(
                14,
                11,
            )
        )

        grid = figure.add_gridspec(
            3,
            1,
            height_ratios=[
                1.0,
                1.4,
                2.2,
            ],
        )

        score_axis = figure.add_subplot(
            grid[0]
        )

        category_axis = figure.add_subplot(
            grid[1]
        )

        deviation_axis = figure.add_subplot(
            grid[2]
        )

        score_axis.barh(
            [0],
            [
                summary.overall_similarity_score
            ],
        )

        score_axis.set_xlim(
            0,
            100,
        )

        score_axis.set_yticks(
            [0],
            labels=[
                "Overall similarity"
            ],
        )

        score_axis.set_xlabel(
            "Score"
        )

        score_axis.set_title(
            (
                f"Personal Shot Score — "
                f"{summary.participant_id} "
                f"{summary.trial_id}"
            )
        )

        score_axis.text(
            summary.overall_similarity_score,
            0,
            (
                f"  "
                f"{summary.overall_similarity_score:.1f}"
            ),
            va="center",
        )

        category_labels = [
            row.category
            for row in category_scores
        ]

        category_values = [
            row.category_similarity_score
            for row in category_scores
        ]

        category_positions = np.arange(
            len(
                category_labels
            )
        )

        category_axis.barh(
            category_positions,
            category_values,
        )

        category_axis.set_yticks(
            category_positions,
            labels=category_labels,
        )

        category_axis.set_xlim(
            0,
            100,
        )

        category_axis.set_xlabel(
            "Category similarity score"
        )

        category_axis.set_title(
            "Category Scores"
        )

        top_rows = scored_features[
            :10
        ]

        feature_labels = [
            row.coach_label
            for row in reversed(
                top_rows
            )
        ]

        feature_values = [
            row.feature_similarity_score
            for row in reversed(
                top_rows
            )
        ]

        feature_positions = np.arange(
            len(
                feature_labels
            )
        )

        deviation_axis.barh(
            feature_positions,
            feature_values,
        )

        deviation_axis.set_yticks(
            feature_positions,
            labels=feature_labels,
            fontsize=8,
        )

        deviation_axis.set_xlim(
            0,
            100,
        )

        deviation_axis.set_xlabel(
            "Similarity to successful baseline"
        )

        deviation_axis.set_title(
            "Lowest-Similarity Personal Features"
        )

        score_axis.grid(
            visible=True,
            axis="x",
            alpha=0.25,
        )

        category_axis.grid(
            visible=True,
            axis="x",
            alpha=0.25,
        )

        deviation_axis.grid(
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
            value = PersonalShotScoringEngine._to_finite_float(
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


def _run_personal_shot_scoring_test() -> None:
    engine = PersonalShotScoringEngine()

    (
        summary,
        scored_features,
        category_scores,
    ) = engine.run()

    assert len(
        scored_features
    ) > 0

    assert len(
        category_scores
    ) > 0

    assert (
        0.0
        <= summary.overall_similarity_score
        <= 100.0
    )

    assert engine.feature_scores_file.exists()
    assert engine.category_scores_file.exists()
    assert engine.summary_file.exists()
    assert engine.report_file.exists()
    assert engine.chart_file.exists()

    print(
        "PERSONAL SHOT SCORING ENGINE TEST PASSED"
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
        "Overall similarity score: "
        f"{summary.overall_similarity_score:.1f} / 100"
    )

    print(
        "Confidence-adjusted score: "
        f"{summary.confidence_adjusted_score:.1f} / 100"
    )

    print(
        "Score confidence: "
        f"{summary.score_confidence}"
    )

    print(
        "Scored features: "
        f"{summary.scored_feature_count}"
    )

    print()

    print(
        "Category scores:"
    )

    for row in category_scores:
        print(
            f"- {row.category}: "
            f"{row.category_similarity_score:.1f} / 100"
        )

    print()

    print(
        "Strongest deviations:"
    )

    for row in scored_features[
        :5
    ]:
        z_text = (
            "N/A"
            if row.absolute_z_score
            is None
            else f"{row.absolute_z_score:.2f} SD"
        )

        print(
            f"- {row.coach_label}: "
            f"similarity "
            f"{row.feature_similarity_score:.1f}, "
            f"deviation {z_text}"
        )

    print()

    print(
        "Scored features: "
        f"{engine.feature_scores_file.resolve()}"
    )

    print(
        "Category scores: "
        f"{engine.category_scores_file.resolve()}"
    )

    print(
        "Summary: "
        f"{engine.summary_file.resolve()}"
    )

    print(
        "Report: "
        f"{engine.report_file.resolve()}"
    )

    print(
        "Dashboard: "
        f"{engine.chart_file.resolve()}"
    )


if __name__ == "__main__":
    _run_personal_shot_scoring_test()