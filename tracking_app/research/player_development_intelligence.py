from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import csv
import numpy as np

from tracking_app.research.feature_metadata import (
    get_feature_metadata,
)
from tracking_app.research.session_history_analyzer import (
    SessionFeatureSummary,
    SessionHistoryAnalyzer,
    SessionHistorySummary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "player_development"
)


@dataclass(frozen=True)
class DevelopmentPriority:
    """
    One long-term development priority for a participant.
    """

    rank: int

    feature: str
    display_name: str
    category: str
    pattern_group: str

    consistency_score: float
    first_to_last_change: float | None
    made_missed_difference: float | None

    development_score: float
    priority_level: str

    diagnosis: str
    why_it_matters: str
    development_focus: str
    suggested_drill: str


@dataclass(frozen=True)
class PlayerDevelopmentSummary:
    """
    Top-level development summary for one participant.
    """

    participant_id: str

    total_shots: int
    made_shots: int
    missed_shots: int
    make_percentage: float

    overall_mechanics_score: float
    consistency_score: float
    timing_score: float
    lower_body_score: float
    upper_body_score: float
    ball_release_score: float

    strongest_area: str
    primary_priority: str
    secondary_priority: str
    third_priority: str

    development_status: str
    evidence_confidence: str


class PlayerDevelopmentIntelligence:
    """
    Convert practice-history measurements into a player-development plan.

    This module analyzes the player across many tracked shots instead of
    diagnosing only one attempt.

    It creates:

    - category-level development scores;
    - long-term priorities;
    - trend-aware diagnoses;
    - coaching focuses;
    - suggested drills;
    - a player-development summary.

    Important:
    ----------
    A numerical increase is not automatically an improvement. This module
    treats low consistency and meaningful change as review signals, then uses
    feature metadata to create cautious coaching language.
    """

    def __init__(
        self,
        output_folder: Path = DEFAULT_OUTPUT_FOLDER,
        maximum_priorities: int = 5,
    ) -> None:
        self.output_folder = Path(
            output_folder
        )

        self.maximum_priorities = max(
            1,
            int(
                maximum_priorities
            ),
        )

        self.summary_file = (
            self.output_folder
            / "player_development_summary.csv"
        )

        self.priority_file = (
            self.output_folder
            / "development_priorities.csv"
        )

        self.report_file = (
            self.output_folder
            / "player_development_report.txt"
        )

    # =====================================================
    # PUBLIC WORKFLOW
    # =====================================================

    def run(
        self,
        participant_id: str = "P0001",
    ) -> tuple[
        PlayerDevelopmentSummary,
        list[DevelopmentPriority],
    ]:
        history_analyzer = (
            SessionHistoryAnalyzer()
        )

        (
            session_summary,
            _,
            feature_summaries,
        ) = history_analyzer.run(
            participant_id=participant_id,
        )

        priorities = self.build_priorities(
            feature_summaries
        )

        summary = self.build_summary(
            session_summary=session_summary,
            feature_summaries=feature_summaries,
            priorities=priorities,
        )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.export_rows(
            self.summary_file,
            [
                summary
            ],
        )

        self.export_rows(
            self.priority_file,
            priorities,
        )

        self.export_report(
            summary=summary,
            priorities=priorities,
        )

        return (
            summary,
            priorities,
        )

    # =====================================================
    # PRIORITIES
    # =====================================================

    def build_priorities(
        self,
        feature_summaries: list[
            SessionFeatureSummary
        ],
    ) -> list[DevelopmentPriority]:
        priorities: list[
            DevelopmentPriority
        ] = []

        for feature_summary in feature_summaries:
            metadata = get_feature_metadata(
                feature_summary.feature
            )

            development_score = (
                self._development_priority_score(
                    feature_summary
                )
            )

            priority_level = (
                self._priority_level(
                    development_score
                )
            )

            diagnosis = self._diagnosis(
                feature_summary,
                metadata.display_name,
            )

            why_it_matters = self._why_it_matters(
                metadata.pattern_group
            )

            development_focus = (
                self._development_focus(
                    metadata.pattern_group
                )
            )

            suggested_drill = (
                self._suggested_drill(
                    metadata.pattern_group
                )
            )

            priorities.append(
                DevelopmentPriority(
                    rank=0,

                    feature=(
                        feature_summary.feature
                    ),
                    display_name=(
                        metadata.display_name
                    ),
                    category=(
                        metadata.category
                    ),
                    pattern_group=(
                        metadata.pattern_group
                    ),

                    consistency_score=(
                        feature_summary
                        .consistency_score
                    ),
                    first_to_last_change=(
                        feature_summary
                        .first_to_last_change
                    ),
                    made_missed_difference=(
                        feature_summary
                        .made_missed_difference
                    ),

                    development_score=(
                        development_score
                    ),
                    priority_level=(
                        priority_level
                    ),

                    diagnosis=diagnosis,
                    why_it_matters=(
                        why_it_matters
                    ),
                    development_focus=(
                        development_focus
                    ),
                    suggested_drill=(
                        suggested_drill
                    ),
                )
            )

        priorities.sort(
            key=lambda row: (
                -row.development_score,
                row.consistency_score,
                row.display_name,
            )
        )

        ranked: list[
            DevelopmentPriority
        ] = []

        for rank, row in enumerate(
            priorities[
                :self.maximum_priorities
            ],
            start=1,
        ):
            ranked.append(
                DevelopmentPriority(
                    rank=rank,

                    feature=row.feature,
                    display_name=row.display_name,
                    category=row.category,
                    pattern_group=row.pattern_group,

                    consistency_score=(
                        row.consistency_score
                    ),
                    first_to_last_change=(
                        row.first_to_last_change
                    ),
                    made_missed_difference=(
                        row.made_missed_difference
                    ),

                    development_score=(
                        row.development_score
                    ),
                    priority_level=(
                        row.priority_level
                    ),

                    diagnosis=row.diagnosis,
                    why_it_matters=(
                        row.why_it_matters
                    ),
                    development_focus=(
                        row.development_focus
                    ),
                    suggested_drill=(
                        row.suggested_drill
                    ),
                )
            )

        return ranked

    @staticmethod
    def _development_priority_score(
        row: SessionFeatureSummary,
    ) -> float:
        """
        Combine inconsistency, trend size, and made/missed separation.
        """

        inconsistency_component = (
            100.0
            - row.consistency_score
        )

        trend_component = 0.0

        if (
            row.first_to_last_change
            is not None
            and row.session_standard_deviation
            > 1e-9
        ):
            normalized_trend = abs(
                row.first_to_last_change
                / row.session_standard_deviation
            )

            trend_component = min(
                100.0,
                normalized_trend
                * 40.0,
            )

        outcome_component = 0.0

        if (
            row.made_missed_difference
            is not None
            and row.session_standard_deviation
            > 1e-9
        ):
            normalized_outcome_difference = abs(
                row.made_missed_difference
                / row.session_standard_deviation
            )

            outcome_component = min(
                100.0,
                normalized_outcome_difference
                * 35.0,
            )

        score = (
            inconsistency_component
            * 0.50
            + trend_component
            * 0.30
            + outcome_component
            * 0.20
        )

        return float(
            np.clip(
                score,
                0.0,
                100.0,
            )
        )

    @staticmethod
    def _priority_level(
        score: float,
    ) -> str:
        if score >= 65:
            return "high"

        if score >= 45:
            return "moderate"

        return "low"

    @staticmethod
    def _diagnosis(
        row: SessionFeatureSummary,
        display_name: str,
    ) -> str:
        change = (
            row.first_to_last_change
        )

        if change is None:
            trend_text = (
                "did not have enough valid data for a trend estimate"
            )

        elif row.trend_direction == "increasing":
            trend_text = (
                "increased from the first ten shots to the last ten"
            )

        elif row.trend_direction == "decreasing":
            trend_text = (
                "decreased from the first ten shots to the last ten"
            )

        else:
            trend_text = (
                "remained relatively stable across the tracked history"
            )

        return (
            f"{display_name} {trend_text}. "
            f"Its consistency score was "
            f"{row.consistency_score:.1f}/100."
        )

    # =====================================================
    # COACHING LANGUAGE
    # =====================================================

    @staticmethod
    def _why_it_matters(
        pattern_group: str,
    ) -> str:
        mapping = {
            "Lower-to-Upper-Body Sequence": (
                "Timing between the legs and shooting arm influences how "
                "smoothly force transfers through the shot."
            ),

            "Lower-Body Sequence": (
                "Hip, pelvis, and knee timing can affect rhythm, balance, "
                "and how much force reaches the ball."
            ),

            "Lower-Body Extension": (
                "Hip and knee extension help create upward force and reduce "
                "the need for the shooting arm to generate everything."
            ),

            "Upper-Body Extension": (
                "Shoulder and elbow extension influence release position, "
                "release height, and the direction of force."
            ),

            "Upper-Body Release Sequence": (
                "Elbow and wrist timing influence the consistency of the "
                "final release action."
            ),

            "Wrist Action": (
                "Wrist speed and timing affect the final force and direction "
                "applied to the ball."
            ),

            "Guide-Hand Action": (
                "Guide-hand consistency helps stabilize the ball without "
                "adding unwanted sideways force."
            ),

            "Ball Launch": (
                "Release angle, speed, and height directly shape the ball's "
                "flight path."
            ),

            "Release Timing": (
                "Takeoff-to-release timing affects body position and force "
                "transfer at release."
            ),

            "Overall Rhythm": (
                "A repeatable shot tempo supports consistent sequencing from "
                "the dip through release."
            ),

            "Jump Timing": (
                "Release-to-landing timing can reflect balance, jump rhythm, "
                "and release position."
            ),
        }

        return mapping.get(
            pattern_group,
            (
                "This feature changed across the tracked history and should "
                "be reviewed alongside the player's successful-shot video."
            ),
        )

    @staticmethod
    def _development_focus(
        pattern_group: str,
    ) -> str:
        mapping = {
            "Lower-to-Upper-Body Sequence": (
                "Build a repeatable rhythm where the lower body starts and "
                "sustains the upward drive before the arm accelerates."
            ),

            "Lower-Body Sequence": (
                "Rehearse smooth hip-to-knee extension without rushing the "
                "transition into the upper body."
            ),

            "Lower-Body Extension": (
                "Improve consistency in completing the normal hip and knee "
                "extension pattern through release."
            ),

            "Upper-Body Extension": (
                "Reproduce the player's normal shoulder and elbow extension "
                "through release without forcing extra range."
            ),

            "Upper-Body Release Sequence": (
                "Build a repeatable elbow-to-wrist-to-release rhythm."
            ),

            "Wrist Action": (
                "Use a relaxed, repeatable wrist finish across every shot."
            ),

            "Guide-Hand Action": (
                "Keep the guide hand stable and remove it cleanly from the ball."
            ),

            "Ball Launch": (
                "Repeat the same release point, release angle, and smooth ball "
                "speed."
            ),

            "Release Timing": (
                "Reproduce the player's normal takeoff-to-release timing."
            ),

            "Overall Rhythm": (
                "Use a consistent tempo from motion start through release."
            ),

            "Jump Timing": (
                "Maintain the same release-to-landing balance pattern."
            ),
        }

        return mapping.get(
            pattern_group,
            (
                "Use slow-motion review to identify the most repeatable "
                "successful version of this movement."
            ),
        )

    @staticmethod
    def _suggested_drill(
        pattern_group: str,
    ) -> str:
        mapping = {
            "Lower-to-Upper-Body Sequence": (
                "Dip-drive-release rhythm drill: feel the legs initiate the "
                "upward motion before the shooting arm accelerates."
            ),

            "Lower-Body Sequence": (
                "No-ball rhythm repetitions followed by slow-tempo free throws."
            ),

            "Lower-Body Extension": (
                "Slow-tempo one-motion form shooting with emphasis on a smooth "
                "rise through the hips and knees."
            ),

            "Upper-Body Extension": (
                "Close-range form shooting with a pause at set point, then a "
                "complete but relaxed shoulder-elbow finish."
            ),

            "Upper-Body Release Sequence": (
                "One-hand form shooting emphasizing elbow extension followed "
                "by a relaxed wrist finish."
            ),

            "Wrist Action": (
                "Close-range one-hand swishes with a relaxed wrist and held "
                "follow-through."
            ),

            "Guide-Hand Action": (
                "Guide-hand stability drill from close range."
            ),

            "Ball Launch": (
                "Arc-control form shooting from close range using a consistent "
                "release point."
            ),

            "Release Timing": (
                "Tempo free throws using the same verbal count from dip to "
                "release."
            ),

            "Overall Rhythm": (
                "Five-shot tempo sets with the same pre-shot and shooting "
                "cadence."
            ),

            "Jump Timing": (
                "Balanced landing free throws with a held finish."
            ),
        }

        return mapping.get(
            pattern_group,
            (
                "Use low-speed repetitions and immediate video feedback."
            ),
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    def build_summary(
        self,
        session_summary: SessionHistorySummary,
        feature_summaries: list[
            SessionFeatureSummary
        ],
        priorities: list[
            DevelopmentPriority
        ],
    ) -> PlayerDevelopmentSummary:
        category_scores = self._category_scores(
            feature_summaries
        )

        overall_mechanics_score = (
            float(
                mean(
                    category_scores.values()
                )
            )
            if category_scores
            else 0.0
        )

        consistency_score = (
            float(
                mean(
                    row.consistency_score
                    for row in feature_summaries
                )
            )
            if feature_summaries
            else 0.0
        )

        strongest_area = (
            max(
                category_scores,
                key=category_scores.get,
            )
            if category_scores
            else "N/A"
        )

        priority_names = [
            row.display_name
            for row in priorities
        ]

        while len(
            priority_names
        ) < 3:
            priority_names.append(
                "N/A"
            )

        evidence_confidence = (
            "moderate"
            if (
                session_summary.total_shots
                >= 100
                and len(
                    feature_summaries
                )
                >= 7
            )
            else (
                "limited"
                if session_summary.total_shots
                >= 30
                else "exploratory"
            )
        )

        development_status = (
            "Established player history"
            if session_summary.total_shots
            >= 100
            else (
                "Developing player history"
                if session_summary.total_shots
                >= 30
                else "Limited player history"
            )
        )

        return PlayerDevelopmentSummary(
            participant_id=(
                session_summary
                .participant_id
            ),

            total_shots=(
                session_summary
                .total_shots
            ),
            made_shots=(
                session_summary
                .made_shots
            ),
            missed_shots=(
                session_summary
                .missed_shots
            ),
            make_percentage=(
                session_summary
                .make_percentage
            ),

            overall_mechanics_score=(
                overall_mechanics_score
            ),
            consistency_score=(
                consistency_score
            ),
            timing_score=category_scores.get(
                "Timing & Coordination",
                0.0,
            ),
            lower_body_score=category_scores.get(
                "Lower Body",
                0.0,
            ),
            upper_body_score=category_scores.get(
                "Upper Body",
                0.0,
            ),
            ball_release_score=category_scores.get(
                "Ball & Release",
                0.0,
            ),

            strongest_area=(
                strongest_area
            ),
            primary_priority=(
                priority_names[0]
            ),
            secondary_priority=(
                priority_names[1]
            ),
            third_priority=(
                priority_names[2]
            ),

            development_status=(
                development_status
            ),
            evidence_confidence=(
                evidence_confidence
            ),
        )

    @staticmethod
    def _category_scores(
        feature_summaries: list[
            SessionFeatureSummary
        ],
    ) -> dict[str, float]:
        grouped: dict[
            str,
            list[float],
        ] = {}

        for row in feature_summaries:
            category = get_feature_metadata(
                row.feature
            ).category

            grouped.setdefault(
                category,
                [],
            ).append(
                row.consistency_score
            )

        return {
            category: float(
                mean(
                    values
                )
            )
            for category, values in grouped.items()
            if values
        }

    # =====================================================
    # EXPORT
    # =====================================================

    @staticmethod
    def export_rows(
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

        with output_file.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    dictionaries[0]
                ),
            )

            writer.writeheader()

            writer.writerows(
                dictionaries
            )

        return output_file

    def export_report(
        self,
        summary: PlayerDevelopmentSummary,
        priorities: list[
            DevelopmentPriority
        ],
    ) -> Path:
        lines = [
            "=" * 92,
            "PLAYER DEVELOPMENT INTELLIGENCE",
            "=" * 92,
            "",
            f"Participant: {summary.participant_id}",
            f"Tracked shots: {summary.total_shots}",
            f"Make percentage: {summary.make_percentage:.1f}%",
            f"Development status: {summary.development_status}",
            f"Evidence confidence: {summary.evidence_confidence}",
            "",
            "DEVELOPMENT SCORES",
            "-" * 92,
            (
                "Overall mechanics consistency: "
                f"{summary.overall_mechanics_score:.1f}/100"
            ),
            (
                "Overall feature consistency: "
                f"{summary.consistency_score:.1f}/100"
            ),
            (
                "Timing and coordination: "
                f"{summary.timing_score:.1f}/100"
            ),
            (
                "Lower body: "
                f"{summary.lower_body_score:.1f}/100"
            ),
            (
                "Upper body: "
                f"{summary.upper_body_score:.1f}/100"
            ),
            (
                "Ball and release: "
                f"{summary.ball_release_score:.1f}/100"
            ),
            "",
            "TOP DEVELOPMENT PRIORITIES",
            "-" * 92,
        ]

        for priority in priorities:
            lines.extend(
                [
                    (
                        f"{priority.rank}. "
                        f"{priority.display_name}"
                    ),
                    (
                        f"   Category: "
                        f"{priority.category}"
                    ),
                    (
                        f"   Priority: "
                        f"{priority.priority_level}"
                    ),
                    (
                        f"   Diagnosis: "
                        f"{priority.diagnosis}"
                    ),
                    (
                        f"   Why it matters: "
                        f"{priority.why_it_matters}"
                    ),
                    (
                        f"   Development focus: "
                        f"{priority.development_focus}"
                    ),
                    (
                        f"   Suggested drill: "
                        f"{priority.suggested_drill}"
                    ),
                    "",
                ]
            )

        self.report_file.write_text(
            "\n".join(
                lines
            ),
            encoding="utf-8",
        )

        return self.report_file


def _run_player_development_test() -> None:
    engine = (
        PlayerDevelopmentIntelligence()
    )

    summary, priorities = engine.run(
        participant_id="P0001",
    )

    assert summary.total_shots > 0
    assert len(
        priorities
    ) > 0

    assert engine.summary_file.exists()
    assert engine.priority_file.exists()
    assert engine.report_file.exists()

    print(
        "PLAYER DEVELOPMENT INTELLIGENCE TEST PASSED"
    )

    print(
        f"Participant: {summary.participant_id}"
    )

    print(
        f"Tracked shots: {summary.total_shots}"
    )

    print(
        f"Make percentage: {summary.make_percentage:.1f}%"
    )

    print(
        "Overall mechanics score: "
        f"{summary.overall_mechanics_score:.1f} / 100"
    )

    print(
        "Consistency score: "
        f"{summary.consistency_score:.1f} / 100"
    )

    print(
        f"Strongest area: {summary.strongest_area}"
    )

    print(
        f"Evidence confidence: {summary.evidence_confidence}"
    )

    print()

    print(
        "Top development priorities:"
    )

    for priority in priorities[
        :3
    ]:
        print(
            f"- {priority.rank}. "
            f"{priority.display_name}: "
            f"{priority.development_score:.1f}, "
            f"{priority.priority_level}"
        )

    print()

    print(
        "Summary: "
        f"{engine.summary_file.resolve()}"
    )

    print(
        "Priorities: "
        f"{engine.priority_file.resolve()}"
    )

    print(
        "Report: "
        f"{engine.report_file.resolve()}"
    )


if __name__ == "__main__":
    _run_player_development_test()