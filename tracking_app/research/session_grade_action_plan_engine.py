from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import numpy as np

from tracking_app.research.session_analysis_engine import (
    SessionAnalysisEngine,
)
from tracking_app.research.session_coach_summary_engine import (
    SessionCoachSummaryEngine,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "session_grade_action_plan"
)


@dataclass(frozen=True)
class SessionGradeComponent:
    """
    One component contributing to the overall session grade.
    """

    component: str
    score: float
    weight: float
    weighted_score: float
    status: str
    explanation: str


@dataclass(frozen=True)
class SessionActionItem:
    """
    One practical next action for the coach or player.
    """

    rank: int
    priority: str
    action_type: str

    title: str
    instruction: str
    reason: str

    related_trials: str
    estimated_minutes: int


@dataclass(frozen=True)
class SessionGradeSummary:
    """
    Overall session grade and coach-ready action plan headline.
    """

    session_id: int
    participant_id: str
    player_display_name: str

    session_name: str
    session_date: str

    overall_grade: float
    grade_label: str
    grade_confidence: str

    execution_score: float
    consistency_score: float
    shot_quality_score: float
    data_completeness_score: float

    primary_focus: str
    recommended_drill: str
    estimated_review_minutes: int

    action_items_created: int


class SessionGradeActionPlanEngine:
    """
    Combine session analysis and coach interpretation into:

    - one overall session grade;
    - transparent component scores;
    - a ranked coach action plan;
    - a suggested drill;
    - estimated review time.

    The overall grade is a session-review tool, not a universal free-throw
    grade or player talent rating.
    """

    COMPONENT_WEIGHTS = {
        "Execution": 0.30,
        "Consistency": 0.35,
        "Shot Quality": 0.25,
        "Data Completeness": 0.10,
    }

    def __init__(
        self,
        output_folder: Path = DEFAULT_OUTPUT_FOLDER,
    ) -> None:
        self.output_folder = Path(
            output_folder
        )

        self.summary_file = (
            self.output_folder
            / "session_grade_summary.csv"
        )

        self.components_file = (
            self.output_folder
            / "session_grade_components.csv"
        )

        self.action_plan_file = (
            self.output_folder
            / "session_action_plan.csv"
        )

        self.report_file = (
            self.output_folder
            / "session_grade_action_plan_report.txt"
        )

    # =====================================================
    # PUBLIC WORKFLOW
    # =====================================================

    def run(
        self,
        session_id: int,
        analysis_engine: SessionAnalysisEngine | None = None,
    ) -> tuple[
        SessionGradeSummary,
        list[SessionGradeComponent],
        list[SessionActionItem],
    ]:
        session_engine = (
            analysis_engine
            if analysis_engine is not None
            else SessionAnalysisEngine()
        )

        (
            session_summary,
            shot_rows,
            category_summaries,
        ) = session_engine.run(
            session_id=session_id
        )

        coach_engine = SessionCoachSummaryEngine(
            output_folder=(
                self.output_folder
                / "_coach_summary"
            )
        )

        (
            coach_summary,
            coach_observations,
        ) = coach_engine.run(
            session_id=session_id,
            analysis_engine=session_engine,
        )

        components = self.build_components(
            session_summary=session_summary,
            shot_rows=shot_rows,
            category_summaries=category_summaries,
        )

        action_items = self.build_action_plan(
            session_summary=session_summary,
            shot_rows=shot_rows,
            category_summaries=category_summaries,
            coach_summary=coach_summary,
            coach_observations=coach_observations,
        )

        overall_grade = sum(
            component.weighted_score
            for component in components
        )

        primary_focus = (
            action_items[
                0
            ].title
            if action_items
            else "Continue collecting session data"
        )

        recommended_drill = self.recommended_drill(
            weakest_category=(
                session_summary.weakest_category
            ),
            category_summaries=(
                category_summaries
            ),
        )

        estimated_review_minutes = sum(
            action.estimated_minutes
            for action in action_items[
                :4
            ]
        )

        summary = SessionGradeSummary(
            session_id=int(
                session_summary.session_id
            ),
            participant_id=str(
                session_summary.participant_id
            ),
            player_display_name=str(
                session_summary.player_display_name
            ),

            session_name=str(
                session_summary.session_name
            ),
            session_date=str(
                session_summary.session_date
            ),

            overall_grade=float(
                overall_grade
            ),
            grade_label=self.grade_label(
                overall_grade
            ),
            grade_confidence=self.grade_confidence(
                assigned_shots=(
                    session_summary.assigned_shots
                ),
                matched_shots=(
                    session_summary.matched_shots
                ),
            ),

            execution_score=self.component_score(
                components,
                "Execution",
            ),
            consistency_score=self.component_score(
                components,
                "Consistency",
            ),
            shot_quality_score=self.component_score(
                components,
                "Shot Quality",
            ),
            data_completeness_score=self.component_score(
                components,
                "Data Completeness",
            ),

            primary_focus=(
                primary_focus
            ),
            recommended_drill=(
                recommended_drill
            ),
            estimated_review_minutes=int(
                estimated_review_minutes
            ),

            action_items_created=len(
                action_items
            ),
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
            self.components_file,
            components,
        )

        self.export_rows(
            self.action_plan_file,
            action_items,
        )

        self.export_report(
            summary=summary,
            components=components,
            action_items=action_items,
        )

        return (
            summary,
            components,
            action_items,
        )

    # =====================================================
    # GRADE COMPONENTS
    # =====================================================

    def build_components(
        self,
        session_summary: object,
        shot_rows: list[object],
        category_summaries: list[object],
    ) -> list[SessionGradeComponent]:
        execution_score = float(
            np.clip(
                session_summary.make_percentage,
                0.0,
                100.0,
            )
        )

        consistency_score = (
            float(
                session_summary.overall_consistency_score
            )
            if session_summary.overall_consistency_score
            is not None
            else 0.0
        )

        similarity_scores = [
            row.overall_similarity_score
            for row in shot_rows
            if row.overall_similarity_score
            is not None
        ]

        adjusted_scores = [
            row.confidence_adjusted_score
            for row in shot_rows
            if row.confidence_adjusted_score
            is not None
        ]

        if similarity_scores:
            shot_quality_score = float(
                mean(
                    similarity_scores
                )
            )

        elif adjusted_scores:
            shot_quality_score = float(
                mean(
                    adjusted_scores
                )
            )

        elif category_summaries:
            shot_quality_score = float(
                mean(
                    row.average_consistency_score
                    for row in category_summaries
                )
            )

        else:
            shot_quality_score = 0.0

        data_completeness_score = (
            session_summary.matched_shots
            / session_summary.assigned_shots
            * 100.0
            if session_summary.assigned_shots
            else 0.0
        )

        raw_components = [
            (
                "Execution",
                execution_score,
                (
                    f"The player made "
                    f"{session_summary.make_percentage:.1f}% of classified "
                    "attempts."
                ),
            ),
            (
                "Consistency",
                consistency_score,
                (
                    "This reflects how repeatable the tracked biomechanical "
                    "categories were during the practice."
                ),
            ),
            (
                "Shot Quality",
                shot_quality_score,
                (
                    "This uses available similarity scores and falls back to "
                    "category repeatability when shot scores are unavailable."
                ),
            ),
            (
                "Data Completeness",
                data_completeness_score,
                (
                    f"{session_summary.matched_shots} of "
                    f"{session_summary.assigned_shots} assigned shots had "
                    "matched feature rows."
                ),
            ),
        ]

        components = []

        for component_name, score, explanation in raw_components:
            weight = self.COMPONENT_WEIGHTS[
                component_name
            ]

            components.append(
                SessionGradeComponent(
                    component=component_name,
                    score=float(
                        np.clip(
                            score,
                            0.0,
                            100.0,
                        )
                    ),
                    weight=float(
                        weight
                    ),
                    weighted_score=float(
                        np.clip(
                            score,
                            0.0,
                            100.0,
                        )
                        * weight
                    ),
                    status=self.component_status(
                        score
                    ),
                    explanation=explanation,
                )
            )

        return components

    # =====================================================
    # ACTION PLAN
    # =====================================================

    def build_action_plan(
        self,
        session_summary: object,
        shot_rows: list[object],
        category_summaries: list[object],
        coach_summary: object,
        coach_observations: list[object],
    ) -> list[SessionActionItem]:
        candidate_actions: list[
            tuple[
                float,
                str,
                str,
                str,
                str,
                str,
                int,
            ]
        ] = []

        ordered_shots = sorted(
            shot_rows,
            key=lambda row: (
                -row.shot_attention_score,
                row.shot_number,
            )
        )

        high_attention_trials = [
            row.trial_id
            for row in ordered_shots
            if row.shot_attention_score
            >= 55.0
        ][
            :3
        ]

        if high_attention_trials:
            candidate_actions.append(
                (
                    90.0,
                    "high",
                    "video_review",
                    "Review the highest-attention attempts",
                    (
                        "Open the listed trials and compare them with a stable "
                        "made attempt from the same session."
                    ),
                    ", ".join(
                        high_attention_trials
                    ),
                    6,
                )
            )

        if session_summary.weakest_category == "Timing & Coordination":
            candidate_actions.append(
                (
                    86.0,
                    "high",
                    "drill",
                    "Run a tempo free-throw block",
                    (
                        "Use a consistent verbal count from dip to release for "
                        "15 to 20 attempts. Preserve the player's normal rhythm "
                        "rather than intentionally slowing the shot."
                    ),
                    "",
                    10,
                )
            )

        elif session_summary.weakest_category == "Lower Body":
            candidate_actions.append(
                (
                    84.0,
                    "high",
                    "drill",
                    "Reinforce lower-body sequencing",
                    (
                        "Complete 12 controlled free throws emphasizing a "
                        "repeatable dip, balanced takeoff, and consistent lower-"
                        "body extension."
                    ),
                    "",
                    8,
                )
            )

        elif session_summary.weakest_category == "Upper Body":
            candidate_actions.append(
                (
                    84.0,
                    "high",
                    "drill",
                    "Stabilize shooting-arm extension",
                    (
                        "Complete 12 close-range rhythm shots before returning "
                        "to the line. Focus on repeating the player's normal "
                        "elbow and wrist extension."
                    ),
                    "",
                    8,
                )
            )

        elif session_summary.weakest_category == "Ball & Release":
            candidate_actions.append(
                (
                    84.0,
                    "high",
                    "drill",
                    "Re-establish the release window",
                    (
                        "Use 12 controlled attempts emphasizing a repeatable "
                        "release point, trajectory, and ball speed."
                    ),
                    "",
                    8,
                )
            )

        if (
            session_summary.matched_shots
            < session_summary.assigned_shots
        ):
            candidate_actions.append(
                (
                    82.0,
                    "high",
                    "data",
                    "Complete feature extraction",
                    (
                        "Process the unmatched shots before treating this as a "
                        "complete practice evaluation."
                    ),
                    "",
                    4,
                )
            )

        if len(
            shot_rows
        ) >= 6:
            split = max(
                1,
                len(
                    shot_rows
                )
                // 3,
            )

            early_attention = mean(
                row.shot_attention_score
                for row in shot_rows[
                    :split
                ]
            )

            late_attention = mean(
                row.shot_attention_score
                for row in shot_rows[
                    -split:
                ]
            )

            if late_attention - early_attention >= 5.0:
                candidate_actions.append(
                    (
                        76.0,
                        "moderate",
                        "fatigue_review",
                        "Review the final third of the workout",
                        (
                            "Compare late attempts with early attempts to check "
                            "whether fatigue, pace, or a changed cue coincided "
                            "with the higher review priority."
                        ),
                        ", ".join(
                            row.trial_id
                            for row in shot_rows[
                                -split:
                            ]
                        ),
                        5,
                    )
                )

        if session_summary.make_percentage < 60.0:
            candidate_actions.append(
                (
                    72.0,
                    "moderate",
                    "retest",
                    "Retest after the targeted drill",
                    (
                        "Finish with a 10-shot free-throw retest and compare "
                        "make percentage and consistency with this session."
                    ),
                    "",
                    7,
                )
            )

        elif session_summary.make_percentage >= 80.0:
            candidate_actions.append(
                (
                    48.0,
                    "low",
                    "preserve",
                    "Preserve the successful session rhythm",
                    (
                        "Record the cue, pace, and setup used in this practice "
                        "so the player can reproduce the same conditions."
                    ),
                    "",
                    3,
                )
            )

        if category_summaries:
            strongest = max(
                category_summaries,
                key=lambda row: (
                    row.average_consistency_score
                ),
            )

            candidate_actions.append(
                (
                    44.0,
                    "low",
                    "reference",
                    (
                        f"Use {strongest.category} as the stable reference"
                    ),
                    (
                        f"{strongest.category} was the most repeatable area. "
                        "Use it as the comparison anchor when reviewing the "
                        "weaker category."
                    ),
                    "",
                    3,
                )
            )

        candidate_actions.sort(
            key=lambda item: (
                -item[
                    0
                ],
                item[
                    3
                ],
            )
        )

        action_items = []

        for rank, (
            _,
            priority,
            action_type,
            title,
            instruction,
            related_trials,
            estimated_minutes,
        ) in enumerate(
            candidate_actions[
                :6
            ],
            start=1,
        ):
            action_items.append(
                SessionActionItem(
                    rank=rank,
                    priority=priority,
                    action_type=action_type,
                    title=title,
                    instruction=instruction,
                    reason=(
                        coach_summary.primary_message
                        if rank == 1
                        else (
                            "This action follows from the session's measured "
                            "review priorities."
                        )
                    ),
                    related_trials=related_trials,
                    estimated_minutes=int(
                        estimated_minutes
                    ),
                )
            )

        return action_items

    # =====================================================
    # HELPERS
    # =====================================================

    @staticmethod
    def recommended_drill(
        weakest_category: str,
        category_summaries: list[object],
    ) -> str:
        lookup = {
            "Timing & Coordination": "Tempo free throws",
            "Lower Body": "Controlled dip-to-extension free throws",
            "Upper Body": "Close-range shooting-arm rhythm",
            "Ball & Release": "Release-window consistency block",
        }

        return lookup.get(
            weakest_category,
            "Baseline rhythm free throws",
        )

    @staticmethod
    def grade_label(
        grade: float,
    ) -> str:
        if grade >= 90.0:
            return "Excellent Session"

        if grade >= 80.0:
            return "Strong Session"

        if grade >= 70.0:
            return "Productive Session"

        if grade >= 60.0:
            return "Mixed Session"

        return "High Review Priority"

    @staticmethod
    def grade_confidence(
        assigned_shots: int,
        matched_shots: int,
    ) -> str:
        if matched_shots >= 30:
            return "strong"

        if matched_shots >= 15:
            return "moderate"

        if matched_shots >= 8:
            return "limited"

        if assigned_shots > 0:
            return "exploratory"

        return "unavailable"

    @staticmethod
    def component_status(
        score: float,
    ) -> str:
        if score >= 85.0:
            return "Strong"

        if score >= 70.0:
            return "Solid"

        if score >= 55.0:
            return "Developing"

        return "Needs Review"

    @staticmethod
    def component_score(
        components: list[
            SessionGradeComponent
        ],
        component_name: str,
    ) -> float:
        for component in components:
            if component.component == component_name:
                return float(
                    component.score
                )

        return 0.0

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
                    dictionaries[
                        0
                    ]
                ),
            )

            writer.writeheader()

            writer.writerows(
                dictionaries
            )

        return output_file

    def export_report(
        self,
        summary: SessionGradeSummary,
        components: list[
            SessionGradeComponent
        ],
        action_items: list[
            SessionActionItem
        ],
    ) -> Path:
        lines = [
            "=" * 96,
            "SESSION GRADE & COACH ACTION PLAN",
            "=" * 96,
            "",
            (
                f"Player: "
                f"{summary.player_display_name} "
                f"({summary.participant_id})"
            ),
            (
                f"Session: "
                f"{summary.session_name}"
            ),
            (
                f"Date: "
                f"{summary.session_date}"
            ),
            "",
            (
                f"Overall grade: "
                f"{summary.overall_grade:.1f}/100"
            ),
            (
                f"Grade label: "
                f"{summary.grade_label}"
            ),
            (
                f"Confidence: "
                f"{summary.grade_confidence}"
            ),
            (
                f"Primary focus: "
                f"{summary.primary_focus}"
            ),
            (
                f"Recommended drill: "
                f"{summary.recommended_drill}"
            ),
            (
                f"Estimated review time: "
                f"{summary.estimated_review_minutes} minutes"
            ),
            "",
            "GRADE COMPONENTS",
            "-" * 96,
        ]

        for component in components:
            lines.append(
                (
                    f"{component.component}: "
                    f"{component.score:.1f}/100 · "
                    f"weight {component.weight:.0%} · "
                    f"{component.status}"
                )
            )

        lines.extend(
            [
                "",
                "COACH ACTION PLAN",
                "-" * 96,
            ]
        )

        for action in action_items:
            lines.extend(
                [
                    (
                        f"{action.rank}. "
                        f"{action.title}"
                    ),
                    (
                        f"   Priority: "
                        f"{action.priority}"
                    ),
                    (
                        f"   Instruction: "
                        f"{action.instruction}"
                    ),
                    (
                        f"   Related trials: "
                        f"{action.related_trials or 'N/A'}"
                    ),
                    (
                        f"   Estimated time: "
                        f"{action.estimated_minutes} minutes"
                    ),
                    "",
                ]
            )

        lines.extend(
            [
                "LIMITATION",
                "-" * 96,
                (
                    "The session grade is a transparent review summary based "
                    "on the available tracked session data. It is not a talent "
                    "rating or universal free-throw grade."
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


def _run_session_grade_action_plan_test() -> None:
    """
    Isolated integration test.
    """

    import json
    import tempfile

    from tracking_app.research.player_history_repository import (
        PlayerHistoryRepository,
    )
    from tracking_app.research.practice_session_repository import (
        PracticeSessionRepository,
    )

    with tempfile.TemporaryDirectory() as temporary_folder:
        root = Path(
            temporary_folder
        )

        database_file = (
            root
            / "history.sqlite3"
        )

        feature_file = (
            root
            / "features.csv"
        )

        history_repository = PlayerHistoryRepository(
            database_file=database_file,
            storage_folder=(
                root
                / "files"
            ),
        )

        player = history_repository.create_or_get_player(
            participant_id="TEST_P1",
            display_name="Grade Test Player",
        )

        uploads = []

        for trial_number in range(
            1,
            11,
        ):
            result = (
                "made"
                if trial_number
                in {
                    1,
                    2,
                    4,
                    5,
                    7,
                    8,
                }
                else "missed"
            )

            payload = {
                "participant_id": "TEST_P1",
                "trial_id": (
                    f"T{trial_number:04d}"
                ),
                "result": result,
                "sampling_rate": 30,
                "tracking": [
                    {
                        "frame": 0,
                        "time": 0,
                        "data": {},
                    }
                ],
            }

            uploads.append(
                (
                    f"trial_{trial_number}.json",
                    json.dumps(
                        payload
                    ).encode(
                        "utf-8"
                    ),
                )
            )

        history_repository.import_json_files(
            uploads,
            selected_player_id=player.player_id,
            enforce_selected_player=True,
        )

        session_repository = PracticeSessionRepository(
            database_file=database_file
        )

        session = session_repository.create_session(
            player_id=player.player_id,
            session_name="Grade Test Session",
            session_date="2026-07-27",
            session_type="Free Throws",
        )

        shots = history_repository.list_shot_files(
            player_id=player.player_id
        )

        session_repository.assign_shots(
            session_id=session.session_id,
            shot_file_ids=[
                shot.shot_file_id
                for shot in shots
            ],
        )

        fieldnames = [
            "participant_id",
            "trial_id",
            "analysis_status",
            "result",
            "overall_similarity_score",
            "confidence_adjusted_score",
            *SessionAnalysisEngine.TRACKED_FEATURES,
        ]

        with feature_file.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for trial_number in range(
                1,
                11,
            ):
                result = (
                    "made"
                    if trial_number
                    in {
                        1,
                        2,
                        4,
                        5,
                        7,
                        8,
                    }
                    else "missed"
                )

                writer.writerow(
                    {
                        "participant_id": "TEST_P1",
                        "trial_id": (
                            f"T{trial_number:04d}"
                        ),
                        "analysis_status": "success",
                        "result": result,
                        "overall_similarity_score": (
                            88
                            - trial_number
                            * 1.8
                        ),
                        "confidence_adjusted_score": (
                            84
                            - trial_number
                            * 1.7
                        ),
                        "release_angle_deg": (
                            51
                            + trial_number
                            * 0.4
                        ),
                        "release_ball_speed_ft_s": (
                            10.5
                            + trial_number
                            * 0.08
                        ),
                        "release_height_ft": (
                            7.1
                            + trial_number
                            * 0.01
                        ),
                        "right_knee_range_of_motion_deg": (
                            36
                            + trial_number
                            * 0.7
                        ),
                        "right_hip_range_of_motion_deg": (
                            26
                            + trial_number
                            * 0.6
                        ),
                        "release_right_elbow_angle_deg": (
                            132
                            + trial_number
                            * 0.5
                        ),
                        "knee_to_elbow_gap_ms": (
                            102
                            - trial_number
                            * 2
                        ),
                        "elbow_to_release_gap_ms": (
                            245
                            - trial_number
                            * 2
                        ),
                        "takeoff_to_release_ms": (
                            155
                            + trial_number
                            * 1.5
                        ),
                    }
                )

        analysis_engine = SessionAnalysisEngine(
            database_file=database_file,
            feature_file=feature_file,
            output_folder=(
                root
                / "analysis"
            ),
        )

        engine = SessionGradeActionPlanEngine(
            output_folder=(
                root
                / "grade"
            )
        )

        summary, components, actions = engine.run(
            session_id=session.session_id,
            analysis_engine=analysis_engine,
        )

        assert 0.0 <= summary.overall_grade <= 100.0
        assert len(
            components
        ) == 4
        assert len(
            actions
        ) > 0

        assert engine.summary_file.exists()
        assert engine.components_file.exists()
        assert engine.action_plan_file.exists()
        assert engine.report_file.exists()

        print(
            "SESSION GRADE & ACTION PLAN TEST PASSED"
        )

        print(
            f"Player: {summary.player_display_name}"
        )

        print(
            f"Session: {summary.session_name}"
        )

        print(
            f"Overall grade: "
            f"{summary.overall_grade:.1f}/100"
        )

        print(
            f"Grade label: "
            f"{summary.grade_label}"
        )

        print(
            f"Grade confidence: "
            f"{summary.grade_confidence}"
        )

        print(
            f"Primary focus: "
            f"{summary.primary_focus}"
        )

        print(
            f"Recommended drill: "
            f"{summary.recommended_drill}"
        )

        print(
            f"Action items created: "
            f"{summary.action_items_created}"
        )

        print(
            "Grade component exports: PASSED"
        )

        print(
            "Coach action-plan exports: PASSED"
        )


if __name__ == "__main__":
    _run_session_grade_action_plan_test()
