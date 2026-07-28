from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import numpy as np

from tracking_app.research.session_analysis_engine import (
    SessionAnalysisEngine,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "session_coach_summary"
)


@dataclass(frozen=True)
class SessionCoachObservation:
    """
    One coach-readable observation about a practice session.
    """

    rank: int
    observation_type: str
    priority: str

    title: str
    summary: str
    evidence: str
    coaching_focus: str


@dataclass(frozen=True)
class SessionCoachSummary:
    """
    Headline coach interpretation for one practice session.
    """

    session_id: int
    participant_id: str
    player_display_name: str

    session_name: str
    session_date: str

    executive_summary: str
    primary_message: str
    secondary_message: str

    strongest_area: str
    main_review_area: str
    highest_attention_trial: str

    evidence_confidence: str
    overall_status: str

    observations_created: int


class SessionCoachSummaryEngine:
    """
    Translate session-analysis outputs into coach-readable language.

    This engine does not change raw measurements. It interprets the existing
    session summary, category consistency, shot progression, and attention
    queue.

    The output is intentionally cautious:
    - it does not claim causation;
    - it does not call one movement universally correct;
    - it identifies review priorities rather than prescribing major changes;
    - it distinguishes missing data from poor performance.
    """

    def __init__(
        self,
        output_folder: Path = DEFAULT_OUTPUT_FOLDER,
    ) -> None:
        self.output_folder = Path(
            output_folder
        )

        self.summary_file = (
            self.output_folder
            / "session_coach_summary.csv"
        )

        self.observations_file = (
            self.output_folder
            / "session_coach_observations.csv"
        )

        self.report_file = (
            self.output_folder
            / "session_coach_report.txt"
        )

    # =====================================================
    # PUBLIC WORKFLOW
    # =====================================================

    def run(
        self,
        session_id: int,
        analysis_engine: SessionAnalysisEngine | None = None,
    ) -> tuple[
        SessionCoachSummary,
        list[SessionCoachObservation],
    ]:
        engine = (
            analysis_engine
            if analysis_engine is not None
            else SessionAnalysisEngine()
        )

        (
            session_summary,
            shot_rows,
            category_summaries,
        ) = engine.run(
            session_id=session_id
        )

        observations = self.build_observations(
            session_summary=session_summary,
            shot_rows=shot_rows,
            category_summaries=category_summaries,
        )

        summary = self.build_summary(
            session_summary=session_summary,
            observations=observations,
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
            self.observations_file,
            observations,
        )

        self.export_report(
            summary=summary,
            observations=observations,
        )

        return (
            summary,
            observations,
        )

    # =====================================================
    # OBSERVATIONS
    # =====================================================

    def build_observations(
        self,
        session_summary: object,
        shot_rows: list[object],
        category_summaries: list[object],
    ) -> list[SessionCoachObservation]:
        raw_observations: list[
            tuple[
                float,
                str,
                str,
                str,
                str,
                str,
            ]
        ] = []

        # ---------------------------------------------
        # Data quality
        # ---------------------------------------------

        if session_summary.assigned_shots == 0:
            raw_observations.append(
                (
                    100.0,
                    "data",
                    "high",
                    "No shots are assigned",
                    (
                        "This session cannot be interpreted because it does "
                        "not contain any active assigned shots."
                    ),
                    "Assigned shots: 0.",
                    (
                        "Return to Sessions & Shot Library and add tracked "
                        "shots to this session."
                    ),
                )
            )

        elif session_summary.matched_shots == 0:
            raw_observations.append(
                (
                    95.0,
                    "data",
                    "high",
                    "Feature extraction is required",
                    (
                        "The session contains shot files, but none of them "
                        "currently match extracted biomechanics rows."
                    ),
                    (
                        f"Assigned shots: {session_summary.assigned_shots}; "
                        "matched shots: 0."
                    ),
                    (
                        "Run feature extraction for the session shots before "
                        "using biomechanical conclusions."
                    ),
                )
            )

        elif session_summary.unmatched_shots > 0:
            raw_observations.append(
                (
                    82.0,
                    "data",
                    "moderate",
                    "Session analysis is partially complete",
                    (
                        "Some shots are included in the session but do not yet "
                        "have matched feature data."
                    ),
                    (
                        f"Matched {session_summary.matched_shots} of "
                        f"{session_summary.assigned_shots} assigned shots."
                    ),
                    (
                        "Extract the remaining shots before treating this as a "
                        "complete practice evaluation."
                    ),
                )
            )

        # ---------------------------------------------
        # Category consistency
        # ---------------------------------------------

        if category_summaries:
            strongest = max(
                category_summaries,
                key=lambda row: (
                    row.average_consistency_score
                ),
            )

            weakest = min(
                category_summaries,
                key=lambda row: (
                    row.average_consistency_score
                ),
            )

            raw_observations.append(
                (
                    38.0,
                    "strength",
                    "low",
                    f"{strongest.category} was the most repeatable area",
                    (
                        f"{strongest.category} produced the highest category "
                        "consistency during this session."
                    ),
                    (
                        f"Category consistency: "
                        f"{strongest.average_consistency_score:.1f}/100. "
                        f"Most repeatable feature: "
                        f"{strongest.strongest_feature}."
                    ),
                    (
                        "Use this area as a stable reference when reviewing "
                        "less repeatable parts of the shot."
                    ),
                )
            )

            weakest_priority = (
                "high"
                if weakest.average_consistency_score
                < 55.0
                else (
                    "moderate"
                    if weakest.average_consistency_score
                    < 75.0
                    else "low"
                )
            )

            raw_observations.append(
                (
                    75.0
                    if weakest_priority == "high"
                    else (
                        58.0
                        if weakest_priority == "moderate"
                        else 35.0
                    ),
                    "development",
                    weakest_priority,
                    f"{weakest.category} deserves the closest review",
                    (
                        f"{weakest.category} was the least repeatable category "
                        "in this practice."
                    ),
                    (
                        f"Category consistency: "
                        f"{weakest.average_consistency_score:.1f}/100. "
                        f"Least repeatable feature: "
                        f"{weakest.weakest_feature}."
                    ),
                    (
                        "Review the highest-attention shots and compare this "
                        "category against the player's successful baseline "
                        "before changing technique."
                    ),
                )
            )

        # ---------------------------------------------
        # Results
        # ---------------------------------------------

        if (
            session_summary.made_shots
            + session_summary.missed_shots
            > 0
        ):
            if session_summary.make_percentage >= 80.0:
                result_title = "Strong session conversion"
                result_priority = "low"
                result_score = 28.0
                focus = (
                    "Preserve the session rhythm and identify which mechanics "
                    "remained most repeatable."
                )

            elif session_summary.make_percentage >= 65.0:
                result_title = "Solid session conversion"
                result_priority = "low"
                result_score = 32.0
                focus = (
                    "Use the review queue to understand what separated the "
                    "misses from the made attempts."
                )

            elif session_summary.make_percentage >= 50.0:
                result_title = "Mixed session conversion"
                result_priority = "moderate"
                result_score = 52.0
                focus = (
                    "Review whether misses clustered around one timing or "
                    "release pattern."
                )

            else:
                result_title = "Low session conversion"
                result_priority = "high"
                result_score = 70.0
                focus = (
                    "Start with the highest-attention misses and compare them "
                    "with the most stable made attempts."
                )

            raw_observations.append(
                (
                    result_score,
                    "performance",
                    result_priority,
                    result_title,
                    (
                        f"The player made "
                        f"{session_summary.make_percentage:.1f}% of classified "
                        "attempts in this session."
                    ),
                    (
                        f"{session_summary.made_shots} makes and "
                        f"{session_summary.missed_shots} misses."
                    ),
                    focus,
                )
            )

        # ---------------------------------------------
        # Shot progression
        # ---------------------------------------------

        matched_rows = [
            row
            for row in shot_rows
            if row.matched_feature_row
        ]

        if len(
            matched_rows
        ) >= 6:
            split_index = max(
                1,
                len(
                    matched_rows
                )
                // 3,
            )

            early_rows = matched_rows[
                :split_index
            ]

            late_rows = matched_rows[
                -split_index:
            ]

            early_attention = float(
                mean(
                    row.shot_attention_score
                    for row in early_rows
                )
            )

            late_attention = float(
                mean(
                    row.shot_attention_score
                    for row in late_rows
                )
            )

            attention_change = (
                late_attention
                - early_attention
            )

            if attention_change >= 10.0:
                raw_observations.append(
                    (
                        72.0,
                        "progression",
                        "high",
                        "Review the end of the session",
                        (
                            "Shot-review priority increased during the final "
                            "portion of the practice."
                        ),
                        (
                            f"Early average attention: "
                            f"{early_attention:.1f}; late average attention: "
                            f"{late_attention:.1f}; change: "
                            f"{attention_change:+.1f}."
                        ),
                        (
                            "Check whether fatigue, pace, or a changed cue "
                            "coincided with the later attempts."
                        ),
                    )
                )

            elif attention_change >= 5.0:
                raw_observations.append(
                    (
                        56.0,
                        "progression",
                        "moderate",
                        "Session quality became less stable late",
                        (
                            "The final shots required somewhat more review "
                            "than the opening shots."
                        ),
                        (
                            f"Early average attention: "
                            f"{early_attention:.1f}; late average attention: "
                            f"{late_attention:.1f}."
                        ),
                        (
                            "Compare early and late shot rhythm before "
                            "concluding that fatigue caused the change."
                        ),
                    )
                )

            elif attention_change <= -5.0:
                raw_observations.append(
                    (
                        42.0,
                        "progression",
                        "low",
                        "Session quality improved over time",
                        (
                            "The final portion of the session required less "
                            "review than the opening portion."
                        ),
                        (
                            f"Early average attention: "
                            f"{early_attention:.1f}; late average attention: "
                            f"{late_attention:.1f}."
                        ),
                        (
                            "Identify which cue or rhythm change preceded the "
                            "more stable late attempts."
                        ),
                    )
                )

            else:
                raw_observations.append(
                    (
                        30.0,
                        "progression",
                        "low",
                        "Session quality remained stable",
                        (
                            "The average review priority was similar in the "
                            "opening and closing portions of practice."
                        ),
                        (
                            f"Early average attention: "
                            f"{early_attention:.1f}; late average attention: "
                            f"{late_attention:.1f}."
                        ),
                        (
                            "Use the individual review queue rather than "
                            "assuming a broad practice-wide decline."
                        ),
                    )
                )

        # ---------------------------------------------
        # Highest-attention shot
        # ---------------------------------------------

        if shot_rows:
            highest_attention = max(
                shot_rows,
                key=lambda row: (
                    row.shot_attention_score
                ),
            )

            raw_observations.append(
                (
                    min(
                        90.0,
                        45.0
                        + highest_attention.shot_attention_score
                        * 0.45,
                    ),
                    "shot_review",
                    (
                        "high"
                        if highest_attention.shot_attention_score
                        >= 65.0
                        else "moderate"
                    ),
                    (
                        f"Start video review with "
                        f"{highest_attention.trial_id}"
                    ),
                    (
                        "This attempt has the highest review priority in the "
                        "session."
                    ),
                    (
                        f"Shot {highest_attention.shot_number}; result "
                        f"{highest_attention.result}; attention score "
                        f"{highest_attention.shot_attention_score:.1f}/100; "
                        f"status {highest_attention.shot_status}."
                    ),
                    (
                        "Open this trial first, then compare it with a stable "
                        "made attempt from the same session."
                    ),
                )
            )

        raw_observations.sort(
            key=lambda item: (
                -item[0],
                item[3],
            )
        )

        observations: list[
            SessionCoachObservation
        ] = []

        for rank, (
            _,
            observation_type,
            priority,
            title,
            summary,
            evidence,
            coaching_focus,
        ) in enumerate(
            raw_observations,
            start=1,
        ):
            observations.append(
                SessionCoachObservation(
                    rank=rank,
                    observation_type=(
                        observation_type
                    ),
                    priority=priority,
                    title=title,
                    summary=summary,
                    evidence=evidence,
                    coaching_focus=(
                        coaching_focus
                    ),
                )
            )

        return observations

    # =====================================================
    # SUMMARY
    # =====================================================

    def build_summary(
        self,
        session_summary: object,
        observations: list[
            SessionCoachObservation
        ],
    ) -> SessionCoachSummary:
        primary_message = (
            observations[
                0
            ].summary
            if observations
            else (
                "No coach interpretation was available for this session."
            )
        )

        secondary_message = (
            observations[
                1
            ].summary
            if len(
                observations
            )
            > 1
            else (
                "Add more analyzed shots to strengthen the session summary."
            )
        )

        evidence_confidence = (
            self._evidence_confidence(
                assigned_shots=(
                    session_summary.assigned_shots
                ),
                matched_shots=(
                    session_summary.matched_shots
                ),
            )
        )

        overall_status = (
            self._overall_status(
                session_summary=session_summary
            )
        )

        executive_summary = (
            f"{session_summary.player_display_name} completed "
            f"{session_summary.assigned_shots} tracked attempts in "
            f"{session_summary.session_name}, making "
            f"{session_summary.make_percentage:.1f}% of classified shots. "
            f"{session_summary.strongest_category} was the most repeatable "
            f"category, while {session_summary.weakest_category} deserves the "
            "closest review. "
            f"Begin detailed review with "
            f"{session_summary.highest_attention_trial}."
        )

        return SessionCoachSummary(
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

            executive_summary=(
                executive_summary
            ),
            primary_message=(
                primary_message
            ),
            secondary_message=(
                secondary_message
            ),

            strongest_area=str(
                session_summary.strongest_category
            ),
            main_review_area=str(
                session_summary.weakest_category
            ),
            highest_attention_trial=str(
                session_summary.highest_attention_trial
            ),

            evidence_confidence=(
                evidence_confidence
            ),
            overall_status=(
                overall_status
            ),

            observations_created=len(
                observations
            ),
        )

    @staticmethod
    def _evidence_confidence(
        assigned_shots: int,
        matched_shots: int,
    ) -> str:
        if matched_shots >= 30:
            return "strong"

        if matched_shots >= 12:
            return "moderate"

        if matched_shots >= 5:
            return "limited"

        if assigned_shots > 0:
            return "exploratory"

        return "unavailable"

    @staticmethod
    def _overall_status(
        session_summary: object,
    ) -> str:
        if session_summary.assigned_shots == 0:
            return "No session data"

        if session_summary.matched_shots == 0:
            return "Feature extraction required"

        if session_summary.unmatched_shots > 0:
            return "Partially analyzed"

        if (
            session_summary.overall_consistency_score
            is not None
            and session_summary.overall_consistency_score
            >= 80.0
        ):
            return "Consistent session"

        if (
            session_summary.overall_consistency_score
            is not None
            and session_summary.overall_consistency_score
            >= 65.0
        ):
            return "Developing consistency"

        return "High review priority"

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
        summary: SessionCoachSummary,
        observations: list[
            SessionCoachObservation
        ],
    ) -> Path:
        lines = [
            "=" * 96,
            "SESSION COACH SUMMARY",
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
            (
                f"Status: "
                f"{summary.overall_status}"
            ),
            (
                f"Evidence confidence: "
                f"{summary.evidence_confidence}"
            ),
            "",
            "EXECUTIVE SUMMARY",
            "-" * 96,
            summary.executive_summary,
            "",
            "COACH OBSERVATIONS",
            "-" * 96,
        ]

        for observation in observations:
            lines.extend(
                [
                    (
                        f"{observation.rank}. "
                        f"{observation.title}"
                    ),
                    (
                        f"   Priority: "
                        f"{observation.priority}"
                    ),
                    (
                        f"   Summary: "
                        f"{observation.summary}"
                    ),
                    (
                        f"   Evidence: "
                        f"{observation.evidence}"
                    ),
                    (
                        f"   Coaching focus: "
                        f"{observation.coaching_focus}"
                    ),
                    "",
                ]
            )

        lines.extend(
            [
                "INTERPRETATION LIMITATION",
                "-" * 96,
                (
                    "These observations are review priorities generated from "
                    "tracked session data. They should be interpreted with "
                    "video, player context, and professional coaching judgment."
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


def _run_session_coach_summary_test() -> None:
    """
    Isolated integration test using the SessionAnalysisEngine test structure.
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

        history_repository = (
            PlayerHistoryRepository(
                database_file=database_file,
                storage_folder=(
                    root
                    / "files"
                ),
            )
        )

        player = history_repository.create_or_get_player(
            participant_id="TEST_P1",
            display_name="Coach Summary Player",
        )

        uploads = []

        for trial_number in range(
            1,
            10,
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
            selected_player_id=(
                player.player_id
            ),
            enforce_selected_player=True,
        )

        session_repository = PracticeSessionRepository(
            database_file=database_file
        )

        session = session_repository.create_session(
            player_id=player.player_id,
            session_name="Coach Summary Test",
            session_date="2026-07-27",
            session_type="Free Throws",
            location="Test Gym",
        )

        shot_files = history_repository.list_shot_files(
            player_id=player.player_id
        )

        session_repository.assign_shots(
            session_id=session.session_id,
            shot_file_ids=[
                shot.shot_file_id
                for shot in shot_files
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
                10,
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
                            * 2
                        ),
                        "confidence_adjusted_score": (
                            84
                            - trial_number
                            * 2
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

        coach_engine = SessionCoachSummaryEngine(
            output_folder=(
                root
                / "coach_summary"
            )
        )

        summary, observations = coach_engine.run(
            session_id=session.session_id,
            analysis_engine=analysis_engine,
        )

        assert summary.session_id == session.session_id
        assert summary.observations_created > 0
        assert len(
            observations
        ) > 0

        assert coach_engine.summary_file.exists()
        assert coach_engine.observations_file.exists()
        assert coach_engine.report_file.exists()

        print(
            "SESSION COACH SUMMARY ENGINE TEST PASSED"
        )

        print(
            f"Player: {summary.player_display_name}"
        )

        print(
            f"Session: {summary.session_name}"
        )

        print(
            f"Overall status: {summary.overall_status}"
        )

        print(
            f"Evidence confidence: "
            f"{summary.evidence_confidence}"
        )

        print(
            f"Observations created: "
            f"{summary.observations_created}"
        )

        print(
            f"Primary message: "
            f"{summary.primary_message}"
        )

        print(
            "Coach interpretation export: PASSED"
        )


if __name__ == "__main__":
    _run_session_coach_summary_test()