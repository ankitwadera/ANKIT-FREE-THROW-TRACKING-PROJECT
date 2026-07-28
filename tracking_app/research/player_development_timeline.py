from __future__ import annotations

import csv
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from statistics import mean

import numpy as np

from tracking_app.research.feature_metadata import (
    get_feature_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "player_history.sqlite3"
)

DEFAULT_FEATURE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "all_shot_features.csv"
)

DEFAULT_OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "development_timeline"
)


@dataclass(frozen=True)
class TimelineSessionRow:
    """
    One practice session on the long-term player-development timeline.
    """

    session_id: int
    player_id: int

    participant_id: str
    player_display_name: str

    session_name: str
    session_date: str
    session_type: str

    assigned_shots: int
    matched_feature_rows: int
    made_shots: int
    missed_shots: int
    make_percentage: float

    overall_consistency_score: float | None
    timing_score: float | None
    lower_body_score: float | None
    upper_body_score: float | None
    ball_release_score: float | None

    strongest_area: str
    weakest_area: str
    data_status: str


@dataclass(frozen=True)
class TimelineFeatureRow:
    """
    One feature summary inside one practice session.
    """

    session_id: int
    participant_id: str
    session_date: str
    session_name: str

    feature: str
    display_name: str
    category: str

    valid_shots: int
    made_shots: int
    missed_shots: int

    session_mean: float
    session_standard_deviation: float
    consistency_score: float

    made_mean: float | None
    missed_mean: float | None
    made_missed_difference: float | None


@dataclass(frozen=True)
class DevelopmentTimelineSummary:
    """
    Headline summary across all available sessions for one player.
    """

    player_id: int
    participant_id: str
    player_display_name: str

    sessions_analyzed: int
    first_session_date: str | None
    most_recent_session_date: str | None

    total_assigned_shots: int
    total_matched_feature_rows: int

    first_consistency_score: float | None
    latest_consistency_score: float | None
    consistency_change: float | None

    first_make_percentage: float | None
    latest_make_percentage: float | None
    make_percentage_change: float | None

    strongest_recent_area: str
    weakest_recent_area: str

    development_direction: str
    evidence_status: str


class PlayerDevelopmentTimelineEngine:
    """
    Build a session-by-session development timeline for one player.

    The engine links:

        Player
            Practice sessions
                Assigned shot files
                    Trial IDs
                        Feature rows

    This lets the platform answer:

    - How did the player change from one practice to the next?
    - Did overall consistency improve or regress?
    - Which category improved most?
    - Which area remains weakest?
    - Is the latest session better than the first tracked session?

    Session assignment comes from player_history.sqlite3.
    Biomechanical features come from outputs/all_shot_features.csv.
    """

    TRACKED_FEATURES = (
        "release_angle_deg",
        "release_ball_speed_ft_s",
        "release_height_ft",
        "right_knee_range_of_motion_deg",
        "right_hip_range_of_motion_deg",
        "release_right_elbow_angle_deg",
        "knee_to_elbow_gap_ms",
        "elbow_to_release_gap_ms",
        "takeoff_to_release_ms",
    )

    def __init__(
        self,
        database_file: Path = DEFAULT_DATABASE_FILE,
        feature_file: Path = DEFAULT_FEATURE_FILE,
        output_folder: Path = DEFAULT_OUTPUT_FOLDER,
    ) -> None:
        self.database_file = Path(
            database_file
        )

        self.feature_file = Path(
            feature_file
        )

        self.output_folder = Path(
            output_folder
        )

        self.session_rows_file = (
            self.output_folder
            / "timeline_sessions.csv"
        )

        self.feature_rows_file = (
            self.output_folder
            / "timeline_features.csv"
        )

        self.summary_file = (
            self.output_folder
            / "timeline_summary.csv"
        )

        self.report_file = (
            self.output_folder
            / "development_timeline_report.txt"
        )

    # =====================================================
    # CONNECTION
    # =====================================================

    def connect(
        self,
    ) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_file
        )

        connection.row_factory = sqlite3.Row

        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        return connection

    @contextmanager
    def open_connection(
        self,
    ):
        connection = self.connect()

        try:
            yield connection

        finally:
            connection.close()

    # =====================================================
    # PUBLIC WORKFLOW
    # =====================================================

    def run(
        self,
        player_id: int,
    ) -> tuple[
        DevelopmentTimelineSummary,
        list[TimelineSessionRow],
        list[TimelineFeatureRow],
    ]:
        player = self._load_player(
            player_id
        )

        feature_records = self._load_feature_records(
            participant_id=(
                player[
                    "participant_id"
                ]
            )
        )

        sessions = self._load_player_sessions(
            player_id
        )

        session_rows: list[
            TimelineSessionRow
        ] = []

        feature_rows: list[
            TimelineFeatureRow
        ] = []

        for session in sessions:
            shot_trial_ids = self._load_session_trial_ids(
                session[
                    "session_id"
                ]
            )

            matched_records = [
                row
                for row in feature_records
                if row.get(
                    "trial_id"
                )
                in shot_trial_ids
            ]

            (
                session_row,
                session_feature_rows,
            ) = self._build_session_rows(
                player=player,
                session=session,
                assigned_shots=len(
                    shot_trial_ids
                ),
                matched_records=matched_records,
            )

            session_rows.append(
                session_row
            )

            feature_rows.extend(
                session_feature_rows
            )

        summary = self._build_summary(
            player=player,
            session_rows=session_rows,
        )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._export_rows(
            self.session_rows_file,
            session_rows,
        )

        self._export_rows(
            self.feature_rows_file,
            feature_rows,
        )

        self._export_rows(
            self.summary_file,
            [
                summary
            ],
        )

        self._export_report(
            summary=summary,
            session_rows=session_rows,
        )

        return (
            summary,
            session_rows,
            feature_rows,
        )

    # =====================================================
    # DATA LOAD
    # =====================================================

    def _load_player(
        self,
        player_id: int,
    ) -> sqlite3.Row:
        with self.open_connection() as connection:
            row = connection.execute(
                """
                SELECT
                    player_id,
                    participant_id,
                    display_name
                FROM players
                WHERE player_id = ?
                """,
                (
                    int(
                        player_id
                    ),
                ),
            ).fetchone()

        if row is None:
            raise ValueError(
                f"No player was found with player_id {player_id}."
            )

        return row

    def _load_player_sessions(
        self,
        player_id: int,
    ) -> list[sqlite3.Row]:
        with self.open_connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    session_id,
                    player_id,
                    session_name,
                    session_date,
                    session_type
                FROM practice_sessions
                WHERE
                    player_id = ?
                    AND active = 1
                ORDER BY
                    session_date,
                    created_at
                """,
                (
                    int(
                        player_id
                    ),
                ),
            ).fetchall()

        return list(
            rows
        )

    def _load_session_trial_ids(
        self,
        session_id: int,
    ) -> set[str]:
        with self.open_connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    sf.trial_id
                FROM practice_session_shots AS pss
                INNER JOIN shot_files AS sf
                    ON sf.shot_file_id = pss.shot_file_id
                WHERE
                    pss.session_id = ?
                    AND sf.active = 1
                ORDER BY
                    pss.display_order,
                    sf.trial_id
                """,
                (
                    int(
                        session_id
                    ),
                ),
            ).fetchall()

        return {
            str(
                row[
                    "trial_id"
                ]
            )
            for row in rows
        }

    def _load_feature_records(
        self,
        participant_id: str,
    ) -> list[dict[str, str]]:
        if not self.feature_file.exists():
            raise FileNotFoundError(
                f"Feature file was not found: {self.feature_file}"
            )

        with self.feature_file.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            rows = list(
                csv.DictReader(
                    file
                )
            )

        return [
            row
            for row in rows
            if (
                str(
                    row.get(
                        "participant_id",
                        "",
                    )
                ).strip()
                == participant_id
                and str(
                    row.get(
                        "analysis_status",
                        "",
                    )
                ).strip().lower()
                == "success"
            )
        ]

    # =====================================================
    # SESSION ANALYSIS
    # =====================================================

    def _build_session_rows(
        self,
        player: sqlite3.Row,
        session: sqlite3.Row,
        assigned_shots: int,
        matched_records: list[
            dict[str, str]
        ],
    ) -> tuple[
        TimelineSessionRow,
        list[TimelineFeatureRow],
    ]:
        made_shots = sum(
            self._normalize_result(
                row.get(
                    "result"
                )
            )
            == "made"
            for row in matched_records
        )

        missed_shots = sum(
            self._normalize_result(
                row.get(
                    "result"
                )
            )
            == "missed"
            for row in matched_records
        )

        classified_shots = (
            made_shots
            + missed_shots
        )

        make_percentage = (
            made_shots
            / classified_shots
            * 100.0
            if classified_shots
            else 0.0
        )

        feature_rows: list[
            TimelineFeatureRow
        ] = []

        category_scores: dict[
            str,
            list[float],
        ] = {}

        for feature in self.TRACKED_FEATURES:
            values_with_results = [
                (
                    self._to_float(
                        row.get(
                            feature
                        )
                    ),
                    self._normalize_result(
                        row.get(
                            "result"
                        )
                    ),
                )
                for row in matched_records
            ]

            values_with_results = [
                (
                    value,
                    result,
                )
                for value, result
                in values_with_results
                if value is not None
            ]

            if len(
                values_with_results
            ) < 2:
                continue

            values = [
                value
                for value, _
                in values_with_results
            ]

            made_values = [
                value
                for value, result
                in values_with_results
                if result == "made"
            ]

            missed_values = [
                value
                for value, result
                in values_with_results
                if result == "missed"
            ]

            session_mean = float(
                mean(
                    values
                )
            )

            session_sd = (
                float(
                    np.std(
                        values,
                        ddof=0,
                    )
                )
                if len(
                    values
                )
                > 1
                else 0.0
            )

            consistency_score = self._consistency_score(
                session_mean=session_mean,
                session_sd=session_sd,
            )

            metadata = get_feature_metadata(
                feature
            )

            category_scores.setdefault(
                metadata.category,
                [],
            ).append(
                consistency_score
            )

            made_mean = (
                float(
                    mean(
                        made_values
                    )
                )
                if made_values
                else None
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

            made_missed_difference = (
                None
                if (
                    made_mean is None
                    or missed_mean is None
                )
                else made_mean
                - missed_mean
            )

            feature_rows.append(
                TimelineFeatureRow(
                    session_id=int(
                        session[
                            "session_id"
                        ]
                    ),
                    participant_id=str(
                        player[
                            "participant_id"
                        ]
                    ),
                    session_date=str(
                        session[
                            "session_date"
                        ]
                    ),
                    session_name=str(
                        session[
                            "session_name"
                        ]
                    ),

                    feature=feature,
                    display_name=(
                        metadata.display_name
                    ),
                    category=(
                        metadata.category
                    ),

                    valid_shots=len(
                        values
                    ),
                    made_shots=len(
                        made_values
                    ),
                    missed_shots=len(
                        missed_values
                    ),

                    session_mean=(
                        session_mean
                    ),
                    session_standard_deviation=(
                        session_sd
                    ),
                    consistency_score=(
                        consistency_score
                    ),

                    made_mean=(
                        made_mean
                    ),
                    missed_mean=(
                        missed_mean
                    ),
                    made_missed_difference=(
                        made_missed_difference
                    ),
                )
            )

        category_means = {
            category: float(
                mean(
                    values
                )
            )
            for category, values
            in category_scores.items()
            if values
        }

        all_consistency_scores = [
            row.consistency_score
            for row in feature_rows
        ]

        overall_consistency_score = (
            float(
                mean(
                    all_consistency_scores
                )
            )
            if all_consistency_scores
            else None
        )

        strongest_area = (
            max(
                category_means,
                key=category_means.get,
            )
            if category_means
            else "N/A"
        )

        weakest_area = (
            min(
                category_means,
                key=category_means.get,
            )
            if category_means
            else "N/A"
        )

        if assigned_shots == 0:
            data_status = "No shots assigned"

        elif not matched_records:
            data_status = "No matched feature rows"

        elif len(
            matched_records
        ) < assigned_shots:
            data_status = "Partially matched"

        elif len(
            feature_rows
        ) < 4:
            data_status = "Limited biomechanics"

        else:
            data_status = "Ready"

        session_row = TimelineSessionRow(
            session_id=int(
                session[
                    "session_id"
                ]
            ),
            player_id=int(
                player[
                    "player_id"
                ]
            ),

            participant_id=str(
                player[
                    "participant_id"
                ]
            ),
            player_display_name=str(
                player[
                    "display_name"
                ]
            ),

            session_name=str(
                session[
                    "session_name"
                ]
            ),
            session_date=str(
                session[
                    "session_date"
                ]
            ),
            session_type=str(
                session[
                    "session_type"
                ]
            ),

            assigned_shots=int(
                assigned_shots
            ),
            matched_feature_rows=len(
                matched_records
            ),
            made_shots=int(
                made_shots
            ),
            missed_shots=int(
                missed_shots
            ),
            make_percentage=float(
                make_percentage
            ),

            overall_consistency_score=(
                overall_consistency_score
            ),
            timing_score=category_means.get(
                "Timing & Coordination"
            ),
            lower_body_score=category_means.get(
                "Lower Body"
            ),
            upper_body_score=category_means.get(
                "Upper Body"
            ),
            ball_release_score=category_means.get(
                "Ball & Release"
            ),

            strongest_area=(
                strongest_area
            ),
            weakest_area=(
                weakest_area
            ),
            data_status=(
                data_status
            ),
        )

        return (
            session_row,
            feature_rows,
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    def _build_summary(
        self,
        player: sqlite3.Row,
        session_rows: list[
            TimelineSessionRow
        ],
    ) -> DevelopmentTimelineSummary:
        analyzable_sessions = [
            row
            for row in session_rows
            if row.overall_consistency_score
            is not None
        ]

        first_session = (
            analyzable_sessions[0]
            if analyzable_sessions
            else None
        )

        latest_session = (
            analyzable_sessions[-1]
            if analyzable_sessions
            else None
        )

        first_consistency_score = (
            None
            if first_session is None
            else (
                first_session
                .overall_consistency_score
            )
        )

        latest_consistency_score = (
            None
            if latest_session is None
            else (
                latest_session
                .overall_consistency_score
            )
        )

        consistency_change = (
            None
            if (
                first_consistency_score is None
                or latest_consistency_score is None
            )
            else (
                latest_consistency_score
                - first_consistency_score
            )
        )

        first_make_percentage = (
            None
            if first_session is None
            else (
                first_session
                .make_percentage
            )
        )

        latest_make_percentage = (
            None
            if latest_session is None
            else (
                latest_session
                .make_percentage
            )
        )

        make_percentage_change = (
            None
            if (
                first_make_percentage is None
                or latest_make_percentage is None
            )
            else (
                latest_make_percentage
                - first_make_percentage
            )
        )

        development_direction = (
            self._development_direction(
                consistency_change
            )
        )

        if len(
            analyzable_sessions
        ) >= 4:
            evidence_status = (
                "Established multi-session history"
            )

        elif len(
            analyzable_sessions
        ) >= 2:
            evidence_status = (
                "Developing multi-session history"
            )

        elif len(
            analyzable_sessions
        ) == 1:
            evidence_status = (
                "Single analyzable session"
            )

        else:
            evidence_status = (
                "No analyzable sessions"
            )

        return DevelopmentTimelineSummary(
            player_id=int(
                player[
                    "player_id"
                ]
            ),
            participant_id=str(
                player[
                    "participant_id"
                ]
            ),
            player_display_name=str(
                player[
                    "display_name"
                ]
            ),

            sessions_analyzed=len(
                analyzable_sessions
            ),
            first_session_date=(
                None
                if not session_rows
                else session_rows[
                    0
                ].session_date
            ),
            most_recent_session_date=(
                None
                if not session_rows
                else session_rows[
                    -1
                ].session_date
            ),

            total_assigned_shots=sum(
                row.assigned_shots
                for row in session_rows
            ),
            total_matched_feature_rows=sum(
                row.matched_feature_rows
                for row in session_rows
            ),

            first_consistency_score=(
                first_consistency_score
            ),
            latest_consistency_score=(
                latest_consistency_score
            ),
            consistency_change=(
                consistency_change
            ),

            first_make_percentage=(
                first_make_percentage
            ),
            latest_make_percentage=(
                latest_make_percentage
            ),
            make_percentage_change=(
                make_percentage_change
            ),

            strongest_recent_area=(
                "N/A"
                if latest_session is None
                else latest_session.strongest_area
            ),
            weakest_recent_area=(
                "N/A"
                if latest_session is None
                else latest_session.weakest_area
            ),

            development_direction=(
                development_direction
            ),
            evidence_status=(
                evidence_status
            ),
        )

    @staticmethod
    def _development_direction(
        consistency_change: float | None,
    ) -> str:
        if consistency_change is None:
            return "Unavailable"

        if consistency_change >= 5:
            return "Improving"

        if consistency_change <= -5:
            return "Regressing"

        return "Stable"

    # =====================================================
    # EXPORT
    # =====================================================

    @staticmethod
    def _export_rows(
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

    def _export_report(
        self,
        summary: DevelopmentTimelineSummary,
        session_rows: list[
            TimelineSessionRow
        ],
    ) -> Path:
        lines = [
            "=" * 96,
            "PLAYER DEVELOPMENT TIMELINE",
            "=" * 96,
            "",
            (
                f"Player: "
                f"{summary.player_display_name} "
                f"({summary.participant_id})"
            ),
            (
                f"Sessions analyzed: "
                f"{summary.sessions_analyzed}"
            ),
            (
                f"Assigned shots: "
                f"{summary.total_assigned_shots}"
            ),
            (
                f"Matched feature rows: "
                f"{summary.total_matched_feature_rows}"
            ),
            (
                f"Development direction: "
                f"{summary.development_direction}"
            ),
            (
                f"Evidence status: "
                f"{summary.evidence_status}"
            ),
            "",
            "FIRST-TO-LATEST CHANGE",
            "-" * 96,
            (
                "Consistency change: "
                + (
                    "N/A"
                    if summary.consistency_change
                    is None
                    else (
                        f"{summary.consistency_change:+.1f}"
                    )
                )
            ),
            (
                "Make percentage change: "
                + (
                    "N/A"
                    if summary.make_percentage_change
                    is None
                    else (
                        f"{summary.make_percentage_change:+.1f} percentage points"
                    )
                )
            ),
            (
                f"Strongest recent area: "
                f"{summary.strongest_recent_area}"
            ),
            (
                f"Weakest recent area: "
                f"{summary.weakest_recent_area}"
            ),
            "",
            "SESSION TIMELINE",
            "-" * 96,
        ]

        for row in session_rows:
            lines.extend(
                [
                    (
                        f"{row.session_date} · "
                        f"{row.session_name}"
                    ),
                    (
                        f"  Assigned shots: "
                        f"{row.assigned_shots}"
                    ),
                    (
                        f"  Matched rows: "
                        f"{row.matched_feature_rows}"
                    ),
                    (
                        f"  Make percentage: "
                        f"{row.make_percentage:.1f}%"
                    ),
                    (
                        "  Consistency: "
                        + (
                            "N/A"
                            if row.overall_consistency_score
                            is None
                            else (
                                f"{row.overall_consistency_score:.1f}/100"
                            )
                        )
                    ),
                    (
                        f"  Strongest area: "
                        f"{row.strongest_area}"
                    ),
                    (
                        f"  Weakest area: "
                        f"{row.weakest_area}"
                    ),
                    (
                        f"  Data status: "
                        f"{row.data_status}"
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

    # =====================================================
    # HELPERS
    # =====================================================

    @staticmethod
    def _to_float(
        value: object,
    ) -> float | None:
        if value is None:
            return None

        cleaned = str(
            value
        ).strip().lower()

        if cleaned in {
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
    def _normalize_result(
        value: object,
    ) -> str:
        cleaned = str(
            value
            if value is not None
            else ""
        ).strip().lower()

        if cleaned in {
            "made",
            "make",
            "successful",
            "hit",
        }:
            return "made"

        if cleaned in {
            "missed",
            "miss",
            "failed",
        }:
            return "missed"

        return "unknown"

    @staticmethod
    def _consistency_score(
        session_mean: float,
        session_sd: float,
    ) -> float:
        if abs(
            session_mean
        ) <= 1e-9:
            return 50.0

        coefficient_of_variation = abs(
            session_sd
            / session_mean
        )

        return float(
            np.clip(
                (
                    1.0
                    - min(
                        1.0,
                        coefficient_of_variation,
                    )
                )
                * 100.0,
                0.0,
                100.0,
            )
        )


def _run_player_development_timeline_test() -> None:
    """
    Isolated integration test with temporary database and feature CSV.
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
            display_name="Timeline Player",
        )

        imported_files = []

        for trial_number in range(
            1,
            7,
        ):
            payload = {
                "participant_id": "TEST_P1",
                "trial_id": (
                    f"T{trial_number:04d}"
                ),
                "result": (
                    "made"
                    if trial_number
                    in {
                        1,
                        2,
                        4,
                        5,
                        6,
                    }
                    else "missed"
                ),
                "sampling_rate": 30,
                "tracking": [
                    {
                        "frame": 0,
                        "time": 0,
                        "data": {},
                    }
                ],
            }

            imported_files.append(
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
            imported_files,
            selected_player_id=(
                player.player_id
            ),
            enforce_selected_player=True,
        )

        session_repository = (
            PracticeSessionRepository(
                database_file=database_file
            )
        )

        first_session = (
            session_repository
            .create_session(
                player_id=player.player_id,
                session_name="Session One",
                session_date="2026-07-20",
            )
        )

        second_session = (
            session_repository
            .create_session(
                player_id=player.player_id,
                session_name="Session Two",
                session_date="2026-07-27",
            )
        )

        shots = history_repository.list_shot_files(
            player_id=player.player_id
        )

        session_repository.assign_shots(
            session_id=first_session.session_id,
            shot_file_ids=[
                shot.shot_file_id
                for shot in shots[
                    :3
                ]
            ],
        )

        session_repository.assign_shots(
            session_id=second_session.session_id,
            shot_file_ids=[
                shot.shot_file_id
                for shot in shots[
                    3:
                ]
            ],
        )

        fieldnames = [
            "participant_id",
            "trial_id",
            "analysis_status",
            "result",
            *PlayerDevelopmentTimelineEngine.TRACKED_FEATURES,
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
                7,
            ):
                writer.writerow(
                    {
                        "participant_id": "TEST_P1",
                        "trial_id": (
                            f"T{trial_number:04d}"
                        ),
                        "analysis_status": "success",
                        "result": (
                            "made"
                            if trial_number
                            in {
                                1,
                                2,
                                4,
                                5,
                                6,
                            }
                            else "missed"
                        ),
                        "release_angle_deg": (
                            50
                            + trial_number
                        ),
                        "release_ball_speed_ft_s": (
                            10
                            + trial_number
                            * 0.1
                        ),
                        "release_height_ft": (
                            7.0
                            + trial_number
                            * 0.02
                        ),
                        "right_knee_range_of_motion_deg": (
                            35
                            + trial_number
                        ),
                        "right_hip_range_of_motion_deg": (
                            25
                            + trial_number
                            * 0.5
                        ),
                        "release_right_elbow_angle_deg": (
                            130
                            + trial_number
                        ),
                        "knee_to_elbow_gap_ms": (
                            100
                            - trial_number
                            * 3
                        ),
                        "elbow_to_release_gap_ms": (
                            250
                            - trial_number
                            * 2
                        ),
                        "takeoff_to_release_ms": (
                            150
                            + trial_number
                        ),
                    }
                )

        engine = PlayerDevelopmentTimelineEngine(
            database_file=database_file,
            feature_file=feature_file,
            output_folder=(
                root
                / "outputs"
            ),
        )

        summary, session_rows, feature_rows = engine.run(
            player_id=player.player_id
        )

        assert summary.sessions_analyzed == 2
        assert len(
            session_rows
        ) == 2
        assert len(
            feature_rows
        ) > 0
        assert summary.total_assigned_shots == 6
        assert summary.total_matched_feature_rows == 6

        assert engine.session_rows_file.exists()
        assert engine.feature_rows_file.exists()
        assert engine.summary_file.exists()
        assert engine.report_file.exists()

        print(
            "PLAYER DEVELOPMENT TIMELINE TEST PASSED"
        )

        print(
            f"Player: "
            f"{summary.player_display_name}"
        )

        print(
            f"Sessions analyzed: "
            f"{summary.sessions_analyzed}"
        )

        print(
            f"Assigned shots: "
            f"{summary.total_assigned_shots}"
        )

        print(
            f"Matched feature rows: "
            f"{summary.total_matched_feature_rows}"
        )

        print(
            f"Development direction: "
            f"{summary.development_direction}"
        )

        print(
            f"Evidence status: "
            f"{summary.evidence_status}"
        )

        print(
            "Session-to-feature matching: PASSED"
        )

        print(
            "Timeline exports: PASSED"
        )


if __name__ == "__main__":
    _run_player_development_timeline_test()