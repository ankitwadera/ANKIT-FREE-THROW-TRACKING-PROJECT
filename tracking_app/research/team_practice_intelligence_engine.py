from __future__ import annotations

import csv
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from tracking_app.research.organization_team_repository import (
    OrganizationTeamRepository,
)
from tracking_app.research.session_analysis_engine import (
    SessionAnalysisEngine,
)
from tracking_app.research.session_grade_action_plan_engine import (
    SessionGradeActionPlanEngine,
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
    / "team_practice_intelligence"
)


@dataclass(frozen=True)
class TeamPracticePlayerRow:
    team_id: int
    player_id: int

    organization_name: str
    team_name: str
    season_name: str

    participant_id: str
    player_display_name: str
    jersey_number: str
    position_group: str
    roster_role: str

    latest_session_id: int | None
    latest_session_name: str
    latest_session_date: str | None

    session_available: bool
    assigned_shots: int
    make_percentage: float | None
    session_grade: float | None
    grade_label: str
    consistency_score: float | None

    primary_focus: str
    recommended_drill: str
    review_first_trial: str

    attention_score: float
    team_status: str


@dataclass(frozen=True)
class TeamPracticeSummary:
    team_id: int
    organization_name: str
    team_name: str
    season_name: str
    level: str

    active_players: int
    players_with_sessions: int
    players_with_grades: int

    total_latest_session_shots: int
    average_make_percentage: float | None
    average_session_grade: float | None
    average_consistency_score: float | None

    highest_grade_player: str
    highest_attention_player: str
    most_common_primary_focus: str

    team_status: str


class TeamPracticeIntelligenceEngine:
    """
    Build one team-wide coaching view from each active roster player's most
    recent practice session.

    This answers:
    - Who practiced most recently?
    - Who has a completed session analysis?
    - Which player needs review first?
    - Who had the strongest latest session?
    - What coaching focus is most common across the roster?
    - What is the team-wide average grade, consistency, and make percentage?

    It does not modify any roster, player, session, or shot data.
    """

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

        self.player_rows_file = (
            self.output_folder
            / "team_practice_players.csv"
        )

        self.summary_file = (
            self.output_folder
            / "team_practice_summary.csv"
        )

        self.report_file = (
            self.output_folder
            / "team_practice_report.txt"
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
        team_id: int,
    ) -> tuple[
        TeamPracticeSummary,
        list[TeamPracticePlayerRow],
    ]:
        organization_repository = (
            OrganizationTeamRepository(
                database_file=self.database_file
            )
        )

        team = organization_repository.get_team(
            team_id
        )

        roster = (
            organization_repository
            .list_team_players(
                team_id=team_id,
                include_inactive=False,
            )
        )

        player_rows = [
            self._build_player_row(
                team=team,
                roster_member=member,
            )
            for member in roster
        ]

        player_rows.sort(
            key=lambda row: (
                -row.attention_score,
                row.player_display_name,
            )
        )

        summary = self._build_summary(
            team=team,
            player_rows=player_rows,
        )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._export_rows(
            self.player_rows_file,
            player_rows,
        )

        self._export_rows(
            self.summary_file,
            [
                summary
            ],
        )

        self._export_report(
            summary=summary,
            player_rows=player_rows,
        )

        return (
            summary,
            player_rows,
        )

    # =====================================================
    # PLAYER ROW
    # =====================================================

    def _build_player_row(
        self,
        team: object,
        roster_member: object,
    ) -> TeamPracticePlayerRow:
        latest_session = self._latest_session(
            roster_member.player_id
        )

        if latest_session is None:
            return TeamPracticePlayerRow(
                team_id=int(
                    team.team_id
                ),
                player_id=int(
                    roster_member.player_id
                ),

                organization_name=str(
                    team.organization_name
                ),
                team_name=str(
                    team.team_name
                ),
                season_name=str(
                    team.season_name
                ),

                participant_id=str(
                    roster_member.participant_id
                ),
                player_display_name=str(
                    roster_member.player_display_name
                ),
                jersey_number=str(
                    roster_member.jersey_number
                ),
                position_group=str(
                    roster_member.position_group
                ),
                roster_role=str(
                    roster_member.roster_role
                ),

                latest_session_id=None,
                latest_session_name="No active session",
                latest_session_date=None,

                session_available=False,
                assigned_shots=0,
                make_percentage=None,
                session_grade=None,
                grade_label="No Session",
                consistency_score=None,

                primary_focus="Create or assign a practice session",
                recommended_drill="N/A",
                review_first_trial="N/A",

                attention_score=100.0,
                team_status="No active session",
            )

        analysis_engine = SessionAnalysisEngine(
            database_file=self.database_file,
            feature_file=self.feature_file,
            output_folder=(
                self.output_folder
                / "_session_analysis"
                / str(
                    latest_session[
                        "session_id"
                    ]
                )
            ),
        )

        grade_engine = SessionGradeActionPlanEngine(
            output_folder=(
                self.output_folder
                / "_session_grade"
                / str(
                    latest_session[
                        "session_id"
                    ]
                )
            )
        )

        try:
            (
                session_summary,
                _,
                _,
            ) = analysis_engine.run(
                session_id=int(
                    latest_session[
                        "session_id"
                    ]
                )
            )

            (
                grade_summary,
                _,
                _,
            ) = grade_engine.run(
                session_id=int(
                    latest_session[
                        "session_id"
                    ]
                ),
                analysis_engine=analysis_engine,
            )

            attention_score = self._attention_score(
                session_grade=(
                    grade_summary.overall_grade
                ),
                consistency_score=(
                    session_summary
                    .overall_consistency_score
                ),
                assigned_shots=(
                    session_summary.assigned_shots
                ),
                matched_shots=(
                    session_summary.matched_shots
                ),
            )

            team_status = self._team_status(
                session_grade=(
                    grade_summary.overall_grade
                ),
                attention_score=attention_score,
            )

            return TeamPracticePlayerRow(
                team_id=int(
                    team.team_id
                ),
                player_id=int(
                    roster_member.player_id
                ),

                organization_name=str(
                    team.organization_name
                ),
                team_name=str(
                    team.team_name
                ),
                season_name=str(
                    team.season_name
                ),

                participant_id=str(
                    roster_member.participant_id
                ),
                player_display_name=str(
                    roster_member.player_display_name
                ),
                jersey_number=str(
                    roster_member.jersey_number
                ),
                position_group=str(
                    roster_member.position_group
                ),
                roster_role=str(
                    roster_member.roster_role
                ),

                latest_session_id=int(
                    latest_session[
                        "session_id"
                    ]
                ),
                latest_session_name=str(
                    latest_session[
                        "session_name"
                    ]
                ),
                latest_session_date=str(
                    latest_session[
                        "session_date"
                    ]
                ),

                session_available=True,
                assigned_shots=int(
                    session_summary.assigned_shots
                ),
                make_percentage=float(
                    session_summary.make_percentage
                ),
                session_grade=float(
                    grade_summary.overall_grade
                ),
                grade_label=str(
                    grade_summary.grade_label
                ),
                consistency_score=(
                    None
                    if session_summary
                    .overall_consistency_score
                    is None
                    else float(
                        session_summary
                        .overall_consistency_score
                    )
                ),

                primary_focus=str(
                    grade_summary.primary_focus
                ),
                recommended_drill=str(
                    grade_summary.recommended_drill
                ),
                review_first_trial=str(
                    session_summary
                    .highest_attention_trial
                ),

                attention_score=float(
                    attention_score
                ),
                team_status=(
                    team_status
                ),
            )

        except Exception as error:
            return TeamPracticePlayerRow(
                team_id=int(
                    team.team_id
                ),
                player_id=int(
                    roster_member.player_id
                ),

                organization_name=str(
                    team.organization_name
                ),
                team_name=str(
                    team.team_name
                ),
                season_name=str(
                    team.season_name
                ),

                participant_id=str(
                    roster_member.participant_id
                ),
                player_display_name=str(
                    roster_member.player_display_name
                ),
                jersey_number=str(
                    roster_member.jersey_number
                ),
                position_group=str(
                    roster_member.position_group
                ),
                roster_role=str(
                    roster_member.roster_role
                ),

                latest_session_id=int(
                    latest_session[
                        "session_id"
                    ]
                ),
                latest_session_name=str(
                    latest_session[
                        "session_name"
                    ]
                ),
                latest_session_date=str(
                    latest_session[
                        "session_date"
                    ]
                ),

                session_available=True,
                assigned_shots=0,
                make_percentage=None,
                session_grade=None,
                grade_label="Analysis Unavailable",
                consistency_score=None,

                primary_focus=(
                    "Complete or repair session analysis"
                ),
                recommended_drill="N/A",
                review_first_trial="N/A",

                attention_score=90.0,
                team_status=(
                    f"Analysis unavailable: "
                    f"{type(error).__name__}"
                ),
            )

    def _latest_session(
        self,
        player_id: int,
    ) -> sqlite3.Row | None:
        with self.open_connection() as connection:
            return connection.execute(
                """
                SELECT
                    session_id,
                    player_id,
                    session_name,
                    session_date,
                    session_type,
                    created_at

                FROM practice_sessions

                WHERE
                    player_id = ?
                    AND active = 1

                ORDER BY
                    session_date DESC,
                    created_at DESC,
                    session_id DESC

                LIMIT 1
                """,
                (
                    int(
                        player_id
                    ),
                ),
            ).fetchone()

    # =====================================================
    # ATTENTION
    # =====================================================

    @staticmethod
    def _attention_score(
        session_grade: float,
        consistency_score: float | None,
        assigned_shots: int,
        matched_shots: int,
    ) -> float:
        grade_component = (
            100.0
            - session_grade
        )

        consistency_component = (
            50.0
            if consistency_score is None
            else (
                100.0
                - consistency_score
            )
        )

        completeness = (
            matched_shots
            / assigned_shots
            if assigned_shots
            else 0.0
        )

        completeness_component = (
            100.0
            - completeness
            * 100.0
        )

        return float(
            max(
                0.0,
                min(
                    100.0,
                    grade_component
                    * 0.55
                    + consistency_component
                    * 0.30
                    + completeness_component
                    * 0.15,
                ),
            )
        )

    @staticmethod
    def _team_status(
        session_grade: float,
        attention_score: float,
    ) -> str:
        if attention_score >= 55.0:
            return "High review priority"

        if attention_score >= 35.0:
            return "Monitor"

        if session_grade >= 85.0:
            return "Strong latest session"

        return "Stable"

    # =====================================================
    # TEAM SUMMARY
    # =====================================================

    def _build_summary(
        self,
        team: object,
        player_rows: list[
            TeamPracticePlayerRow
        ],
    ) -> TeamPracticeSummary:
        session_rows = [
            row
            for row in player_rows
            if row.session_available
        ]

        grade_rows = [
            row
            for row in player_rows
            if row.session_grade
            is not None
        ]

        make_rows = [
            row
            for row in player_rows
            if row.make_percentage
            is not None
        ]

        consistency_rows = [
            row
            for row in player_rows
            if row.consistency_score
            is not None
        ]

        highest_grade_player = (
            max(
                grade_rows,
                key=lambda row: (
                    row.session_grade
                ),
            ).player_display_name
            if grade_rows
            else "N/A"
        )

        highest_attention_player = (
            max(
                player_rows,
                key=lambda row: (
                    row.attention_score
                ),
            ).player_display_name
            if player_rows
            else "N/A"
        )

        focus_counts: dict[
            str,
            int,
        ] = {}

        for row in grade_rows:
            focus = (
                row.primary_focus.strip()
                or "Unspecified"
            )

            focus_counts[
                focus
            ] = (
                focus_counts.get(
                    focus,
                    0,
                )
                + 1
            )

        most_common_focus = (
            max(
                focus_counts,
                key=focus_counts.get,
            )
            if focus_counts
            else "N/A"
        )

        if not player_rows:
            team_status = "No active roster"

        elif not session_rows:
            team_status = "No player sessions available"

        elif len(
            grade_rows
        ) < len(
            player_rows
        ):
            team_status = "Partially analyzed"

        else:
            team_status = "Ready"

        return TeamPracticeSummary(
            team_id=int(
                team.team_id
            ),
            organization_name=str(
                team.organization_name
            ),
            team_name=str(
                team.team_name
            ),
            season_name=str(
                team.season_name
            ),
            level=str(
                team.level
            ),

            active_players=len(
                player_rows
            ),
            players_with_sessions=len(
                session_rows
            ),
            players_with_grades=len(
                grade_rows
            ),

            total_latest_session_shots=sum(
                row.assigned_shots
                for row in grade_rows
            ),
            average_make_percentage=(
                float(
                    mean(
                        row.make_percentage
                        for row in make_rows
                    )
                )
                if make_rows
                else None
            ),
            average_session_grade=(
                float(
                    mean(
                        row.session_grade
                        for row in grade_rows
                    )
                )
                if grade_rows
                else None
            ),
            average_consistency_score=(
                float(
                    mean(
                        row.consistency_score
                        for row in consistency_rows
                    )
                )
                if consistency_rows
                else None
            ),

            highest_grade_player=(
                highest_grade_player
            ),
            highest_attention_player=(
                highest_attention_player
            ),
            most_common_primary_focus=(
                most_common_focus
            ),

            team_status=(
                team_status
            ),
        )

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

    def _export_report(
        self,
        summary: TeamPracticeSummary,
        player_rows: list[
            TeamPracticePlayerRow
        ],
    ) -> Path:
        lines = [
            "=" * 96,
            "TEAM PRACTICE INTELLIGENCE",
            "=" * 96,
            "",
            (
                f"Organization: "
                f"{summary.organization_name}"
            ),
            (
                f"Team: "
                f"{summary.team_name}"
            ),
            (
                f"Season: "
                f"{summary.season_name or 'N/A'}"
            ),
            "",
            (
                f"Active players: "
                f"{summary.active_players}"
            ),
            (
                f"Players with sessions: "
                f"{summary.players_with_sessions}"
            ),
            (
                f"Players with grades: "
                f"{summary.players_with_grades}"
            ),
            (
                "Average session grade: "
                + (
                    "N/A"
                    if summary.average_session_grade
                    is None
                    else (
                        f"{summary.average_session_grade:.1f}"
                    )
                )
            ),
            (
                "Average make percentage: "
                + (
                    "N/A"
                    if summary.average_make_percentage
                    is None
                    else (
                        f"{summary.average_make_percentage:.1f}%"
                    )
                )
            ),
            (
                "Average consistency: "
                + (
                    "N/A"
                    if summary.average_consistency_score
                    is None
                    else (
                        f"{summary.average_consistency_score:.1f}"
                    )
                )
            ),
            (
                f"Highest-grade player: "
                f"{summary.highest_grade_player}"
            ),
            (
                f"Highest-attention player: "
                f"{summary.highest_attention_player}"
            ),
            (
                f"Most common focus: "
                f"{summary.most_common_primary_focus}"
            ),
            "",
            "ROSTER",
            "-" * 96,
        ]

        for row in player_rows:
            lines.extend(
                [
                    (
                        f"{row.player_display_name} "
                        f"({row.participant_id})"
                    ),
                    (
                        f"  Latest session: "
                        f"{row.latest_session_name}"
                    ),
                    (
                        f"  Grade: "
                        + (
                            "N/A"
                            if row.session_grade
                            is None
                            else (
                                f"{row.session_grade:.1f}"
                            )
                        )
                    ),
                    (
                        f"  Make %: "
                        + (
                            "N/A"
                            if row.make_percentage
                            is None
                            else (
                                f"{row.make_percentage:.1f}%"
                            )
                        )
                    ),
                    (
                        f"  Primary focus: "
                        f"{row.primary_focus}"
                    ),
                    (
                        f"  Recommended drill: "
                        f"{row.recommended_drill}"
                    ),
                    (
                        f"  Status: "
                        f"{row.team_status}"
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


def _run_team_practice_intelligence_test() -> None:
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

        history_repository = (
            PlayerHistoryRepository(
                database_file=database_file,
                storage_folder=(
                    root
                    / "files"
                ),
            )
        )

        players = []

        for player_number in range(
            1,
            3,
        ):
            player = (
                history_repository
                .create_or_get_player(
                    participant_id=(
                        f"TEST_P{player_number}"
                    ),
                    display_name=(
                        f"Team Player {player_number}"
                    ),
                )
            )

            players.append(
                player
            )

        uploads_by_player = {}

        for player in players:
            uploads = []

            for trial_number in range(
                1,
                6,
            ):
                result = (
                    "made"
                    if trial_number
                    in {
                        1,
                        2,
                        4,
                    }
                    else "missed"
                )

                payload = {
                    "participant_id": (
                        player.participant_id
                    ),
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
                        (
                            f"{player.participant_id}_"
                            f"{trial_number}.json"
                        ),
                        json.dumps(
                            payload
                        ).encode(
                            "utf-8"
                        ),
                    )
                )

            uploads_by_player[
                player.player_id
            ] = uploads

            history_repository.import_json_files(
                uploads,
                selected_player_id=player.player_id,
                enforce_selected_player=True,
            )

        session_repository = (
            PracticeSessionRepository(
                database_file=database_file
            )
        )

        sessions = []

        for index, player in enumerate(
            players,
            start=1,
        ):
            session = (
                session_repository
                .create_session(
                    player_id=player.player_id,
                    session_name=(
                        f"Latest Session {index}"
                    ),
                    session_date=(
                        f"2026-07-2{index}"
                    ),
                    session_type="Free Throws",
                )
            )

            sessions.append(
                session
            )

            shots = (
                history_repository
                .list_shot_files(
                    player_id=player.player_id
                )
            )

            session_repository.assign_shots(
                session_id=session.session_id,
                shot_file_ids=[
                    shot.shot_file_id
                    for shot in shots
                ],
            )

        organization_repository = (
            OrganizationTeamRepository(
                database_file=database_file
            )
        )

        organization = (
            organization_repository
            .create_organization(
                organization_name=(
                    "Test Organization"
                ),
                organization_type=(
                    "Professional Team"
                ),
            )
        )

        team = (
            organization_repository
            .create_team(
                organization_id=(
                    organization
                    .organization_id
                ),
                team_name="Test Team",
                season_name="2026",
                level="Professional",
            )
        )

        for player in players:
            organization_repository.assign_player_to_team(
                team_id=team.team_id,
                player_id=player.player_id,
                roster_role="Player",
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

            for player_index, player in enumerate(
                players,
                start=1,
            ):
                for trial_number in range(
                    1,
                    6,
                ):
                    result = (
                        "made"
                        if trial_number
                        in {
                            1,
                            2,
                            4,
                        }
                        else "missed"
                    )

                    writer.writerow(
                        {
                            "participant_id": (
                                player.participant_id
                            ),
                            "trial_id": (
                                f"T{trial_number:04d}"
                            ),
                            "analysis_status": (
                                "success"
                            ),
                            "result": result,
                            "overall_similarity_score": (
                                80
                                + player_index
                                * 3
                                - trial_number
                            ),
                            "confidence_adjusted_score": (
                                76
                                + player_index
                                * 3
                                - trial_number
                            ),
                            "release_angle_deg": (
                                50
                                + trial_number
                                * 0.5
                            ),
                            "release_ball_speed_ft_s": (
                                10
                                + trial_number
                                * 0.1
                            ),
                            "release_height_ft": (
                                7
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
                            ),
                        }
                    )

        engine = TeamPracticeIntelligenceEngine(
            database_file=database_file,
            feature_file=feature_file,
            output_folder=(
                root
                / "outputs"
            ),
        )

        summary, player_rows = engine.run(
            team_id=team.team_id
        )

        assert summary.active_players == 2
        assert summary.players_with_sessions == 2
        assert summary.players_with_grades == 2
        assert len(
            player_rows
        ) == 2
        assert summary.average_session_grade is not None

        assert engine.player_rows_file.exists()
        assert engine.summary_file.exists()
        assert engine.report_file.exists()

        print(
            "TEAM PRACTICE INTELLIGENCE TEST PASSED"
        )

        print(
            f"Team: {summary.team_name}"
        )

        print(
            f"Active players: "
            f"{summary.active_players}"
        )

        print(
            f"Players with grades: "
            f"{summary.players_with_grades}"
        )

        print(
            f"Average session grade: "
            f"{summary.average_session_grade:.1f}"
        )

        print(
            f"Highest-grade player: "
            f"{summary.highest_grade_player}"
        )

        print(
            f"Highest-attention player: "
            f"{summary.highest_attention_player}"
        )

        print(
            "Team aggregation: PASSED"
        )

        print(
            "Team practice exports: PASSED"
        )


if __name__ == "__main__":
    _run_team_practice_intelligence_test()