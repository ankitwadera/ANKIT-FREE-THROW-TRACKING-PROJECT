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
from tracking_app.research.player_development_intelligence import (
    PlayerDevelopmentIntelligence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "player_history.sqlite3"
)

DEFAULT_OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "team_dashboard"
)


@dataclass(frozen=True)
class TeamPlayerDashboardRow:
    """
    One player row used by the coaching dashboard.
    """

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

    active_shot_files: int
    made_shots: int
    missed_shots: int
    make_percentage: float

    active_sessions: int
    most_recent_session_date: str | None

    development_available: bool
    overall_mechanics_score: float | None
    consistency_score: float | None
    timing_score: float | None
    lower_body_score: float | None
    upper_body_score: float | None
    ball_release_score: float | None

    primary_priority: str
    strongest_area: str
    evidence_confidence: str

    attention_score: float
    dashboard_status: str


@dataclass(frozen=True)
class TeamDashboardSummary:
    """
    Team-level headline metrics.
    """

    team_id: int
    organization_name: str
    team_name: str
    season_name: str
    level: str

    active_players: int

    total_shot_files: int
    total_made_shots: int
    total_missed_shots: int
    team_make_percentage: float

    total_sessions: int

    players_with_development_data: int
    average_mechanics_score: float | None
    average_consistency_score: float | None

    highest_attention_player: str
    strongest_consistency_player: str

    dashboard_status: str


class TeamDashboardEngine:
    """
    Build a coach-facing dashboard from the organization, team, player,
    session, shot-history, and player-development systems.

    The dashboard answers questions such as:

    - How many active players are on the roster?
    - How many tracked shots and practice sessions exist?
    - What is the roster's make percentage?
    - Which player has the strongest consistency?
    - Which player currently deserves the most review attention?
    - What is each player's primary long-term development priority?

    It does not alter any player or team data.
    """

    def __init__(
        self,
        database_file: Path = DEFAULT_DATABASE_FILE,
        output_folder: Path = DEFAULT_OUTPUT_FOLDER,
    ) -> None:
        self.database_file = Path(
            database_file
        )

        self.output_folder = Path(
            output_folder
        )

        self.player_rows_file = (
            self.output_folder
            / "team_player_dashboard.csv"
        )

        self.team_summary_file = (
            self.output_folder
            / "team_dashboard_summary.csv"
        )

        self.report_file = (
            self.output_folder
            / "team_dashboard_report.txt"
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
        TeamDashboardSummary,
        list[TeamPlayerDashboardRow],
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
            self.team_summary_file,
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
    # PLAYER ROWS
    # =====================================================

    def _build_player_row(
        self,
        team: object,
        roster_member: object,
    ) -> TeamPlayerDashboardRow:
        history = self._player_history_counts(
            roster_member.player_id
        )

        development_available = False
        development_summary = None

        try:
            development_engine = (
                PlayerDevelopmentIntelligence()
            )

            (
                development_summary,
                _,
            ) = development_engine.run(
                participant_id=(
                    roster_member.participant_id
                )
            )

            development_available = True

        except Exception:
            development_available = False
            development_summary = None

        attention_score = self._attention_score(
            development_available=(
                development_available
            ),
            development_summary=(
                development_summary
            ),
            shot_count=(
                history[
                    "active_shot_files"
                ]
            ),
            session_count=(
                history[
                    "active_sessions"
                ]
            ),
        )

        dashboard_status = self._dashboard_status(
            development_available=(
                development_available
            ),
            attention_score=attention_score,
            active_shot_files=(
                history[
                    "active_shot_files"
                ]
            ),
        )

        return TeamPlayerDashboardRow(
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

            active_shot_files=int(
                history[
                    "active_shot_files"
                ]
            ),
            made_shots=int(
                history[
                    "made_shots"
                ]
            ),
            missed_shots=int(
                history[
                    "missed_shots"
                ]
            ),
            make_percentage=float(
                history[
                    "make_percentage"
                ]
            ),

            active_sessions=int(
                history[
                    "active_sessions"
                ]
            ),
            most_recent_session_date=(
                history[
                    "most_recent_session_date"
                ]
            ),

            development_available=(
                development_available
            ),
            overall_mechanics_score=(
                None
                if development_summary is None
                else float(
                    development_summary
                    .overall_mechanics_score
                )
            ),
            consistency_score=(
                None
                if development_summary is None
                else float(
                    development_summary
                    .consistency_score
                )
            ),
            timing_score=(
                None
                if development_summary is None
                else float(
                    development_summary
                    .timing_score
                )
            ),
            lower_body_score=(
                None
                if development_summary is None
                else float(
                    development_summary
                    .lower_body_score
                )
            ),
            upper_body_score=(
                None
                if development_summary is None
                else float(
                    development_summary
                    .upper_body_score
                )
            ),
            ball_release_score=(
                None
                if development_summary is None
                else float(
                    development_summary
                    .ball_release_score
                )
            ),

            primary_priority=(
                "Insufficient development data"
                if development_summary is None
                else str(
                    development_summary
                    .primary_priority
                )
            ),
            strongest_area=(
                "N/A"
                if development_summary is None
                else str(
                    development_summary
                    .strongest_area
                )
            ),
            evidence_confidence=(
                "unavailable"
                if development_summary is None
                else str(
                    development_summary
                    .evidence_confidence
                )
            ),

            attention_score=float(
                attention_score
            ),
            dashboard_status=(
                dashboard_status
            ),
        )

    def _player_history_counts(
        self,
        player_id: int,
    ) -> dict[str, object]:
        with self.open_connection() as connection:
            shot_row = connection.execute(
                """
                SELECT
                    COUNT(*) AS active_shot_files,

                    SUM(
                        CASE
                            WHEN LOWER(recorded_result) = 'made'
                            THEN 1
                            ELSE 0
                        END
                    ) AS made_shots,

                    SUM(
                        CASE
                            WHEN LOWER(recorded_result) = 'missed'
                            THEN 1
                            ELSE 0
                        END
                    ) AS missed_shots

                FROM shot_files

                WHERE
                    player_id = ?
                    AND active = 1
                """,
                (
                    int(
                        player_id
                    ),
                ),
            ).fetchone()

            session_row = connection.execute(
                """
                SELECT
                    COUNT(*) AS active_sessions,
                    MAX(session_date) AS most_recent_session_date

                FROM practice_sessions

                WHERE
                    player_id = ?
                    AND active = 1
                """,
                (
                    int(
                        player_id
                    ),
                ),
            ).fetchone()

        active_shot_files = int(
            shot_row[
                "active_shot_files"
            ]
            or 0
        )

        made_shots = int(
            shot_row[
                "made_shots"
            ]
            or 0
        )

        missed_shots = int(
            shot_row[
                "missed_shots"
            ]
            or 0
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

        return {
            "active_shot_files": (
                active_shot_files
            ),
            "made_shots": made_shots,
            "missed_shots": missed_shots,
            "make_percentage": (
                make_percentage
            ),
            "active_sessions": int(
                session_row[
                    "active_sessions"
                ]
                or 0
            ),
            "most_recent_session_date": (
                session_row[
                    "most_recent_session_date"
                ]
            ),
        }

    # =====================================================
    # ATTENTION SCORING
    # =====================================================

    @staticmethod
    def _attention_score(
        development_available: bool,
        development_summary: object | None,
        shot_count: int,
        session_count: int,
    ) -> float:
        """
        Higher means the player deserves more coach review attention.

        This is not a talent or performance grade. It is a workflow priority.
        """

        if not development_available:
            missing_data_component = (
                55.0
                if shot_count < 30
                else 35.0
            )

            session_component = (
                20.0
                if session_count == 0
                else 5.0
            )

            return float(
                min(
                    100.0,
                    missing_data_component
                    + session_component
                )
            )

        consistency_component = (
            100.0
            - float(
                development_summary
                .consistency_score
            )
        )

        mechanics_component = (
            100.0
            - float(
                development_summary
                .overall_mechanics_score
            )
        )

        data_component = (
            20.0
            if shot_count < 30
            else (
                10.0
                if shot_count < 75
                else 0.0
            )
        )

        score = (
            consistency_component
            * 0.55
            + mechanics_component
            * 0.30
            + data_component
            * 0.15
        )

        return float(
            max(
                0.0,
                min(
                    100.0,
                    score,
                ),
            )
        )

    @staticmethod
    def _dashboard_status(
        development_available: bool,
        attention_score: float,
        active_shot_files: int,
    ) -> str:
        if active_shot_files == 0:
            return "No tracked shots"

        if not development_available:
            return "Collect more development data"

        if attention_score >= 55:
            return "High review priority"

        if attention_score >= 35:
            return "Monitor"

        return "Stable"

    # =====================================================
    # TEAM SUMMARY
    # =====================================================

    def _build_summary(
        self,
        team: object,
        player_rows: list[
            TeamPlayerDashboardRow
        ],
    ) -> TeamDashboardSummary:
        total_shot_files = sum(
            row.active_shot_files
            for row in player_rows
        )

        total_made_shots = sum(
            row.made_shots
            for row in player_rows
        )

        total_missed_shots = sum(
            row.missed_shots
            for row in player_rows
        )

        classified_shots = (
            total_made_shots
            + total_missed_shots
        )

        team_make_percentage = (
            total_made_shots
            / classified_shots
            * 100.0
            if classified_shots
            else 0.0
        )

        total_sessions = sum(
            row.active_sessions
            for row in player_rows
        )

        development_rows = [
            row
            for row in player_rows
            if row.development_available
        ]

        average_mechanics_score = (
            float(
                mean(
                    row.overall_mechanics_score
                    for row in development_rows
                    if row.overall_mechanics_score
                    is not None
                )
            )
            if development_rows
            else None
        )

        average_consistency_score = (
            float(
                mean(
                    row.consistency_score
                    for row in development_rows
                    if row.consistency_score
                    is not None
                )
            )
            if development_rows
            else None
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

        consistency_candidates = [
            row
            for row in player_rows
            if row.consistency_score
            is not None
        ]

        strongest_consistency_player = (
            max(
                consistency_candidates,
                key=lambda row: (
                    row.consistency_score
                ),
            ).player_display_name
            if consistency_candidates
            else "N/A"
        )

        if not player_rows:
            dashboard_status = "No active roster"

        elif total_shot_files == 0:
            dashboard_status = "Roster created; no tracked shots"

        elif len(
            development_rows
        ) < len(
            player_rows
        ):
            dashboard_status = "Partially populated"

        else:
            dashboard_status = "Ready"

        return TeamDashboardSummary(
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

            total_shot_files=(
                total_shot_files
            ),
            total_made_shots=(
                total_made_shots
            ),
            total_missed_shots=(
                total_missed_shots
            ),
            team_make_percentage=float(
                team_make_percentage
            ),

            total_sessions=(
                total_sessions
            ),

            players_with_development_data=len(
                development_rows
            ),
            average_mechanics_score=(
                average_mechanics_score
            ),
            average_consistency_score=(
                average_consistency_score
            ),

            highest_attention_player=(
                highest_attention_player
            ),
            strongest_consistency_player=(
                strongest_consistency_player
            ),

            dashboard_status=(
                dashboard_status
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
        summary: TeamDashboardSummary,
        player_rows: list[
            TeamPlayerDashboardRow
        ],
    ) -> Path:
        lines = [
            "=" * 94,
            "TEAM COACHING DASHBOARD",
            "=" * 94,
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
            (
                f"Level: "
                f"{summary.level or 'N/A'}"
            ),
            "",
            (
                f"Active players: "
                f"{summary.active_players}"
            ),
            (
                f"Tracked shot files: "
                f"{summary.total_shot_files}"
            ),
            (
                f"Team make percentage: "
                f"{summary.team_make_percentage:.1f}%"
            ),
            (
                f"Practice sessions: "
                f"{summary.total_sessions}"
            ),
            (
                f"Dashboard status: "
                f"{summary.dashboard_status}"
            ),
            "",
            "COACHING HIGHLIGHTS",
            "-" * 94,
            (
                "Highest attention player: "
                f"{summary.highest_attention_player}"
            ),
            (
                "Strongest consistency player: "
                f"{summary.strongest_consistency_player}"
            ),
            "",
            "PLAYER ROWS",
            "-" * 94,
        ]

        for row in player_rows:
            lines.extend(
                [
                    (
                        f"{row.player_display_name} "
                        f"({row.participant_id})"
                    ),
                    (
                        f"  Shots: {row.active_shot_files} · "
                        f"Make %: {row.make_percentage:.1f} · "
                        f"Sessions: {row.active_sessions}"
                    ),
                    (
                        f"  Development status: "
                        f"{row.dashboard_status}"
                    ),
                    (
                        f"  Primary priority: "
                        f"{row.primary_priority}"
                    ),
                    (
                        f"  Attention score: "
                        f"{row.attention_score:.1f}/100"
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


def _run_team_dashboard_engine_test() -> None:
    """
    Run an isolated structural test with a temporary team database.

    Development scores are optional in this test because the temporary player
    IDs do not exist in the project's master feature CSV.
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
            / "test_history.sqlite3"
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

        player_one = (
            history_repository
            .create_or_get_player(
                participant_id="TEST_P1",
                display_name="Test Player One",
            )
        )

        player_two = (
            history_repository
            .create_or_get_player(
                participant_id="TEST_P2",
                display_name="Test Player Two",
            )
        )

        for player, result in (
            (
                player_one,
                "made",
            ),
            (
                player_two,
                "missed",
            ),
        ):
            payload = {
                "participant_id": (
                    player.participant_id
                ),
                "trial_id": "T0001",
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

            history_repository.import_json_files(
                [
                    (
                        (
                            f"{player.participant_id}"
                            "_T0001.json"
                        ),
                        json.dumps(
                            payload
                        ).encode(
                            "utf-8"
                        ),
                    )
                ],
                selected_player_id=(
                    player.player_id
                ),
                enforce_selected_player=True,
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

        organization_repository.assign_player_to_team(
            team_id=team.team_id,
            player_id=player_one.player_id,
            jersey_number="1",
            position_group="Guard",
        )

        organization_repository.assign_player_to_team(
            team_id=team.team_id,
            player_id=player_two.player_id,
            jersey_number="2",
            position_group="Forward",
        )

        session_repository = (
            PracticeSessionRepository(
                database_file=database_file
            )
        )

        session_repository.create_session(
            player_id=player_one.player_id,
            session_name="Test Practice",
            session_date="2026-07-27",
        )

        engine = TeamDashboardEngine(
            database_file=database_file,
            output_folder=(
                root
                / "outputs"
            ),
        )

        summary, player_rows = engine.run(
            team_id=team.team_id
        )

        assert summary.active_players == 2
        assert summary.total_shot_files == 2
        assert summary.total_made_shots == 1
        assert summary.total_missed_shots == 1
        assert summary.team_make_percentage == 50.0
        assert len(
            player_rows
        ) == 2

        assert engine.player_rows_file.exists()
        assert engine.team_summary_file.exists()
        assert engine.report_file.exists()

        print(
            "TEAM DASHBOARD ENGINE TEST PASSED"
        )

        print(
            f"Team: {summary.team_name}"
        )

        print(
            f"Active players: {summary.active_players}"
        )

        print(
            f"Tracked shots: {summary.total_shot_files}"
        )

        print(
            f"Team make percentage: "
            f"{summary.team_make_percentage:.1f}%"
        )

        print(
            f"Practice sessions: "
            f"{summary.total_sessions}"
        )

        print(
            f"Dashboard status: "
            f"{summary.dashboard_status}"
        )

        print(
            "Player isolation and roster aggregation: PASSED"
        )

        print(
            "Dashboard exports: PASSED"
        )


if __name__ == "__main__":
    _run_team_dashboard_engine_test()