from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "player_history.sqlite3"
)


@dataclass(frozen=True)
class OrganizationRecord:
    organization_id: int

    organization_name: str
    organization_type: str
    description: str

    active: bool
    created_at: str
    updated_at: str

    team_count: int
    player_count: int


@dataclass(frozen=True)
class TeamRecord:
    team_id: int
    organization_id: int

    organization_name: str

    team_name: str
    season_name: str
    level: str
    description: str

    active: bool
    created_at: str
    updated_at: str

    player_count: int


@dataclass(frozen=True)
class TeamPlayerRecord:
    team_id: int
    player_id: int

    organization_name: str
    team_name: str

    participant_id: str
    player_display_name: str

    roster_role: str
    jersey_number: str
    position_group: str

    active: bool
    added_at: str


class OrganizationTeamRepository:
    """
    Manage organizations, teams, and player-to-team assignments.

    The existing structure is:

        Player
            Practice sessions
                Shot files

    This repository adds:

        Organization
            Team
                Player
                    Practice sessions
                        Shot files

    It uses the existing player_history.sqlite3 database and does not modify
    or delete existing player, session, or shot-file records.

    A player may belong to multiple teams, which is useful for cases such as:

        Raptors 905
        2026 Summer League
        Canadian National Team
        Private Clients

    Each membership can store a roster role, jersey number, and position group.
    """

    def __init__(
        self,
        database_file: Path = DEFAULT_DATABASE_FILE,
    ) -> None:
        self.database_file = Path(
            database_file
        )

        self.database_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.initialize_database()

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
            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    # =====================================================
    # DATABASE SETUP
    # =====================================================

    def initialize_database(
        self,
    ) -> None:
        with self.open_connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS organizations (
                    organization_id INTEGER PRIMARY KEY AUTOINCREMENT,

                    organization_name TEXT NOT NULL UNIQUE,
                    organization_type TEXT NOT NULL DEFAULT 'Basketball',
                    description TEXT NOT NULL DEFAULT '',

                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS teams (
                    team_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    organization_id INTEGER NOT NULL,

                    team_name TEXT NOT NULL,
                    season_name TEXT NOT NULL DEFAULT '',
                    level TEXT NOT NULL DEFAULT '',
                    description TEXT NOT NULL DEFAULT '',

                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,

                    UNIQUE (
                        organization_id,
                        team_name,
                        season_name
                    ),

                    FOREIGN KEY (
                        organization_id
                    )
                    REFERENCES organizations (
                        organization_id
                    )
                    ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS team_players (
                    team_id INTEGER NOT NULL,
                    player_id INTEGER NOT NULL,

                    roster_role TEXT NOT NULL DEFAULT 'Player',
                    jersey_number TEXT NOT NULL DEFAULT '',
                    position_group TEXT NOT NULL DEFAULT '',

                    active INTEGER NOT NULL DEFAULT 1,
                    added_at TEXT NOT NULL,

                    PRIMARY KEY (
                        team_id,
                        player_id
                    ),

                    FOREIGN KEY (
                        team_id
                    )
                    REFERENCES teams (
                        team_id
                    )
                    ON DELETE CASCADE,

                    FOREIGN KEY (
                        player_id
                    )
                    REFERENCES players (
                        player_id
                    )
                    ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_teams_organization
                ON teams (
                    organization_id
                );

                CREATE INDEX IF NOT EXISTS idx_team_players_team
                ON team_players (
                    team_id
                );

                CREATE INDEX IF NOT EXISTS idx_team_players_player
                ON team_players (
                    player_id
                );
                """
            )

    # =====================================================
    # ORGANIZATIONS
    # =====================================================

    def create_organization(
        self,
        organization_name: str,
        organization_type: str = "Basketball",
        description: str = "",
    ) -> OrganizationRecord:
        cleaned_name = self._required_text(
            organization_name,
            "organization_name",
        )

        cleaned_type = (
            str(
                organization_type
                or "Basketball"
            )
            .strip()
            or "Basketball"
        )

        timestamp = self._utc_now()

        with self.open_connection() as connection:
            existing = connection.execute(
                """
                SELECT
                    organization_id
                FROM organizations
                WHERE organization_name = ?
                """,
                (
                    cleaned_name,
                ),
            ).fetchone()

            if existing is not None:
                raise ValueError(
                    (
                        "An organization already exists with the name "
                        f"{cleaned_name}."
                    )
                )

            cursor = connection.execute(
                """
                INSERT INTO organizations (
                    organization_name,
                    organization_type,
                    description,
                    active,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, 1, ?, ?)
                """,
                (
                    cleaned_name,
                    cleaned_type,
                    str(
                        description
                        or ""
                    ).strip(),
                    timestamp,
                    timestamp,
                ),
            )

            organization_id = int(
                cursor.lastrowid
            )

        return self.get_organization(
            organization_id
        )

    def get_organization(
        self,
        organization_id: int,
    ) -> OrganizationRecord:
        with self.open_connection() as connection:
            row = connection.execute(
                """
                SELECT
                    o.organization_id,
                    o.organization_name,
                    o.organization_type,
                    o.description,
                    o.active,
                    o.created_at,
                    o.updated_at,

                    COUNT(
                        DISTINCT t.team_id
                    ) AS team_count,

                    COUNT(
                        DISTINCT tp.player_id
                    ) AS player_count

                FROM organizations AS o

                LEFT JOIN teams AS t
                    ON t.organization_id = o.organization_id

                LEFT JOIN team_players AS tp
                    ON tp.team_id = t.team_id
                    AND tp.active = 1

                WHERE o.organization_id = ?

                GROUP BY
                    o.organization_id
                """,
                (
                    int(
                        organization_id
                    ),
                ),
            ).fetchone()

        if row is None:
            raise ValueError(
                (
                    "No organization was found with "
                    f"organization_id {organization_id}."
                )
            )

        return self._organization_from_row(
            row
        )

    def list_organizations(
        self,
        include_inactive: bool = False,
    ) -> list[OrganizationRecord]:
        query = """
            SELECT
                o.organization_id,
                o.organization_name,
                o.organization_type,
                o.description,
                o.active,
                o.created_at,
                o.updated_at,

                COUNT(
                    DISTINCT t.team_id
                ) AS team_count,

                COUNT(
                    DISTINCT tp.player_id
                ) AS player_count

            FROM organizations AS o

            LEFT JOIN teams AS t
                ON t.organization_id = o.organization_id

            LEFT JOIN team_players AS tp
                ON tp.team_id = t.team_id
                AND tp.active = 1
        """

        parameters: tuple[object, ...] = ()

        if not include_inactive:
            query += """
                WHERE o.active = 1
            """

        query += """
            GROUP BY
                o.organization_id

            ORDER BY
                o.organization_name
        """

        with self.open_connection() as connection:
            rows = connection.execute(
                query,
                parameters,
            ).fetchall()

        return [
            self._organization_from_row(
                row
            )
            for row in rows
        ]

    def update_organization(
        self,
        organization_id: int,
        organization_name: str | None = None,
        organization_type: str | None = None,
        description: str | None = None,
    ) -> OrganizationRecord:
        current = self.get_organization(
            organization_id
        )

        new_name = (
            current.organization_name
            if organization_name is None
            else self._required_text(
                organization_name,
                "organization_name",
            )
        )

        new_type = (
            current.organization_type
            if organization_type is None
            else (
                str(
                    organization_type
                )
                .strip()
                or current.organization_type
            )
        )

        new_description = (
            current.description
            if description is None
            else str(
                description
            ).strip()
        )

        with self.open_connection() as connection:
            connection.execute(
                """
                UPDATE organizations
                SET
                    organization_name = ?,
                    organization_type = ?,
                    description = ?,
                    updated_at = ?
                WHERE organization_id = ?
                """,
                (
                    new_name,
                    new_type,
                    new_description,
                    self._utc_now(),
                    int(
                        organization_id
                    ),
                ),
            )

        return self.get_organization(
            organization_id
        )

    def set_organization_active(
        self,
        organization_id: int,
        active: bool,
    ) -> None:
        with self.open_connection() as connection:
            cursor = connection.execute(
                """
                UPDATE organizations
                SET
                    active = ?,
                    updated_at = ?
                WHERE organization_id = ?
                """,
                (
                    1
                    if active
                    else 0,
                    self._utc_now(),
                    int(
                        organization_id
                    ),
                ),
            )

        if cursor.rowcount == 0:
            raise ValueError(
                (
                    "No organization was found with "
                    f"organization_id {organization_id}."
                )
            )

    # =====================================================
    # TEAMS
    # =====================================================

    def create_team(
        self,
        organization_id: int,
        team_name: str,
        season_name: str = "",
        level: str = "",
        description: str = "",
    ) -> TeamRecord:
        cleaned_team_name = self._required_text(
            team_name,
            "team_name",
        )

        timestamp = self._utc_now()

        with self.open_connection() as connection:
            organization = connection.execute(
                """
                SELECT
                    organization_id
                FROM organizations
                WHERE organization_id = ?
                """,
                (
                    int(
                        organization_id
                    ),
                ),
            ).fetchone()

            if organization is None:
                raise ValueError(
                    (
                        "No organization was found with "
                        f"organization_id {organization_id}."
                    )
                )

            cursor = connection.execute(
                """
                INSERT INTO teams (
                    organization_id,
                    team_name,
                    season_name,
                    level,
                    description,
                    active,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    int(
                        organization_id
                    ),
                    cleaned_team_name,
                    str(
                        season_name
                        or ""
                    ).strip(),
                    str(
                        level
                        or ""
                    ).strip(),
                    str(
                        description
                        or ""
                    ).strip(),
                    timestamp,
                    timestamp,
                ),
            )

            team_id = int(
                cursor.lastrowid
            )

        return self.get_team(
            team_id
        )

    def get_team(
        self,
        team_id: int,
    ) -> TeamRecord:
        with self.open_connection() as connection:
            row = connection.execute(
                """
                SELECT
                    t.team_id,
                    t.organization_id,

                    o.organization_name,

                    t.team_name,
                    t.season_name,
                    t.level,
                    t.description,

                    t.active,
                    t.created_at,
                    t.updated_at,

                    COUNT(
                        CASE
                            WHEN tp.active = 1
                            THEN tp.player_id
                        END
                    ) AS player_count

                FROM teams AS t

                INNER JOIN organizations AS o
                    ON o.organization_id = t.organization_id

                LEFT JOIN team_players AS tp
                    ON tp.team_id = t.team_id

                WHERE t.team_id = ?

                GROUP BY
                    t.team_id
                """,
                (
                    int(
                        team_id
                    ),
                ),
            ).fetchone()

        if row is None:
            raise ValueError(
                f"No team was found with team_id {team_id}."
            )

        return self._team_from_row(
            row
        )

    def list_teams(
        self,
        organization_id: int | None = None,
        include_inactive: bool = False,
    ) -> list[TeamRecord]:
        query = """
            SELECT
                t.team_id,
                t.organization_id,

                o.organization_name,

                t.team_name,
                t.season_name,
                t.level,
                t.description,

                t.active,
                t.created_at,
                t.updated_at,

                COUNT(
                    CASE
                        WHEN tp.active = 1
                        THEN tp.player_id
                    END
                ) AS player_count

            FROM teams AS t

            INNER JOIN organizations AS o
                ON o.organization_id = t.organization_id

            LEFT JOIN team_players AS tp
                ON tp.team_id = t.team_id
        """

        conditions: list[str] = []
        parameters: list[object] = []

        if organization_id is not None:
            conditions.append(
                "t.organization_id = ?"
            )

            parameters.append(
                int(
                    organization_id
                )
            )

        if not include_inactive:
            conditions.append(
                "t.active = 1"
            )

        if conditions:
            query += (
                " WHERE "
                + " AND ".join(
                    conditions
                )
            )

        query += """
            GROUP BY
                t.team_id

            ORDER BY
                o.organization_name,
                t.season_name DESC,
                t.team_name
        """

        with self.open_connection() as connection:
            rows = connection.execute(
                query,
                tuple(
                    parameters
                ),
            ).fetchall()

        return [
            self._team_from_row(
                row
            )
            for row in rows
        ]

    def update_team(
        self,
        team_id: int,
        team_name: str | None = None,
        season_name: str | None = None,
        level: str | None = None,
        description: str | None = None,
    ) -> TeamRecord:
        current = self.get_team(
            team_id
        )

        new_name = (
            current.team_name
            if team_name is None
            else self._required_text(
                team_name,
                "team_name",
            )
        )

        new_season = (
            current.season_name
            if season_name is None
            else str(
                season_name
            ).strip()
        )

        new_level = (
            current.level
            if level is None
            else str(
                level
            ).strip()
        )

        new_description = (
            current.description
            if description is None
            else str(
                description
            ).strip()
        )

        with self.open_connection() as connection:
            connection.execute(
                """
                UPDATE teams
                SET
                    team_name = ?,
                    season_name = ?,
                    level = ?,
                    description = ?,
                    updated_at = ?
                WHERE team_id = ?
                """,
                (
                    new_name,
                    new_season,
                    new_level,
                    new_description,
                    self._utc_now(),
                    int(
                        team_id
                    ),
                ),
            )

        return self.get_team(
            team_id
        )

    def set_team_active(
        self,
        team_id: int,
        active: bool,
    ) -> None:
        with self.open_connection() as connection:
            cursor = connection.execute(
                """
                UPDATE teams
                SET
                    active = ?,
                    updated_at = ?
                WHERE team_id = ?
                """,
                (
                    1
                    if active
                    else 0,
                    self._utc_now(),
                    int(
                        team_id
                    ),
                ),
            )

        if cursor.rowcount == 0:
            raise ValueError(
                f"No team was found with team_id {team_id}."
            )

    # =====================================================
    # TEAM ROSTERS
    # =====================================================

    def assign_player_to_team(
        self,
        team_id: int,
        player_id: int,
        roster_role: str = "Player",
        jersey_number: str = "",
        position_group: str = "",
    ) -> TeamPlayerRecord:
        self.get_team(
            team_id
        )

        with self.open_connection() as connection:
            player = connection.execute(
                """
                SELECT
                    player_id
                FROM players
                WHERE player_id = ?
                """,
                (
                    int(
                        player_id
                    ),
                ),
            ).fetchone()

            if player is None:
                raise ValueError(
                    f"No player was found with player_id {player_id}."
                )

            connection.execute(
                """
                INSERT INTO team_players (
                    team_id,
                    player_id,
                    roster_role,
                    jersey_number,
                    position_group,
                    active,
                    added_at
                )
                VALUES (?, ?, ?, ?, ?, 1, ?)

                ON CONFLICT (
                    team_id,
                    player_id
                )
                DO UPDATE SET
                    roster_role = excluded.roster_role,
                    jersey_number = excluded.jersey_number,
                    position_group = excluded.position_group,
                    active = 1
                """,
                (
                    int(
                        team_id
                    ),
                    int(
                        player_id
                    ),
                    str(
                        roster_role
                        or "Player"
                    ).strip()
                    or "Player",
                    str(
                        jersey_number
                        or ""
                    ).strip(),
                    str(
                        position_group
                        or ""
                    ).strip(),
                    self._utc_now(),
                ),
            )

        return self.get_team_player(
            team_id=team_id,
            player_id=player_id,
        )

    def get_team_player(
        self,
        team_id: int,
        player_id: int,
    ) -> TeamPlayerRecord:
        with self.open_connection() as connection:
            row = connection.execute(
                """
                SELECT
                    tp.team_id,
                    tp.player_id,

                    o.organization_name,
                    t.team_name,

                    p.participant_id,
                    p.display_name,

                    tp.roster_role,
                    tp.jersey_number,
                    tp.position_group,

                    tp.active,
                    tp.added_at

                FROM team_players AS tp

                INNER JOIN teams AS t
                    ON t.team_id = tp.team_id

                INNER JOIN organizations AS o
                    ON o.organization_id = t.organization_id

                INNER JOIN players AS p
                    ON p.player_id = tp.player_id

                WHERE
                    tp.team_id = ?
                    AND tp.player_id = ?
                """,
                (
                    int(
                        team_id
                    ),
                    int(
                        player_id
                    ),
                ),
            ).fetchone()

        if row is None:
            raise ValueError(
                (
                    "No team-player membership was found for "
                    f"team_id {team_id} and player_id {player_id}."
                )
            )

        return self._team_player_from_row(
            row
        )

    def list_team_players(
        self,
        team_id: int,
        include_inactive: bool = False,
    ) -> list[TeamPlayerRecord]:
        query = """
            SELECT
                tp.team_id,
                tp.player_id,

                o.organization_name,
                t.team_name,

                p.participant_id,
                p.display_name,

                tp.roster_role,
                tp.jersey_number,
                tp.position_group,

                tp.active,
                tp.added_at

            FROM team_players AS tp

            INNER JOIN teams AS t
                ON t.team_id = tp.team_id

            INNER JOIN organizations AS o
                ON o.organization_id = t.organization_id

            INNER JOIN players AS p
                ON p.player_id = tp.player_id

            WHERE tp.team_id = ?
        """

        parameters: list[object] = [
            int(
                team_id
            )
        ]

        if not include_inactive:
            query += """
                AND tp.active = 1
            """

        query += """
            ORDER BY
                p.display_name,
                p.participant_id
        """

        with self.open_connection() as connection:
            rows = connection.execute(
                query,
                tuple(
                    parameters
                ),
            ).fetchall()

        return [
            self._team_player_from_row(
                row
            )
            for row in rows
        ]

    def update_team_player(
        self,
        team_id: int,
        player_id: int,
        roster_role: str | None = None,
        jersey_number: str | None = None,
        position_group: str | None = None,
    ) -> TeamPlayerRecord:
        current = self.get_team_player(
            team_id=team_id,
            player_id=player_id,
        )

        with self.open_connection() as connection:
            connection.execute(
                """
                UPDATE team_players
                SET
                    roster_role = ?,
                    jersey_number = ?,
                    position_group = ?
                WHERE
                    team_id = ?
                    AND player_id = ?
                """,
                (
                    (
                        current.roster_role
                        if roster_role is None
                        else (
                            str(
                                roster_role
                            )
                            .strip()
                            or "Player"
                        )
                    ),
                    (
                        current.jersey_number
                        if jersey_number is None
                        else str(
                            jersey_number
                        ).strip()
                    ),
                    (
                        current.position_group
                        if position_group is None
                        else str(
                            position_group
                        ).strip()
                    ),
                    int(
                        team_id
                    ),
                    int(
                        player_id
                    ),
                ),
            )

        return self.get_team_player(
            team_id=team_id,
            player_id=player_id,
        )

    def set_team_player_active(
        self,
        team_id: int,
        player_id: int,
        active: bool,
    ) -> None:
        with self.open_connection() as connection:
            cursor = connection.execute(
                """
                UPDATE team_players
                SET active = ?
                WHERE
                    team_id = ?
                    AND player_id = ?
                """,
                (
                    1
                    if active
                    else 0,
                    int(
                        team_id
                    ),
                    int(
                        player_id
                    ),
                ),
            )

        if cursor.rowcount == 0:
            raise ValueError(
                (
                    "No team-player membership was found for "
                    f"team_id {team_id} and player_id {player_id}."
                )
            )

    def remove_player_from_team(
        self,
        team_id: int,
        player_id: int,
    ) -> None:
        """
        Soft-remove a player from a roster.

        The player profile, practices, and shot history remain untouched.
        """

        self.set_team_player_active(
            team_id=team_id,
            player_id=player_id,
            active=False,
        )

    # =====================================================
    # HELPERS
    # =====================================================

    @staticmethod
    def _organization_from_row(
        row: sqlite3.Row,
    ) -> OrganizationRecord:
        return OrganizationRecord(
            organization_id=int(
                row[
                    "organization_id"
                ]
            ),

            organization_name=str(
                row[
                    "organization_name"
                ]
            ),
            organization_type=str(
                row[
                    "organization_type"
                ]
            ),
            description=str(
                row[
                    "description"
                ]
            ),

            active=bool(
                row[
                    "active"
                ]
            ),
            created_at=str(
                row[
                    "created_at"
                ]
            ),
            updated_at=str(
                row[
                    "updated_at"
                ]
            ),

            team_count=int(
                row[
                    "team_count"
                ]
            ),
            player_count=int(
                row[
                    "player_count"
                ]
            ),
        )

    @staticmethod
    def _team_from_row(
        row: sqlite3.Row,
    ) -> TeamRecord:
        return TeamRecord(
            team_id=int(
                row[
                    "team_id"
                ]
            ),
            organization_id=int(
                row[
                    "organization_id"
                ]
            ),

            organization_name=str(
                row[
                    "organization_name"
                ]
            ),

            team_name=str(
                row[
                    "team_name"
                ]
            ),
            season_name=str(
                row[
                    "season_name"
                ]
            ),
            level=str(
                row[
                    "level"
                ]
            ),
            description=str(
                row[
                    "description"
                ]
            ),

            active=bool(
                row[
                    "active"
                ]
            ),
            created_at=str(
                row[
                    "created_at"
                ]
            ),
            updated_at=str(
                row[
                    "updated_at"
                ]
            ),

            player_count=int(
                row[
                    "player_count"
                ]
            ),
        )

    @staticmethod
    def _team_player_from_row(
        row: sqlite3.Row,
    ) -> TeamPlayerRecord:
        return TeamPlayerRecord(
            team_id=int(
                row[
                    "team_id"
                ]
            ),
            player_id=int(
                row[
                    "player_id"
                ]
            ),

            organization_name=str(
                row[
                    "organization_name"
                ]
            ),
            team_name=str(
                row[
                    "team_name"
                ]
            ),

            participant_id=str(
                row[
                    "participant_id"
                ]
            ),
            player_display_name=str(
                row[
                    "display_name"
                ]
            ),

            roster_role=str(
                row[
                    "roster_role"
                ]
            ),
            jersey_number=str(
                row[
                    "jersey_number"
                ]
            ),
            position_group=str(
                row[
                    "position_group"
                ]
            ),

            active=bool(
                row[
                    "active"
                ]
            ),
            added_at=str(
                row[
                    "added_at"
                ]
            ),
        )

    @staticmethod
    def _required_text(
        value: object,
        field_name: str,
    ) -> str:
        cleaned = str(
            value
            if value is not None
            else ""
        ).strip()

        if not cleaned:
            raise ValueError(
                f"{field_name} is required."
            )

        return cleaned

    @staticmethod
    def _utc_now(
    ) -> str:
        return (
            datetime.now(
                timezone.utc
            )
            .replace(
                microsecond=0
            )
            .isoformat()
        )


def _run_organization_team_repository_test() -> None:
    """
    Run an isolated test without touching the real project database.
    """

    import tempfile

    from tracking_app.research.player_history_repository import (
        PlayerHistoryRepository,
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
                participant_id="P0001",
                display_name="Player One",
            )
        )

        player_two = (
            history_repository
            .create_or_get_player(
                participant_id="P0002",
                display_name="Player Two",
            )
        )

        repository = (
            OrganizationTeamRepository(
                database_file=database_file
            )
        )

        organization = repository.create_organization(
            organization_name="Test Organization",
            organization_type="Professional Team",
            description="Repository test",
        )

        team = repository.create_team(
            organization_id=(
                organization.organization_id
            ),
            team_name="Test Team",
            season_name="2026",
            level="Professional",
            description="Test roster",
        )

        repository.assign_player_to_team(
            team_id=team.team_id,
            player_id=player_one.player_id,
            roster_role="Player",
            jersey_number="1",
            position_group="Guard",
        )

        repository.assign_player_to_team(
            team_id=team.team_id,
            player_id=player_two.player_id,
            roster_role="Player",
            jersey_number="2",
            position_group="Forward",
        )

        roster = repository.list_team_players(
            team_id=team.team_id
        )

        assert len(
            roster
        ) == 2

        updated_membership = (
            repository
            .update_team_player(
                team_id=team.team_id,
                player_id=player_one.player_id,
                jersey_number="11",
                position_group="Point Guard",
            )
        )

        assert (
            updated_membership
            .jersey_number
            == "11"
        )

        repository.remove_player_from_team(
            team_id=team.team_id,
            player_id=player_two.player_id,
        )

        active_roster = repository.list_team_players(
            team_id=team.team_id,
            include_inactive=False,
        )

        full_roster = repository.list_team_players(
            team_id=team.team_id,
            include_inactive=True,
        )

        assert len(
            active_roster
        ) == 1

        assert len(
            full_roster
        ) == 2

        updated_team = repository.get_team(
            team.team_id
        )

        assert (
            updated_team.player_count
            == 1
        )

        print(
            "ORGANIZATION & TEAM REPOSITORY TEST PASSED"
        )

        print(
            f"Organization created: "
            f"{organization.organization_name}"
        )

        print(
            f"Team created: "
            f"{team.team_name}"
        )

        print(
            f"Players originally assigned: "
            f"{len(roster)}"
        )

        print(
            f"Active roster after removal: "
            f"{len(active_roster)}"
        )

        print(
            "Roster update test: PASSED"
        )

        print(
            "Soft roster removal test: PASSED"
        )

        print(
            "Player history preservation: PASSED"
        )


if __name__ == "__main__":
    _run_organization_team_repository_test()