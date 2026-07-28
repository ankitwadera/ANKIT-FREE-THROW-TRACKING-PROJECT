from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "player_history.sqlite3"
)


@dataclass(frozen=True)
class PracticeSessionRecord:
    session_id: int
    player_id: int

    participant_id: str
    player_display_name: str

    session_name: str
    session_date: str
    session_type: str
    location: str
    notes: str

    active: bool
    created_at: str
    updated_at: str

    shot_count: int


@dataclass(frozen=True)
class SessionShotRecord:
    session_id: int
    shot_file_id: int

    participant_id: str
    player_display_name: str

    trial_id: str
    original_filename: str
    recorded_result: str
    total_frames: int

    added_at: str
    display_order: int


class PracticeSessionRepository:
    """
    Organize a player's shot files into named practices and sessions.

    Examples
    --------
    Player: P0001

        July 27 Free Throws
            T0001
            T0002
            T0003

        July 29 Fatigue Test
            T0004
            T0005

    This repository uses the existing player_history.sqlite3 database created
    by PlayerHistoryRepository. It adds session tables without changing or
    deleting existing player and shot-file records.

    A shot can belong to one session at a time. It can be removed from one
    session and assigned to another without deleting the underlying JSON file.
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
                CREATE TABLE IF NOT EXISTS practice_sessions (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id INTEGER NOT NULL,

                    session_name TEXT NOT NULL,
                    session_date TEXT NOT NULL,
                    session_type TEXT NOT NULL DEFAULT 'Free Throws',
                    location TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '',

                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,

                    FOREIGN KEY (
                        player_id
                    )
                    REFERENCES players (
                        player_id
                    )
                    ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS practice_session_shots (
                    session_id INTEGER NOT NULL,
                    shot_file_id INTEGER NOT NULL,

                    added_at TEXT NOT NULL,
                    display_order INTEGER NOT NULL DEFAULT 0,

                    PRIMARY KEY (
                        session_id,
                        shot_file_id
                    ),

                    UNIQUE (
                        shot_file_id
                    ),

                    FOREIGN KEY (
                        session_id
                    )
                    REFERENCES practice_sessions (
                        session_id
                    )
                    ON DELETE CASCADE,

                    FOREIGN KEY (
                        shot_file_id
                    )
                    REFERENCES shot_files (
                        shot_file_id
                    )
                    ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_practice_sessions_player
                ON practice_sessions (
                    player_id
                );

                CREATE INDEX IF NOT EXISTS idx_practice_sessions_date
                ON practice_sessions (
                    session_date
                );

                CREATE INDEX IF NOT EXISTS idx_practice_session_shots_session
                ON practice_session_shots (
                    session_id
                );
                """
            )

    # =====================================================
    # SESSION MANAGEMENT
    # =====================================================

    def create_session(
        self,
        player_id: int,
        session_name: str,
        session_date: str | date | None = None,
        session_type: str = "Free Throws",
        location: str = "",
        notes: str = "",
    ) -> PracticeSessionRecord:
        cleaned_name = self._required_text(
            session_name,
            "session_name",
        )

        cleaned_date = self._normalize_date(
            session_date
        )

        cleaned_type = (
            str(
                session_type
                or "Free Throws"
            )
            .strip()
            or "Free Throws"
        )

        timestamp = self._utc_now()

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

            cursor = connection.execute(
                """
                INSERT INTO practice_sessions (
                    player_id,
                    session_name,
                    session_date,
                    session_type,
                    location,
                    notes,
                    active,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    int(
                        player_id
                    ),
                    cleaned_name,
                    cleaned_date,
                    cleaned_type,
                    str(
                        location
                        or ""
                    ).strip(),
                    str(
                        notes
                        or ""
                    ).strip(),
                    timestamp,
                    timestamp,
                ),
            )

            session_id = int(
                cursor.lastrowid
            )

        return self.get_session(
            session_id
        )

    def get_session(
        self,
        session_id: int,
    ) -> PracticeSessionRecord:
        with self.open_connection() as connection:
            row = connection.execute(
                """
                SELECT
                    ps.session_id,
                    ps.player_id,

                    p.participant_id,
                    p.display_name,

                    ps.session_name,
                    ps.session_date,
                    ps.session_type,
                    ps.location,
                    ps.notes,

                    ps.active,
                    ps.created_at,
                    ps.updated_at,

                    COUNT(
                        pss.shot_file_id
                    ) AS shot_count

                FROM practice_sessions AS ps

                INNER JOIN players AS p
                    ON p.player_id = ps.player_id

                LEFT JOIN practice_session_shots AS pss
                    ON pss.session_id = ps.session_id

                WHERE ps.session_id = ?

                GROUP BY
                    ps.session_id
                """,
                (
                    int(
                        session_id
                    ),
                ),
            ).fetchone()

        if row is None:
            raise ValueError(
                f"No practice session was found with session_id {session_id}."
            )

        return self._session_from_row(
            row
        )

    def list_sessions(
        self,
        player_id: int | None = None,
        include_inactive: bool = False,
    ) -> list[PracticeSessionRecord]:
        query = """
            SELECT
                ps.session_id,
                ps.player_id,

                p.participant_id,
                p.display_name,

                ps.session_name,
                ps.session_date,
                ps.session_type,
                ps.location,
                ps.notes,

                ps.active,
                ps.created_at,
                ps.updated_at,

                COUNT(
                    pss.shot_file_id
                ) AS shot_count

            FROM practice_sessions AS ps

            INNER JOIN players AS p
                ON p.player_id = ps.player_id

            LEFT JOIN practice_session_shots AS pss
                ON pss.session_id = ps.session_id
        """

        conditions: list[str] = []
        parameters: list[object] = []

        if player_id is not None:
            conditions.append(
                "ps.player_id = ?"
            )

            parameters.append(
                int(
                    player_id
                )
            )

        if not include_inactive:
            conditions.append(
                "ps.active = 1"
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
                ps.session_id

            ORDER BY
                ps.session_date DESC,
                ps.created_at DESC
        """

        with self.open_connection() as connection:
            rows = connection.execute(
                query,
                tuple(
                    parameters
                ),
            ).fetchall()

        return [
            self._session_from_row(
                row
            )
            for row in rows
        ]

    def update_session(
        self,
        session_id: int,
        session_name: str | None = None,
        session_date: str | date | None = None,
        session_type: str | None = None,
        location: str | None = None,
        notes: str | None = None,
    ) -> PracticeSessionRecord:
        current = self.get_session(
            session_id
        )

        new_name = (
            current.session_name
            if session_name is None
            else self._required_text(
                session_name,
                "session_name",
            )
        )

        new_date = (
            current.session_date
            if session_date is None
            else self._normalize_date(
                session_date
            )
        )

        new_type = (
            current.session_type
            if session_type is None
            else (
                str(
                    session_type
                )
                .strip()
                or current.session_type
            )
        )

        new_location = (
            current.location
            if location is None
            else str(
                location
            ).strip()
        )

        new_notes = (
            current.notes
            if notes is None
            else str(
                notes
            ).strip()
        )

        with self.open_connection() as connection:
            connection.execute(
                """
                UPDATE practice_sessions
                SET
                    session_name = ?,
                    session_date = ?,
                    session_type = ?,
                    location = ?,
                    notes = ?,
                    updated_at = ?
                WHERE session_id = ?
                """,
                (
                    new_name,
                    new_date,
                    new_type,
                    new_location,
                    new_notes,
                    self._utc_now(),
                    int(
                        session_id
                    ),
                ),
            )

        return self.get_session(
            session_id
        )

    def set_session_active(
        self,
        session_id: int,
        active: bool,
    ) -> None:
        with self.open_connection() as connection:
            cursor = connection.execute(
                """
                UPDATE practice_sessions
                SET
                    active = ?,
                    updated_at = ?
                WHERE session_id = ?
                """,
                (
                    1
                    if active
                    else 0,
                    self._utc_now(),
                    int(
                        session_id
                    ),
                ),
            )

        if cursor.rowcount == 0:
            raise ValueError(
                f"No practice session was found with session_id {session_id}."
            )

    def delete_session(
        self,
        session_id: int,
    ) -> None:
        """
        Permanently delete a session container.

        This does not delete shot files. It only removes their session
        assignments and the session record.
        """

        with self.open_connection() as connection:
            cursor = connection.execute(
                """
                DELETE FROM practice_sessions
                WHERE session_id = ?
                """,
                (
                    int(
                        session_id
                    ),
                ),
            )

        if cursor.rowcount == 0:
            raise ValueError(
                f"No practice session was found with session_id {session_id}."
            )

    # =====================================================
    # SHOT ASSIGNMENT
    # =====================================================

    def assign_shots(
        self,
        session_id: int,
        shot_file_ids: list[int],
    ) -> int:
        """
        Assign shot files to a session.

        A shot file can belong to one session at a time. Reassigning a shot
        moves it from its previous session into the selected session.
        """

        if not shot_file_ids:
            return 0

        session = self.get_session(
            session_id
        )

        unique_ids = list(
            dict.fromkeys(
                int(
                    shot_file_id
                )
                for shot_file_id in shot_file_ids
            )
        )

        with self.open_connection() as connection:
            placeholders = ",".join(
                "?"
                for _ in unique_ids
            )

            rows = connection.execute(
                f"""
                SELECT
                    shot_file_id,
                    player_id,
                    active
                FROM shot_files
                WHERE shot_file_id IN (
                    {placeholders}
                )
                """,
                tuple(
                    unique_ids
                ),
            ).fetchall()

            found_ids = {
                int(
                    row[
                        "shot_file_id"
                    ]
                )
                for row in rows
            }

            missing_ids = [
                shot_file_id
                for shot_file_id in unique_ids
                if shot_file_id
                not in found_ids
            ]

            if missing_ids:
                raise ValueError(
                    (
                        "The following shot files were not found: "
                        + ", ".join(
                            str(
                                value
                            )
                            for value in missing_ids
                        )
                    )
                )

            wrong_player_ids = [
                int(
                    row[
                        "shot_file_id"
                    ]
                )
                for row in rows
                if int(
                    row[
                        "player_id"
                    ]
                )
                != session.player_id
            ]

            if wrong_player_ids:
                raise ValueError(
                    (
                        "These shot files belong to another player: "
                        + ", ".join(
                            str(
                                value
                            )
                            for value in wrong_player_ids
                        )
                    )
                )

            inactive_ids = [
                int(
                    row[
                        "shot_file_id"
                    ]
                )
                for row in rows
                if not bool(
                    row[
                        "active"
                    ]
                )
            ]

            if inactive_ids:
                raise ValueError(
                    (
                        "Removed shot files cannot be assigned: "
                        + ", ".join(
                            str(
                                value
                            )
                            for value in inactive_ids
                        )
                    )
                )

            current_max = connection.execute(
                """
                SELECT
                    COALESCE(
                        MAX(
                            display_order
                        ),
                        0
                    )
                FROM practice_session_shots
                WHERE session_id = ?
                """,
                (
                    int(
                        session_id
                    ),
                ),
            ).fetchone()[0]

            next_order = int(
                current_max
            ) + 1

            assigned_count = 0

            for shot_file_id in unique_ids:
                connection.execute(
                    """
                    DELETE FROM practice_session_shots
                    WHERE shot_file_id = ?
                    """,
                    (
                        shot_file_id,
                    ),
                )

                connection.execute(
                    """
                    INSERT INTO practice_session_shots (
                        session_id,
                        shot_file_id,
                        added_at,
                        display_order
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        int(
                            session_id
                        ),
                        shot_file_id,
                        self._utc_now(),
                        next_order,
                    ),
                )

                next_order += 1
                assigned_count += 1

        return assigned_count

    def remove_shots(
        self,
        session_id: int,
        shot_file_ids: list[int],
    ) -> int:
        if not shot_file_ids:
            return 0

        unique_ids = list(
            dict.fromkeys(
                int(
                    shot_file_id
                )
                for shot_file_id in shot_file_ids
            )
        )

        placeholders = ",".join(
            "?"
            for _ in unique_ids
        )

        parameters = [
            int(
                session_id
            ),
            *unique_ids,
        ]

        with self.open_connection() as connection:
            cursor = connection.execute(
                f"""
                DELETE FROM practice_session_shots
                WHERE
                    session_id = ?
                    AND shot_file_id IN (
                        {placeholders}
                    )
                """,
                tuple(
                    parameters
                ),
            )

        return int(
            cursor.rowcount
        )

    def list_session_shots(
        self,
        session_id: int,
    ) -> list[SessionShotRecord]:
        with self.open_connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    pss.session_id,
                    pss.shot_file_id,

                    p.participant_id,
                    p.display_name,

                    sf.trial_id,
                    sf.original_filename,
                    sf.recorded_result,
                    sf.total_frames,

                    pss.added_at,
                    pss.display_order

                FROM practice_session_shots AS pss

                INNER JOIN shot_files AS sf
                    ON sf.shot_file_id = pss.shot_file_id

                INNER JOIN players AS p
                    ON p.player_id = sf.player_id

                WHERE pss.session_id = ?

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

        return [
            SessionShotRecord(
                session_id=int(
                    row[
                        "session_id"
                    ]
                ),
                shot_file_id=int(
                    row[
                        "shot_file_id"
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

                trial_id=str(
                    row[
                        "trial_id"
                    ]
                ),
                original_filename=str(
                    row[
                        "original_filename"
                    ]
                ),
                recorded_result=str(
                    row[
                        "recorded_result"
                    ]
                ),
                total_frames=int(
                    row[
                        "total_frames"
                    ]
                ),

                added_at=str(
                    row[
                        "added_at"
                    ]
                ),
                display_order=int(
                    row[
                        "display_order"
                    ]
                ),
            )
            for row in rows
        ]

    def list_unassigned_shot_ids(
        self,
        player_id: int,
    ) -> list[int]:
        with self.open_connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    sf.shot_file_id

                FROM shot_files AS sf

                LEFT JOIN practice_session_shots AS pss
                    ON pss.shot_file_id = sf.shot_file_id

                WHERE
                    sf.player_id = ?
                    AND sf.active = 1
                    AND pss.shot_file_id IS NULL

                ORDER BY
                    sf.imported_at,
                    sf.trial_id
                """,
                (
                    int(
                        player_id
                    ),
                ),
            ).fetchall()

        return [
            int(
                row[
                    "shot_file_id"
                ]
            )
            for row in rows
        ]

    # =====================================================
    # HELPERS
    # =====================================================

    @staticmethod
    def _session_from_row(
        row: sqlite3.Row,
    ) -> PracticeSessionRecord:
        return PracticeSessionRecord(
            session_id=int(
                row[
                    "session_id"
                ]
            ),
            player_id=int(
                row[
                    "player_id"
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

            session_name=str(
                row[
                    "session_name"
                ]
            ),
            session_date=str(
                row[
                    "session_date"
                ]
            ),
            session_type=str(
                row[
                    "session_type"
                ]
            ),
            location=str(
                row[
                    "location"
                ]
            ),
            notes=str(
                row[
                    "notes"
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

            shot_count=int(
                row[
                    "shot_count"
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
    def _normalize_date(
        value: str | date | None,
    ) -> str:
        if value is None:
            return date.today().isoformat()

        if isinstance(
            value,
            date,
        ):
            return value.isoformat()

        cleaned = str(
            value
        ).strip()

        try:
            return date.fromisoformat(
                cleaned
            ).isoformat()

        except ValueError as error:
            raise ValueError(
                "session_date must use YYYY-MM-DD format."
            ) from error

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


def _run_practice_session_repository_test() -> None:
    """
    Isolated test using a temporary player-history database.
    """

    import json
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

        p1_files = []

        for trial_number in range(
            1,
            4,
        ):
            payload = {
                "participant_id": "P0001",
                "trial_id": (
                    f"T{trial_number:04d}"
                ),
                "result": (
                    "made"
                    if trial_number
                    % 2
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

            p1_files.append(
                (
                    f"p1_t{trial_number}.json",
                    json.dumps(
                        payload
                    ).encode(
                        "utf-8"
                    ),
                )
            )

        p2_payload = {
            "participant_id": "P0002",
            "trial_id": "T0001",
            "result": "made",
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
            p1_files,
            selected_player_id=(
                player_one.player_id
            ),
            enforce_selected_player=True,
        )

        history_repository.import_json_files(
            [
                (
                    "p2_t1.json",
                    json.dumps(
                        p2_payload
                    ).encode(
                        "utf-8"
                    ),
                )
            ],
            selected_player_id=(
                player_two.player_id
            ),
            enforce_selected_player=True,
        )

        session_repository = (
            PracticeSessionRepository(
                database_file=database_file
            )
        )

        session = session_repository.create_session(
            player_id=(
                player_one.player_id
            ),
            session_name=(
                "July 27 Free Throws"
            ),
            session_date="2026-07-27",
            session_type="Free Throws",
            location="Practice Gym",
            notes="Baseline session",
        )

        player_one_shots = (
            history_repository
            .list_shot_files(
                player_id=(
                    player_one.player_id
                )
            )
        )

        assigned = session_repository.assign_shots(
            session_id=(
                session.session_id
            ),
            shot_file_ids=[
                shot.shot_file_id
                for shot in player_one_shots
            ],
        )

        assert assigned == 3

        session_shots = (
            session_repository
            .list_session_shots(
                session.session_id
            )
        )

        assert len(
            session_shots
        ) == 3

        player_two_shots = (
            history_repository
            .list_shot_files(
                player_id=(
                    player_two.player_id
                )
            )
        )

        wrong_player_error = False

        try:
            session_repository.assign_shots(
                session_id=(
                    session.session_id
                ),
                shot_file_ids=[
                    player_two_shots[
                        0
                    ].shot_file_id
                ],
            )

        except ValueError:
            wrong_player_error = True

        assert wrong_player_error

        removed = session_repository.remove_shots(
            session_id=(
                session.session_id
            ),
            shot_file_ids=[
                session_shots[
                    0
                ].shot_file_id
            ],
        )

        assert removed == 1

        assert len(
            session_repository
            .list_session_shots(
                session.session_id
            )
        ) == 2

        assert len(
            session_repository
            .list_unassigned_shot_ids(
                player_one.player_id
            )
        ) == 1

        updated = session_repository.update_session(
            session_id=(
                session.session_id
            ),
            session_name=(
                "Updated Free-Throw Session"
            ),
        )

        assert (
            updated.session_name
            == "Updated Free-Throw Session"
        )

        print(
            "PRACTICE SESSION REPOSITORY TEST PASSED"
        )

        print(
            f"Session created: "
            f"{updated.session_name}"
        )

        print(
            f"Shots originally assigned: "
            f"{assigned}"
        )

        print(
            f"Shots remaining: "
            f"{len(session_repository.list_session_shots(session.session_id))}"
        )

        print(
            "Wrong-player protection test: PASSED"
        )

        print(
            "Remove from session test: PASSED"
        )

        print(
            "Session update test: PASSED"
        )


if __name__ == "__main__":
    _run_practice_session_repository_test()