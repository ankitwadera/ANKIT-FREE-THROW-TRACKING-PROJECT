from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "player_history.sqlite3"
)

DEFAULT_FILE_STORAGE_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "player_history_files"
)


@dataclass(frozen=True)
class PlayerRecord:
    player_id: int
    participant_id: str
    display_name: str
    active: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ShotFileRecord:
    shot_file_id: int
    player_id: int

    participant_id: str
    display_name: str

    trial_id: str
    original_filename: str
    stored_filename: str
    stored_path: str

    recorded_result: str
    sampling_rate: float | None
    total_frames: int

    file_hash: str
    imported_at: str
    active: bool


@dataclass(frozen=True)
class BulkImportResult:
    total_files: int
    imported_files: int
    skipped_duplicates: int
    failed_files: int

    imported: tuple[str, ...]
    duplicates: tuple[str, ...]
    failures: tuple[str, ...]


class PlayerHistoryRepository:
    """
    Persistent player-and-shot history manager.

    This is the foundation for an editable history system.

    It keeps players separate from their shot files, so:

    - P0001's files cannot accidentally become P0002's files;
    - many JSON files can be imported at once;
    - files can be added later;
    - files can be removed from history;
    - players can be renamed or deactivated;
    - duplicate uploads can be detected safely.

    SQLite is used because it is built into Python and keeps the project
    portable. The stored JSON files remain ordinary files on disk, while the
    database tracks ownership and metadata.
    """

    def __init__(
        self,
        database_file: Path = DEFAULT_DATABASE_FILE,
        storage_folder: Path = DEFAULT_FILE_STORAGE_FOLDER,
    ) -> None:
        self.database_file = Path(
            database_file
        )

        self.storage_folder = Path(
            storage_folder
        )

        self.database_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.storage_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.initialize_database()

    # =====================================================
    # DATABASE SETUP
    # =====================================================

    def connect(
        self,
    ) -> sqlite3.Connection:
        """
        Create a configured SQLite connection.

        Callers should normally use open_connection(), which guarantees that
        Windows releases the database file handle immediately after use.
        """

        connection = sqlite3.connect(
            self.database_file
        )

        connection.row_factory = (
            sqlite3.Row
        )

        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        return connection

    @contextmanager
    def open_connection(
        self,
    ):
        """
        Open, commit or roll back, and always close a SQLite connection.

        sqlite3.Connection's built-in context manager commits or rolls back,
        but it does not reliably close the file handle immediately. On Windows,
        that can leave a temporary database locked during folder cleanup.
        """

        connection = self.connect()

        try:
            yield connection
            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    def initialize_database(
        self,
    ) -> None:
        with self.open_connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS players (
                    player_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    participant_id TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS shot_files (
                    shot_file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id INTEGER NOT NULL,

                    trial_id TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    stored_filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,

                    recorded_result TEXT NOT NULL,
                    sampling_rate REAL,
                    total_frames INTEGER NOT NULL,

                    file_hash TEXT NOT NULL UNIQUE,
                    imported_at TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,

                    FOREIGN KEY (
                        player_id
                    )
                    REFERENCES players (
                        player_id
                    )
                    ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_shot_files_player_id
                ON shot_files (
                    player_id
                );

                CREATE INDEX IF NOT EXISTS idx_shot_files_trial_id
                ON shot_files (
                    trial_id
                );

                CREATE INDEX IF NOT EXISTS idx_shot_files_active
                ON shot_files (
                    active
                );
                """
            )

    # =====================================================
    # PLAYER MANAGEMENT
    # =====================================================

    def create_or_get_player(
        self,
        participant_id: str,
        display_name: str | None = None,
    ) -> PlayerRecord:
        cleaned_participant_id = self._clean_required_text(
            participant_id,
            "participant_id",
        )

        cleaned_display_name = (
            self._clean_required_text(
                display_name,
                "display_name",
            )
            if display_name is not None
            else cleaned_participant_id
        )

        timestamp = self._utc_now()

        with self.open_connection() as connection:
            existing = connection.execute(
                """
                SELECT
                    player_id,
                    participant_id,
                    display_name,
                    active,
                    created_at,
                    updated_at
                FROM players
                WHERE participant_id = ?
                """,
                (
                    cleaned_participant_id,
                ),
            ).fetchone()

            if existing is not None:
                return self._player_from_row(
                    existing
                )

            cursor = connection.execute(
                """
                INSERT INTO players (
                    participant_id,
                    display_name,
                    active,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, 1, ?, ?)
                """,
                (
                    cleaned_participant_id,
                    cleaned_display_name,
                    timestamp,
                    timestamp,
                ),
            )

            row = connection.execute(
                """
                SELECT
                    player_id,
                    participant_id,
                    display_name,
                    active,
                    created_at,
                    updated_at
                FROM players
                WHERE player_id = ?
                """,
                (
                    cursor.lastrowid,
                ),
            ).fetchone()

        if row is None:
            raise RuntimeError(
                "The player was created but could not be read back."
            )

        return self._player_from_row(
            row
        )

    def list_players(
        self,
        include_inactive: bool = False,
    ) -> list[PlayerRecord]:
        query = """
            SELECT
                player_id,
                participant_id,
                display_name,
                active,
                created_at,
                updated_at
            FROM players
        """

        parameters: tuple[object, ...] = ()

        if not include_inactive:
            query += """
                WHERE active = 1
            """

        query += """
            ORDER BY display_name, participant_id
        """

        with self.open_connection() as connection:
            rows = connection.execute(
                query,
                parameters,
            ).fetchall()

        return [
            self._player_from_row(
                row
            )
            for row in rows
        ]

    def rename_player(
        self,
        player_id: int,
        new_display_name: str,
    ) -> PlayerRecord:
        cleaned_name = self._clean_required_text(
            new_display_name,
            "new_display_name",
        )

        timestamp = self._utc_now()

        with self.open_connection() as connection:
            connection.execute(
                """
                UPDATE players
                SET
                    display_name = ?,
                    updated_at = ?
                WHERE player_id = ?
                """,
                (
                    cleaned_name,
                    timestamp,
                    int(
                        player_id
                    ),
                ),
            )

            row = connection.execute(
                """
                SELECT
                    player_id,
                    participant_id,
                    display_name,
                    active,
                    created_at,
                    updated_at
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

        return self._player_from_row(
            row
        )

    def set_player_active(
        self,
        player_id: int,
        active: bool,
    ) -> None:
        timestamp = self._utc_now()

        with self.open_connection() as connection:
            cursor = connection.execute(
                """
                UPDATE players
                SET
                    active = ?,
                    updated_at = ?
                WHERE player_id = ?
                """,
                (
                    1
                    if active
                    else 0,
                    timestamp,
                    int(
                        player_id
                    ),
                ),
            )

        if cursor.rowcount == 0:
            raise ValueError(
                f"No player was found with player_id {player_id}."
            )

    # =====================================================
    # JSON IMPORT
    # =====================================================

    def import_json_files(
        self,
        json_files: Iterable[tuple[str, bytes]],
        selected_player_id: int | None = None,
        enforce_selected_player: bool = True,
    ) -> BulkImportResult:
        """
        Import many JSON files in one operation.

        Parameters
        ----------
        json_files:
            Iterable of (filename, raw_bytes).

        selected_player_id:
            Optional player selected in the application.

        enforce_selected_player:
            When True, every JSON participant_id must match the selected
            player's participant_id. This prevents Player 1's files from being
            assigned to Player 2 accidentally.
        """

        files = list(
            json_files
        )

        imported: list[
            str
        ] = []

        duplicates: list[
            str
        ] = []

        failures: list[
            str
        ] = []

        selected_player = (
            self.get_player(
                selected_player_id
            )
            if selected_player_id
            is not None
            else None
        )

        for filename, raw_bytes in files:
            try:
                parsed = json.loads(
                    raw_bytes
                )

                if not isinstance(
                    parsed,
                    dict,
                ):
                    raise ValueError(
                        "The JSON root must be an object."
                    )

                participant_id = self._clean_required_text(
                    parsed.get(
                        "participant_id"
                    ),
                    "participant_id",
                )

                trial_id = self._clean_required_text(
                    parsed.get(
                        "trial_id"
                    ),
                    "trial_id",
                )

                tracking = parsed.get(
                    "tracking"
                )

                if not isinstance(
                    tracking,
                    list,
                ):
                    raise ValueError(
                        "The JSON does not contain a valid tracking list."
                    )

                if (
                    selected_player is not None
                    and enforce_selected_player
                    and participant_id
                    != selected_player.participant_id
                ):
                    raise ValueError(
                        (
                            f"File participant {participant_id} does not match "
                            f"selected player {selected_player.participant_id}."
                        )
                    )

                player = (
                    selected_player
                    if selected_player
                    is not None
                    else self.create_or_get_player(
                        participant_id=(
                            participant_id
                        ),
                        display_name=(
                            participant_id
                        ),
                    )
                )

                file_hash = hashlib.sha256(
                    raw_bytes
                ).hexdigest()

                existing_file = self.get_shot_file_by_hash(
                    file_hash
                )

                player_folder = (
                    self.storage_folder
                    / player.participant_id
                )

                player_folder.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                safe_filename = self._safe_filename(
                    filename
                )

                stored_filename = (
                    f"{trial_id}__"
                    f"{file_hash[:12]}__"
                    f"{safe_filename}"
                )

                stored_path = (
                    player_folder
                    / stored_filename
                )

                if existing_file is not None:
                    existing_path = Path(
                        existing_file.stored_path
                    )

                    if (
                        existing_file.active
                        and existing_path.exists()
                    ):
                        duplicates.append(
                            filename
                        )

                        continue

                    # The file was previously removed from active history,
                    # or its physical JSON file is missing. Re-uploading must
                    # restore the existing record instead of being rejected as
                    # a duplicate.
                    stored_path.write_bytes(
                        raw_bytes
                    )

                    self.restore_or_repair_shot_file(
                        shot_file_id=(
                            existing_file.shot_file_id
                        ),
                        player_id=player.player_id,
                        trial_id=trial_id,
                        original_filename=filename,
                        stored_filename=stored_filename,
                        stored_path=stored_path,
                        recorded_result=(
                            self._normalize_result(
                                parsed.get(
                                    "result"
                                )
                            )
                        ),
                        sampling_rate=(
                            self._optional_float(
                                parsed.get(
                                    "sampling_rate"
                                )
                            )
                        ),
                        total_frames=len(
                            tracking
                        ),
                    )

                    if (
                        existing_path.resolve()
                        != stored_path.resolve()
                    ):
                        existing_path.unlink(
                            missing_ok=True
                        )

                    imported.append(
                        filename
                    )

                    continue

                stored_path.write_bytes(
                    raw_bytes
                )

                try:
                    self._insert_shot_file(
                        player_id=(
                            player.player_id
                        ),
                        trial_id=trial_id,
                        original_filename=filename,
                        stored_filename=(
                            stored_filename
                        ),
                        stored_path=stored_path,
                        recorded_result=(
                            self._normalize_result(
                                parsed.get(
                                    "result"
                                )
                            )
                        ),
                        sampling_rate=(
                            self._optional_float(
                                parsed.get(
                                    "sampling_rate"
                                )
                            )
                        ),
                        total_frames=len(
                            tracking
                        ),
                        file_hash=file_hash,
                    )

                except Exception:
                    stored_path.unlink(
                        missing_ok=True
                    )

                    raise

                imported.append(
                    filename
                )

            except Exception as error:
                failures.append(
                    (
                        f"{filename}: "
                        f"{type(error).__name__}: "
                        f"{error}"
                    )
                )

        return BulkImportResult(
            total_files=len(
                files
            ),
            imported_files=len(
                imported
            ),
            skipped_duplicates=len(
                duplicates
            ),
            failed_files=len(
                failures
            ),

            imported=tuple(
                imported
            ),
            duplicates=tuple(
                duplicates
            ),
            failures=tuple(
                failures
            ),
        )

    def _insert_shot_file(
        self,
        player_id: int,
        trial_id: str,
        original_filename: str,
        stored_filename: str,
        stored_path: Path,
        recorded_result: str,
        sampling_rate: float | None,
        total_frames: int,
        file_hash: str,
    ) -> None:
        with self.open_connection() as connection:
            connection.execute(
                """
                INSERT INTO shot_files (
                    player_id,
                    trial_id,
                    original_filename,
                    stored_filename,
                    stored_path,
                    recorded_result,
                    sampling_rate,
                    total_frames,
                    file_hash,
                    imported_at,
                    active
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    int(
                        player_id
                    ),
                    trial_id,
                    original_filename,
                    stored_filename,
                    str(
                        stored_path.resolve()
                    ),
                    recorded_result,
                    sampling_rate,
                    int(
                        total_frames
                    ),
                    file_hash,
                    self._utc_now(),
                ),
            )

    # =====================================================
    # SHOT HISTORY MANAGEMENT
    # =====================================================

    def list_shot_files(
        self,
        player_id: int | None = None,
        include_inactive: bool = False,
    ) -> list[ShotFileRecord]:
        query = """
            SELECT
                sf.shot_file_id,
                sf.player_id,

                p.participant_id,
                p.display_name,

                sf.trial_id,
                sf.original_filename,
                sf.stored_filename,
                sf.stored_path,

                sf.recorded_result,
                sf.sampling_rate,
                sf.total_frames,

                sf.file_hash,
                sf.imported_at,
                sf.active

            FROM shot_files AS sf

            INNER JOIN players AS p
                ON p.player_id = sf.player_id
        """

        conditions: list[
            str
        ] = []

        parameters: list[
            object
        ] = []

        if player_id is not None:
            conditions.append(
                "sf.player_id = ?"
            )

            parameters.append(
                int(
                    player_id
                )
            )

        if not include_inactive:
            conditions.append(
                "sf.active = 1"
            )

        if conditions:
            query += (
                " WHERE "
                + " AND ".join(
                    conditions
                )
            )

        query += """
            ORDER BY
                p.display_name,
                sf.imported_at,
                sf.trial_id
        """

        with self.open_connection() as connection:
            rows = connection.execute(
                query,
                tuple(
                    parameters
                ),
            ).fetchall()

        return [
            self._shot_file_from_row(
                row
            )
            for row in rows
        ]

    def get_player(
        self,
        player_id: int,
    ) -> PlayerRecord:
        with self.open_connection() as connection:
            row = connection.execute(
                """
                SELECT
                    player_id,
                    participant_id,
                    display_name,
                    active,
                    created_at,
                    updated_at
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

        return self._player_from_row(
            row
        )

    def get_shot_file_by_hash(
        self,
        file_hash: str,
    ) -> ShotFileRecord | None:
        """
        Return the existing database record for a file hash.

        Inactive records are included so a re-upload can restore a previously
        removed shot instead of incorrectly rejecting it as a duplicate.
        """

        with self.open_connection() as connection:
            row = connection.execute(
                """
                SELECT
                    sf.shot_file_id,
                    sf.player_id,

                    p.participant_id,
                    p.display_name,

                    sf.trial_id,
                    sf.original_filename,
                    sf.stored_filename,
                    sf.stored_path,

                    sf.recorded_result,
                    sf.sampling_rate,
                    sf.total_frames,

                    sf.file_hash,
                    sf.imported_at,
                    sf.active

                FROM shot_files AS sf

                INNER JOIN players AS p
                    ON p.player_id = sf.player_id

                WHERE sf.file_hash = ?
                """,
                (
                    file_hash,
                ),
            ).fetchone()

        if row is None:
            return None

        return self._shot_file_from_row(
            row
        )

    def restore_or_repair_shot_file(
        self,
        shot_file_id: int,
        player_id: int,
        trial_id: str,
        original_filename: str,
        stored_filename: str,
        stored_path: Path,
        recorded_result: str,
        sampling_rate: float | None,
        total_frames: int,
    ) -> None:
        """
        Reactivate a soft-deleted record and refresh its stored-file metadata.

        This also repairs active database records whose physical JSON file was
        manually deleted or moved.
        """

        with self.open_connection() as connection:
            cursor = connection.execute(
                """
                UPDATE shot_files
                SET
                    player_id = ?,
                    trial_id = ?,
                    original_filename = ?,
                    stored_filename = ?,
                    stored_path = ?,
                    recorded_result = ?,
                    sampling_rate = ?,
                    total_frames = ?,
                    imported_at = ?,
                    active = 1
                WHERE shot_file_id = ?
                """,
                (
                    int(
                        player_id
                    ),
                    trial_id,
                    original_filename,
                    stored_filename,
                    str(
                        stored_path.resolve()
                    ),
                    recorded_result,
                    sampling_rate,
                    int(
                        total_frames
                    ),
                    self._utc_now(),
                    int(
                        shot_file_id
                    ),
                ),
            )

        if cursor.rowcount == 0:
            raise ValueError(
                (
                    "No shot file was found with "
                    f"shot_file_id {shot_file_id}."
                )
            )

    def file_hash_exists(
        self,
        file_hash: str,
    ) -> bool:
        with self.open_connection() as connection:
            row = connection.execute(
                """
                SELECT
                    shot_file_id
                FROM shot_files
                WHERE file_hash = ?
                """,
                (
                    file_hash,
                ),
            ).fetchone()

        return row is not None

    def remove_shot_file(
        self,
        shot_file_id: int,
        delete_physical_file: bool = False,
    ) -> None:
        """
        Remove a shot from active history.

        By default, this is a soft delete. The database record and JSON file
        remain recoverable. Set delete_physical_file=True only when permanent
        file deletion is intended.
        """

        with self.open_connection() as connection:
            row = connection.execute(
                """
                SELECT
                    stored_path
                FROM shot_files
                WHERE shot_file_id = ?
                """,
                (
                    int(
                        shot_file_id
                    ),
                ),
            ).fetchone()

            if row is None:
                raise ValueError(
                    (
                        "No shot file was found with "
                        f"shot_file_id {shot_file_id}."
                    )
                )

            if delete_physical_file:
                stored_path = Path(
                    row[
                        "stored_path"
                    ]
                )

                stored_path.unlink(
                    missing_ok=True
                )

                connection.execute(
                    """
                    DELETE FROM shot_files
                    WHERE shot_file_id = ?
                    """,
                    (
                        int(
                            shot_file_id
                        ),
                    ),
                )

            else:
                connection.execute(
                    """
                    UPDATE shot_files
                    SET active = 0
                    WHERE shot_file_id = ?
                    """,
                    (
                        int(
                            shot_file_id
                        ),
                    ),
                )

    def restore_shot_file(
        self,
        shot_file_id: int,
    ) -> None:
        with self.open_connection() as connection:
            cursor = connection.execute(
                """
                UPDATE shot_files
                SET active = 1
                WHERE shot_file_id = ?
                """,
                (
                    int(
                        shot_file_id
                    ),
                ),
            )

        if cursor.rowcount == 0:
            raise ValueError(
                (
                    "No shot file was found with "
                    f"shot_file_id {shot_file_id}."
                )
            )

    def move_shot_file(
        self,
        shot_file_id: int,
        destination_player_id: int,
    ) -> ShotFileRecord:
        """
        Move an accidentally assigned file to the correct player.

        The JSON participant_id is checked before the move.
        """

        destination_player = self.get_player(
            destination_player_id
        )

        with self.open_connection() as connection:
            row = connection.execute(
                """
                SELECT
                    stored_path
                FROM shot_files
                WHERE shot_file_id = ?
                """,
                (
                    int(
                        shot_file_id
                    ),
                ),
            ).fetchone()

        if row is None:
            raise ValueError(
                (
                    "No shot file was found with "
                    f"shot_file_id {shot_file_id}."
                )
            )

        current_path = Path(
            row[
                "stored_path"
            ]
        )

        parsed = json.loads(
            current_path.read_bytes()
        )

        file_participant_id = self._clean_required_text(
            parsed.get(
                "participant_id"
            ),
            "participant_id",
        )

        if (
            file_participant_id
            != destination_player.participant_id
        ):
            raise ValueError(
                (
                    f"The JSON belongs to {file_participant_id}, "
                    f"not {destination_player.participant_id}."
                )
            )

        destination_folder = (
            self.storage_folder
            / destination_player.participant_id
        )

        destination_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination_path = (
            destination_folder
            / current_path.name
        )

        if (
            current_path.resolve()
            != destination_path.resolve()
        ):
            destination_path.write_bytes(
                current_path.read_bytes()
            )

            current_path.unlink(
                missing_ok=True
            )

        with self.open_connection() as connection:
            connection.execute(
                """
                UPDATE shot_files
                SET
                    player_id = ?,
                    stored_path = ?
                WHERE shot_file_id = ?
                """,
                (
                    destination_player.player_id,
                    str(
                        destination_path.resolve()
                    ),
                    int(
                        shot_file_id
                    ),
                ),
            )

        return self.get_shot_file(
            shot_file_id
        )

    def get_shot_file(
        self,
        shot_file_id: int,
    ) -> ShotFileRecord:
        with self.open_connection() as connection:
            row = connection.execute(
                """
                SELECT
                    sf.shot_file_id,
                    sf.player_id,

                    p.participant_id,
                    p.display_name,

                    sf.trial_id,
                    sf.original_filename,
                    sf.stored_filename,
                    sf.stored_path,

                    sf.recorded_result,
                    sf.sampling_rate,
                    sf.total_frames,

                    sf.file_hash,
                    sf.imported_at,
                    sf.active

                FROM shot_files AS sf

                INNER JOIN players AS p
                    ON p.player_id = sf.player_id

                WHERE sf.shot_file_id = ?
                """,
                (
                    int(
                        shot_file_id
                    ),
                ),
            ).fetchone()

        if row is None:
            raise ValueError(
                (
                    "No shot file was found with "
                    f"shot_file_id {shot_file_id}."
                )
            )

        return self._shot_file_from_row(
            row
        )

    # =====================================================
    # HELPERS
    # =====================================================

    @staticmethod
    def _player_from_row(
        row: sqlite3.Row,
    ) -> PlayerRecord:
        return PlayerRecord(
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
            display_name=str(
                row[
                    "display_name"
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
        )

    @staticmethod
    def _shot_file_from_row(
        row: sqlite3.Row,
    ) -> ShotFileRecord:
        sampling_rate_value = row[
            "sampling_rate"
        ]

        return ShotFileRecord(
            shot_file_id=int(
                row[
                    "shot_file_id"
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
            display_name=str(
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
            stored_filename=str(
                row[
                    "stored_filename"
                ]
            ),
            stored_path=str(
                row[
                    "stored_path"
                ]
            ),

            recorded_result=str(
                row[
                    "recorded_result"
                ]
            ),
            sampling_rate=(
                None
                if sampling_rate_value
                is None
                else float(
                    sampling_rate_value
                )
            ),
            total_frames=int(
                row[
                    "total_frames"
                ]
            ),

            file_hash=str(
                row[
                    "file_hash"
                ]
            ),
            imported_at=str(
                row[
                    "imported_at"
                ]
            ),
            active=bool(
                row[
                    "active"
                ]
            ),
        )

    @staticmethod
    def _clean_required_text(
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
    def _safe_filename(
        filename: str,
    ) -> str:
        cleaned = "".join(
            character
            if (
                character.isalnum()
                or character
                in {
                    ".",
                    "-",
                    "_",
                }
            )
            else "_"
            for character in filename
        )

        return (
            cleaned
            or "shot.json"
        )

    @staticmethod
    def _normalize_result(
        value: object,
    ) -> str:
        normalized = str(
            value
            if value is not None
            else ""
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

    @staticmethod
    def _optional_float(
        value: object,
    ) -> float | None:
        if value is None:
            return None

        try:
            return float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            return None

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


def _run_player_history_repository_test() -> None:
    """
    Run an isolated test without touching the real player-history database.
    """

    import tempfile

    with tempfile.TemporaryDirectory() as temporary_folder:
        root = Path(
            temporary_folder
        )

        repository = PlayerHistoryRepository(
            database_file=(
                root
                / "test_history.sqlite3"
            ),
            storage_folder=(
                root
                / "files"
            ),
        )

        player_one = repository.create_or_get_player(
            participant_id="P0001",
            display_name="Player One",
        )

        player_two = repository.create_or_get_player(
            participant_id="P0002",
            display_name="Player Two",
        )

        player_one_json = {
            "participant_id": "P0001",
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

        player_two_json = {
            "participant_id": "P0002",
            "trial_id": "T0001",
            "result": "missed",
            "sampling_rate": 30,
            "tracking": [
                {
                    "frame": 0,
                    "time": 0,
                    "data": {},
                }
            ],
        }

        import_result = repository.import_json_files(
            json_files=[
                (
                    "p1_t1.json",
                    json.dumps(
                        player_one_json
                    ).encode(
                        "utf-8"
                    ),
                ),
                (
                    "p2_t1.json",
                    json.dumps(
                        player_two_json
                    ).encode(
                        "utf-8"
                    ),
                ),
            ],
            selected_player_id=None,
            enforce_selected_player=True,
        )

        assert import_result.imported_files == 2
        assert import_result.failed_files == 0

        p1_files = repository.list_shot_files(
            player_id=(
                player_one.player_id
            )
        )

        p2_files = repository.list_shot_files(
            player_id=(
                player_two.player_id
            )
        )

        assert len(
            p1_files
        ) == 1

        assert len(
            p2_files
        ) == 1

        mismatch_result = repository.import_json_files(
            json_files=[
                (
                    "wrong_player.json",
                    json.dumps(
                        player_two_json
                    ).encode(
                        "utf-8"
                    ),
                )
            ],
            selected_player_id=(
                player_one.player_id
            ),
            enforce_selected_player=True,
        )

        assert mismatch_result.imported_files == 0
        assert mismatch_result.failed_files == 1

        repository.remove_shot_file(
            p1_files[
                0
            ].shot_file_id
        )

        assert len(
            repository.list_shot_files(
                player_id=(
                    player_one.player_id
                )
            )
        ) == 0

        reupload_result = repository.import_json_files(
            json_files=[
                (
                    "p1_t1_reuploaded.json",
                    json.dumps(
                        player_one_json
                    ).encode(
                        "utf-8"
                    ),
                )
            ],
            selected_player_id=(
                player_one.player_id
            ),
            enforce_selected_player=True,
        )

        assert reupload_result.imported_files == 1
        assert reupload_result.skipped_duplicates == 0

        restored_files = repository.list_shot_files(
            player_id=(
                player_one.player_id
            )
        )

        assert len(
            restored_files
        ) == 1

        assert restored_files[
            0
        ].active

        print(
            "PLAYER HISTORY REPOSITORY TEST PASSED"
        )

        print(
            f"Players created: "
            f"{len(repository.list_players())}"
        )

        print(
            f"Files imported: "
            f"{import_result.imported_files}"
        )

        print(
            "Player isolation test: PASSED"
        )

        print(
            "Wrong-player protection test: PASSED"
        )

        print(
            "Remove and re-upload restoration test: PASSED"
        )


if __name__ == "__main__":
    _run_player_history_repository_test()