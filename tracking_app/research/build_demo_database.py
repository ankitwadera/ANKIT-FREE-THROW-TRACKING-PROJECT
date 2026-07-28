from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from tracking_app.research.organization_team_repository import (
    OrganizationTeamRepository,
)
from tracking_app.research.player_history_repository import (
    PlayerHistoryRepository,
)
from tracking_app.research.practice_session_repository import (
    PracticeSessionRepository,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_FOLDER = (
    PROJECT_ROOT
    / "SPL-Open-Data-main"
    / "basketball"
    / "freethrow"
    / "data"
)

DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "player_history.sqlite3"
)

STORAGE_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "player_history_files"
)

BACKUP_ROOT = (
    PROJECT_ROOT
    / "backups"
    / "demo_database_rebuilds"
)

SESSION_SIZE = 15

ORGANIZATION_NAME = "Toronto Raptors Basketball Operations"
ORGANIZATION_TYPE = "Professional Basketball Organization"
SEASON_NAME = "2026 Demo Environment"

TEAM_RAPTORS_905 = "Raptors 905"
TEAM_TORONTO_RAPTORS = "Toronto Raptors"

SESSION_NAMES = (
    "Baseline Evaluation",
    "Foundation Mechanics",
    "Lower-Body Sequencing",
    "Release Alignment",
    "Rhythm and Timing",
    "Shot Repeatability",
    "Upper-Body Coordination",
    "Fatigue Response",
    "Pressure Free Throws",
    "Consistency Checkpoint",
    "Mechanics Refinement",
    "Performance Validation",
    "Advanced Shot Review",
    "Final Development Review",
    "Latest Performance",
)

PLAYER_PROFILES = {
    "P0001": {
        "display_name": "Chucky Hepburn",
        "team_name": TEAM_RAPTORS_905,
        "jersey_number": "24",
        "position_group": "Guard",
        "roster_role": "Two-Way Player",
    },
    "P0002": {
        "display_name": "Alijah Martin",
        "team_name": TEAM_RAPTORS_905,
        "jersey_number": "55",
        "position_group": "Guard",
        "roster_role": "Two-Way Player",
    },
    "P0003": {
        "display_name": "Scottie Barnes",
        "team_name": TEAM_TORONTO_RAPTORS,
        "jersey_number": "4",
        "position_group": "Forward-Guard",
        "roster_role": "Starter / Core Player",
    },
    "P0004": {
        "display_name": "Ja'Kobe Walter",
        "team_name": TEAM_TORONTO_RAPTORS,
        "jersey_number": "14",
        "position_group": "Guard",
        "roster_role": "Rotation Player",
    },
    "P0005": {
        "display_name": "Collin Murray-Boyles",
        "team_name": TEAM_TORONTO_RAPTORS,
        "jersey_number": "30",
        "position_group": "Forward",
        "roster_role": "Rotation Player",
    },
}


@dataclass(frozen=True)
class SourceTrial:
    path: Path
    participant_id: str
    trial_id: str
    result: str
    source_date: str


def trial_sort_key(trial_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", str(trial_id))
    number = int(match.group(1)) if match else 10**9
    return number, str(trial_id)


def read_source_trials(
    dataset_folder: Path,
) -> dict[str, list[SourceTrial]]:
    json_files = sorted(
        dataset_folder.rglob("BB_FT_*.json")
    )

    if not json_files:
        raise FileNotFoundError(
            f"No BB_FT JSON files were found under: {dataset_folder}"
        )

    grouped: dict[str, list[SourceTrial]] = {}
    failures: list[str] = []

    for json_file in json_files:
        try:
            payload = json.loads(
                json_file.read_text(
                    encoding="utf-8"
                )
            )

            participant_id = str(
                payload.get(
                    "participant_id",
                    "",
                )
            ).strip()

            trial_id = str(
                payload.get(
                    "trial_id",
                    "",
                )
            ).strip()

            result = str(
                payload.get(
                    "result",
                    "unknown",
                )
            ).strip().lower()

            if not participant_id:
                raise ValueError(
                    "participant_id is missing"
                )

            if not trial_id:
                raise ValueError(
                    "trial_id is missing"
                )

            source_date = ""

            for parent in json_file.parents:
                try:
                    source_date = (
                        date.fromisoformat(
                            parent.name
                        ).isoformat()
                    )
                    break
                except ValueError:
                    continue

            grouped.setdefault(
                participant_id,
                [],
            ).append(
                SourceTrial(
                    path=json_file,
                    participant_id=participant_id,
                    trial_id=trial_id,
                    result=result,
                    source_date=source_date,
                )
            )

        except Exception as error:
            failures.append(
                (
                    f"{json_file.name}: "
                    f"{type(error).__name__}: "
                    f"{error}"
                )
            )

    for participant_id in grouped:
        grouped[
            participant_id
        ].sort(
            key=lambda row: (
                row.source_date
                or "9999-12-31",
                trial_sort_key(
                    row.trial_id
                ),
                row.path.name,
            )
        )

    if failures:
        print()
        print(
            "FILES THAT COULD NOT BE READ"
        )
        print("-" * 78)

        for failure in failures[:20]:
            print(failure)

        if len(failures) > 20:
            print(
                (
                    f"... plus "
                    f"{len(failures) - 20} more"
                )
            )

    return grouped


def backup_and_reset_existing_data() -> Path:
    """
    Back up the current database and stored uploaded JSON files, then remove
    them so the rebuilt demo starts with no previous players, sessions,
    organizations, teams, rosters, or uploaded shot records.
    """

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_folder = (
        BACKUP_ROOT
        / f"before_rebuild_{timestamp}"
    )

    backup_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    if DATABASE_FILE.exists():
        shutil.copy2(
            DATABASE_FILE,
            backup_folder
            / DATABASE_FILE.name,
        )

    if STORAGE_FOLDER.exists():
        shutil.copytree(
            STORAGE_FOLDER,
            backup_folder
            / STORAGE_FOLDER.name,
            dirs_exist_ok=True,
        )

    if DATABASE_FILE.exists():
        DATABASE_FILE.unlink()

    if STORAGE_FOLDER.exists():
        shutil.rmtree(
            STORAGE_FOLDER
        )

    DATABASE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    STORAGE_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    return backup_folder


def create_organization_and_teams(
    repository: OrganizationTeamRepository,
):
    organization = (
        repository.create_organization(
            organization_name=(
                ORGANIZATION_NAME
            ),
            organization_type=(
                ORGANIZATION_TYPE
            ),
            description=(
                "Front-office demonstration environment using SPL Open Data. "
                "Player identities are fictional demo names and do not claim "
                "to represent tracking data from actual Raptors athletes."
            ),
        )
    )

    raptors_905 = (
        repository.create_team(
            organization_id=(
                organization.organization_id
            ),
            team_name=(
                TEAM_RAPTORS_905
            ),
            season_name=(
                SEASON_NAME
            ),
            level="G League",
            description=(
                "Development roster used for the free-throw analysis demo."
            ),
        )
    )

    toronto_raptors = (
        repository.create_team(
            organization_id=(
                organization.organization_id
            ),
            team_name=(
                TEAM_TORONTO_RAPTORS
            ),
            season_name=(
                SEASON_NAME
            ),
            level="NBA",
            description=(
                "NBA roster used for the free-throw analysis demo."
            ),
        )
    )

    return (
        organization,
        {
            TEAM_RAPTORS_905: raptors_905,
            TEAM_TORONTO_RAPTORS: toronto_raptors,
        },
    )


def base_session_date(
    source_trials: list[SourceTrial],
) -> date:
    valid_dates = []

    for row in source_trials:
        if not row.source_date:
            continue

        try:
            valid_dates.append(
                date.fromisoformat(
                    row.source_date
                )
            )
        except ValueError:
            continue

    if valid_dates:
        return min(
            valid_dates
        )

    return date(
        2024,
        8,
        28,
    )


def session_name_for_index(
    chunk_index: int,
) -> str:
    if chunk_index < len(
        SESSION_NAMES
    ):
        name = SESSION_NAMES[
            chunk_index
        ]
    else:
        name = (
            f"Extended Development "
            f"{chunk_index + 1}"
        )

    return (
        f"{chunk_index + 1:02d} · "
        f"{name}"
    )


def build_demo_database() -> None:
    if not DATASET_FOLDER.exists():
        raise FileNotFoundError(
            (
                "Dataset folder was not found: "
                f"{DATASET_FOLDER}"
            )
        )

    grouped_trials = read_source_trials(
        DATASET_FOLDER
    )

    missing_profiles = sorted(
        set(grouped_trials)
        - set(PLAYER_PROFILES)
    )

    if missing_profiles:
        raise ValueError(
            (
                "No demo player profiles were configured for: "
                + ", ".join(
                    missing_profiles
                )
            )
        )

    total_source_files = sum(
        len(trials)
        for trials in grouped_trials.values()
    )

    print("=" * 78)
    print(
        "ANKIT'S FREE THROW ANALYSIS SOFTWARE"
    )
    print(
        "AUTHENTIC DEMO DATABASE REBUILDER"
    )
    print("=" * 78)
    print(
        f"Dataset folder: {DATASET_FOLDER}"
    )
    print(
        f"Participants found: {len(grouped_trials)}"
    )
    print(
        f"JSON files found: {total_source_files}"
    )
    print()
    print(
        "WARNING: This rebuild removes all current players, sessions, teams, "
        "organizations, rosters, and uploaded shot records from the active "
        "database after creating a backup."
    )
    print()

    backup_folder = (
        backup_and_reset_existing_data()
    )

    print(
        f"Backup created: {backup_folder}"
    )
    print(
        "Existing active database and stored uploads removed."
    )
    print()

    history = PlayerHistoryRepository(
        database_file=DATABASE_FILE,
        storage_folder=STORAGE_FOLDER,
    )

    sessions = PracticeSessionRepository(
        database_file=DATABASE_FILE
    )

    organizations = (
        OrganizationTeamRepository(
            database_file=DATABASE_FILE
        )
    )

    (
        organization,
        teams,
    ) = create_organization_and_teams(
        organizations
    )

    imported_total = 0
    duplicate_total = 0
    failed_total = 0
    session_total = 0
    assigned_total = 0

    for player_index, participant_id in enumerate(
        sorted(grouped_trials),
        start=1,
    ):
        source_trials = (
            grouped_trials[
                participant_id
            ]
        )

        profile = (
            PLAYER_PROFILES[
                participant_id
            ]
        )

        print("-" * 78)
        print(
            (
                f"PLAYER {player_index}/"
                f"{len(grouped_trials)}: "
                f"{profile['display_name']} "
                f"({participant_id}) · "
                f"{len(source_trials)} files"
            )
        )

        player = history.create_or_get_player(
            participant_id=participant_id,
            display_name=(
                profile[
                    "display_name"
                ]
            ),
        )

        selected_team = teams[
            profile[
                "team_name"
            ]
        ]

        organizations.assign_player_to_team(
            team_id=(
                selected_team.team_id
            ),
            player_id=(
                player.player_id
            ),
            roster_role=(
                profile[
                    "roster_role"
                ]
            ),
            jersey_number=(
                profile[
                    "jersey_number"
                ]
            ),
            position_group=(
                profile[
                    "position_group"
                ]
            ),
        )

        uploads = [
            (
                trial.path.name,
                trial.path.read_bytes(),
            )
            for trial in source_trials
        ]

        import_result = (
            history.import_json_files(
                json_files=uploads,
                selected_player_id=(
                    player.player_id
                ),
                enforce_selected_player=True,
            )
        )

        imported_total += (
            import_result.imported_files
        )
        duplicate_total += (
            import_result.skipped_duplicates
        )
        failed_total += (
            import_result.failed_files
        )

        print(
            (
                f"Team: {profile['team_name']} · "
                f"#{profile['jersey_number']} · "
                f"{profile['position_group']}"
            )
        )

        print(
            (
                f"Imported: "
                f"{import_result.imported_files} · "
                f"Duplicates skipped: "
                f"{import_result.skipped_duplicates} · "
                f"Failed: "
                f"{import_result.failed_files}"
            )
        )

        stored_shots = (
            history.list_shot_files(
                player_id=(
                    player.player_id
                ),
                include_inactive=False,
            )
        )

        source_order = {
            trial.trial_id: index
            for index, trial in enumerate(
                source_trials
            )
        }

        stored_shots.sort(
            key=lambda shot: (
                source_order.get(
                    shot.trial_id,
                    10**9,
                ),
                trial_sort_key(
                    shot.trial_id
                ),
            )
        )

        first_date = base_session_date(
            source_trials
        )

        for chunk_index, start_index in enumerate(
            range(
                0,
                len(stored_shots),
                SESSION_SIZE,
            )
        ):
            shot_chunk = stored_shots[
                start_index:
                start_index + SESSION_SIZE
            ]

            if not shot_chunk:
                continue

            session_name = (
                session_name_for_index(
                    chunk_index
                )
            )

            session_date = (
                first_date
                + timedelta(
                    days=(
                        chunk_index
                        * 7
                    )
                )
            ).isoformat()

            session = (
                sessions.create_session(
                    player_id=(
                        player.player_id
                    ),
                    session_name=(
                        session_name
                    ),
                    session_date=(
                        session_date
                    ),
                    session_type=(
                        "Free Throw Development"
                    ),
                    location=(
                        "OVO Athletic Centre"
                        if profile[
                            "team_name"
                        ]
                        == TEAM_TORONTO_RAPTORS
                        else "Paramount Fine Foods Centre"
                    ),
                    notes=(
                        "Automatically generated demonstration session using "
                        "ordered SPL Open Data trials. Player identity is a "
                        "fictional demo label."
                    ),
                )
            )

            assigned = (
                sessions.assign_shots(
                    session_id=(
                        session.session_id
                    ),
                    shot_file_ids=[
                        shot.shot_file_id
                        for shot in shot_chunk
                    ],
                )
            )

            session_total += 1
            assigned_total += assigned

            print(
                (
                    f"  {session_name}: "
                    f"{assigned} shots · "
                    f"{session_date}"
                )
            )

    print()
    print("=" * 78)
    print(
        "DEMO DATABASE REBUILD COMPLETE"
    )
    print("=" * 78)
    print(
        f"Backup folder: {backup_folder}"
    )
    print(
        f"Organization: {organization.organization_name}"
    )
    print(
        f"Teams: {TEAM_RAPTORS_905}, {TEAM_TORONTO_RAPTORS}"
    )
    print(
        f"Players processed: {len(grouped_trials)}"
    )
    print(
        f"Source JSON files: {total_source_files}"
    )
    print(
        f"New files imported: {imported_total}"
    )
    print(
        f"Duplicates skipped: {duplicate_total}"
    )
    print(
        f"Failed files: {failed_total}"
    )
    print(
        f"Sessions created: {session_total}"
    )
    print(
        f"Shot assignments completed: {assigned_total}"
    )
    print()
    print(
        "Roster summary:"
    )

    for participant_id in sorted(
        PLAYER_PROFILES
    ):
        profile = (
            PLAYER_PROFILES[
                participant_id
            ]
        )

        print(
            (
                f"  {profile['team_name']} · "
                f"#{profile['jersey_number']} · "
                f"{profile['display_name']} · "
                f"{profile['position_group']} · "
                f"{participant_id}"
            )
        )

    print()
    print(
        "Important: The names are fictional demo identities. The tracking data "
        "comes from SPL Open Data and is not represented as data from actual "
        "Toronto Raptors or Raptors 905 players."
    )


if __name__ == "__main__":
    build_demo_database()