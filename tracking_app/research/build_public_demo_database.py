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

DEPLOYMENT_ROOT = (
    PROJECT_ROOT
    / "deployment_data"
)

DATABASE_FILE = (
    DEPLOYMENT_ROOT
    / "player_history.sqlite3"
)

STORAGE_FOLDER = (
    DEPLOYMENT_ROOT
    / "player_history_files"
)

BACKUP_ROOT = (
    PROJECT_ROOT
    / "backups"
    / "public_demo_rebuilds"
)

ORGANIZATION_NAME = "Toronto Raptors Basketball Operations"
ORGANIZATION_TYPE = "Professional Basketball Organization"
SEASON_NAME = "Public Portfolio Demo"

TEAM_RAPTORS_905 = "Raptors 905"
TEAM_TORONTO_RAPTORS = "Toronto Raptors"

SHOTS_PER_SESSION = 5
SESSIONS_PER_PLAYER = 3
SHOTS_PER_PLAYER = (
    SHOTS_PER_SESSION
    * SESSIONS_PER_PLAYER
)

SESSION_DEFINITIONS = (
    {
        "name": "01 · Baseline Evaluation",
        "type": "Free Throw Evaluation",
        "objective": (
            "Establish the player's starting movement profile and "
            "successful-shot reference."
        ),
    },
    {
        "name": "02 · Mechanics Development",
        "type": "Free Throw Development",
        "objective": (
            "Review lower-body sequencing, release timing, and movement "
            "repeatability."
        ),
    },
    {
        "name": "03 · Latest Performance",
        "type": "Performance Review",
        "objective": (
            "Evaluate the most recent tracked attempts and identify the "
            "next coaching priority."
        ),
    },
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


def trial_sort_key(
    trial_id: str,
) -> tuple[int, str]:
    match = re.search(
        r"(\d+)$",
        str(trial_id),
    )

    number = (
        int(match.group(1))
        if match
        else 10**9
    )

    return (
        number,
        str(trial_id),
    )


def normalize_result(
    value: object,
) -> str:
    normalized = str(value).strip().lower()

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


def read_source_trials(
    dataset_folder: Path,
) -> dict[str, list[SourceTrial]]:
    json_files = sorted(
        dataset_folder.rglob(
            "BB_FT_*.json"
        )
    )

    if not json_files:
        raise FileNotFoundError(
            (
                "No BB_FT JSON files were found under: "
                f"{dataset_folder}"
            )
        )

    grouped: dict[
        str,
        list[SourceTrial],
    ] = {}

    failures: list[str] = []

    for json_file in json_files:
        try:
            payload = json.loads(
                json_file.read_text(
                    encoding="utf-8",
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

            result = normalize_result(
                payload.get(
                    "result",
                    "unknown",
                )
            )

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


def balanced_five_shot_group(
    candidates: list[SourceTrial],
) -> list[SourceTrial]:
    """
    Select five ordered shots while trying to include both makes and misses.

    The output remains deterministic. No random assignment is used.
    """

    if len(candidates) <= SHOTS_PER_SESSION:
        return list(candidates)

    made = [
        row
        for row in candidates
        if row.result == "made"
    ]

    missed = [
        row
        for row in candidates
        if row.result == "missed"
    ]

    selected: list[SourceTrial] = []

    for pool, target_count in (
        (made, 3),
        (missed, 2),
    ):
        if not pool:
            continue

        if len(pool) <= target_count:
            selected.extend(pool)
            continue

        step = (
            len(pool)
            - 1
        ) / max(
            1,
            target_count - 1,
        )

        indexes = {
            round(
                index
                * step
            )
            for index in range(
                target_count
            )
        }

        selected.extend(
            pool[index]
            for index in sorted(
                indexes
            )
        )

    selected_ids = {
        row.trial_id
        for row in selected
    }

    for row in candidates:
        if len(selected) >= SHOTS_PER_SESSION:
            break

        if row.trial_id in selected_ids:
            continue

        selected.append(row)
        selected_ids.add(
            row.trial_id
        )

    selected.sort(
        key=lambda row: (
            row.source_date
            or "9999-12-31",
            trial_sort_key(
                row.trial_id
            ),
        )
    )

    return selected[
        :SHOTS_PER_SESSION
    ]


def select_public_demo_trials(
    source_trials: list[SourceTrial],
) -> list[list[SourceTrial]]:
    """
    Build three deterministic five-shot sessions:

    1. Early trials
    2. Middle trials
    3. Latest trials
    """

    if len(source_trials) < SHOTS_PER_PLAYER:
        raise ValueError(
            (
                f"At least {SHOTS_PER_PLAYER} trials are required, "
                f"but only {len(source_trials)} were found."
            )
        )

    total = len(
        source_trials
    )

    early_window = source_trials[
        :min(
            total,
            20,
        )
    ]

    middle_center = (
        total
        // 2
    )

    middle_start = max(
        0,
        middle_center
        - 10,
    )

    middle_end = min(
        total,
        middle_start
        + 20,
    )

    middle_window = source_trials[
        middle_start:
        middle_end
    ]

    latest_window = source_trials[
        max(
            0,
            total
            - 20,
        ):
    ]

    groups = [
        balanced_five_shot_group(
            early_window
        ),
        balanced_five_shot_group(
            middle_window
        ),
        balanced_five_shot_group(
            latest_window
        ),
    ]

    used_trial_ids: set[str] = set()

    for group_index, group in enumerate(
        groups
    ):
        unique_group: list[
            SourceTrial
        ] = []

        for trial in group:
            if trial.trial_id in used_trial_ids:
                continue

            unique_group.append(
                trial
            )

            used_trial_ids.add(
                trial.trial_id
            )

        if len(unique_group) < SHOTS_PER_SESSION:
            preferred_pool = (
                early_window
                if group_index == 0
                else (
                    middle_window
                    if group_index == 1
                    else latest_window
                )
            )

            fallback_pool = (
                list(preferred_pool)
                + list(source_trials)
            )

            for trial in fallback_pool:
                if len(unique_group) >= SHOTS_PER_SESSION:
                    break

                if trial.trial_id in used_trial_ids:
                    continue

                unique_group.append(
                    trial
                )

                used_trial_ids.add(
                    trial.trial_id
                )

        unique_group.sort(
            key=lambda row: (
                row.source_date
                or "9999-12-31",
                trial_sort_key(
                    row.trial_id
                ),
            )
        )

        groups[
            group_index
        ] = unique_group[
            :SHOTS_PER_SESSION
        ]

    return groups


def backup_and_reset_deployment_data() -> Path:
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

    DEPLOYMENT_ROOT.mkdir(
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
                "Public portfolio demonstration environment using curated "
                "SPL Open Data free-throw trials."
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
                "Curated player-development roster for the public portfolio "
                "demonstration."
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
                "Curated NBA roster for the public portfolio demonstration."
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


def earliest_trial_date(
    trials: list[SourceTrial],
) -> date:
    valid_dates: list[date] = []

    for trial in trials:
        if not trial.source_date:
            continue

        try:
            valid_dates.append(
                date.fromisoformat(
                    trial.source_date
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


def build_public_demo_database() -> None:
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
                "No public-demo player profiles were configured for: "
                + ", ".join(
                    missing_profiles
                )
            )
        )

    print("=" * 78)
    print(
        "ANKIT'S FREE THROW ANALYSIS SOFTWARE"
    )
    print(
        "PUBLIC PORTFOLIO DEMO DATABASE BUILDER"
    )
    print("=" * 78)
    print(
        f"Source dataset: {DATASET_FOLDER}"
    )
    print(
        f"Deployment folder: {DEPLOYMENT_ROOT}"
    )
    print(
        f"Participants found: {len(grouped_trials)}"
    )
    print(
        (
            f"Target: {SHOTS_PER_PLAYER} shots per player · "
            f"{SESSIONS_PER_PLAYER} sessions per player · "
            f"{SHOTS_PER_SESSION} shots per session"
        )
    )
    print()

    backup_folder = (
        backup_and_reset_deployment_data()
    )

    print(
        f"Previous deployment data backup: {backup_folder}"
    )
    print(
        "Your full local data folder was not changed."
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

    total_imported = 0
    total_failed = 0
    total_assigned = 0
    total_sessions = 0

    for player_index, participant_id in enumerate(
        sorted(grouped_trials),
        start=1,
    ):
        profile = (
            PLAYER_PROFILES[
                participant_id
            ]
        )

        source_trials = (
            grouped_trials[
                participant_id
            ]
        )

        selected_groups = (
            select_public_demo_trials(
                source_trials
            )
        )

        selected_trials = [
            trial
            for group in selected_groups
            for trial in group
        ]

        print("-" * 78)
        print(
            (
                f"PLAYER {player_index}/"
                f"{len(grouped_trials)}: "
                f"{profile['display_name']} "
                f"({participant_id})"
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

        upload_files = [
            (
                trial.path.name,
                trial.path.read_bytes(),
            )
            for trial in selected_trials
        ]

        import_result = (
            history.import_json_files(
                json_files=upload_files,
                selected_player_id=(
                    player.player_id
                ),
                enforce_selected_player=True,
            )
        )

        total_imported += (
            import_result.imported_files
        )

        total_failed += (
            import_result.failed_files
        )

        stored_shots = (
            history.list_shot_files(
                player_id=(
                    player.player_id
                ),
                include_inactive=False,
            )
        )

        stored_by_trial = {
            shot.trial_id: shot
            for shot in stored_shots
        }

        first_date = earliest_trial_date(
            source_trials
        )

        for session_index, (
            session_definition,
            selected_group,
        ) in enumerate(
            zip(
                SESSION_DEFINITIONS,
                selected_groups,
            )
        ):
            session_date = (
                first_date
                + timedelta(
                    days=(
                        session_index
                        * 28
                    )
                )
            ).isoformat()

            location = (
                "OVO Athletic Centre"
                if profile[
                    "team_name"
                ]
                == TEAM_TORONTO_RAPTORS
                else "Paramount Fine Foods Centre"
            )

            session = (
                sessions.create_session(
                    player_id=(
                        player.player_id
                    ),
                    session_name=(
                        session_definition[
                            "name"
                        ]
                    ),
                    session_date=(
                        session_date
                    ),
                    session_type=(
                        session_definition[
                            "type"
                        ]
                    ),
                    location=location,
                    notes=(
                        session_definition[
                            "objective"
                        ]
                        + " Public portfolio demo session created from "
                        "curated SPL Open Data trials."
                    ),
                )
            )

            shot_file_ids = []

            for trial in selected_group:
                stored_shot = (
                    stored_by_trial.get(
                        trial.trial_id
                    )
                )

                if stored_shot is None:
                    raise RuntimeError(
                        (
                            "Imported trial was not found in the deployment "
                            f"database: {participant_id} {trial.trial_id}"
                        )
                    )

                shot_file_ids.append(
                    stored_shot.shot_file_id
                )

            assigned = sessions.assign_shots(
                session_id=(
                    session.session_id
                ),
                shot_file_ids=(
                    shot_file_ids
                ),
            )

            total_assigned += assigned
            total_sessions += 1

            made_count = sum(
                1
                for trial in selected_group
                if trial.result == "made"
            )

            missed_count = sum(
                1
                for trial in selected_group
                if trial.result == "missed"
            )

            print(
                (
                    f"  {session_definition['name']}: "
                    f"{assigned} shots · "
                    f"{made_count} made · "
                    f"{missed_count} missed · "
                    f"{session_date}"
                )
            )

    stored_files = list(
        STORAGE_FOLDER.rglob(
            "*.json"
        )
    )

    stored_bytes = sum(
        file_path.stat().st_size
        for file_path in stored_files
    )

    print()
    print("=" * 78)
    print(
        "PUBLIC DEMO DATABASE BUILD COMPLETE"
    )
    print("=" * 78)
    print(
        f"Organization: {organization.organization_name}"
    )
    print(
        f"Teams: {TEAM_RAPTORS_905}, {TEAM_TORONTO_RAPTORS}"
    )
    print(
        f"Players: {len(PLAYER_PROFILES)}"
    )
    print(
        f"Sessions created: {total_sessions}"
    )
    print(
        f"Files imported: {total_imported}"
    )
    print(
        f"Shot assignments: {total_assigned}"
    )
    print(
        f"Failed files: {total_failed}"
    )
    print(
        (
            f"Stored public-demo size: "
            f"{stored_bytes / (1024 * 1024):.2f} MB "
            f"across {len(stored_files)} JSON files"
        )
    )
    print(
        f"Database: {DATABASE_FILE}"
    )
    print(
        f"Stored files: {STORAGE_FOLDER}"
    )
    print()
    print(
        "The full local 583-shot database and local stored files were not "
        "changed."
    )
    print()
    print(
        "Important: Published player names are portfolio labels only. The "
        "underlying biomechanics originate from SPL Open Data and are not "
        "represented as tracking data collected from these named athletes."
    )


if __name__ == "__main__":
    build_public_demo_database()