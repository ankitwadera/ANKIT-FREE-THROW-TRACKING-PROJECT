from __future__ import annotations

import json
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_open_data_folder() -> Path:
    candidates = [
        PROJECT_ROOT / "basketball" / "freethrow" / "data",
        PROJECT_ROOT / "SPL-Open-Data" / "basketball" / "freethrow" / "data",
        PROJECT_ROOT / "data" / "basketball" / "freethrow",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "SPL Open Data free-throw files were not found. "
        "Place the dataset inside the project before running this script."
    )


def main() -> None:
    data_folder = find_open_data_folder()

    json_files = sorted(
        data_folder.glob("*.json")
    )

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files were found in {data_folder}."
        )

    selected_by_participant: dict[str, list[Path]] = {}

    for json_file in json_files:
        try:
            payload = json.loads(
                json_file.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:
            continue

        participant_id = str(
            payload.get(
                "participant_id",
                "",
            )
        ).strip()

        if not participant_id:
            continue

        selected_by_participant.setdefault(
            participant_id,
            [],
        )

        if len(
            selected_by_participant[
                participant_id
            ]
        ) < 18:
            selected_by_participant[
                participant_id
            ].append(
                json_file
            )

        if (
            len(
                selected_by_participant
            )
            >= 2
            and all(
                len(
                    files
                )
                >= 18
                for files in list(
                    selected_by_participant.values()
                )[
                    :2
                ]
            )
        ):
            break

    selected_participants = list(
        selected_by_participant
    )[
        :2
    ]

    if not selected_participants:
        raise RuntimeError(
            "No usable participant files were found."
        )

    history = PlayerHistoryRepository()
    sessions = PracticeSessionRepository(
        database_file=history.database_file
    )
    organizations = OrganizationTeamRepository(
        database_file=history.database_file
    )

    organization = organizations.create_organization(
        organization_name="Public Demo Organization",
        organization_type="Basketball Analytics Demo",
        notes=(
            "Synthetic showcase organization created from public open-data files."
        ),
    )

    team = organizations.create_team(
        organization_id=organization.organization_id,
        team_name="Public Demo Team",
        season_name="Version 1.0",
        level="Demonstration",
        notes="Read-only public showcase roster.",
    )

    for participant_id in selected_participants:
        player = history.create_or_get_player(
            participant_id=participant_id,
            display_name=f"Demo Player {participant_id}",
        )

        organizations.assign_player_to_team(
            team_id=team.team_id,
            player_id=player.player_id,
            roster_role="Demo Player",
            jersey_number="",
            position_group="",
        )

        selected_files = selected_by_participant[
            participant_id
        ]

        uploads = [
            (
                json_file.name,
                json_file.read_bytes(),
            )
            for json_file in selected_files
        ]

        history.import_json_files(
            uploads,
            selected_player_id=player.player_id,
            enforce_selected_player=True,
        )

        stored_shots = history.list_shot_files(
            player_id=player.player_id,
            include_inactive=False,
        )

        split_index = max(
            1,
            len(
                stored_shots
            )
            // 2,
        )

        session_groups = [
            (
                "Demo Baseline Session",
                stored_shots[
                    :split_index
                ],
            ),
            (
                "Demo Follow-Up Session",
                stored_shots[
                    split_index:
                ],
            ),
        ]

        for session_name, session_shots in session_groups:
            if not session_shots:
                continue

            session = sessions.create_session(
                player_id=player.player_id,
                session_name=session_name,
                session_date="2026-07-27",
                session_type="Public Demonstration",
                location="Demo Environment",
                notes=(
                    "Seeded from SPL Open Data for the public showcase workflow."
                ),
            )

            sessions.assign_shots(
                session_id=session.session_id,
                shot_file_ids=[
                    shot.shot_file_id
                    for shot in session_shots
                ],
            )

    print("PUBLIC DEMO DATASET SEEDED")
    print(f"Organization: {organization.organization_name}")
    print(f"Team: {team.team_name}")
    print(f"Players: {len(selected_participants)}")
    print("Open Streamlit and use Executive Demo or Showcase Mode.")


if __name__ == "__main__":
    main()
