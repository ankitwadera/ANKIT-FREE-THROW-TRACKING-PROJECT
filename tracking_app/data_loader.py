from pathlib import Path
import json


def get_project_folder() -> Path:
    """
    Return the main ankit_tracking_project folder.
    """

    return Path(__file__).resolve().parent.parent


def get_data_folder() -> Path:
    """
    Return the folder containing the SPL free-throw JSON files.
    """

    return (
        get_project_folder()
        / "SPL-Open-Data-main"
        / "basketball"
        / "freethrow"
        / "data"
    )


def find_trial_files() -> list[Path]:
    """
    Find every free-throw JSON file in the dataset.
    """

    data_folder = get_data_folder()

    if not data_folder.exists():
        raise FileNotFoundError(
            "The SPL free-throw data folder was not found.\n"
            f"Python looked here:\n{data_folder}"
        )

    trial_files = sorted(data_folder.rglob("*.json"))

    if not trial_files:
        raise FileNotFoundError(
            "The data folder exists, but no JSON trial files were found."
        )

    return trial_files


def load_trial(trial_file: Path) -> dict:
    """
    Open one free-throw JSON file and return its contents.
    """

    if not trial_file.exists():
        raise FileNotFoundError(
            f"The requested trial file does not exist:\n{trial_file}"
        )

    with trial_file.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_first_trial() -> tuple[Path, dict]:
    """
    Find and open the first free throw in the dataset.
    """

    trial_files = find_trial_files()
    first_trial_file = trial_files[0]
    trial_data = load_trial(first_trial_file)

    return first_trial_file, trial_data