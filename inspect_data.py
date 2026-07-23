from pathlib import Path
import json
import sys


# Find the folder containing this Python file.
project_folder = Path(__file__).resolve().parent


# Build the path to the free-throw data folder.
data_folder = (
    project_folder
    / "SPL-Open-Data-main"
    / "basketball"
    / "freethrow"
    / "data"
)


# Stop safely if the folder is missing.
if not data_folder.exists():
    print("ERROR: The free-throw data folder was not found.")
    print(data_folder)
    sys.exit(1)


# Find all JSON tracking files.
json_files = sorted(data_folder.rglob("*.json"))


# Stop safely if no files were found.
if not json_files:
    print("ERROR: No JSON tracking files were found.")
    sys.exit(1)


# Select the first trial.
first_file = json_files[0]


# Open the trial.
with first_file.open("r", encoding="utf-8") as file:
    trial_data = json.load(file)


# Get the tracking frames.
tracking_data = trial_data.get("tracking")


if not isinstance(tracking_data, list) or not tracking_data:
    print("ERROR: Tracking data is missing or is not a list.")
    sys.exit(1)


# Get the first frame and its data.
first_frame = tracking_data[0]
frame_data = first_frame.get("data", {})

ball_data = frame_data.get("ball")
player_data = frame_data.get("player")


print("BALL AND PLAYER STRUCTURE")
print("=" * 60)

print()
print("File:")
print(first_file.name)

print()
print("BALL DATA TYPE:")
print(type(ball_data).__name__)

if isinstance(ball_data, dict):
    print("Ball fields:")
    for key in ball_data.keys():
        print(f"- {key}")
else:
    print("Ball contents:")
    print(ball_data)


print()
print("PLAYER DATA TYPE:")
print(type(player_data).__name__)

if isinstance(player_data, dict):
    print("Player fields:")
    for key in player_data.keys():
        print(f"- {key}")

elif isinstance(player_data, list):
    print("Number of player items:")
    print(len(player_data))

    if player_data:
        print()
        print("First player item:")
        print(player_data[0])

else:
    print("Player contents:")
    print(player_data)


print()
print("=" * 60)
print("BALL AND PLAYER INSPECTION COMPLETE")