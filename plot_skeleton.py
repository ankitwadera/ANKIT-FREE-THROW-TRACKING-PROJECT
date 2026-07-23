from pathlib import Path
import json
import math
import sys

import matplotlib.pyplot as plt


# ---------------------------------------------------------
# FIND THE DATA
# ---------------------------------------------------------

project_folder = Path(__file__).resolve().parent

data_folder = (
    project_folder
    / "SPL-Open-Data-main"
    / "basketball"
    / "freethrow"
    / "data"
)

if not data_folder.exists():
    print("ERROR: The free-throw data folder was not found.")
    print(data_folder)
    sys.exit(1)

json_files = sorted(data_folder.rglob("*.json"))

if not json_files:
    print("ERROR: No JSON tracking files were found.")
    sys.exit(1)


# ---------------------------------------------------------
# OPEN THE FIRST FREE THROW
# ---------------------------------------------------------

trial_file = json_files[0]

with trial_file.open("r", encoding="utf-8") as file:
    trial_data = json.load(file)

tracking_frames = trial_data.get("tracking", [])

if not tracking_frames:
    print("ERROR: No tracking frames were found.")
    sys.exit(1)


# ---------------------------------------------------------
# CHECK WHETHER A 3D POINT IS VALID
# ---------------------------------------------------------

def is_valid_point(point):
    if not isinstance(point, list) or len(point) != 3:
        return False

    return all(
        isinstance(value, (int, float))
        and not math.isnan(value)
        for value in point
    )


# ---------------------------------------------------------
# MAJOR BODY CONNECTIONS
# ---------------------------------------------------------

body_connections = [
    ("NOSE", "NECK"),

    ("NECK", "LEFT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),

    ("NECK", "RIGHT_SHOULDER"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),

    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),

    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),

    ("LEFT_HIP", "RIGHT_HIP"),

    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),

    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),

    ("LEFT_ANKLE", "LEFT_HEEL"),
    ("LEFT_ANKLE", "LEFT_BIG_TOE"),

    ("RIGHT_ANKLE", "RIGHT_HEEL"),
    ("RIGHT_ANKLE", "RIGHT_BIG_TOE"),
]


# Create a set containing only the joints used in the skeleton.
major_joints = {
    joint
    for connection in body_connections
    for joint in connection
}


# ---------------------------------------------------------
# FIND A CLEAR FRAME
# ---------------------------------------------------------

selected_frame = None

# Start searching later in the trial instead of using frame 0.
for frame in tracking_frames[30:]:
    frame_data = frame.get("data", {})
    player = frame_data.get("player", {})

    valid_major_joints = sum(
        1
        for joint_name in major_joints
        if is_valid_point(player.get(joint_name))
    )

    if valid_major_joints >= 15:
        selected_frame = frame
        break

if selected_frame is None:
    print("ERROR: No usable body frame was found.")
    sys.exit(1)


# ---------------------------------------------------------
# GET PLAYER AND BALL DATA
# ---------------------------------------------------------

frame_data = selected_frame.get("data", {})
player = frame_data.get("player", {})
ball = frame_data.get("ball")


# ---------------------------------------------------------
# CREATE THE 3D FIGURE
# ---------------------------------------------------------

figure = plt.figure(figsize=(10, 9))
axis = figure.add_subplot(111, projection="3d")


# ---------------------------------------------------------
# DRAW ONLY THE MAJOR BODY JOINTS
# ---------------------------------------------------------

for joint_name in major_joints:
    point = player.get(joint_name)

    if not is_valid_point(point):
        continue

    x, y, z = point
    axis.scatter(x, y, z, s=35)


# ---------------------------------------------------------
# DRAW THE SKELETON LINES
# ---------------------------------------------------------

for joint_a, joint_b in body_connections:
    point_a = player.get(joint_a)
    point_b = player.get(joint_b)

    if not is_valid_point(point_a):
        continue

    if not is_valid_point(point_b):
        continue

    axis.plot(
        [point_a[0], point_b[0]],
        [point_a[1], point_b[1]],
        [point_a[2], point_b[2]],
        linewidth=3,
    )


# ---------------------------------------------------------
# DRAW THE BALL IF VISIBLE
# ---------------------------------------------------------

if is_valid_point(ball):
    axis.scatter(
        ball[0],
        ball[1],
        ball[2],
        s=180,
        marker="o",
        label="Basketball",
    )


# ---------------------------------------------------------
# CENTER THE VIEW AROUND THE PLAYER
# ---------------------------------------------------------

left_hip = player.get("LEFT_HIP")
right_hip = player.get("RIGHT_HIP")

if is_valid_point(left_hip) and is_valid_point(right_hip):
    center_x = (left_hip[0] + right_hip[0]) / 2
    center_y = (left_hip[1] + right_hip[1]) / 2
else:
    center_x = 0
    center_y = 0


# Give the player room around the body.
axis.set_xlim(center_x - 4, center_x + 4)
axis.set_ylim(center_y - 4, center_y + 4)
axis.set_zlim(0, 8)


# Make one foot look like the same distance on every axis.
axis.set_box_aspect((1, 1, 1))


# ---------------------------------------------------------
# LABEL THE PLOT
# ---------------------------------------------------------

participant = trial_data.get("participant_id", "Unknown")
trial = trial_data.get("trial_id", "Unknown")
frame_number = selected_frame.get("frame", "Unknown")
frame_time = selected_frame.get("time", "Unknown")

axis.set_title(
    "3D Free-Throw Skeleton\n"
    f"Participant {participant} | Trial {trial} | Frame {frame_number}"
)

axis.set_xlabel("Court X")
axis.set_ylabel("Court Y")
axis.set_zlabel("Height")

axis.view_init(elev=15, azim=40)

plt.tight_layout()


# ---------------------------------------------------------
# SAVE AND SHOW
# ---------------------------------------------------------

output_file = project_folder / "first_skeleton_fixed.png"

plt.savefig(output_file, dpi=200)

print("SUCCESS: The corrected skeleton was created.")
print(f"Trial file: {trial_file.name}")
print(f"Frame number: {frame_number}")
print(f"Frame time: {frame_time}")
print(f"Image saved here: {output_file}")

plt.show()