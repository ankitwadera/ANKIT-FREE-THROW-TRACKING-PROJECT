from __future__ import annotations

import math

import numpy as np

from tracking_app.skeleton import is_valid_point


def calculate_joint_angle(
    point_a: object,
    joint_point: object,
    point_c: object,
) -> float:
    """
    Calculate the angle at the middle joint.

    The three points form:

    point_a  →  joint_point  →  point_c

    Example for the right elbow:

    RIGHT_SHOULDER
          ↓
    RIGHT_ELBOW
          ↓
    RIGHT_WRIST

    The returned angle is measured in degrees.
    """

    if not is_valid_point(point_a):
        raise ValueError("Point A is missing or invalid.")

    if not is_valid_point(joint_point):
        raise ValueError("The middle joint is missing or invalid.")

    if not is_valid_point(point_c):
        raise ValueError("Point C is missing or invalid.")

    point_a_array = np.array(point_a, dtype=float)
    joint_array = np.array(joint_point, dtype=float)
    point_c_array = np.array(point_c, dtype=float)

    # Create two vectors that begin at the middle joint.
    vector_one = point_a_array - joint_array
    vector_two = point_c_array - joint_array

    vector_one_length = float(np.linalg.norm(vector_one))
    vector_two_length = float(np.linalg.norm(vector_two))

    if vector_one_length == 0 or vector_two_length == 0:
        raise ValueError(
            "A joint angle cannot be calculated when two points overlap."
        )

    cosine_angle = float(
        np.dot(vector_one, vector_two)
        / (vector_one_length * vector_two_length)
    )

    # Floating-point calculations can produce values such as
    # 1.00000001, which would break arccos.
    cosine_angle = float(
        np.clip(cosine_angle, -1.0, 1.0)
    )

    angle_radians = math.acos(cosine_angle)

    return math.degrees(angle_radians)


def calculate_player_angles(
    player_data: dict[str, object],
) -> dict[str, float | None]:
    """
    Calculate important basketball biomechanics angles
    for one tracking frame.

    A missing angle is returned as None rather than crashing
    the entire analysis.
    """

    angle_definitions = {
        "right_elbow_angle": (
            "RIGHT_SHOULDER",
            "RIGHT_ELBOW",
            "RIGHT_WRIST",
        ),
        "left_elbow_angle": (
            "LEFT_SHOULDER",
            "LEFT_ELBOW",
            "LEFT_WRIST",
        ),
        "right_knee_angle": (
            "RIGHT_HIP",
            "RIGHT_KNEE",
            "RIGHT_ANKLE",
        ),
        "left_knee_angle": (
            "LEFT_HIP",
            "LEFT_KNEE",
            "LEFT_ANKLE",
        ),
        "right_hip_angle": (
            "RIGHT_SHOULDER",
            "RIGHT_HIP",
            "RIGHT_KNEE",
        ),
        "left_hip_angle": (
            "LEFT_SHOULDER",
            "LEFT_HIP",
            "LEFT_KNEE",
        ),
        "right_shoulder_angle": (
            "RIGHT_ELBOW",
            "RIGHT_SHOULDER",
            "RIGHT_HIP",
        ),
        "left_shoulder_angle": (
            "LEFT_ELBOW",
            "LEFT_SHOULDER",
            "LEFT_HIP",
        ),
    }

    results: dict[str, float | None] = {}

    for angle_name, joint_names in angle_definitions.items():
        point_a_name, joint_name, point_c_name = joint_names

        point_a = player_data.get(point_a_name)
        joint_point = player_data.get(joint_name)
        point_c = player_data.get(point_c_name)

        try:
            results[angle_name] = calculate_joint_angle(
                point_a=point_a,
                joint_point=joint_point,
                point_c=point_c,
            )

        except ValueError:
            results[angle_name] = None

    return results


def calculate_angles_at_frame(
    tracking_frames: list[dict],
    frame_index: int,
) -> dict[str, float | None]:
    """
    Calculate all supported body angles at one selected frame.
    """

    if frame_index < 0 or frame_index >= len(tracking_frames):
        raise IndexError(
            f"Frame index {frame_index} is outside the available range."
        )

    frame = tracking_frames[frame_index]

    player_data = (
        frame
        .get("data", {})
        .get("player", {})
    )

    if not isinstance(player_data, dict) or not player_data:
        raise ValueError(
            "The selected frame contains no usable player data."
        )

    return calculate_player_angles(player_data)