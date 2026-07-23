from __future__ import annotations

import math
from typing import Iterable


# Each pair tells the viewer which two body joints to connect.
SKELETON_CONNECTIONS: list[tuple[str, str]] = [
    # Head
    ("RIGHT_EYE", "LEFT_EYE"),
    ("RIGHT_EYE", "NOSE"),
    ("LEFT_EYE", "NOSE"),
    ("RIGHT_EYE", "RIGHT_EAR"),
    ("LEFT_EYE", "LEFT_EAR"),
    ("NOSE", "NECK"),

    # Shoulders and arms
    ("RIGHT_SHOULDER", "LEFT_SHOULDER"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),

    # Torso
    ("NECK", "MID_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_HIP", "LEFT_HIP"),

    # Right leg
    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),

    # Left leg
    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),

    # Right foot
    ("RIGHT_ANKLE", "RIGHT_BIG_TOE"),
    ("RIGHT_ANKLE", "RIGHT_SMALL_TOE"),
    ("RIGHT_ANKLE", "RIGHT_HEEL"),
    ("RIGHT_BIG_TOE", "RIGHT_SMALL_TOE"),

    # Left foot
    ("LEFT_ANKLE", "LEFT_BIG_TOE"),
    ("LEFT_ANKLE", "LEFT_SMALL_TOE"),
    ("LEFT_ANKLE", "LEFT_HEEL"),
    ("LEFT_BIG_TOE", "LEFT_SMALL_TOE"),

    # Simplified hands
    ("RIGHT_WRIST", "RIGHT_THUMB"),
    ("RIGHT_WRIST", "RIGHT_PINKY"),
    ("LEFT_WRIST", "LEFT_THUMB"),
    ("LEFT_WRIST", "LEFT_PINKY"),
]


def is_valid_point(point: object) -> bool:
    """
    Check whether a body point contains three usable coordinates:
    [X, Y, Z]
    """

    if not isinstance(point, list) or len(point) != 3:
        return False

    return all(
        isinstance(value, (int, float))
        and not math.isnan(value)
        for value in point
    )


def get_usable_connections(
    available_joint_names: Iterable[str],
) -> list[tuple[str, str]]:
    """
    Keep only connections where both joints exist in the trial.
    """

    available = set(available_joint_names)

    return [
        connection
        for connection in SKELETON_CONNECTIONS
        if connection[0] in available
        and connection[1] in available
    ]


def count_valid_joints(player_data: dict[str, object]) -> int:
    """
    Count the usable body joints in one tracking frame.
    """

    return sum(
        1
        for point in player_data.values()
        if is_valid_point(point)
    )