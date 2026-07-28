from __future__ import annotations

import math

import numpy as np

from tracking_app.skeleton import is_valid_point


def point_to_array(point: object) -> np.ndarray:
    """
    Convert a valid [X, Y, Z] tracking point into a NumPy array.

    Example:
    [28.1, -0.7, 7.3]
    """

    if not is_valid_point(point):
        raise ValueError(
            "The tracking point must contain three valid numbers."
        )

    return np.array(point, dtype=float)


def calculate_velocity_vector(
    previous_point: object,
    current_point: object,
    sampling_rate: float,
) -> np.ndarray:
    """
    Calculate three-dimensional velocity.

    The returned values represent:

    [X velocity, Y velocity, Z velocity]

    Units:
    feet per second
    """

    if not isinstance(sampling_rate, (int, float)):
        raise TypeError("Sampling rate must be a number.")

    if sampling_rate <= 0:
        raise ValueError(
            "Sampling rate must be greater than zero."
        )

    previous_array = point_to_array(previous_point)
    current_array = point_to_array(current_point)

    position_change = current_array - previous_array

    return position_change * sampling_rate


def calculate_speed(
    velocity_vector: np.ndarray,
) -> float:
    """
    Calculate total three-dimensional speed.

    Speed does not include direction. It tells us only how
    quickly the object is moving.
    """

    if not isinstance(velocity_vector, np.ndarray):
        raise TypeError(
            "Velocity vector must be a NumPy array."
        )

    return float(
        np.linalg.norm(velocity_vector)
    )


def calculate_horizontal_speed(
    velocity_vector: np.ndarray,
) -> float:
    """
    Calculate movement speed parallel to the court floor.

    This uses only X and Y movement.
    """

    x_velocity = float(velocity_vector[0])
    y_velocity = float(velocity_vector[1])

    return math.sqrt(
        x_velocity**2 + y_velocity**2
    )


def calculate_vertical_speed(
    velocity_vector: np.ndarray,
) -> float:
    """
    Return vertical velocity.

    Positive:
        moving upward

    Negative:
        moving downward
    """

    return float(velocity_vector[2])


def calculate_frame_velocity(
    tracking_frames: list[dict],
    frame_index: int,
    object_name: str,
    sampling_rate: float,
    joint_name: str | None = None,
) -> dict | None:
    """
    Calculate velocity for either the basketball or a body joint.

    Examples
    --------
    Basketball:

    calculate_frame_velocity(
        tracking_frames,
        frame_index=172,
        object_name="ball",
        sampling_rate=30,
    )

    Right wrist:

    calculate_frame_velocity(
        tracking_frames,
        frame_index=172,
        object_name="player",
        joint_name="RIGHT_WRIST",
        sampling_rate=30,
    )
    """

    if frame_index <= 0:
        return None

    if frame_index >= len(tracking_frames):
        return None

    previous_frame = tracking_frames[frame_index - 1]
    current_frame = tracking_frames[frame_index]

    previous_data = previous_frame.get("data", {})
    current_data = current_frame.get("data", {})

    if object_name == "ball":
        previous_point = previous_data.get("ball")
        current_point = current_data.get("ball")

    elif object_name == "player":
        if not joint_name:
            raise ValueError(
                "joint_name is required when object_name is 'player'."
            )

        previous_player = previous_data.get("player", {})
        current_player = current_data.get("player", {})

        previous_point = previous_player.get(joint_name)
        current_point = current_player.get(joint_name)

    else:
        raise ValueError(
            "object_name must be either 'ball' or 'player'."
        )

    if (
        not is_valid_point(previous_point)
        or not is_valid_point(current_point)
    ):
        return None

    velocity_vector = calculate_velocity_vector(
        previous_point=previous_point,
        current_point=current_point,
        sampling_rate=sampling_rate,
    )

    return {
        "frame_index": frame_index,
        "velocity_x": float(velocity_vector[0]),
        "velocity_y": float(velocity_vector[1]),
        "velocity_z": float(velocity_vector[2]),
        "speed": calculate_speed(velocity_vector),
        "horizontal_speed": calculate_horizontal_speed(
            velocity_vector
        ),
        "vertical_speed": calculate_vertical_speed(
            velocity_vector
        ),
    }