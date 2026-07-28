from __future__ import annotations

import math

import numpy as np

from tracking_app.skeleton import is_valid_point


def distance_between(
    point_a: list[float],
    point_b: list[float],
) -> float:
    """
    Calculate the straight-line distance between two 3D points.
    """

    array_a = np.array(point_a, dtype=float)
    array_b = np.array(point_b, dtype=float)

    return float(np.linalg.norm(array_b - array_a))


def calculate_vertical_velocity(
    previous_ball: list[float],
    current_ball: list[float],
    sampling_rate: float,
) -> float:
    """
    Calculate the basketball's vertical velocity in feet per second.
    """

    previous_z = float(previous_ball[2])
    current_z = float(current_ball[2])

    return (current_z - previous_z) * sampling_rate


def detect_release_frame(
    tracking_frames: list[dict],
    sampling_rate: float,
    shooting_wrist: str = "RIGHT_WRIST",
    minimum_ball_height: float = 6.0,
    release_distance: float = 0.75,
    minimum_vertical_velocity: float = 7.0,
    required_increasing_frames: int = 2,
) -> int | None:
    """
    Estimate the basketball release frame using several signals.

    Release requirements
    --------------------
    1. The ball must be at least six feet high.
    2. The ball must be moving upward quickly.
    3. The ball must be separating from the shooting wrist.
    4. The separation must continue increasing.

    This is more reliable than using distance alone.
    """

    if not isinstance(sampling_rate, (int, float)):
        raise TypeError("Sampling rate must be a number.")

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than zero.")

    previous_ball: list[float] | None = None
    previous_separation: float | None = None

    increasing_separation_count = 0
    candidate_release_index: int | None = None

    for frame_index, frame in enumerate(tracking_frames):
        frame_data = frame.get("data", {})
        player_data = frame_data.get("player", {})

        ball = frame_data.get("ball")
        wrist = player_data.get(shooting_wrist)

        if not is_valid_point(ball) or not is_valid_point(wrist):
            previous_ball = None
            previous_separation = None
            increasing_separation_count = 0
            candidate_release_index = None
            continue

        separation = distance_between(ball, wrist)

        if not math.isfinite(separation):
            continue

        if previous_ball is None:
            previous_ball = ball
            previous_separation = separation
            continue

        vertical_velocity = calculate_vertical_velocity(
            previous_ball=previous_ball,
            current_ball=ball,
            sampling_rate=sampling_rate,
        )

        ball_height = float(ball[2])

        separation_is_increasing = (
            previous_separation is not None
            and separation > previous_separation
        )

        meets_release_conditions = (
            ball_height >= minimum_ball_height
            and separation >= release_distance
            and vertical_velocity >= minimum_vertical_velocity
            and separation_is_increasing
        )

        if meets_release_conditions:
            if increasing_separation_count == 0:
                candidate_release_index = frame_index

            increasing_separation_count += 1

            if (
                increasing_separation_count
                >= required_increasing_frames
            ):
                return candidate_release_index

        else:
            increasing_separation_count = 0
            candidate_release_index = None

        previous_ball = ball
        previous_separation = separation

    return None


def get_release_details(
    tracking_frames: list[dict],
    release_frame_index: int,
    sampling_rate: float,
    shooting_wrist: str = "RIGHT_WRIST",
) -> dict:
    """
    Return detailed information from the detected release frame.
    """

    if (
        release_frame_index < 0
        or release_frame_index >= len(tracking_frames)
    ):
        raise IndexError("Release-frame index is out of range.")

    frame = tracking_frames[release_frame_index]
    frame_data = frame.get("data", {})

    player_data = frame_data.get("player", {})
    ball = frame_data.get("ball")
    wrist = player_data.get(shooting_wrist)

    if not is_valid_point(ball) or not is_valid_point(wrist):
        raise ValueError(
            "The release frame does not contain valid ball "
            "and wrist coordinates."
        )

    separation = distance_between(ball, wrist)

    vertical_velocity = math.nan

    if release_frame_index > 0:
        previous_ball = (
            tracking_frames[release_frame_index - 1]
            .get("data", {})
            .get("ball")
        )

        if is_valid_point(previous_ball):
            vertical_velocity = calculate_vertical_velocity(
                previous_ball=previous_ball,
                current_ball=ball,
                sampling_rate=sampling_rate,
            )

    return {
        "frame_index": release_frame_index,
        "frame_number": frame.get(
            "frame",
            release_frame_index,
        ),
        "time": frame.get("time"),
        "ball_position": ball,
        "wrist_position": wrist,
        "ball_height": float(ball[2]),
        "ball_wrist_separation": separation,
        "vertical_velocity": vertical_velocity,
    }