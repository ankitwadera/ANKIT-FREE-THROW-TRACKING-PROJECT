from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from tracking_app.release import detect_release_frame
from tracking_app.skeleton import is_valid_point


@dataclass
class ShotEvents:
    """
    Stores the major detected moments in one free throw.
    """

    motion_start_frame: int | None
    dip_frame: int | None
    takeoff_frame: int | None
    release_frame: int | None
    ball_apex_frame: int | None
    landing_frame: int | None


def get_player_data(frame: dict) -> dict:
    """
    Return the player-tracking dictionary from one frame.
    """

    player_data = (
        frame
        .get("data", {})
        .get("player", {})
    )

    if not isinstance(player_data, dict):
        return {}

    return player_data


def get_pelvis_height(frame: dict) -> float | None:
    """
    Return the player's pelvis height.

    Preferred source:
        MID_HIP

    Fallback:
        Average of LEFT_HIP and RIGHT_HIP

    Some SPL trials contain the MID_HIP field but do not contain
    usable MID_HIP coordinates. The two-hip average gives us a
    reliable estimated pelvis center.
    """

    player = get_player_data(frame)

    mid_hip = player.get("MID_HIP")

    if is_valid_point(mid_hip):
        return float(mid_hip[2])

    left_hip = player.get("LEFT_HIP")
    right_hip = player.get("RIGHT_HIP")

    if (
        not is_valid_point(left_hip)
        or not is_valid_point(right_hip)
    ):
        return None

    return (
        float(left_hip[2])
        + float(right_hip[2])
    ) / 2


def get_mid_hip_height(frame: dict) -> float | None:
    """
    Backward-compatible name for pelvis height.

    Existing diagnostic code can continue using this function.
    """

    return get_pelvis_height(frame)


def get_average_ankle_height(frame: dict) -> float | None:
    """
    Return the average vertical position of both ankles.
    """

    player = get_player_data(frame)

    left_ankle = player.get("LEFT_ANKLE")
    right_ankle = player.get("RIGHT_ANKLE")

    if (
        not is_valid_point(left_ankle)
        or not is_valid_point(right_ankle)
    ):
        return None

    return (
        float(left_ankle[2])
        + float(right_ankle[2])
    ) / 2


def get_ball_height(frame: dict) -> float | None:
    """
    Return the basketball height for one frame.
    """

    ball = (
        frame
        .get("data", {})
        .get("ball")
    )

    if not is_valid_point(ball):
        return None

    return float(ball[2])


def get_valid_height_values(
    tracking_frames: list[dict],
    measurement_function: Callable[[dict], float | None],
    start_frame: int,
    end_frame: int,
) -> list[tuple[int, float]]:
    """
    Return valid frame-and-height pairs within a frame range.
    """

    values: list[tuple[int, float]] = []

    safe_start = max(0, start_frame)
    safe_end = min(
        len(tracking_frames) - 1,
        end_frame,
    )

    for frame_index in range(safe_start, safe_end + 1):
        height = measurement_function(
            tracking_frames[frame_index]
        )

        if height is not None:
            values.append((frame_index, height))

    return values


def smooth_height_values(
    values: list[tuple[int, float]],
    window_size: int = 5,
) -> list[tuple[int, float]]:
    """
    Smooth a height series using a centered moving average.

    This reduces small frame-to-frame tracking noise before event
    detection.
    """

    if not values:
        return []

    if window_size < 1:
        raise ValueError(
            "Moving-average window size must be at least one."
        )

    frame_numbers = [
        frame_number
        for frame_number, _ in values
    ]

    heights = np.array(
        [
            height
            for _, height in values
        ],
        dtype=float,
    )

    half_window = window_size // 2
    smoothed_values: list[tuple[int, float]] = []

    for index, frame_number in enumerate(frame_numbers):
        start_index = max(0, index - half_window)
        end_index = min(
            len(heights),
            index + half_window + 1,
        )

        smoothed_height = float(
            np.mean(
                heights[start_index:end_index]
            )
        )

        smoothed_values.append(
            (frame_number, smoothed_height)
        )

    return smoothed_values


def detect_motion_start(
    tracking_frames: list[dict],
    release_frame: int,
    search_window_frames: int = 120,
    baseline_frames: int = 20,
    minimum_pelvis_change: float = 0.04,
    required_frames: int = 3,
) -> int | None:
    """
    Detect the beginning of meaningful shooting motion.

    A resting pelvis baseline is calculated from the start of the
    search window. Motion begins when pelvis height moves far enough
    from that baseline for several consecutive frames.
    """

    search_start = max(
        0,
        release_frame - search_window_frames,
    )

    pelvis_values = get_valid_height_values(
        tracking_frames=tracking_frames,
        measurement_function=get_pelvis_height,
        start_frame=search_start,
        end_frame=release_frame,
    )

    pelvis_values = smooth_height_values(
        pelvis_values,
        window_size=5,
    )

    if not pelvis_values:
        return None

    baseline_source = pelvis_values[
        :min(baseline_frames, len(pelvis_values))
    ]

    baseline_height = float(
        np.median(
            [
                height
                for _, height in baseline_source
            ]
        )
    )

    consecutive_frames = 0
    candidate_frame: int | None = None

    for frame_number, pelvis_height in pelvis_values:
        change_from_baseline = abs(
            pelvis_height - baseline_height
        )

        if change_from_baseline >= minimum_pelvis_change:
            if consecutive_frames == 0:
                candidate_frame = frame_number

            consecutive_frames += 1

            if consecutive_frames >= required_frames:
                return candidate_frame

        else:
            consecutive_frames = 0
            candidate_frame = None

    return None


def detect_dip_frame(
    tracking_frames: list[dict],
    release_frame: int,
    motion_start_frame: int | None,
    search_window_frames: int = 100,
) -> int | None:
    """
    Find the lowest smoothed pelvis position before release.
    """

    search_start = (
        motion_start_frame
        if motion_start_frame is not None
        else max(
            0,
            release_frame - search_window_frames,
        )
    )

    pelvis_values = get_valid_height_values(
        tracking_frames=tracking_frames,
        measurement_function=get_pelvis_height,
        start_frame=search_start,
        end_frame=release_frame,
    )

    pelvis_values = smooth_height_values(
        pelvis_values,
        window_size=5,
    )

    if not pelvis_values:
        return None

    dip_frame, _ = min(
        pelvis_values,
        key=lambda item: item[1],
    )

    return dip_frame


def detect_takeoff_frame(
    tracking_frames: list[dict],
    dip_frame: int,
    release_frame: int,
    baseline_window_frames: int = 12,
    minimum_ankle_rise: float = 0.08,
    required_frames: int = 2,
) -> int | None:
    """
    Detect when the feet rise meaningfully from their lowest position.

    An ankle rise of 0.08 feet is approximately one inch.

    Returning None remains valid for a grounded free throw.
    """

    baseline_start = max(
        0,
        dip_frame - baseline_window_frames,
    )

    baseline_end = min(
        len(tracking_frames) - 1,
        dip_frame + 2,
    )

    baseline_values = get_valid_height_values(
        tracking_frames=tracking_frames,
        measurement_function=get_average_ankle_height,
        start_frame=baseline_start,
        end_frame=baseline_end,
    )

    if not baseline_values:
        return None

    baseline_height = float(
        np.min(
            [
                height
                for _, height in baseline_values
            ]
        )
    )

    consecutive_frames = 0
    candidate_frame: int | None = None

    for frame_index in range(
        dip_frame,
        min(
            release_frame + 1,
            len(tracking_frames),
        ),
    ):
        ankle_height = get_average_ankle_height(
            tracking_frames[frame_index]
        )

        if ankle_height is None:
            consecutive_frames = 0
            candidate_frame = None
            continue

        ankle_rise = ankle_height - baseline_height

        if ankle_rise >= minimum_ankle_rise:
            if consecutive_frames == 0:
                candidate_frame = frame_index

            consecutive_frames += 1

            if consecutive_frames >= required_frames:
                return candidate_frame

        else:
            consecutive_frames = 0
            candidate_frame = None

    return None


def detect_ball_apex_frame(
    tracking_frames: list[dict],
    release_frame: int,
) -> int | None:
    """
    Find the highest valid basketball position after release.
    """

    valid_ball_heights = get_valid_height_values(
        tracking_frames=tracking_frames,
        measurement_function=get_ball_height,
        start_frame=release_frame,
        end_frame=len(tracking_frames) - 1,
    )

    if not valid_ball_heights:
        return None

    apex_frame, _ = max(
        valid_ball_heights,
        key=lambda item: item[1],
    )

    return apex_frame


def detect_landing_frame(
    tracking_frames: list[dict],
    takeoff_frame: int | None,
    baseline_window_frames: int = 15,
    minimum_airborne_rise: float = 0.08,
    landing_tolerance: float = 0.04,
    required_landing_frames: int = 2,
) -> int | None:
    """
    Detect when the feet return near their pre-takeoff height.

    If no takeoff occurred, landing remains None.
    """

    if takeoff_frame is None:
        return None

    baseline_values = get_valid_height_values(
        tracking_frames=tracking_frames,
        measurement_function=get_average_ankle_height,
        start_frame=max(
            0,
            takeoff_frame - baseline_window_frames,
        ),
        end_frame=takeoff_frame - 1,
    )

    if not baseline_values:
        return None

    baseline_height = float(
        np.min(
            [
                height
                for _, height in baseline_values
            ]
        )
    )

    airborne_seen = False
    landing_count = 0
    candidate_landing_frame: int | None = None

    for frame_index in range(
        takeoff_frame,
        len(tracking_frames),
    ):
        ankle_height = get_average_ankle_height(
            tracking_frames[frame_index]
        )

        if ankle_height is None:
            landing_count = 0
            candidate_landing_frame = None
            continue

        ankle_rise = ankle_height - baseline_height

        if ankle_rise >= minimum_airborne_rise:
            airborne_seen = True

        returned_to_floor = (
            airborne_seen
            and abs(ankle_height - baseline_height)
            <= landing_tolerance
        )

        if returned_to_floor:
            if landing_count == 0:
                candidate_landing_frame = frame_index

            landing_count += 1

            if landing_count >= required_landing_frames:
                return candidate_landing_frame

        else:
            landing_count = 0
            candidate_landing_frame = None

    return None


def detect_shot_events(
    tracking_frames: list[dict],
    sampling_rate: float,
    shooting_wrist: str = "RIGHT_WRIST",
) -> ShotEvents:
    """
    Detect the major phases of one free throw.
    """

    release_frame = detect_release_frame(
        tracking_frames=tracking_frames,
        sampling_rate=sampling_rate,
        shooting_wrist=shooting_wrist,
        minimum_ball_height=6.0,
        release_distance=0.75,
        minimum_vertical_velocity=7.0,
        required_increasing_frames=2,
    )

    if release_frame is None:
        return ShotEvents(
            motion_start_frame=None,
            dip_frame=None,
            takeoff_frame=None,
            release_frame=None,
            ball_apex_frame=None,
            landing_frame=None,
        )

    motion_start_frame = detect_motion_start(
        tracking_frames=tracking_frames,
        release_frame=release_frame,
    )

    dip_frame = detect_dip_frame(
        tracking_frames=tracking_frames,
        release_frame=release_frame,
        motion_start_frame=motion_start_frame,
    )

    takeoff_frame = None

    if dip_frame is not None:
        takeoff_frame = detect_takeoff_frame(
            tracking_frames=tracking_frames,
            dip_frame=dip_frame,
            release_frame=release_frame,
        )

    ball_apex_frame = detect_ball_apex_frame(
        tracking_frames=tracking_frames,
        release_frame=release_frame,
    )

    landing_frame = detect_landing_frame(
        tracking_frames=tracking_frames,
        takeoff_frame=takeoff_frame,
    )

    return ShotEvents(
        motion_start_frame=motion_start_frame,
        dip_frame=dip_frame,
        takeoff_frame=takeoff_frame,
        release_frame=release_frame,
        ball_apex_frame=ball_apex_frame,
        landing_frame=landing_frame,
    )