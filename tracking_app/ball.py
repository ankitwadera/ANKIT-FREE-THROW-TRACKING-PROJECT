from __future__ import annotations

import math

import numpy as np

from tracking_app.skeleton import is_valid_point


def distance_between(
    point_a: np.ndarray,
    point_b: np.ndarray,
) -> float:
    """
    Calculate the straight-line distance between two 3D points.
    """

    return float(
        np.linalg.norm(point_b - point_a)
    )


def collect_clean_ball_segments(
    tracking_frames: list[dict],
    maximum_frame_jump: float = 2.5,
    minimum_segment_length: int = 4,
) -> list[np.ndarray]:
    """
    Convert raw ball coordinates into clean continuous segments.

    A new segment begins when:

    - the ball coordinate is missing;
    - the ball moves an impossible distance between frames;
    - the coordinate is not a valid [X, Y, Z] point.

    Parameters
    ----------
    maximum_frame_jump:
        Maximum allowed movement in feet from one frame to
        the next. Larger jumps are treated as tracking errors.

    minimum_segment_length:
        Minimum number of points required for a segment to
        be displayed.
    """

    clean_segments: list[np.ndarray] = []
    current_segment: list[np.ndarray] = []

    previous_point: np.ndarray | None = None

    for frame in tracking_frames:
        ball_data = (
            frame
            .get("data", {})
            .get("ball")
        )

        if not is_valid_point(ball_data):
            if len(current_segment) >= minimum_segment_length:
                clean_segments.append(
                    np.array(current_segment, dtype=float)
                )

            current_segment = []
            previous_point = None
            continue

        current_point = np.array(
            ball_data,
            dtype=float,
        )

        if previous_point is not None:
            frame_jump = distance_between(
                previous_point,
                current_point,
            )

            if (
                not math.isfinite(frame_jump)
                or frame_jump > maximum_frame_jump
            ):
                if len(current_segment) >= minimum_segment_length:
                    clean_segments.append(
                        np.array(current_segment, dtype=float)
                    )

                current_segment = []

        current_segment.append(current_point)
        previous_point = current_point

    if len(current_segment) >= minimum_segment_length:
        clean_segments.append(
            np.array(current_segment, dtype=float)
        )

    return clean_segments


def get_longest_ball_segment(
    tracking_frames: list[dict],
    maximum_frame_jump: float = 2.5,
    minimum_segment_length: int = 4,
) -> np.ndarray | None:
    """
    Return the longest continuous clean ball-tracking segment.

    This is normally the most useful section for displaying
    the shooting motion and early ball flight.
    """

    segments = collect_clean_ball_segments(
        tracking_frames=tracking_frames,
        maximum_frame_jump=maximum_frame_jump,
        minimum_segment_length=minimum_segment_length,
    )

    if not segments:
        return None

    return max(
        segments,
        key=len,
    )