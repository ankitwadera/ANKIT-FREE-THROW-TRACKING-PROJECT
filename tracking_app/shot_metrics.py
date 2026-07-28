from __future__ import annotations

import math


def calculate_release_angle(
    horizontal_speed: float,
    vertical_speed: float,
) -> float:
    """
    Calculate the ball's release angle in degrees.

    0 degrees:
        moving completely parallel to the floor

    90 degrees:
        moving completely straight upward
    """

    if horizontal_speed < 0:
        raise ValueError(
            "Horizontal speed cannot be negative."
        )

    return math.degrees(
        math.atan2(
            vertical_speed,
            horizontal_speed,
        )
    )