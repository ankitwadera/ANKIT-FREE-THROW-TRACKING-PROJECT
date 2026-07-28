from __future__ import annotations

from tracking_app.data_loader import load_first_trial
from tracking_app.events import (
    get_average_ankle_height,
    get_mid_hip_height,
)
from tracking_app.release import detect_release_frame


def find_valid_values(
    tracking_frames: list[dict],
    measurement_function,
    start_frame: int,
    end_frame: int,
) -> list[tuple[int, float]]:
    """
    Collect valid frame numbers and measurement values.
    """

    values: list[tuple[int, float]] = []

    for frame_index in range(start_frame, end_frame + 1):
        value = measurement_function(
            tracking_frames[frame_index]
        )

        if value is not None:
            values.append((frame_index, value))

    return values


def print_frame_window(
    tracking_frames: list[dict],
    start_frame: int,
    end_frame: int,
) -> None:
    """
    Print hip and ankle height around the shooting motion.
    """

    print()
    print("FRAME-BY-FRAME LOWER-BODY MOVEMENT")
    print("-" * 70)
    print(
        f"{'Frame':>7}"
        f"{'MID_HIP Z':>15}"
        f"{'ANKLE Z':>15}"
    )
    print("-" * 70)

    for frame_index in range(start_frame, end_frame + 1):
        hip_height = get_mid_hip_height(
            tracking_frames[frame_index]
        )

        ankle_height = get_average_ankle_height(
            tracking_frames[frame_index]
        )

        hip_text = (
            f"{hip_height:.4f}"
            if hip_height is not None
            else "Missing"
        )

        ankle_text = (
            f"{ankle_height:.4f}"
            if ankle_height is not None
            else "Missing"
        )

        print(
            f"{frame_index:>7}"
            f"{hip_text:>15}"
            f"{ankle_text:>15}"
        )


def diagnose_lower_body_events() -> None:
    """
    Inspect the movement ranges needed for event detection.
    """

    trial_file, trial_data = load_first_trial()

    tracking_frames = trial_data.get("tracking", [])
    sampling_rate = trial_data.get("sampling_rate", 30)

    if not tracking_frames:
        raise ValueError(
            "The selected trial contains no tracking frames."
        )

    release_frame = detect_release_frame(
        tracking_frames=tracking_frames,
        sampling_rate=sampling_rate,
        shooting_wrist="RIGHT_WRIST",
        minimum_ball_height=6.0,
        release_distance=0.75,
        minimum_vertical_velocity=7.0,
        required_increasing_frames=2,
    )

    if release_frame is None:
        raise ValueError(
            "Release must be detected before event diagnostics."
        )

    search_start = max(0, release_frame - 120)
    search_end = min(
        len(tracking_frames) - 1,
        release_frame + 50,
    )

    hip_values = find_valid_values(
        tracking_frames=tracking_frames,
        measurement_function=get_mid_hip_height,
        start_frame=search_start,
        end_frame=release_frame,
    )

    ankle_values = find_valid_values(
        tracking_frames=tracking_frames,
        measurement_function=get_average_ankle_height,
        start_frame=search_start,
        end_frame=search_end,
    )

    print("=" * 70)
    print("LOWER-BODY EVENT DIAGNOSTICS")
    print("=" * 70)

    print()
    print(f"Trial: {trial_file.name}")
    print(f"Release frame: {release_frame}")
    print(
        f"Inspection range: "
        f"{search_start} through {search_end}"
    )

    if hip_values:
        minimum_hip_frame, minimum_hip_height = min(
            hip_values,
            key=lambda item: item[1],
        )

        maximum_hip_frame, maximum_hip_height = max(
            hip_values,
            key=lambda item: item[1],
        )

        hip_range = (
            maximum_hip_height - minimum_hip_height
        )

        print()
        print("MID-HIP MOVEMENT")
        print("-" * 70)
        print(
            f"Lowest height: "
            f"{minimum_hip_height:.4f} ft "
            f"at Frame {minimum_hip_frame}"
        )
        print(
            f"Highest height before release: "
            f"{maximum_hip_height:.4f} ft "
            f"at Frame {maximum_hip_frame}"
        )
        print(
            f"Total vertical range: "
            f"{hip_range:.4f} ft"
        )
        print(
            f"Total vertical range: "
            f"{hip_range * 12:.2f} inches"
        )

    else:
        print()
        print("No valid MID_HIP values were found.")

    if ankle_values:
        minimum_ankle_frame, minimum_ankle_height = min(
            ankle_values,
            key=lambda item: item[1],
        )

        maximum_ankle_frame, maximum_ankle_height = max(
            ankle_values,
            key=lambda item: item[1],
        )

        ankle_range = (
            maximum_ankle_height - minimum_ankle_height
        )

        print()
        print("ANKLE MOVEMENT")
        print("-" * 70)
        print(
            f"Lowest height: "
            f"{minimum_ankle_height:.4f} ft "
            f"at Frame {minimum_ankle_frame}"
        )
        print(
            f"Highest height: "
            f"{maximum_ankle_height:.4f} ft "
            f"at Frame {maximum_ankle_frame}"
        )
        print(
            f"Total vertical range: "
            f"{ankle_range:.4f} ft"
        )
        print(
            f"Total vertical range: "
            f"{ankle_range * 12:.2f} inches"
        )

    else:
        print()
        print("No valid ankle values were found.")

    # Print a focused window around the likely dip and release.
    print_frame_window(
        tracking_frames=tracking_frames,
        start_frame=max(0, release_frame - 50),
        end_frame=min(
            len(tracking_frames) - 1,
            release_frame + 15,
        ),
    )

    print()
    print("=" * 70)
    print("DIAGNOSTICS COMPLETE")


if __name__ == "__main__":
    diagnose_lower_body_events()