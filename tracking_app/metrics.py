from __future__ import annotations

from tracking_app.data_loader import load_first_trial
from tracking_app.pipeline import AnalysisPipeline


def format_frame(value: int | None) -> str:
    """
    Format an optional frame number.
    """

    if value is None:
        return "Not detected"

    return str(value)


def test_analysis_pipeline() -> None:
    """
    Run the complete analysis pipeline on the first free throw.
    """

    trial_file, trial_data = load_first_trial()

    pipeline = AnalysisPipeline(
        trial_data=trial_data,
        trial_file=trial_file,
        shooting_wrist="RIGHT_WRIST",
    )

    result = pipeline.run()

    analysis = result.analysis
    events = result.events

    print("=" * 65)
    print("BASKETBALL MOTION ANALYSIS PIPELINE")
    print("=" * 65)

    print(f"\nTrial file: {analysis.trial_file}")
    print(f"Participant: {analysis.participant_id}")
    print(f"Trial: {analysis.trial_id}")
    print(f"Result: {analysis.result}")

    print("\nSHOT EVENTS")
    print("-" * 65)
    print(
        f"Motion start: "
        f"{format_frame(events.motion_start_frame)}"
    )
    print(f"Dip: {format_frame(events.dip_frame)}")
    print(f"Takeoff: {format_frame(events.takeoff_frame)}")
    print(f"Release: {format_frame(events.release_frame)}")
    print(
        f"Ball apex: "
        f"{format_frame(events.ball_apex_frame)}"
    )
    print(f"Landing: {format_frame(events.landing_frame)}")

    print("\nRELEASE METRICS")
    print("-" * 65)
    print(f"Release frame: {analysis.release_frame}")
    print(f"Release time: {analysis.release_time}")
    print(f"Release height: {analysis.release_height:.2f} ft")
    print(f"Ball speed: {analysis.ball_speed:.2f} ft/s")
    print(
        f"Horizontal ball speed: "
        f"{analysis.ball_horizontal_speed:.2f} ft/s"
    )
    print(
        f"Vertical ball speed: "
        f"{analysis.ball_vertical_speed:.2f} ft/s"
    )
    print(f"Release angle: {analysis.release_angle:.2f}°")

    print("\nRELEASE BIOMECHANICS")
    print("-" * 65)
    print(
        f"Right elbow: "
        f"{analysis.right_elbow_angle:.2f}°"
        if analysis.right_elbow_angle is not None
        else "Right elbow: Unavailable"
    )
    print(
        f"Right knee: "
        f"{analysis.right_knee_angle:.2f}°"
        if analysis.right_knee_angle is not None
        else "Right knee: Unavailable"
    )
    print(
        f"Left knee: "
        f"{analysis.left_knee_angle:.2f}°"
        if analysis.left_knee_angle is not None
        else "Left knee: Unavailable"
    )
    print(
        f"Right hip: "
        f"{analysis.right_hip_angle:.2f}°"
        if analysis.right_hip_angle is not None
        else "Right hip: Unavailable"
    )

    print("\n" + "=" * 65)
    print("PIPELINE TEST COMPLETE")


if __name__ == "__main__":
    test_analysis_pipeline()