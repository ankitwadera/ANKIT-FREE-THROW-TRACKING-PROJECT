from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureMetadata:
    """
    Coach-facing information for one technical feature.

    This keeps technical field names, categories, wording, units,
    interpretation, and review frames in one central location.
    """

    display_name: str
    category: str
    unit: str

    higher_meaning: str
    lower_meaning: str

    review_event: str
    pattern_group: str

    explanation: str


FEATURE_METADATA: dict[str, FeatureMetadata] = {
    "takeoff_to_release_ms": FeatureMetadata(
        display_name="Time from takeoff to release",
        category="Timing & Coordination",
        unit="ms",
        higher_meaning="a later release after takeoff",
        lower_meaning="an earlier release after takeoff",
        review_event="takeoff_to_release",
        pattern_group="Release Timing",
        explanation=(
            "The time between leaving the floor and releasing the ball."
        ),
    ),

    "release_to_landing_ms": FeatureMetadata(
        display_name="Time from release to landing",
        category="Timing & Coordination",
        unit="ms",
        higher_meaning="more time in the air after release",
        lower_meaning="less time in the air after release",
        review_event="release_to_landing",
        pattern_group="Jump Timing",
        explanation=(
            "The time between ball release and the player returning "
            "to the floor."
        ),
    ),

    "motion_to_release_ms": FeatureMetadata(
        display_name="Time from motion start to release",
        category="Timing & Coordination",
        unit="ms",
        higher_meaning="a slower overall shooting motion",
        lower_meaning="a faster overall shooting motion",
        review_event="motion_to_release",
        pattern_group="Overall Rhythm",
        explanation=(
            "The total time from the beginning of the shooting motion "
            "to ball release."
        ),
    ),

    "knee_to_elbow_gap_ms": FeatureMetadata(
        display_name="Knee-to-elbow timing",
        category="Timing & Coordination",
        unit="ms",
        higher_meaning="the elbow began later relative to the knee",
        lower_meaning="the elbow began earlier relative to the knee",
        review_event="knee_to_elbow",
        pattern_group="Lower-to-Upper-Body Sequence",
        explanation=(
            "The timing relationship between knee extension and "
            "elbow extension."
        ),
    ),

    "hip_to_knee_gap_ms": FeatureMetadata(
        display_name="Hip-to-knee timing",
        category="Timing & Coordination",
        unit="ms",
        higher_meaning="the knee began later relative to the hip",
        lower_meaning="the knee began earlier relative to the hip",
        review_event="hip_to_knee",
        pattern_group="Lower-Body Sequence",
        explanation=(
            "The timing relationship between hip extension and "
            "knee extension."
        ),
    ),

    "elbow_to_release_gap_ms": FeatureMetadata(
        display_name="Elbow-to-release timing",
        category="Timing & Coordination",
        unit="ms",
        higher_meaning="more time between elbow extension and release",
        lower_meaning="less time between elbow extension and release",
        review_event="elbow_to_release",
        pattern_group="Upper-Body Release Sequence",
        explanation=(
            "The time between elbow-extension onset and ball release."
        ),
    ),

    "wrist_to_release_gap_ms": FeatureMetadata(
        display_name="Wrist-to-release timing",
        category="Timing & Coordination",
        unit="ms",
        higher_meaning="more time between wrist propulsion and release",
        lower_meaning="less time between wrist propulsion and release",
        review_event="wrist_to_release",
        pattern_group="Upper-Body Release Sequence",
        explanation=(
            "The time between wrist propulsion onset and ball release."
        ),
    ),

    "release_right_elbow_angle_deg": FeatureMetadata(
        display_name="Shooting elbow angle at release",
        category="Upper Body",
        unit="degrees",
        higher_meaning="a more extended shooting elbow",
        lower_meaning="a more flexed shooting elbow",
        review_event="release",
        pattern_group="Upper-Body Extension",
        explanation=(
            "The angle of the shooting elbow at ball release."
        ),
    ),

    "release_left_elbow_angle_deg": FeatureMetadata(
        display_name="Guide elbow angle at release",
        category="Upper Body",
        unit="degrees",
        higher_meaning="a more extended guide elbow",
        lower_meaning="a more flexed guide elbow",
        review_event="release",
        pattern_group="Upper-Body Extension",
        explanation=(
            "The angle of the guide-side elbow at ball release."
        ),
    ),

    "right_elbow_range_of_motion_deg": FeatureMetadata(
        display_name="Shooting elbow extension range",
        category="Upper Body",
        unit="degrees",
        higher_meaning="more shooting-elbow movement",
        lower_meaning="less shooting-elbow movement",
        review_event="motion_to_release",
        pattern_group="Upper-Body Extension",
        explanation=(
            "The total shooting-elbow angle change from motion start "
            "to release."
        ),
    ),

    "left_elbow_range_of_motion_deg": FeatureMetadata(
        display_name="Guide elbow movement range",
        category="Upper Body",
        unit="degrees",
        higher_meaning="more guide-elbow movement",
        lower_meaning="less guide-elbow movement",
        review_event="motion_to_release",
        pattern_group="Upper-Body Extension",
        explanation=(
            "The total guide-elbow angle change from motion start "
            "to release."
        ),
    ),

    "release_right_shoulder_angle_deg": FeatureMetadata(
        display_name="Shooting shoulder angle at release",
        category="Upper Body",
        unit="degrees",
        higher_meaning="a more open shooting shoulder position",
        lower_meaning="a more closed shooting shoulder position",
        review_event="release",
        pattern_group="Upper-Body Extension",
        explanation=(
            "The shooting-side shoulder angle at ball release."
        ),
    ),

    "release_left_shoulder_angle_deg": FeatureMetadata(
        display_name="Guide shoulder angle at release",
        category="Upper Body",
        unit="degrees",
        higher_meaning="a more open guide shoulder position",
        lower_meaning="a more closed guide shoulder position",
        review_event="release",
        pattern_group="Upper-Body Extension",
        explanation=(
            "The guide-side shoulder angle at ball release."
        ),
    ),

    "right_shoulder_range_of_motion_deg": FeatureMetadata(
        display_name="Shooting shoulder movement range",
        category="Upper Body",
        unit="degrees",
        higher_meaning="more shooting-shoulder movement",
        lower_meaning="less shooting-shoulder movement",
        review_event="motion_to_release",
        pattern_group="Upper-Body Extension",
        explanation=(
            "The total shooting-shoulder angle change from motion start "
            "to release."
        ),
    ),

    "left_shoulder_range_of_motion_deg": FeatureMetadata(
        display_name="Guide shoulder movement range",
        category="Upper Body",
        unit="degrees",
        higher_meaning="more guide-shoulder movement",
        lower_meaning="less guide-shoulder movement",
        review_event="motion_to_release",
        pattern_group="Upper-Body Extension",
        explanation=(
            "The total guide-shoulder angle change from motion start "
            "to release."
        ),
    ),

    "peak_right_wrist_speed_ft_s": FeatureMetadata(
        display_name="Peak shooting-wrist speed",
        category="Upper Body",
        unit="ft/s",
        higher_meaning="a faster shooting wrist",
        lower_meaning="a slower shooting wrist",
        review_event="wrist_propulsion",
        pattern_group="Wrist Action",
        explanation=(
            "The fastest tracked shooting-wrist movement before release."
        ),
    ),

    "peak_left_wrist_speed_ft_s": FeatureMetadata(
        display_name="Peak guide-wrist speed",
        category="Upper Body",
        unit="ft/s",
        higher_meaning="a faster guide wrist",
        lower_meaning="a slower guide wrist",
        review_event="wrist_propulsion",
        pattern_group="Guide-Hand Action",
        explanation=(
            "The fastest tracked guide-wrist movement before release."
        ),
    ),

    "release_right_hip_angle_deg": FeatureMetadata(
        display_name="Shooting-side hip angle at release",
        category="Lower Body",
        unit="degrees",
        higher_meaning="a more extended shooting-side hip",
        lower_meaning="a more flexed shooting-side hip",
        review_event="release",
        pattern_group="Lower-Body Extension",
        explanation=(
            "The shooting-side hip angle at ball release."
        ),
    ),

    "release_left_hip_angle_deg": FeatureMetadata(
        display_name="Guide-side hip angle at release",
        category="Lower Body",
        unit="degrees",
        higher_meaning="a more extended guide-side hip",
        lower_meaning="a more flexed guide-side hip",
        review_event="release",
        pattern_group="Lower-Body Extension",
        explanation=(
            "The guide-side hip angle at ball release."
        ),
    ),

    "right_hip_range_of_motion_deg": FeatureMetadata(
        display_name="Shooting-side hip movement range",
        category="Lower Body",
        unit="degrees",
        higher_meaning="more shooting-side hip movement",
        lower_meaning="less shooting-side hip movement",
        review_event="dip_to_release",
        pattern_group="Lower-Body Extension",
        explanation=(
            "The total shooting-side hip angle change from motion start "
            "to release."
        ),
    ),

    "left_hip_range_of_motion_deg": FeatureMetadata(
        display_name="Guide-side hip movement range",
        category="Lower Body",
        unit="degrees",
        higher_meaning="more guide-side hip movement",
        lower_meaning="less guide-side hip movement",
        review_event="dip_to_release",
        pattern_group="Lower-Body Extension",
        explanation=(
            "The total guide-side hip angle change from motion start "
            "to release."
        ),
    ),

    "release_right_knee_angle_deg": FeatureMetadata(
        display_name="Shooting-side knee angle at release",
        category="Lower Body",
        unit="degrees",
        higher_meaning="a more extended shooting-side knee",
        lower_meaning="a more flexed shooting-side knee",
        review_event="release",
        pattern_group="Lower-Body Extension",
        explanation=(
            "The shooting-side knee angle at ball release."
        ),
    ),

    "release_left_knee_angle_deg": FeatureMetadata(
        display_name="Guide-side knee angle at release",
        category="Lower Body",
        unit="degrees",
        higher_meaning="a more extended guide-side knee",
        lower_meaning="a more flexed guide-side knee",
        review_event="release",
        pattern_group="Lower-Body Extension",
        explanation=(
            "The guide-side knee angle at ball release."
        ),
    ),

    "right_knee_range_of_motion_deg": FeatureMetadata(
        display_name="Shooting-side knee movement range",
        category="Lower Body",
        unit="degrees",
        higher_meaning="more shooting-side knee movement",
        lower_meaning="less shooting-side knee movement",
        review_event="dip_to_release",
        pattern_group="Lower-Body Extension",
        explanation=(
            "The total shooting-side knee angle change from motion start "
            "to release."
        ),
    ),

    "left_knee_range_of_motion_deg": FeatureMetadata(
        display_name="Guide-side knee movement range",
        category="Lower Body",
        unit="degrees",
        higher_meaning="more guide-side knee movement",
        lower_meaning="less guide-side knee movement",
        review_event="dip_to_release",
        pattern_group="Lower-Body Extension",
        explanation=(
            "The total guide-side knee angle change from motion start "
            "to release."
        ),
    ),

    "pelvis_vertical_displacement_ft": FeatureMetadata(
        display_name="Vertical pelvis movement",
        category="Lower Body",
        unit="ft",
        higher_meaning="more upward pelvis movement",
        lower_meaning="less upward pelvis movement",
        review_event="dip_to_release",
        pattern_group="Lower-Body Extension",
        explanation=(
            "The total upward displacement of the pelvis during the shot."
        ),
    ),

    "release_angle_deg": FeatureMetadata(
        display_name="Ball release angle",
        category="Ball & Release",
        unit="degrees",
        higher_meaning="a steeper release angle",
        lower_meaning="a flatter release angle",
        review_event="release",
        pattern_group="Ball Launch",
        explanation=(
            "The upward launch angle of the ball at release."
        ),
    ),

    "release_ball_speed_ft_s": FeatureMetadata(
        display_name="Ball speed at release",
        category="Ball & Release",
        unit="ft/s",
        higher_meaning="a faster ball at release",
        lower_meaning="a slower ball at release",
        review_event="release",
        pattern_group="Ball Launch",
        explanation=(
            "The three-dimensional speed of the ball at release."
        ),
    ),

    "release_vertical_speed_ft_s": FeatureMetadata(
        display_name="Vertical ball speed at release",
        category="Ball & Release",
        unit="ft/s",
        higher_meaning="more upward ball speed",
        lower_meaning="less upward ball speed",
        review_event="release",
        pattern_group="Ball Launch",
        explanation=(
            "The upward component of ball speed at release."
        ),
    ),

    "release_height_ft": FeatureMetadata(
        display_name="Ball release height",
        category="Ball & Release",
        unit="ft",
        higher_meaning="a higher release point",
        lower_meaning="a lower release point",
        review_event="release",
        pattern_group="Ball Launch",
        explanation=(
            "The vertical position of the ball at release."
        ),
    ),

    "peak_ball_speed_ft_s": FeatureMetadata(
        display_name="Peak ball speed",
        category="Ball & Release",
        unit="ft/s",
        higher_meaning="a higher peak ball speed",
        lower_meaning="a lower peak ball speed",
        review_event="ball_propulsion",
        pattern_group="Ball Launch",
        explanation=(
            "The fastest tracked ball movement during the shot."
        ),
    ),

    "release_wrist_speed_ft_s": FeatureMetadata(
        display_name="Shooting-wrist speed at release",
        category="Upper Body",
        unit="ft/s",
        higher_meaning="a faster shooting wrist at release",
        lower_meaning="a slower shooting wrist at release",
        review_event="release",
        pattern_group="Wrist Action",
        explanation=(
            "The speed of the shooting wrist at the exact release frame."
        ),
    ),

    "wrist_propulsion_duration_ms": FeatureMetadata(
        display_name="Wrist propulsion duration",
        category="Timing & Coordination",
        unit="ms",
        higher_meaning="a longer wrist propulsion phase",
        lower_meaning="a shorter wrist propulsion phase",
        review_event="wrist_to_release",
        pattern_group="Upper-Body Release Sequence",
        explanation=(
            "The duration of the wrist propulsion phase before release."
        ),
    ),

    "elbow_propulsion_duration_ms": FeatureMetadata(
        display_name="Elbow propulsion duration",
        category="Timing & Coordination",
        unit="ms",
        higher_meaning="a longer elbow propulsion phase",
        lower_meaning="a shorter elbow propulsion phase",
        review_event="elbow_to_release",
        pattern_group="Upper-Body Release Sequence",
        explanation=(
            "The duration of the elbow extension phase before release."
        ),
    ),

    "knee_propulsion_duration_ms": FeatureMetadata(
        display_name="Knee propulsion duration",
        category="Timing & Coordination",
        unit="ms",
        higher_meaning="a longer knee extension phase",
        lower_meaning="a shorter knee extension phase",
        review_event="knee_to_release",
        pattern_group="Lower-Body Sequence",
        explanation=(
            "The duration of the knee extension phase before release."
        ),
    ),

    "hip_propulsion_duration_ms": FeatureMetadata(
        display_name="Hip propulsion duration",
        category="Timing & Coordination",
        unit="ms",
        higher_meaning="a longer hip extension phase",
        lower_meaning="a shorter hip extension phase",
        review_event="hip_to_release",
        pattern_group="Lower-Body Sequence",
        explanation=(
            "The duration of the hip extension phase before release."
        ),
    ),

    "pelvis_propulsion_duration_ms": FeatureMetadata(
        display_name="Pelvis propulsion duration",
        category="Timing & Coordination",
        unit="ms",
        higher_meaning="a longer upward pelvis phase",
        lower_meaning="a shorter upward pelvis phase",
        review_event="pelvis_to_release",
        pattern_group="Lower-Body Sequence",
        explanation=(
            "The duration of upward pelvis movement before release."
        ),
    ),
}


def get_feature_metadata(
    feature_name: str,
) -> FeatureMetadata:
    """
    Return configured metadata or a safe fallback for an unknown feature.
    """

    if feature_name in FEATURE_METADATA:
        return FEATURE_METADATA[
            feature_name
        ]

    fallback_display_name = (
        feature_name
        .replace(
            "_deg",
            "",
        )
        .replace(
            "_ft_s",
            "",
        )
        .replace(
            "_ms",
            "",
        )
        .replace(
            "_ft",
            "",
        )
        .replace(
            "_",
            " ",
        )
        .strip()
        .title()
    )

    return FeatureMetadata(
        display_name=fallback_display_name,
        category="Other",
        unit="",
        higher_meaning="a higher value",
        lower_meaning="a lower value",
        review_event="full_shot",
        pattern_group="Other",
        explanation=(
            "No custom coach-facing explanation has been configured yet."
        ),
    )


def describe_direction(
    feature_name: str,
    current_value: float,
    baseline_value: float,
) -> str:
    """
    Translate a numeric direction into basketball language.
    """

    metadata = get_feature_metadata(
        feature_name
    )

    if current_value > baseline_value:
        return metadata.higher_meaning

    if current_value < baseline_value:
        return metadata.lower_meaning

    return "very similar to the successful baseline"