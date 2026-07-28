from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from tracking_app.research.feature_metadata import (
    get_feature_metadata,
)


@dataclass(frozen=True)
class CoachDiagnosis:
    """
    One coach-facing diagnosis built from a grouped movement pattern.
    """

    rank: int

    pattern_name: str
    category: str
    priority_level: str

    diagnosis_title: str
    diagnosis_summary: str

    why_it_matters: str
    coaching_focus: str
    recommended_drill: str

    supporting_features: tuple[str, ...]

    review_start_frame: int | None
    review_end_frame: int | None
    review_instruction: str

    grouped_similarity_score: float
    pattern_priority_score: float
    evidence_confidence: str


@dataclass(frozen=True)
class CoachIntelligenceSummary:
    """
    Top-level coach-facing interpretation for one analyzed shot.
    """

    primary_diagnosis: str
    primary_summary: str

    what_stayed_consistent: str
    session_focus: str

    diagnosis_count: int
    confidence: str


class CoachIntelligenceLayer:
    """
    Convert grouped biomechanical patterns into concise coaching language.

    This layer is intentionally different from the feature extractor and
    scoring engine. It does not create new biomechanics. It translates the
    existing measurements into:

    - one primary diagnosis;
    - why the pattern matters;
    - one practical coaching focus;
    - one suggested drill;
    - the exact animation frames to review.

    The language remains cautious. It describes a measurable pattern and
    coaching hypothesis rather than claiming a proven cause of the result.
    """

    def __init__(
        self,
        maximum_diagnoses: int = 3,
    ) -> None:
        self.maximum_diagnoses = max(
            1,
            int(
                maximum_diagnoses
            ),
        )

    def build(
        self,
        movement_patterns: list[object],
        scored_features: list[object],
        category_scores: list[object],
        feature_record: object,
    ) -> tuple[
        CoachIntelligenceSummary,
        list[CoachDiagnosis],
    ]:
        """
        Build the coach summary and diagnosis cards.
        """

        diagnoses: list[
            CoachDiagnosis
        ] = []

        for rank, pattern in enumerate(
            movement_patterns[
                :self.maximum_diagnoses
            ],
            start=1,
        ):
            (
                review_start,
                review_end,
            ) = self._review_window(
                review_event=(
                    pattern.review_event
                ),
                feature_record=(
                    feature_record
                ),
            )

            diagnosis_title = (
                self._diagnosis_title(
                    pattern
                )
            )

            diagnosis_summary = (
                self._diagnosis_summary(
                    pattern=pattern,
                    scored_features=(
                        scored_features
                    ),
                )
            )

            why_it_matters = (
                self._why_it_matters(
                    pattern.pattern_name
                )
            )

            coaching_focus = (
                self._coaching_focus(
                    pattern=pattern,
                    scored_features=(
                        scored_features
                    ),
                )
            )

            recommended_drill = (
                self._recommended_drill(
                    pattern.pattern_name
                )
            )

            review_instruction = (
                self._review_instruction(
                    review_start,
                    review_end,
                )
            )

            diagnoses.append(
                CoachDiagnosis(
                    rank=rank,

                    pattern_name=(
                        pattern.pattern_name
                    ),
                    category=(
                        pattern.category
                    ),
                    priority_level=(
                        pattern.pattern_level
                    ),

                    diagnosis_title=(
                        diagnosis_title
                    ),
                    diagnosis_summary=(
                        diagnosis_summary
                    ),

                    why_it_matters=(
                        why_it_matters
                    ),
                    coaching_focus=(
                        coaching_focus
                    ),
                    recommended_drill=(
                        recommended_drill
                    ),

                    supporting_features=(
                        tuple(
                            pattern
                            .supporting_features
                        )
                    ),

                    review_start_frame=(
                        review_start
                    ),
                    review_end_frame=(
                        review_end
                    ),
                    review_instruction=(
                        review_instruction
                    ),

                    grouped_similarity_score=float(
                        pattern
                        .average_similarity_score
                    ),
                    pattern_priority_score=float(
                        pattern
                        .pattern_priority_score
                    ),
                    evidence_confidence=(
                        self._evidence_confidence(
                            pattern
                        )
                    ),
                )
            )

        summary = self._build_summary(
            diagnoses=diagnoses,
            category_scores=category_scores,
        )

        return (
            summary,
            diagnoses,
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    def _build_summary(
        self,
        diagnoses: list[CoachDiagnosis],
        category_scores: list[object],
    ) -> CoachIntelligenceSummary:
        if diagnoses:
            primary = diagnoses[0]

            primary_diagnosis = (
                primary.diagnosis_title
            )

            primary_summary = (
                primary.diagnosis_summary
            )

            session_focus = (
                primary.coaching_focus
            )

            confidence = (
                primary.evidence_confidence
            )

        else:
            primary_diagnosis = (
                "No stable movement diagnosis"
            )

            primary_summary = (
                "The current feature set did not produce a stable grouped "
                "movement pattern for this shot."
            )

            session_focus = (
                "Use the animation and objective metrics for manual review."
            )

            confidence = (
                "exploratory"
            )

        if category_scores:
            best_category = max(
                category_scores,
                key=lambda row: (
                    row
                    .category_similarity_score
                ),
            )

            what_stayed_consistent = (
                f"{best_category.category} most closely matched the "
                f"player's successful baseline at "
                f"{best_category.category_similarity_score:.1f}/100."
            )

        else:
            what_stayed_consistent = (
                "No category-level consistency result was available."
            )

        return CoachIntelligenceSummary(
            primary_diagnosis=(
                primary_diagnosis
            ),
            primary_summary=(
                primary_summary
            ),

            what_stayed_consistent=(
                what_stayed_consistent
            ),
            session_focus=(
                session_focus
            ),

            diagnosis_count=len(
                diagnoses
            ),
            confidence=confidence,
        )

    # =====================================================
    # DIAGNOSIS LANGUAGE
    # =====================================================

    @staticmethod
    def _diagnosis_title(
        pattern: object,
    ) -> str:
        pattern_name = (
            pattern.pattern_name
        )

        if pattern_name == "Upper-Body Extension":
            return (
                "Reduced upper-body extension"
            )

        if pattern_name == "Lower-Body Extension":
            return (
                "Reduced lower-body extension"
            )

        if pattern_name == "Lower-to-Upper-Body Sequence":
            return (
                "Early upper-body involvement"
            )

        if pattern_name == "Lower-Body Sequence":
            return (
                "Lower-body timing changed"
            )

        if pattern_name == "Upper-Body Release Sequence":
            return (
                "Release sequence timing changed"
            )

        if pattern_name == "Release Timing":
            return (
                "Release timing changed"
            )

        if pattern_name == "Overall Rhythm":
            return (
                "Overall shot rhythm changed"
            )

        if pattern_name == "Wrist Action":
            return (
                "Shooting-wrist action changed"
            )

        if pattern_name == "Guide-Hand Action":
            return (
                "Guide-hand action changed"
            )

        if pattern_name == "Ball Launch":
            return (
                "Ball-launch pattern changed"
            )

        if pattern_name == "Jump Timing":
            return (
                "Jump and landing timing changed"
            )

        return (
            f"Difference in {pattern_name.lower()}"
        )

    def _diagnosis_summary(
        self,
        pattern: object,
        scored_features: list[object],
    ) -> str:
        related = self._related_features(
            pattern_name=(
                pattern.pattern_name
            ),
            scored_features=(
                scored_features
            ),
        )

        if not related:
            return (
                pattern.summary
            )

        strongest = related[0]

        metadata = get_feature_metadata(
            strongest.feature
        )

        direction = (
            metadata.higher_meaning
            if strongest.shot_value
            > strongest.successful_mean
            else metadata.lower_meaning
        )

        if len(
            related
        ) == 1:
            return (
                f"The strongest measured difference showed {direction}. "
                f"This pattern matched the successful baseline at "
                f"{pattern.average_similarity_score:.1f}/100."
            )

        return (
            f"Several related measurements changed together. The clearest "
            f"difference showed {direction}. As a group, these measurements "
            f"matched the successful baseline at "
            f"{pattern.average_similarity_score:.1f}/100."
        )

    @staticmethod
    def _why_it_matters(
        pattern_name: str,
    ) -> str:
        explanations = {
            "Upper-Body Extension": (
                "A more flexed or shortened upper-body position at release "
                "may change release height, direction, or how force reaches "
                "the ball."
            ),

            "Lower-Body Extension": (
                "Reduced hip or knee extension may limit upward force transfer "
                "and place more responsibility on the shooting arm."
            ),

            "Lower-to-Upper-Body Sequence": (
                "When the arm begins too early relative to the legs, the shot "
                "may rely more on upper-body effort and lose whole-body rhythm."
            ),

            "Lower-Body Sequence": (
                "Changes in hip, pelvis, and knee timing can alter how smoothly "
                "force travels upward through the shot."
            ),

            "Upper-Body Release Sequence": (
                "Changes in elbow and wrist timing may affect release rhythm "
                "and the consistency of the final ball action."
            ),

            "Release Timing": (
                "Releasing earlier or later than the player's successful "
                "pattern can change body position and force transfer at release."
            ),

            "Overall Rhythm": (
                "A faster or slower overall motion may disrupt the player's "
                "normal coordination pattern."
            ),

            "Wrist Action": (
                "A different wrist-speed pattern may change the final force "
                "and direction applied to the ball."
            ),

            "Guide-Hand Action": (
                "A change in guide-hand movement may influence ball stability "
                "or add unwanted sideways force."
            ),

            "Ball Launch": (
                "Changes in release angle, speed, or height directly affect "
                "the ball's flight path."
            ),

            "Jump Timing": (
                "Different release-to-landing timing may indicate a change in "
                "jump rhythm, balance, or release position."
            ),
        }

        return explanations.get(
            pattern_name,
            (
                "This pattern differs from the player's successful history "
                "and should be reviewed alongside the animation."
            ),
        )

    def _coaching_focus(
        self,
        pattern: object,
        scored_features: list[object],
    ) -> str:
        related = self._related_features(
            pattern_name=(
                pattern.pattern_name
            ),
            scored_features=(
                scored_features
            ),
        )

        pattern_name = (
            pattern.pattern_name
        )

        if pattern_name == "Upper-Body Extension":
            return (
                "Review whether the shooting shoulder and elbow finish their "
                "normal extension through release. Avoid forcing extra range; "
                "the goal is to reproduce the player's successful pattern."
            )

        if pattern_name == "Lower-Body Extension":
            return (
                "Focus on completing the normal upward drive through the hips "
                "and knees before release."
            )

        if pattern_name == "Lower-to-Upper-Body Sequence":
            return (
                "Allow the lower body to begin and sustain the upward drive "
                "before accelerating the shooting arm."
            )

        if pattern_name == "Lower-Body Sequence":
            return (
                "Rehearse a smooth hip-to-knee extension sequence without "
                "rushing the transition into the upper body."
            )

        if pattern_name == "Upper-Body Release Sequence":
            return (
                "Rehearse the normal elbow-to-wrist-to-release rhythm at a "
                "comfortable speed."
            )

        if pattern_name == "Release Timing":
            return (
                "Match the player's normal takeoff-to-release rhythm rather "
                "than intentionally holding or rushing the ball."
            )

        if pattern_name == "Overall Rhythm":
            return (
                "Use a repeatable tempo from motion start through release."
            )

        if pattern_name == "Wrist Action":
            return (
                "Focus on a relaxed, repeatable wrist finish rather than "
                "trying to create extra speed."
            )

        if pattern_name == "Guide-Hand Action":
            return (
                "Keep the guide hand stable and allow it to leave the ball "
                "without adding sideways force."
            )

        if pattern_name == "Ball Launch":
            return (
                "Review the release frame and reproduce the player's normal "
                "release angle, height, and ball-speed pattern."
            )

        if pattern_name == "Jump Timing":
            return (
                "Maintain balance and reproduce the player's normal timing "
                "from release through landing."
            )

        if related:
            return (
                f"Begin by reviewing "
                f"{get_feature_metadata(related[0].feature).display_name.lower()}."
            )

        return (
            "Review the grouped pattern on the animation before selecting "
            "a technical change."
        )

    @staticmethod
    def _recommended_drill(
        pattern_name: str,
    ) -> str:
        drills = {
            "Upper-Body Extension": (
                "Form shooting close to the basket: pause at set point, then "
                "finish through the normal shoulder-elbow extension."
            ),

            "Lower-Body Extension": (
                "Slow-tempo one-motion form shooting: emphasize a smooth rise "
                "through the hips and knees into release."
            ),

            "Lower-to-Upper-Body Sequence": (
                "Dip-drive-release rhythm drill: clearly feel legs initiate "
                "the upward motion before the arm accelerates."
            ),

            "Lower-Body Sequence": (
                "No-ball rhythm reps: rehearse hip and knee extension together "
                "before adding the shooting motion."
            ),

            "Upper-Body Release Sequence": (
                "One-hand form shooting: rehearse elbow extension followed by "
                "a relaxed wrist finish."
            ),

            "Release Timing": (
                "Tempo free throws using a consistent verbal count from dip "
                "to release."
            ),

            "Overall Rhythm": (
                "Five-shot tempo sets: repeat the same pre-shot and shooting "
                "cadence on every attempt."
            ),

            "Wrist Action": (
                "Close-range one-hand swishes with a relaxed wrist and held "
                "follow-through."
            ),

            "Guide-Hand Action": (
                "Guide-hand stability drill: shoot close to the basket while "
                "keeping the guide hand quiet."
            ),

            "Ball Launch": (
                "Close-range arc-control shooting using the same release point "
                "and smooth ball speed."
            ),

            "Jump Timing": (
                "Balanced landing free throws: hold the finish and land in the "
                "same position after every shot."
            ),
        }

        return drills.get(
            pattern_name,
            (
                "Use slow-motion video review and low-speed repetitions to "
                "reproduce the player's successful pattern."
            ),
        )

    # =====================================================
    # EVIDENCE AND REVIEW WINDOW
    # =====================================================

    @staticmethod
    def _related_features(
        pattern_name: str,
        scored_features: list[object],
    ) -> list[object]:
        rows = [
            row
            for row in scored_features
            if get_feature_metadata(
                row.feature
            ).pattern_group
            == pattern_name
        ]

        rows.sort(
            key=lambda row: (
                row.feature_similarity_score,
                -row.evidence_weight,
            )
        )

        return rows

    @staticmethod
    def _evidence_confidence(
        pattern: object,
    ) -> str:
        if (
            pattern.pattern_priority_score
            >= 65
            and pattern.feature_count
            >= 3
        ):
            return (
                "moderate"
            )

        if (
            pattern.pattern_priority_score
            >= 45
            and pattern.feature_count
            >= 2
        ):
            return (
                "limited"
            )

        return (
            "exploratory"
        )

    @staticmethod
    def _review_window(
        review_event: str,
        feature_record: object,
    ) -> tuple[
        int | None,
        int | None,
    ]:
        motion = (
            feature_record
            .motion_start_frame
        )

        dip = (
            feature_record
            .dip_frame
        )

        takeoff = (
            feature_record
            .takeoff_frame
        )

        release = (
            feature_record
            .release_frame
        )

        landing = (
            feature_record
            .landing_frame
        )

        if review_event == "release":
            return (
                None
                if release is None
                else max(
                    0,
                    release - 6,
                ),
                release,
            )

        if review_event == "takeoff_to_release":
            return (
                takeoff,
                release,
            )

        if review_event == "release_to_landing":
            return (
                release,
                landing,
            )

        if review_event == "motion_to_release":
            return (
                motion,
                release,
            )

        if review_event == "dip_to_release":
            return (
                dip,
                release,
            )

        if review_event in {
            "knee_to_elbow",
            "hip_to_knee",
            "elbow_to_release",
            "wrist_to_release",
            "knee_to_release",
            "hip_to_release",
            "pelvis_to_release",
            "wrist_propulsion",
            "ball_propulsion",
        }:
            return (
                None
                if release is None
                else max(
                    0,
                    release - 15,
                ),
                release,
            )

        return (
            motion,
            release,
        )

    @staticmethod
    def _review_instruction(
        start_frame: int | None,
        end_frame: int | None,
    ) -> str:
        if (
            start_frame is None
            and end_frame is None
        ):
            return (
                "Review the complete shot."
            )

        if start_frame is None:
            return (
                f"Review frame {end_frame}."
            )

        if end_frame is None:
            return (
                f"Review from frame {start_frame} onward."
            )

        if start_frame == end_frame:
            return (
                f"Review frame {start_frame}."
            )

        return (
            f"Review frames {start_frame}–{end_frame}."
        )


def _run_coach_intelligence_layer_test() -> None:
    """
    Structural test using lightweight fake records.
    """

    @dataclass(frozen=True)
    class FakePattern:
        pattern_name: str
        category: str
        pattern_level: str
        average_similarity_score: float
        pattern_priority_score: float
        review_event: str
        feature_count: int
        supporting_features: tuple[str, ...]
        summary: str

    @dataclass(frozen=True)
    class FakeFeature:
        feature: str
        shot_value: float
        successful_mean: float
        feature_similarity_score: float
        evidence_weight: float

    @dataclass(frozen=True)
    class FakeCategory:
        category: str
        category_similarity_score: float

    @dataclass(frozen=True)
    class FakeFeatureRecord:
        motion_start_frame: int
        dip_frame: int
        takeoff_frame: int
        release_frame: int
        landing_frame: int

    patterns = [
        FakePattern(
            pattern_name=(
                "Upper-Body Extension"
            ),
            category="Upper Body",
            pattern_level="high",
            average_similarity_score=42.0,
            pattern_priority_score=70.0,
            review_event="release",
            feature_count=3,
            supporting_features=(
                "Shooting elbow angle at release",
                "Shooting elbow extension range",
                "Shooting shoulder angle at release",
            ),
            summary=(
                "Several related upper-body measurements differed from "
                "the successful-shot baseline."
            ),
        )
    ]

    features = [
        FakeFeature(
            feature=(
                "release_right_elbow_angle_deg"
            ),
            shot_value=118.0,
            successful_mean=137.0,
            feature_similarity_score=35.0,
            evidence_weight=0.75,
        ),
        FakeFeature(
            feature=(
                "right_elbow_range_of_motion_deg"
            ),
            shot_value=39.0,
            successful_mean=60.0,
            feature_similarity_score=25.0,
            evidence_weight=0.78,
        ),
    ]

    categories = [
        FakeCategory(
            category=(
                "Ball & Release"
            ),
            category_similarity_score=91.0,
        )
    ]

    feature_record = FakeFeatureRecord(
        motion_start_frame=110,
        dip_frame=163,
        takeoff_frame=171,
        release_frame=173,
        landing_frame=185,
    )

    layer = CoachIntelligenceLayer()

    summary, diagnoses = layer.build(
        movement_patterns=patterns,
        scored_features=features,
        category_scores=categories,
        feature_record=feature_record,
    )

    assert len(
        diagnoses
    ) == 1

    assert (
        diagnoses[0]
        .diagnosis_title
        == "Reduced upper-body extension"
    )

    assert (
        diagnoses[0]
        .review_start_frame
        == 167
    )

    assert (
        summary.primary_diagnosis
        == "Reduced upper-body extension"
    )

    print(
        "COACH INTELLIGENCE LAYER TEST PASSED"
    )

    print(
        f"Diagnoses created: {len(diagnoses)}"
    )

    print(
        f"Primary diagnosis: "
        f"{summary.primary_diagnosis}"
    )

    print(
        f"Review window: "
        f"{diagnoses[0].review_start_frame} "
        f"to "
        f"{diagnoses[0].review_end_frame}"
    )

    print(
        f"Evidence confidence: "
        f"{diagnoses[0].evidence_confidence}"
    )


if __name__ == "__main__":
    _run_coach_intelligence_layer_test()