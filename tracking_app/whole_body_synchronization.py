from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tracking_app.basketball_event_mapper import BasketballEventType
from tracking_app.data_loader import load_first_trial
from tracking_app.shot_timeline_builder import (
    ShotTimeline,
    ShotTimelineBuilder,
    TimelineSource,
)
from tracking_app.analyzer import ShotAnalyzer
from tracking_app.basketball_event_mapper import BasketballEventMapper
from tracking_app.phase_biomechanics import RefinedPhaseBiomechanicsAnalyzer
from tracking_app.timeseries import TimeSeriesAnalyzer


class SynchronizationClass(str, Enum):
    """
    Interpretation categories for timing relationships.
    """

    SIMULTANEOUS = "simultaneous"
    OVERLAPPING = "overlapping"
    SEQUENTIAL = "sequential"
    EARLY = "early"
    LATE = "late"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class SynchronizationRelationship:
    """
    Timing relationship between two authoritative shot events.
    """

    relationship_name: str

    first_event: BasketballEventType
    second_event: BasketballEventType

    first_frame: int | None
    second_frame: int | None

    frame_gap: int | None
    milliseconds_gap: float | None

    classification: SynchronizationClass
    confidence: float

    first_source: TimelineSource | None
    second_source: TimelineSource | None

    interpretation: str


@dataclass(frozen=True)
class WholeBodySynchronizationAnalysis:
    """
    Complete whole-body coordination report for one shot.
    """

    sampling_rate: float
    release_frame: int

    relationships: tuple[
        SynchronizationRelationship,
        ...
    ]

    simultaneous_count: int
    overlapping_count: int
    sequential_count: int
    early_count: int
    late_count: int
    unavailable_count: int

    synchronization_score: float


class WholeBodySynchronizationEngine:
    """
    Analyze how authoritative biomechanical events relate in time.

    The engine does not decide whether one universal movement pattern
    is correct. It describes coordination transparently and produces
    an initial consistency-oriented score.

    Classification rules
    --------------------
    Simultaneous:
        Events occur within one tracking frame.

    Overlapping:
        Events are separated by 2-3 frames.

    Sequential:
        Events occur in the expected order with a moderate gap.

    Early:
        The second event occurs before the first event.

    Late:
        The gap is larger than the configured maximum expected gap.

    Unavailable:
        One or both events are missing.
    """

    RELATIONSHIP_DEFINITIONS = (
        (
            "Hip to knee extension",
            BasketballEventType.HIP_EXTENSION_ONSET,
            BasketballEventType.KNEE_EXTENSION_ONSET,
            8,
        ),
        (
            "Pelvis to knee extension",
            BasketballEventType.PELVIS_UPWARD_MOTION_ONSET,
            BasketballEventType.KNEE_EXTENSION_ONSET,
            8,
        ),
        (
            "Shoulder to elbow propulsion",
            BasketballEventType.SHOULDER_PROPULSION_ONSET,
            BasketballEventType.ELBOW_EXTENSION_ONSET,
            8,
        ),
        (
            "Knee to elbow extension",
            BasketballEventType.KNEE_EXTENSION_ONSET,
            BasketballEventType.ELBOW_EXTENSION_ONSET,
            12,
        ),
        (
            "Wrist propulsion to release",
            BasketballEventType.WRIST_PROPULSION_ONSET,
            BasketballEventType.RELEASE,
            12,
        ),
        (
            "Ball upward motion to release",
            BasketballEventType.BALL_UPWARD_PROPULSION_ONSET,
            BasketballEventType.RELEASE,
            12,
        ),
        (
            "Pelvis upward motion to release",
            BasketballEventType.PELVIS_UPWARD_MOTION_ONSET,
            BasketballEventType.RELEASE,
            16,
        ),
        (
            "Shoulder propulsion to release",
            BasketballEventType.SHOULDER_PROPULSION_ONSET,
            BasketballEventType.RELEASE,
            16,
        ),
        (
            "Elbow extension to release",
            BasketballEventType.ELBOW_EXTENSION_ONSET,
            BasketballEventType.RELEASE,
            12,
        ),
        (
            "Ball lowest point to release",
            BasketballEventType.BALL_LOWEST_POINT,
            BasketballEventType.RELEASE,
            60,
        ),
    )

    def __init__(
        self,
        timeline: ShotTimeline,
        sampling_rate: float,
        simultaneous_tolerance_frames: int = 1,
        overlap_tolerance_frames: int = 3,
    ) -> None:
        if sampling_rate <= 0:
            raise ValueError(
                "sampling_rate must be greater than zero."
            )

        self.timeline = timeline
        self.sampling_rate = float(
            sampling_rate
        )

        self.simultaneous_tolerance_frames = max(
            0,
            int(
                simultaneous_tolerance_frames
            ),
        )

        self.overlap_tolerance_frames = max(
            self.simultaneous_tolerance_frames,
            int(
                overlap_tolerance_frames
            ),
        )

    # =====================================================
    # PUBLIC ANALYSIS
    # =====================================================

    def analyze(
        self,
    ) -> WholeBodySynchronizationAnalysis:
        """
        Build all configured timing relationships.
        """

        relationships: list[
            SynchronizationRelationship
        ] = []

        for (
            relationship_name,
            first_event_type,
            second_event_type,
            maximum_expected_gap,
        ) in self.RELATIONSHIP_DEFINITIONS:
            relationships.append(
                self._analyze_relationship(
                    relationship_name=(
                        relationship_name
                    ),
                    first_event_type=(
                        first_event_type
                    ),
                    second_event_type=(
                        second_event_type
                    ),
                    maximum_expected_gap=(
                        maximum_expected_gap
                    ),
                )
            )

        simultaneous_count = sum(
            relationship.classification
            == SynchronizationClass.SIMULTANEOUS
            for relationship in relationships
        )

        overlapping_count = sum(
            relationship.classification
            == SynchronizationClass.OVERLAPPING
            for relationship in relationships
        )

        sequential_count = sum(
            relationship.classification
            == SynchronizationClass.SEQUENTIAL
            for relationship in relationships
        )

        early_count = sum(
            relationship.classification
            == SynchronizationClass.EARLY
            for relationship in relationships
        )

        late_count = sum(
            relationship.classification
            == SynchronizationClass.LATE
            for relationship in relationships
        )

        unavailable_count = sum(
            relationship.classification
            == SynchronizationClass.UNAVAILABLE
            for relationship in relationships
        )

        score = self._calculate_score(
            relationships
        )

        return WholeBodySynchronizationAnalysis(
            sampling_rate=self.sampling_rate,
            release_frame=(
                self.timeline.release_frame
            ),
            relationships=tuple(
                relationships
            ),
            simultaneous_count=(
                simultaneous_count
            ),
            overlapping_count=(
                overlapping_count
            ),
            sequential_count=(
                sequential_count
            ),
            early_count=early_count,
            late_count=late_count,
            unavailable_count=(
                unavailable_count
            ),
            synchronization_score=score,
        )

    # =====================================================
    # RELATIONSHIP ANALYSIS
    # =====================================================

    def _analyze_relationship(
        self,
        relationship_name: str,
        first_event_type: BasketballEventType,
        second_event_type: BasketballEventType,
        maximum_expected_gap: int,
    ) -> SynchronizationRelationship:
        """
        Analyze one pair of authoritative events.
        """

        first_event = self.timeline.get_event(
            first_event_type
        )

        second_event = self.timeline.get_event(
            second_event_type
        )

        if (
            first_event is None
            or second_event is None
        ):
            return SynchronizationRelationship(
                relationship_name=(
                    relationship_name
                ),
                first_event=(
                    first_event_type
                ),
                second_event=(
                    second_event_type
                ),
                first_frame=(
                    None
                    if first_event is None
                    else first_event.frame
                ),
                second_frame=(
                    None
                    if second_event is None
                    else second_event.frame
                ),
                frame_gap=None,
                milliseconds_gap=None,
                classification=(
                    SynchronizationClass
                    .UNAVAILABLE
                ),
                confidence=0.0,
                first_source=(
                    None
                    if first_event is None
                    else first_event.source
                ),
                second_source=(
                    None
                    if second_event is None
                    else second_event.source
                ),
                interpretation=(
                    "One or both required events "
                    "were unavailable."
                ),
            )

        frame_gap = (
            second_event.frame
            - first_event.frame
        )

        milliseconds_gap = (
            frame_gap
            / self.sampling_rate
            * 1000.0
        )

        classification = (
            self._classify_gap(
                frame_gap=frame_gap,
                maximum_expected_gap=(
                    maximum_expected_gap
                ),
            )
        )

        confidence = min(
            first_event.confidence,
            second_event.confidence,
        )

        interpretation = (
            self._build_interpretation(
                relationship_name=(
                    relationship_name
                ),
                classification=(
                    classification
                ),
                frame_gap=frame_gap,
                milliseconds_gap=(
                    milliseconds_gap
                ),
            )
        )

        return SynchronizationRelationship(
            relationship_name=(
                relationship_name
            ),
            first_event=(
                first_event_type
            ),
            second_event=(
                second_event_type
            ),
            first_frame=(
                first_event.frame
            ),
            second_frame=(
                second_event.frame
            ),
            frame_gap=frame_gap,
            milliseconds_gap=(
                milliseconds_gap
            ),
            classification=(
                classification
            ),
            confidence=confidence,
            first_source=(
                first_event.source
            ),
            second_source=(
                second_event.source
            ),
            interpretation=(
                interpretation
            ),
        )

    def _classify_gap(
        self,
        frame_gap: int,
        maximum_expected_gap: int,
    ) -> SynchronizationClass:
        """
        Classify one event gap.
        """

        if frame_gap < 0:
            return SynchronizationClass.EARLY

        if (
            frame_gap
            <= self.simultaneous_tolerance_frames
        ):
            return (
                SynchronizationClass
                .SIMULTANEOUS
            )

        if (
            frame_gap
            <= self.overlap_tolerance_frames
        ):
            return (
                SynchronizationClass
                .OVERLAPPING
            )

        if frame_gap <= maximum_expected_gap:
            return (
                SynchronizationClass
                .SEQUENTIAL
            )

        return SynchronizationClass.LATE

    @staticmethod
    def _build_interpretation(
        relationship_name: str,
        classification: SynchronizationClass,
        frame_gap: int,
        milliseconds_gap: float,
    ) -> str:
        """
        Build coach-readable timing text.
        """

        absolute_ms = abs(
            milliseconds_gap
        )

        if (
            classification
            == SynchronizationClass
            .SIMULTANEOUS
        ):
            return (
                f"{relationship_name} occurred "
                f"within {absolute_ms:.1f} ms."
            )

        if (
            classification
            == SynchronizationClass
            .OVERLAPPING
        ):
            return (
                f"{relationship_name} overlapped "
                f"with a {absolute_ms:.1f} ms gap."
            )

        if (
            classification
            == SynchronizationClass
            .SEQUENTIAL
        ):
            return (
                f"{relationship_name} followed "
                f"in sequence after "
                f"{absolute_ms:.1f} ms."
            )

        if (
            classification
            == SynchronizationClass.EARLY
        ):
            return (
                f"The second event in "
                f"{relationship_name} occurred "
                f"{absolute_ms:.1f} ms before "
                f"the first event."
            )

        if (
            classification
            == SynchronizationClass.LATE
        ):
            return (
                f"The second event in "
                f"{relationship_name} occurred "
                f"{absolute_ms:.1f} ms later, "
                f"outside the initial expected "
                f"timing window."
            )

        return (
            f"{relationship_name} could not "
            f"be evaluated."
        )

    # =====================================================
    # SCORE
    # =====================================================

    @staticmethod
    def _calculate_score(
        relationships: list[
            SynchronizationRelationship
        ],
    ) -> float:
        """
        Calculate an initial transparent 0-100 score.

        This score is provisional and intended for consistency
        analysis, not a universal judgment of shooting quality.

        Weights
        -------
        simultaneous: 1.00
        overlapping:  0.90
        sequential:   0.80
        early:        0.40
        late:         0.40
        unavailable:  0.00
        """

        weights = {
            SynchronizationClass.SIMULTANEOUS: 1.00,
            SynchronizationClass.OVERLAPPING: 0.90,
            SynchronizationClass.SEQUENTIAL: 0.80,
            SynchronizationClass.EARLY: 0.40,
            SynchronizationClass.LATE: 0.40,
            SynchronizationClass.UNAVAILABLE: 0.00,
        }

        if not relationships:
            return 0.0

        weighted_total = 0.0
        confidence_total = 0.0

        for relationship in relationships:
            confidence = max(
                0.0,
                min(
                    1.0,
                    relationship.confidence,
                ),
            )

            weighted_total += (
                weights[
                    relationship.classification
                ]
                * confidence
            )

            confidence_total += confidence

        if confidence_total <= 0:
            return 0.0

        return (
            weighted_total
            / confidence_total
            * 100.0
        )


def _run_whole_body_synchronization_test() -> None:
    """
    Analyze whole-body event timing for the first free throw.
    """

    trial_file, trial_data = load_first_trial()

    shot_analysis = ShotAnalyzer(
        trial_data=trial_data,
        trial_file=trial_file,
        shooting_wrist="RIGHT_WRIST",
    ).analyze()

    time_series = TimeSeriesAnalyzer(
        trial_data=trial_data,
        smoothing_window=5,
    ).analyze()

    basketball_timeline = (
        BasketballEventMapper(
            time_series=time_series,
            sampling_rate=float(
                trial_data.get(
                    "sampling_rate",
                    30,
                )
            ),
            motion_start_frame=(
                shot_analysis
                .motion_start_frame
            ),
            dip_frame=(
                shot_analysis.dip_frame
            ),
            takeoff_frame=(
                shot_analysis.takeoff_frame
            ),
            release_frame=(
                shot_analysis.release_frame
            ),
            ball_apex_frame=(
                shot_analysis.ball_apex_frame
            ),
            shooting_side="RIGHT",
        ).build_timeline()
    )

    phase_analysis = (
        RefinedPhaseBiomechanicsAnalyzer(
            time_series=time_series,
            sampling_rate=float(
                trial_data.get(
                    "sampling_rate",
                    30,
                )
            ),
            motion_start_frame=(
                shot_analysis
                .motion_start_frame
            ),
            dip_frame=(
                shot_analysis.dip_frame
            ),
            takeoff_frame=(
                shot_analysis.takeoff_frame
            ),
            release_frame=(
                shot_analysis.release_frame
            ),
            shooting_side="RIGHT",
        ).analyze()
    )

    timeline = ShotTimelineBuilder(
        basketball_timeline=(
            basketball_timeline
        ),
        phase_analysis=phase_analysis,
        landing_frame=(
            shot_analysis.landing_frame
        ),
    ).build()

    result = WholeBodySynchronizationEngine(
        timeline=timeline,
        sampling_rate=float(
            trial_data.get(
                "sampling_rate",
                30,
            )
        ),
    ).analyze()

    assert len(
        result.relationships
    ) > 0

    assert (
        0.0
        <= result.synchronization_score
        <= 100.0
    )

    print(
        "WHOLE-BODY SYNCHRONIZATION TEST PASSED"
    )

    print(
        f"Trial file: {trial_file.name}"
    )

    print(
        "Synchronization score: "
        f"{result.synchronization_score:.1f} / 100"
    )

    print()

    for relationship in (
        result.relationships
    ):
        milliseconds = (
            "N/A"
            if relationship.milliseconds_gap
            is None
            else (
                f"{relationship.milliseconds_gap:.1f} ms"
            )
        )

        print(
            f"{relationship.relationship_name}: "
            f"{relationship.classification.value}, "
            f"gap {milliseconds}, "
            f"confidence "
            f"{relationship.confidence:.2f}"
        )

    print()

    print(
        "Classification counts:"
    )

    print(
        f"Simultaneous: "
        f"{result.simultaneous_count}"
    )

    print(
        f"Overlapping: "
        f"{result.overlapping_count}"
    )

    print(
        f"Sequential: "
        f"{result.sequential_count}"
    )

    print(
        f"Early: {result.early_count}"
    )

    print(
        f"Late: {result.late_count}"
    )

    print(
        f"Unavailable: "
        f"{result.unavailable_count}"
    )


if __name__ == "__main__":
    _run_whole_body_synchronization_test()