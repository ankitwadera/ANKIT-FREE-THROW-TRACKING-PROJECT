from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tracking_app.analyzer import ShotAnalyzer
from tracking_app.basketball_event_mapper import (
    BasketballEvent,
    BasketballEventMapper,
    BasketballEventTimeline,
    BasketballEventType,
)
from tracking_app.data_loader import load_first_trial
from tracking_app.phase_biomechanics import (
    MovementPhase,
    RefinedPhaseBiomechanicsAnalyzer,
)
from tracking_app.timeseries import TimeSeriesAnalyzer


class TimelineSource(str, Enum):
    """
    Source used for the final authoritative event.
    """

    SHOT_ANALYZER = "shot_analyzer"
    BASKETBALL_MAPPER = "basketball_mapper"
    PHASE_ANALYZER = "phase_analyzer"


@dataclass(frozen=True)
class ShotTimelineEvent:
    """
    One authoritative event in the final shot timeline.
    """

    event_type: BasketballEventType
    frame: int
    confidence: float
    source: TimelineSource
    source_signal: str
    explanation: str


@dataclass(frozen=True)
class ShotTimeline:
    """
    Complete chronological event timeline for one shot.
    """

    motion_start_frame: int
    dip_frame: int | None
    takeoff_frame: int | None
    release_frame: int
    ball_apex_frame: int | None
    landing_frame: int | None

    events: tuple[ShotTimelineEvent, ...]

    def get_event(
        self,
        event_type: BasketballEventType,
    ) -> ShotTimelineEvent | None:
        for event in self.events:
            if event.event_type == event_type:
                return event

        return None


class ShotTimelineBuilder:
    """
    Build one authoritative basketball shot timeline.

    Priority rules
    --------------
    1. Direct shot-analyzer events are treated as authoritative.
    2. Basketball-mapper events are preferred for interpreted landmarks.
    3. Refined phase landmarks provide fallback events when mapping fails.

    This creates one central event record for reports, scoring,
    comparisons, and coaching feedback.
    """

    def __init__(
        self,
        basketball_timeline: BasketballEventTimeline,
        phase_analysis: object,
        landing_frame: int | None = None,
    ) -> None:
        self.basketball_timeline = basketball_timeline
        self.phase_analysis = phase_analysis
        self.landing_frame = landing_frame

    # =====================================================
    # PUBLIC WORKFLOW
    # =====================================================

    def build(self) -> ShotTimeline:
        """
        Merge direct, mapped, and fallback events.
        """

        final_events: dict[
            BasketballEventType,
            ShotTimelineEvent,
        ] = {}

        # First, use all mapped basketball events.
        for event in self.basketball_timeline.events:
            final_events[
                event.event_type
            ] = self._from_mapped_event(
                event
            )

        # Direct analyzer events always override interpreted sources.
        final_events[
            BasketballEventType.RELEASE
        ] = ShotTimelineEvent(
            event_type=BasketballEventType.RELEASE,
            frame=self.basketball_timeline.release_frame,
            confidence=1.0,
            source=TimelineSource.SHOT_ANALYZER,
            source_signal="Shot analyzer",
            explanation=(
                "Release frame supplied directly by the shot analyzer."
            ),
        )

        if (
            self.basketball_timeline.ball_apex_frame
            is not None
        ):
            final_events[
                BasketballEventType.BALL_APEX
            ] = ShotTimelineEvent(
                event_type=BasketballEventType.BALL_APEX,
                frame=(
                    self.basketball_timeline
                    .ball_apex_frame
                ),
                confidence=1.0,
                source=TimelineSource.SHOT_ANALYZER,
                source_signal="Shot analyzer",
                explanation=(
                    "Ball apex supplied directly by the shot analyzer."
                ),
            )

        # Fill missing interpreted landmarks from refined phases.
        self._add_phase_fallback(
            final_events=final_events,
            event_type=(
                BasketballEventType
                .KNEE_EXTENSION_ONSET
            ),
            phase=self.phase_analysis.knee_phase,
            phase_field="propulsion_onset_frame",
            source_signal="Knee phase",
            explanation=(
                "Fallback knee extension onset from the refined "
                "phase detector."
            ),
        )

        self._add_phase_fallback(
            final_events=final_events,
            event_type=(
                BasketballEventType
                .HIP_EXTENSION_ONSET
            ),
            phase=self.phase_analysis.hip_phase,
            phase_field="propulsion_onset_frame",
            source_signal="Hip phase",
            explanation=(
                "Fallback hip extension onset from the refined "
                "phase detector."
            ),
        )

        self._add_phase_fallback(
            final_events=final_events,
            event_type=(
                BasketballEventType
                .PELVIS_UPWARD_MOTION_ONSET
            ),
            phase=self.phase_analysis.pelvis_phase,
            phase_field="propulsion_onset_frame",
            source_signal="Pelvis phase",
            explanation=(
                "Fallback pelvis upward-motion onset from the refined "
                "phase detector."
            ),
        )

        self._add_phase_fallback(
            final_events=final_events,
            event_type=(
                BasketballEventType
                .SHOULDER_PROPULSION_ONSET
            ),
            phase=self.phase_analysis.shoulder_phase,
            phase_field="propulsion_onset_frame",
            source_signal="Shoulder phase",
            explanation=(
                "Fallback shoulder propulsion onset from the refined "
                "phase detector."
            ),
        )

        self._add_phase_fallback(
            final_events=final_events,
            event_type=(
                BasketballEventType
                .ELBOW_EXTENSION_ONSET
            ),
            phase=self.phase_analysis.elbow_phase,
            phase_field="propulsion_onset_frame",
            source_signal="Elbow phase",
            explanation=(
                "Fallback elbow extension onset from the refined "
                "phase detector."
            ),
        )

        self._add_phase_fallback(
            final_events=final_events,
            event_type=(
                BasketballEventType
                .WRIST_PROPULSION_ONSET
            ),
            phase=self.phase_analysis.wrist_phase,
            phase_field="propulsion_onset_frame",
            source_signal="Wrist phase",
            explanation=(
                "Fallback wrist propulsion onset from the refined "
                "phase detector."
            ),
        )

        self._add_phase_fallback(
            final_events=final_events,
            event_type=(
                BasketballEventType
                .BALL_UPWARD_PROPULSION_ONSET
            ),
            phase=self.phase_analysis.ball_phase,
            phase_field="propulsion_onset_frame",
            source_signal="Ball phase",
            explanation=(
                "Fallback ball upward-propulsion onset from the refined "
                "phase detector."
            ),
        )

        ordered_events = tuple(
            sorted(
                final_events.values(),
                key=lambda event: (
                    event.frame,
                    event.event_type.value,
                ),
            )
        )

        return ShotTimeline(
            motion_start_frame=(
                self.basketball_timeline
                .motion_start_frame
            ),
            dip_frame=(
                self.basketball_timeline
                .dip_frame
            ),
            takeoff_frame=(
                self.basketball_timeline
                .takeoff_frame
            ),
            release_frame=(
                self.basketball_timeline
                .release_frame
            ),
            ball_apex_frame=(
                self.basketball_timeline
                .ball_apex_frame
            ),
            landing_frame=self.landing_frame,
            events=ordered_events,
        )

    # =====================================================
    # SOURCE CONVERSION
    # =====================================================

    @staticmethod
    def _from_mapped_event(
        event: BasketballEvent,
    ) -> ShotTimelineEvent:
        """
        Convert a mapper event into a final timeline event.
        """

        source = (
            TimelineSource.SHOT_ANALYZER
            if event.source_motion_event is None
            and event.source_signal == "Shot analyzer"
            else TimelineSource.BASKETBALL_MAPPER
        )

        return ShotTimelineEvent(
            event_type=event.event_type,
            frame=event.frame,
            confidence=event.confidence,
            source=source,
            source_signal=event.source_signal,
            explanation=event.explanation,
        )

    @staticmethod
    def _add_phase_fallback(
        final_events: dict[
            BasketballEventType,
            ShotTimelineEvent,
        ],
        event_type: BasketballEventType,
        phase: MovementPhase,
        phase_field: str,
        source_signal: str,
        explanation: str,
    ) -> None:
        """
        Add a phase event only when no mapped event exists.
        """

        if event_type in final_events:
            return

        frame = getattr(
            phase,
            phase_field,
        )

        if frame is None:
            return

        final_events[event_type] = (
            ShotTimelineEvent(
                event_type=event_type,
                frame=int(frame),
                confidence=0.75,
                source=(
                    TimelineSource
                    .PHASE_ANALYZER
                ),
                source_signal=source_signal,
                explanation=explanation,
            )
        )


def _run_shot_timeline_builder_test() -> None:
    """
    Build the authoritative timeline for the first free throw.
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

    mapped_timeline = BasketballEventMapper(
        time_series=time_series,
        sampling_rate=float(
            trial_data.get(
                "sampling_rate",
                30,
            )
        ),
        motion_start_frame=(
            shot_analysis.motion_start_frame
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
                shot_analysis.motion_start_frame
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
        basketball_timeline=mapped_timeline,
        phase_analysis=phase_analysis,
        landing_frame=(
            shot_analysis.landing_frame
        ),
    ).build()

    assert len(timeline.events) > 0

    assert timeline.get_event(
        BasketballEventType.RELEASE
    ) is not None

    assert timeline.get_event(
        BasketballEventType.HIP_EXTENSION_ONSET
    ) is not None

    assert timeline.get_event(
        BasketballEventType.SHOULDER_PROPULSION_ONSET
    ) is not None

    print(
        "SHOT TIMELINE BUILDER TEST PASSED"
    )

    print(
        f"Trial file: {trial_file.name}"
    )

    print(
        f"Final timeline events: "
        f"{len(timeline.events)}"
    )

    print()

    for event in timeline.events:
        print(
            f"Frame {event.frame}: "
            f"{event.event_type.value}, "
            f"source {event.source.value}, "
            f"signal {event.source_signal}, "
            f"confidence {event.confidence:.2f}"
        )


if __name__ == "__main__":
    _run_shot_timeline_builder_test()