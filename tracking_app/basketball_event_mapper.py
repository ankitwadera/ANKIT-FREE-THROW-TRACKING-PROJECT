from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from tracking_app.analyzer import ShotAnalyzer
from tracking_app.data_loader import load_first_trial
from tracking_app.motion_event_detector import (
    MotionEvent,
    MotionEventDetector,
    MotionEventType,
    MotionSignalAnalysis,
)
from tracking_app.timeseries import ShotTimeSeries, TimeSeriesAnalyzer


class BasketballEventType(str, Enum):
    MAX_KNEE_FLEXION = "max_knee_flexion"
    KNEE_EXTENSION_ONSET = "knee_extension_onset"
    MAX_HIP_FLEXION = "max_hip_flexion"
    HIP_EXTENSION_ONSET = "hip_extension_onset"
    PELVIS_LOWEST_POINT = "pelvis_lowest_point"
    PELVIS_UPWARD_MOTION_ONSET = "pelvis_upward_motion_onset"
    SHOULDER_PROPULSION_ONSET = "shoulder_propulsion_onset"
    ELBOW_EXTENSION_ONSET = "elbow_extension_onset"
    WRIST_PROPULSION_ONSET = "wrist_propulsion_onset"
    BALL_LOWEST_POINT = "ball_lowest_point"
    BALL_UPWARD_PROPULSION_ONSET = "ball_upward_propulsion_onset"
    RELEASE = "release"
    BALL_APEX = "ball_apex"


@dataclass(frozen=True)
class BasketballEvent:
    event_type: BasketballEventType
    frame: int
    source_signal: str
    source_motion_event: MotionEventType | None
    value: float | None
    confidence: float
    explanation: str


@dataclass(frozen=True)
class BasketballEventTimeline:
    motion_start_frame: int
    dip_frame: int | None
    takeoff_frame: int | None
    release_frame: int
    ball_apex_frame: int | None
    events: tuple[BasketballEvent, ...]

    def get_event(
        self,
        event_type: BasketballEventType,
    ) -> BasketballEvent | None:
        for event in self.events:
            if event.event_type == event_type:
                return event
        return None


class BasketballEventMapper:
    """
    Convert generic signal events into basketball-specific landmarks.
    """

    def __init__(
        self,
        time_series: ShotTimeSeries,
        sampling_rate: float,
        motion_start_frame: int,
        release_frame: int,
        dip_frame: int | None = None,
        takeoff_frame: int | None = None,
        ball_apex_frame: int | None = None,
        shooting_side: str = "RIGHT",
    ) -> None:
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be greater than zero.")

        side = shooting_side.strip().upper()
        if side not in {"RIGHT", "LEFT"}:
            raise ValueError("shooting_side must be RIGHT or LEFT.")

        self.time_series = time_series
        self.sampling_rate = float(sampling_rate)
        self.motion_start_frame = max(0, int(motion_start_frame))
        self.release_frame = min(
            int(release_frame),
            time_series.total_frames - 1,
        )
        self.dip_frame = None if dip_frame is None else int(dip_frame)
        self.takeoff_frame = (
            None if takeoff_frame is None else int(takeoff_frame)
        )
        self.ball_apex_frame = (
            None if ball_apex_frame is None else int(ball_apex_frame)
        )
        self.shooting_side = side

        self.detector = MotionEventDetector(
            sampling_rate=self.sampling_rate,
            smoothing_window=7,
            extrema_window_radius=4,
            minimum_prominence_ratio=0.03,
            sign_change_threshold_ratio=0.05,
            plateau_velocity_ratio=0.04,
            minimum_plateau_frames=4,
            minimum_event_separation_frames=4,
        )

    def build_timeline(self) -> BasketballEventTimeline:
        knee_angle = self._side_series(
            self.time_series.right_knee_angle,
            self.time_series.left_knee_angle,
        )
        hip_angle = self._side_series(
            self.time_series.right_hip_angle,
            self.time_series.left_hip_angle,
        )
        shoulder_angle = self._side_series(
            self.time_series.right_shoulder_angle,
            self.time_series.left_shoulder_angle,
        )
        elbow_angle = self._side_series(
            self.time_series.right_elbow_angle,
            self.time_series.left_elbow_angle,
        )
        wrist_speed = self._side_series(
            self.time_series.right_wrist_speed,
            self.time_series.left_wrist_speed,
        )

        knee = self._analyze_signal("Shooting-side knee angle", knee_angle)
        hip = self._analyze_signal("Shooting-side hip angle", hip_angle)
        pelvis = self._analyze_signal(
            "Pelvis height",
            self.time_series.pelvis_height,
        )
        shoulder = self._analyze_signal(
            "Shooting-side shoulder angle",
            shoulder_angle,
        )
        elbow = self._analyze_signal(
            "Shooting-side elbow angle",
            elbow_angle,
        )
        wrist = self._analyze_signal(
            "Shooting-side wrist speed",
            wrist_speed,
        )
        ball_height = self._analyze_signal(
            "Ball height",
            self.time_series.ball_height,
        )
        ball_vertical_speed = self._analyze_signal(
            "Ball vertical speed",
            self.time_series.ball_vertical_speed,
        )

        events: list[BasketballEvent] = []

        knee_minimum = self._map_local_minimum(
            knee,
            BasketballEventType.MAX_KNEE_FLEXION,
            self.dip_frame,
            "Lowest knee-angle event; interpreted as maximum knee flexion.",
        )
        if knee_minimum is not None:
            events.append(knee_minimum)
            knee_onset = self._map_post_minimum_reversal(
                knee,
                knee_minimum.frame,
                BasketballEventType.KNEE_EXTENSION_ONSET,
                "First direction reversal after maximum knee flexion.",
            )
            if knee_onset is not None:
                events.append(knee_onset)

        hip_minimum = self._map_local_minimum(
            hip,
            BasketballEventType.MAX_HIP_FLEXION,
            self.dip_frame,
            "Lowest hip-angle event; interpreted as maximum hip flexion.",
        )
        if hip_minimum is not None:
            events.append(hip_minimum)
            hip_onset = self._map_post_minimum_reversal(
                hip,
                hip_minimum.frame,
                BasketballEventType.HIP_EXTENSION_ONSET,
                "First direction reversal after maximum hip flexion.",
            )
            if hip_onset is not None:
                events.append(hip_onset)

        pelvis_lowest = self._map_local_minimum(
            pelvis,
            BasketballEventType.PELVIS_LOWEST_POINT,
            self.dip_frame,
            "Lowest pelvis-height event.",
        )
        if pelvis_lowest is not None:
            events.append(pelvis_lowest)
            pelvis_upward = self._map_post_minimum_reversal(
                pelvis,
                pelvis_lowest.frame,
                BasketballEventType.PELVIS_UPWARD_MOTION_ONSET,
                "First upward reversal after the pelvis lowest point.",
            )
            if pelvis_upward is not None:
                events.append(pelvis_upward)

        shoulder_onset = self._map_velocity_reversal_near(
            shoulder,
            BasketballEventType.SHOULDER_PROPULSION_ONSET,
            self.dip_frame or self.motion_start_frame,
            "Shoulder direction reversal near propulsion.",
        )
        if shoulder_onset is not None:
            events.append(shoulder_onset)

        elbow_onset = self._map_velocity_reversal_near(
            elbow,
            BasketballEventType.ELBOW_EXTENSION_ONSET,
            self.takeoff_frame or max(
                self.motion_start_frame,
                self.release_frame - 6,
            ),
            "Elbow direction reversal near takeoff and release.",
        )
        if elbow_onset is not None:
            events.append(elbow_onset)

        wrist_onset = self._map_signal_rise_onset(
            wrist,
            BasketballEventType.WRIST_PROPULSION_ONSET,
            self.dip_frame or self.motion_start_frame,
            0.30,
            "Sustained rise in wrist speed during propulsion.",
        )
        if wrist_onset is not None:
            events.append(wrist_onset)

        ball_lowest = self._map_local_minimum(
            ball_height,
            BasketballEventType.BALL_LOWEST_POINT,
            self.dip_frame,
            "Lowest tracked ball-height event before release.",
        )
        if ball_lowest is not None:
            events.append(ball_lowest)

        ball_upward = self._map_signal_rise_onset(
            ball_vertical_speed,
            BasketballEventType.BALL_UPWARD_PROPULSION_ONSET,
            self.dip_frame or self.motion_start_frame,
            0.30,
            "Sustained increase in positive ball vertical speed.",
        )
        if ball_upward is not None:
            events.append(ball_upward)

        events.append(
            BasketballEvent(
                event_type=BasketballEventType.RELEASE,
                frame=self.release_frame,
                source_signal="Shot analyzer",
                source_motion_event=None,
                value=None,
                confidence=1.0,
                explanation="Release frame supplied by the shot analyzer.",
            )
        )

        if self.ball_apex_frame is not None:
            events.append(
                BasketballEvent(
                    event_type=BasketballEventType.BALL_APEX,
                    frame=self.ball_apex_frame,
                    source_signal="Shot analyzer",
                    source_motion_event=None,
                    value=None,
                    confidence=1.0,
                    explanation="Ball apex supplied by the shot analyzer.",
                )
            )

        events.sort(
            key=lambda event: (
                event.frame,
                event.event_type.value,
            )
        )

        return BasketballEventTimeline(
            motion_start_frame=self.motion_start_frame,
            dip_frame=self.dip_frame,
            takeoff_frame=self.takeoff_frame,
            release_frame=self.release_frame,
            ball_apex_frame=self.ball_apex_frame,
            events=tuple(events),
        )

    def _analyze_signal(
        self,
        signal_name: str,
        values: np.ndarray,
    ) -> MotionSignalAnalysis:
        return self.detector.analyze(
            signal_name=signal_name,
            values=values,
            start_frame=self.motion_start_frame,
            end_frame=self.release_frame,
        )

    def _map_local_minimum(
        self,
        analysis: MotionSignalAnalysis,
        event_type: BasketballEventType,
        preferred_frame: int | None,
        explanation: str,
    ) -> BasketballEvent | None:
        candidates = list(
            analysis.events_of_type(MotionEventType.LOCAL_MINIMUM)
        )
        if not candidates:
            return None

        if preferred_frame is None:
            selected = min(
                candidates,
                key=lambda event: (
                    event.value,
                    event.frame,
                ),
            )
        else:
            selected = min(
                candidates,
                key=lambda event: (
                    abs(event.frame - preferred_frame),
                    event.value,
                ),
            )

        return self._convert_event(
            selected,
            event_type,
            analysis.signal_name,
            explanation,
        )

    def _map_post_minimum_reversal(
        self,
        analysis: MotionSignalAnalysis,
        reference_frame: int,
        event_type: BasketballEventType,
        explanation: str,
    ) -> BasketballEvent | None:
        candidates = [
            event
            for event in analysis.events_of_type(
                MotionEventType.VELOCITY_REVERSAL
            )
            if event.frame >= reference_frame
        ]
        if not candidates:
            return None

        selected = min(
            candidates,
            key=lambda event: event.frame,
        )
        return self._convert_event(
            selected,
            event_type,
            analysis.signal_name,
            explanation,
        )

    def _map_velocity_reversal_near(
        self,
        analysis: MotionSignalAnalysis,
        event_type: BasketballEventType,
        preferred_frame: int,
        explanation: str,
    ) -> BasketballEvent | None:
        candidates = list(
            analysis.events_of_type(
                MotionEventType.VELOCITY_REVERSAL
            )
        )
        if not candidates:
            return None

        selected = min(
            candidates,
            key=lambda event: (
                abs(event.frame - preferred_frame),
                event.frame,
            ),
        )
        return self._convert_event(
            selected,
            event_type,
            analysis.signal_name,
            explanation,
        )

    def _map_signal_rise_onset(
        self,
        analysis: MotionSignalAnalysis,
        event_type: BasketballEventType,
        search_start: int,
        threshold_ratio: float,
        explanation: str,
    ) -> BasketballEvent | None:
        values = analysis.values
        sample = values[
            search_start:self.release_frame + 1
        ]
        valid = sample[~np.isnan(sample)]

        if valid.size == 0:
            return None

        peak_value = float(np.max(valid))
        if peak_value <= 0:
            return None

        threshold = peak_value * threshold_ratio
        sustained_frames = 3

        for frame in range(
            search_start,
            self.release_frame - sustained_frames + 2,
        ):
            window = np.asarray(
                values[frame:frame + sustained_frames],
                dtype=float,
            )
            if (
                window.size != sustained_frames
                or np.any(np.isnan(window))
            ):
                continue

            if (
                np.all(window >= threshold)
                and np.all(np.diff(window) >= 0)
            ):
                return BasketballEvent(
                    event_type=event_type,
                    frame=frame,
                    source_signal=analysis.signal_name,
                    source_motion_event=None,
                    value=float(values[frame]),
                    confidence=0.70,
                    explanation=explanation,
                )

        return None

    @staticmethod
    def _convert_event(
        event: MotionEvent,
        event_type: BasketballEventType,
        source_signal: str,
        explanation: str,
    ) -> BasketballEvent:
        return BasketballEvent(
            event_type=event_type,
            frame=event.frame,
            source_signal=source_signal,
            source_motion_event=event.event_type,
            value=event.value,
            confidence=event.confidence,
            explanation=explanation,
        )

    def _side_series(
        self,
        right: np.ndarray,
        left: np.ndarray,
    ) -> np.ndarray:
        return right if self.shooting_side == "RIGHT" else left


def _run_basketball_event_mapper_test() -> None:
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

    timeline = BasketballEventMapper(
        time_series=time_series,
        sampling_rate=float(
            trial_data.get("sampling_rate", 30)
        ),
        motion_start_frame=shot_analysis.motion_start_frame,
        dip_frame=shot_analysis.dip_frame,
        takeoff_frame=shot_analysis.takeoff_frame,
        release_frame=shot_analysis.release_frame,
        ball_apex_frame=shot_analysis.ball_apex_frame,
        shooting_side="RIGHT",
    ).build_timeline()

    assert len(timeline.events) > 0
    release_event = timeline.get_event(
        BasketballEventType.RELEASE
    )
    assert release_event is not None
    assert release_event.frame == shot_analysis.release_frame

    print("BASKETBALL EVENT MAPPER TEST PASSED")
    print(f"Trial file: {trial_file.name}")
    print(
        f"Analysis window: "
        f"{timeline.motion_start_frame} to {timeline.release_frame}"
    )
    print(f"Basketball events: {len(timeline.events)}")
    print()

    for event in timeline.events:
        source_event = (
            "direct"
            if event.source_motion_event is None
            else event.source_motion_event.value
        )
        value_text = (
            "N/A"
            if event.value is None
            else f"{event.value:.2f}"
        )
        print(
            f"Frame {event.frame}: "
            f"{event.event_type.value}, "
            f"source {source_event}, "
            f"value {value_text}, "
            f"confidence {event.confidence:.2f}"
        )


if __name__ == "__main__":
    _run_basketball_event_mapper_test()