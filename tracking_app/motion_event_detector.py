from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from tracking_app.analyzer import ShotAnalyzer
from tracking_app.data_loader import load_first_trial
from tracking_app.timeseries import TimeSeriesAnalyzer


class MotionEventType(str, Enum):
    """
    Generic motion-event categories independent of basketball.
    """

    LOCAL_MINIMUM = "local_minimum"
    LOCAL_MAXIMUM = "local_maximum"
    ZERO_CROSSING_UP = "zero_crossing_up"
    ZERO_CROSSING_DOWN = "zero_crossing_down"
    VELOCITY_REVERSAL = "velocity_reversal"
    ACCELERATION_REVERSAL = "acceleration_reversal"
    INFLECTION_POINT = "inflection_point"
    PLATEAU_START = "plateau_start"
    PLATEAU_END = "plateau_end"


@dataclass(frozen=True)
class MotionEvent:
    """
    One meaningful event detected in a one-dimensional signal.
    """

    event_type: MotionEventType
    frame: int
    value: float
    velocity: float | None
    acceleration: float | None
    prominence: float | None
    confidence: float


@dataclass(frozen=True)
class MotionSignalAnalysis:
    """
    Complete generic event analysis for one signal.
    """

    signal_name: str
    sampling_rate: float
    analysis_start_frame: int
    analysis_end_frame: int

    values: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray

    events: tuple[MotionEvent, ...]

    @property
    def total_events(self) -> int:
        return len(self.events)

    def events_of_type(
        self,
        event_type: MotionEventType,
    ) -> tuple[MotionEvent, ...]:
        return tuple(
            event
            for event in self.events
            if event.event_type == event_type
        )


class MotionEventDetector:
    """
    Generic one-dimensional motion-event detector.

    Version 2 improvements
    ----------------------
    - Uses a configurable analysis window.
    - Detects extrema across a neighborhood, not only adjacent frames.
    - Uses noise-aware sign-change thresholds.
    - Separates acceleration reversals from inflection points.
    - Requires plateaus to persist for several frames.
    - Reduces duplicate micro-events caused by tracking noise.
    """

    def __init__(
        self,
        sampling_rate: float,
        smoothing_window: int = 7,
        extrema_window_radius: int = 4,
        minimum_prominence_ratio: float = 0.04,
        sign_change_threshold_ratio: float = 0.04,
        plateau_velocity_ratio: float = 0.04,
        minimum_plateau_frames: int = 4,
        minimum_event_separation_frames: int = 4,
    ) -> None:
        if sampling_rate <= 0:
            raise ValueError(
                "sampling_rate must be greater than zero."
            )

        self.sampling_rate = float(
            sampling_rate
        )

        self.smoothing_window = max(
            1,
            int(smoothing_window),
        )

        if self.smoothing_window % 2 == 0:
            self.smoothing_window += 1

        self.extrema_window_radius = max(
            2,
            int(extrema_window_radius),
        )

        self.minimum_prominence_ratio = max(
            0.0,
            float(minimum_prominence_ratio),
        )

        self.sign_change_threshold_ratio = max(
            0.0,
            float(sign_change_threshold_ratio),
        )

        self.plateau_velocity_ratio = max(
            0.0,
            float(plateau_velocity_ratio),
        )

        self.minimum_plateau_frames = max(
            2,
            int(minimum_plateau_frames),
        )

        self.minimum_event_separation_frames = max(
            1,
            int(minimum_event_separation_frames),
        )

    # =====================================================
    # PUBLIC ANALYSIS
    # =====================================================

    def analyze(
        self,
        signal_name: str,
        values: np.ndarray,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> MotionSignalAnalysis:
        """
        Detect generic motion events in one scalar signal.
        """

        numeric_values = np.asarray(
            values,
            dtype=float,
        )

        total_frames = len(
            numeric_values
        )

        if total_frames == 0:
            raise ValueError(
                "Motion-event analysis requires at least one frame."
            )

        resolved_start = (
            0
            if start_frame is None
            else max(
                0,
                min(
                    int(start_frame),
                    total_frames - 1,
                ),
            )
        )

        resolved_end = (
            total_frames - 1
            if end_frame is None
            else max(
                resolved_start,
                min(
                    int(end_frame),
                    total_frames - 1,
                ),
            )
        )

        smoothed = self._smooth(
            numeric_values
        )

        velocity = self._smooth(
            self._derivative(
                smoothed
            )
        )

        acceleration = self._smooth(
            self._derivative(
                velocity
            )
        )

        jerk = self._smooth(
            self._derivative(
                acceleration
            )
        )

        candidates: list[
            MotionEvent
        ] = []

        candidates.extend(
            self._detect_local_extrema(
                values=smoothed,
                velocity=velocity,
                acceleration=acceleration,
                start_frame=resolved_start,
                end_frame=resolved_end,
            )
        )

        candidates.extend(
            self._detect_zero_crossings(
                values=smoothed,
                velocity=velocity,
                acceleration=acceleration,
                start_frame=resolved_start,
                end_frame=resolved_end,
            )
        )

        candidates.extend(
            self._detect_velocity_reversals(
                values=smoothed,
                velocity=velocity,
                acceleration=acceleration,
                start_frame=resolved_start,
                end_frame=resolved_end,
            )
        )

        # Acceleration reversal now means a local extremum in
        # acceleration, identified by a jerk sign change.
        candidates.extend(
            self._detect_acceleration_reversals(
                values=smoothed,
                velocity=velocity,
                acceleration=acceleration,
                jerk=jerk,
                start_frame=resolved_start,
                end_frame=resolved_end,
            )
        )

        # Inflection point remains an acceleration zero crossing.
        candidates.extend(
            self._detect_inflection_points(
                values=smoothed,
                velocity=velocity,
                acceleration=acceleration,
                start_frame=resolved_start,
                end_frame=resolved_end,
            )
        )

        candidates.extend(
            self._detect_plateaus(
                values=smoothed,
                velocity=velocity,
                acceleration=acceleration,
                start_frame=resolved_start,
                end_frame=resolved_end,
            )
        )

        filtered = self._deduplicate_events(
            candidates
        )

        filtered.sort(
            key=lambda event: (
                event.frame,
                event.event_type.value,
            )
        )

        return MotionSignalAnalysis(
            signal_name=signal_name,
            sampling_rate=self.sampling_rate,
            analysis_start_frame=resolved_start,
            analysis_end_frame=resolved_end,
            values=smoothed,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            events=tuple(filtered),
        )

    # =====================================================
    # LOCAL EXTREMA
    # =====================================================

    def _detect_local_extrema(
        self,
        values: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        start_frame: int,
        end_frame: int,
    ) -> list[MotionEvent]:
        """
        Detect extrema using a local neighborhood and prominence.
        """

        events: list[
            MotionEvent
        ] = []

        analysis_sample = values[
            start_frame:
            end_frame + 1
        ]

        valid_values = analysis_sample[
            ~np.isnan(analysis_sample)
        ]

        if valid_values.size == 0:
            return events

        signal_range = float(
            np.max(valid_values)
            - np.min(valid_values)
        )

        minimum_prominence = (
            signal_range
            * self.minimum_prominence_ratio
        )

        radius = self.extrema_window_radius

        first_frame = max(
            start_frame + radius,
            radius,
        )

        last_frame = min(
            end_frame - radius,
            len(values) - radius - 1,
        )

        for frame in range(
            first_frame,
            last_frame + 1,
        ):
            current = values[
                frame
            ]

            if np.isnan(current):
                continue

            left = values[
                frame - radius:
                frame
            ]

            right = values[
                frame + 1:
                frame + radius + 1
            ]

            left = left[
                ~np.isnan(left)
            ]

            right = right[
                ~np.isnan(right)
            ]

            if (
                left.size == 0
                or right.size == 0
            ):
                continue

            surrounding = np.concatenate(
                (
                    left,
                    right,
                )
            )

            local_maximum = (
                current
                >= np.max(left)
                and current
                >= np.max(right)
            )

            local_minimum = (
                current
                <= np.min(left)
                and current
                <= np.min(right)
            )

            if local_maximum:
                prominence = float(
                    current
                    - np.min(surrounding)
                )

                if prominence >= minimum_prominence:
                    events.append(
                        self._make_event(
                            event_type=(
                                MotionEventType
                                .LOCAL_MAXIMUM
                            ),
                            frame=frame,
                            values=values,
                            velocity=velocity,
                            acceleration=acceleration,
                            prominence=prominence,
                            signal_range=signal_range,
                        )
                    )

            if local_minimum:
                prominence = float(
                    np.max(surrounding)
                    - current
                )

                if prominence >= minimum_prominence:
                    events.append(
                        self._make_event(
                            event_type=(
                                MotionEventType
                                .LOCAL_MINIMUM
                            ),
                            frame=frame,
                            values=values,
                            velocity=velocity,
                            acceleration=acceleration,
                            prominence=prominence,
                            signal_range=signal_range,
                        )
                    )

        return events

    # =====================================================
    # ZERO CROSSINGS
    # =====================================================

    def _detect_zero_crossings(
        self,
        values: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        start_frame: int,
        end_frame: int,
    ) -> list[MotionEvent]:
        """
        Detect the original signal crossing zero.
        """

        return self._detect_sign_crossings(
            signal=values,
            event_type_up=(
                MotionEventType
                .ZERO_CROSSING_UP
            ),
            event_type_down=(
                MotionEventType
                .ZERO_CROSSING_DOWN
            ),
            values=values,
            velocity=velocity,
            acceleration=acceleration,
            start_frame=start_frame,
            end_frame=end_frame,
            minimum_magnitude=0.0,
        )

    # =====================================================
    # REVERSALS AND INFLECTIONS
    # =====================================================

    def _detect_velocity_reversals(
        self,
        values: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        start_frame: int,
        end_frame: int,
    ) -> list[MotionEvent]:
        """
        Detect meaningful movement-direction reversals.
        """

        threshold = self._signal_threshold(
            velocity[
                start_frame:
                end_frame + 1
            ]
        )

        return self._detect_single_type_sign_changes(
            signal=velocity,
            event_type=(
                MotionEventType
                .VELOCITY_REVERSAL
            ),
            values=values,
            velocity=velocity,
            acceleration=acceleration,
            start_frame=start_frame,
            end_frame=end_frame,
            minimum_magnitude=threshold,
        )

    def _detect_acceleration_reversals(
        self,
        values: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        jerk: np.ndarray,
        start_frame: int,
        end_frame: int,
    ) -> list[MotionEvent]:
        """
        Detect acceleration extrema through jerk sign changes.
        """

        threshold = self._signal_threshold(
            jerk[
                start_frame:
                end_frame + 1
            ]
        )

        return self._detect_single_type_sign_changes(
            signal=jerk,
            event_type=(
                MotionEventType
                .ACCELERATION_REVERSAL
            ),
            values=values,
            velocity=velocity,
            acceleration=acceleration,
            start_frame=start_frame,
            end_frame=end_frame,
            minimum_magnitude=threshold,
        )

    def _detect_inflection_points(
        self,
        values: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        start_frame: int,
        end_frame: int,
    ) -> list[MotionEvent]:
        """
        Detect curvature changes through acceleration zero crossings.
        """

        threshold = self._signal_threshold(
            acceleration[
                start_frame:
                end_frame + 1
            ]
        )

        return self._detect_single_type_sign_changes(
            signal=acceleration,
            event_type=(
                MotionEventType
                .INFLECTION_POINT
            ),
            values=values,
            velocity=velocity,
            acceleration=acceleration,
            start_frame=start_frame,
            end_frame=end_frame,
            minimum_magnitude=threshold,
        )

    def _detect_single_type_sign_changes(
        self,
        signal: np.ndarray,
        event_type: MotionEventType,
        values: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        start_frame: int,
        end_frame: int,
        minimum_magnitude: float,
    ) -> list[MotionEvent]:
        """
        Detect thresholded sign changes for one event type.
        """

        events: list[
            MotionEvent
        ] = []

        for frame in range(
            max(
                start_frame + 1,
                1,
            ),
            end_frame + 1,
        ):
            previous = signal[
                frame - 1
            ]

            current = signal[
                frame
            ]

            if (
                np.isnan(previous)
                or np.isnan(current)
            ):
                continue

            if (
                max(
                    abs(previous),
                    abs(current),
                )
                < minimum_magnitude
            ):
                continue

            crossed = (
                previous < 0
                and current >= 0
            ) or (
                previous > 0
                and current <= 0
            )

            if crossed:
                events.append(
                    self._make_event(
                        event_type=event_type,
                        frame=frame,
                        values=values,
                        velocity=velocity,
                        acceleration=acceleration,
                        prominence=abs(
                            current
                            - previous
                        ),
                        signal_range=None,
                    )
                )

        return events

    def _detect_sign_crossings(
        self,
        signal: np.ndarray,
        event_type_up: MotionEventType,
        event_type_down: MotionEventType,
        values: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        start_frame: int,
        end_frame: int,
        minimum_magnitude: float,
    ) -> list[MotionEvent]:
        """
        Detect upward and downward sign crossings separately.
        """

        events: list[
            MotionEvent
        ] = []

        for frame in range(
            max(
                start_frame + 1,
                1,
            ),
            end_frame + 1,
        ):
            previous = signal[
                frame - 1
            ]

            current = signal[
                frame
            ]

            if (
                np.isnan(previous)
                or np.isnan(current)
            ):
                continue

            if (
                max(
                    abs(previous),
                    abs(current),
                )
                < minimum_magnitude
            ):
                continue

            if (
                previous < 0
                and current >= 0
            ):
                event_type = (
                    event_type_up
                )

            elif (
                previous > 0
                and current <= 0
            ):
                event_type = (
                    event_type_down
                )

            else:
                continue

            events.append(
                self._make_event(
                    event_type=event_type,
                    frame=frame,
                    values=values,
                    velocity=velocity,
                    acceleration=acceleration,
                    prominence=abs(
                        current
                        - previous
                    ),
                    signal_range=None,
                )
            )

        return events

    # =====================================================
    # PLATEAUS
    # =====================================================

    def _detect_plateaus(
        self,
        values: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        start_frame: int,
        end_frame: int,
    ) -> list[MotionEvent]:
        """
        Detect low-velocity intervals that persist long enough.
        """

        sample = velocity[
            start_frame:
            end_frame + 1
        ]

        valid_velocity = sample[
            ~np.isnan(sample)
        ]

        if valid_velocity.size == 0:
            return []

        velocity_scale = float(
            np.max(
                np.abs(
                    valid_velocity
                )
            )
        )

        threshold = (
            velocity_scale
            * self.plateau_velocity_ratio
        )

        plateau_mask = np.zeros(
            len(velocity),
            dtype=bool,
        )

        for frame in range(
            start_frame,
            end_frame + 1,
        ):
            value = velocity[
                frame
            ]

            if np.isnan(value):
                continue

            plateau_mask[
                frame
            ] = (
                abs(value)
                <= threshold
            )

        events: list[
            MotionEvent
        ] = []

        frame = start_frame

        while frame <= end_frame:
            if not plateau_mask[
                frame
            ]:
                frame += 1
                continue

            plateau_start = frame

            while (
                frame <= end_frame
                and plateau_mask[frame]
            ):
                frame += 1

            plateau_end = frame - 1

            duration = (
                plateau_end
                - plateau_start
                + 1
            )

            if (
                duration
                < self.minimum_plateau_frames
            ):
                continue

            events.append(
                self._make_event(
                    event_type=(
                        MotionEventType
                        .PLATEAU_START
                    ),
                    frame=plateau_start,
                    values=values,
                    velocity=velocity,
                    acceleration=acceleration,
                    prominence=None,
                    signal_range=None,
                )
            )

            events.append(
                self._make_event(
                    event_type=(
                        MotionEventType
                        .PLATEAU_END
                    ),
                    frame=plateau_end,
                    values=values,
                    velocity=velocity,
                    acceleration=acceleration,
                    prominence=None,
                    signal_range=None,
                )
            )

        return events

    # =====================================================
    # EVENT CREATION AND FILTERING
    # =====================================================

    def _make_event(
        self,
        event_type: MotionEventType,
        frame: int,
        values: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        prominence: float | None,
        signal_range: float | None,
    ) -> MotionEvent:
        """
        Build one event with a transparent confidence value.
        """

        confidence = 0.50

        if (
            prominence is not None
            and signal_range is not None
            and signal_range > 0
        ):
            confidence = min(
                1.0,
                0.50
                + prominence
                / signal_range,
            )

        value = float(
            values[frame]
        )

        velocity_value = (
            None
            if np.isnan(
                velocity[frame]
            )
            else float(
                velocity[frame]
            )
        )

        acceleration_value = (
            None
            if np.isnan(
                acceleration[frame]
            )
            else float(
                acceleration[frame]
            )
        )

        return MotionEvent(
            event_type=event_type,
            frame=int(frame),
            value=value,
            velocity=velocity_value,
            acceleration=(
                acceleration_value
            ),
            prominence=prominence,
            confidence=float(
                confidence
            ),
        )

    def _deduplicate_events(
        self,
        events: list[MotionEvent],
    ) -> list[MotionEvent]:
        """
        Remove nearby duplicates of the same event type.
        """

        grouped: dict[
            MotionEventType,
            list[MotionEvent],
        ] = {}

        for event in events:
            grouped.setdefault(
                event.event_type,
                [],
            ).append(event)

        output: list[
            MotionEvent
        ] = []

        for group in grouped.values():
            group.sort(
                key=lambda event: (
                    event.frame,
                    -event.confidence,
                )
            )

            accepted: list[
                MotionEvent
            ] = []

            for event in group:
                if not accepted:
                    accepted.append(
                        event
                    )
                    continue

                previous = accepted[
                    -1
                ]

                if (
                    event.frame
                    - previous.frame
                    < self.minimum_event_separation_frames
                ):
                    if (
                        event.confidence
                        > previous.confidence
                    ):
                        accepted[
                            -1
                        ] = event

                    continue

                accepted.append(
                    event
                )

            output.extend(
                accepted
            )

        return output

    # =====================================================
    # NUMERIC HELPERS
    # =====================================================

    def _signal_threshold(
        self,
        values: np.ndarray,
    ) -> float:
        """
        Build a scale-aware threshold for sign-change filtering.
        """

        valid = values[
            ~np.isnan(values)
        ]

        if valid.size == 0:
            return 0.0

        scale = float(
            np.max(
                np.abs(
                    valid
                )
            )
        )

        return (
            scale
            * self.sign_change_threshold_ratio
        )

    def _derivative(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """
        Frame-to-frame derivative.
        """

        derivative = np.full_like(
            values,
            np.nan,
            dtype=float,
        )

        for frame in range(
            1,
            len(values),
        ):
            previous = values[
                frame - 1
            ]

            current = values[
                frame
            ]

            if (
                np.isnan(previous)
                or np.isnan(current)
            ):
                continue

            derivative[frame] = (
                current - previous
            ) * self.sampling_rate

        return derivative

    def _smooth(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """
        Centered NaN-aware moving average.
        """

        if self.smoothing_window <= 1:
            return values.copy()

        half_window = (
            self.smoothing_window // 2
        )

        output = np.full_like(
            values,
            np.nan,
            dtype=float,
        )

        for frame in range(
            len(values)
        ):
            start = max(
                0,
                frame - half_window,
            )

            end = min(
                len(values),
                frame
                + half_window
                + 1,
            )

            sample = values[
                start:end
            ]

            valid = sample[
                ~np.isnan(sample)
            ]

            if valid.size > 0:
                output[frame] = float(
                    np.mean(valid)
                )

        return output


def _run_motion_event_detector_test() -> None:
    """
    Test the refined detector on the active shooting window.
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

    detector = MotionEventDetector(
        sampling_rate=float(
            trial_data.get(
                "sampling_rate",
                30,
            )
        ),
        smoothing_window=7,
        extrema_window_radius=4,
        minimum_prominence_ratio=0.03,
        sign_change_threshold_ratio=0.05,
        plateau_velocity_ratio=0.04,
        minimum_plateau_frames=4,
        minimum_event_separation_frames=4,
    )

    result = detector.analyze(
        signal_name=(
            "Right knee angle"
        ),
        values=(
            time_series
            .right_knee_angle
        ),
        start_frame=(
            shot_analysis.motion_start_frame
        ),
        end_frame=(
            shot_analysis.release_frame
        ),
    )

    assert result.total_events > 0
    assert len(result.values) == (
        time_series.total_frames
    )

    print(
        "REFINED MOTION EVENT DETECTOR TEST PASSED"
    )

    print(
        f"Trial file: {trial_file.name}"
    )

    print(
        f"Signal: {result.signal_name}"
    )

    print(
        "Analysis window: "
        f"{result.analysis_start_frame} "
        f"to {result.analysis_end_frame}"
    )

    print(
        f"Total events: {result.total_events}"
    )

    print()

    for event_type in MotionEventType:
        count = len(
            result.events_of_type(
                event_type
            )
        )

        print(
            f"{event_type.value}: "
            f"{count}"
        )

    print()
    print("Detected events:")

    for event in result.events:
        print(
            f"Frame {event.frame}: "
            f"{event.event_type.value}, "
            f"value {event.value:.2f}, "
            f"confidence "
            f"{event.confidence:.2f}"
        )


if __name__ == "__main__":
    _run_motion_event_detector_test()