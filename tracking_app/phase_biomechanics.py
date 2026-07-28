from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tracking_app.analyzer import ShotAnalyzer
from tracking_app.data_loader import load_first_trial
from tracking_app.timeseries import ShotTimeSeries, TimeSeriesAnalyzer


@dataclass(frozen=True)
class MovementPhase:
    """
    Refined phase landmarks for one body segment.
    """

    segment: str

    preparatory_onset_frame: int | None
    propulsion_onset_frame: int | None

    peak_velocity_frame: int | None
    peak_acceleration_frame: int | None
    deceleration_start_frame: int | None
    end_frame: int | None

    peak_velocity_value: float | None
    peak_acceleration_value: float | None

    preparatory_to_propulsion_ms: float | None
    propulsion_to_peak_ms: float | None
    peak_to_release_ms: float | None
    propulsion_duration_ms: float | None


@dataclass(frozen=True)
class PhaseBasedBiomechanicsAnalysis:
    """
    Complete refined phase-based analysis for one shot.
    """

    motion_start_frame: int
    dip_frame: int
    takeoff_frame: int
    release_frame: int
    sampling_rate: float

    knee_phase: MovementPhase
    hip_phase: MovementPhase
    pelvis_phase: MovementPhase
    shoulder_phase: MovementPhase
    elbow_phase: MovementPhase
    wrist_phase: MovementPhase
    ball_phase: MovementPhase

    preparatory_onset_sequence: tuple[str, ...]
    propulsion_onset_sequence: tuple[str, ...]
    peak_sequence: tuple[str, ...]


class RefinedPhaseBiomechanicsAnalyzer:
    """
    Detect preparatory and propulsive movement phases.

    Improvements over the original phase detector
    ---------------------------------------------
    - Uses a baseline-noise threshold.
    - Requires sustained movement over several frames.
    - Uses segment-specific propulsion search windows.
    - Separates preparatory onset from propulsion onset.
    - Limits ball propulsion analysis to the dip-to-release phase.
    - Uses direction-aware joint propulsion signals.
    - Uses vertical ball speed to suppress early handling peaks.
    - Applies a robust baseline threshold that cannot exceed the signal peak.
    - Uses a trend-based fallback for short, steadily rising propulsion bursts.
    - Uses release-relative fallbacks when dip or takeoff is not detected.
    - Keeps all calculations independent of the viewer.
    """

    SEGMENT_SETTINGS = {
        "Knee": {
            "onset_ratio": 0.20,
            "sustained_frames": 3,
            "propulsion_start": "dip",
        },
        "Hip": {
            "onset_ratio": 0.18,
            "sustained_frames": 3,
            "propulsion_start": "dip",
        },
        "Pelvis": {
            "onset_ratio": 0.18,
            "sustained_frames": 3,
            "propulsion_start": "dip",
        },
        "Shoulder": {
            "onset_ratio": 0.22,
            "sustained_frames": 3,
            "propulsion_start": "dip",
        },
        "Elbow": {
            "onset_ratio": 0.25,
            "sustained_frames": 3,
            "propulsion_start": "takeoff_minus",
        },
        "Wrist": {
            "onset_ratio": 0.25,
            "sustained_frames": 3,
            "propulsion_start": "dip",
        },
        "Ball": {
            "onset_ratio": 0.30,
            "sustained_frames": 3,
            "propulsion_start": "dip",
        },
    }

    def __init__(
        self,
        time_series: ShotTimeSeries,
        sampling_rate: float,
        motion_start_frame: int | None,
        dip_frame: int | None,
        takeoff_frame: int | None,
        release_frame: int | None,
        shooting_side: str = "RIGHT",
        smoothing_window: int = 5,
        baseline_frame_count: int = 15,
        noise_multiplier: float = 3.0,
    ) -> None:
        if sampling_rate <= 0:
            raise ValueError(
                "sampling_rate must be greater than zero."
            )

        side = shooting_side.strip().upper()

        if side not in {"RIGHT", "LEFT"}:
            raise ValueError(
                "shooting_side must be RIGHT or LEFT."
            )

        total_frames = time_series.total_frames

        self.time_series = time_series
        self.sampling_rate = float(sampling_rate)

        if release_frame is None:
            raise ValueError(
                "Refined phase analysis requires a detected release frame."
            )

        self.release_frame = self._clamp_frame(
            release_frame,
            total_frames,
        )

        if motion_start_frame is None:
            self.motion_start_frame = max(
                0,
                self.release_frame
                - int(round(self.sampling_rate * 2.5)),
            )
        else:
            self.motion_start_frame = self._clamp_frame(
                motion_start_frame,
                total_frames,
            )

        # Some trials do not produce a reliable dip or takeoff event.
        # Use release-relative fallbacks so phase analysis can continue
        # without inventing values outside a realistic shot window.
        if dip_frame is None:
            resolved_dip = max(
                self.motion_start_frame,
                self.release_frame
                - int(round(self.sampling_rate * 0.35)),
            )
        else:
            resolved_dip = self._clamp_frame(
                dip_frame,
                total_frames,
            )

        if takeoff_frame is None:
            resolved_takeoff = max(
                resolved_dip,
                self.release_frame
                - max(
                    1,
                    int(round(self.sampling_rate * 0.08)),
                ),
            )
        else:
            resolved_takeoff = self._clamp_frame(
                takeoff_frame,
                total_frames,
            )

        self.dip_frame = max(
            self.motion_start_frame,
            min(
                resolved_dip,
                self.release_frame,
            ),
        )

        self.takeoff_frame = max(
            self.dip_frame,
            min(
                resolved_takeoff,
                self.release_frame,
            ),
        )

        self.shooting_side = side

        self.smoothing_window = max(
            1,
            int(smoothing_window),
        )

        if self.smoothing_window % 2 == 0:
            self.smoothing_window += 1

        self.baseline_frame_count = max(
            5,
            int(baseline_frame_count),
        )

        self.noise_multiplier = max(
            1.0,
            float(noise_multiplier),
        )

    # =====================================================
    # PUBLIC ANALYSIS
    # =====================================================

    def analyze(
        self,
    ) -> PhaseBasedBiomechanicsAnalysis:
        """
        Analyze refined movement phases for all segments.
        """

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

        knee_velocity = self._smooth(
            self._derivative(knee_angle)
        )

        hip_velocity = self._smooth(
            self._derivative(hip_angle)
        )

        pelvis_velocity = self._smooth(
            self._derivative(
                self.time_series.pelvis_height
            )
        )

        shoulder_velocity = self._smooth(
            self._derivative(shoulder_angle)
        )

        elbow_velocity = self._smooth(
            self._derivative(elbow_angle)
        )

        wrist_velocity = self._smooth(
            wrist_speed
        )

        # Vertical ball speed is a better propulsion signal than
        # total ball speed because early horizontal handling can create
        # false peaks before the shot-driving phase.
        ball_velocity = self._smooth(
            self.time_series.ball_vertical_speed
        )

        knee_phase = self._analyze_segment(
            "Knee",
            self._orient_propulsion_signal(
                knee_velocity
            ),
        )

        hip_phase = self._analyze_segment(
            "Hip",
            self._orient_propulsion_signal(
                hip_velocity
            ),
        )

        pelvis_phase = self._analyze_segment(
            "Pelvis",
            self._orient_propulsion_signal(
                pelvis_velocity
            ),
        )

        shoulder_phase = self._analyze_segment(
            "Shoulder",
            self._orient_propulsion_signal(
                shoulder_velocity
            ),
        )

        elbow_phase = self._analyze_segment(
            "Elbow",
            self._orient_propulsion_signal(
                elbow_velocity
            ),
        )

        wrist_phase = self._analyze_segment(
            "Wrist",
            wrist_velocity,
        )

        ball_phase = self._analyze_segment(
            "Ball",
            ball_velocity,
        )

        phases = (
            knee_phase,
            hip_phase,
            pelvis_phase,
            shoulder_phase,
            elbow_phase,
            wrist_phase,
            ball_phase,
        )

        preparatory_sequence = self._build_sequence(
            phases,
            "preparatory_onset_frame",
        )

        propulsion_sequence = self._build_sequence(
            phases,
            "propulsion_onset_frame",
        )

        peak_sequence = self._build_sequence(
            phases,
            "peak_velocity_frame",
        )

        return PhaseBasedBiomechanicsAnalysis(
            motion_start_frame=self.motion_start_frame,
            dip_frame=self.dip_frame,
            takeoff_frame=self.takeoff_frame,
            release_frame=self.release_frame,
            sampling_rate=self.sampling_rate,
            knee_phase=knee_phase,
            hip_phase=hip_phase,
            pelvis_phase=pelvis_phase,
            shoulder_phase=shoulder_phase,
            elbow_phase=elbow_phase,
            wrist_phase=wrist_phase,
            ball_phase=ball_phase,
            preparatory_onset_sequence=(
                preparatory_sequence
            ),
            propulsion_onset_sequence=(
                propulsion_sequence
            ),
            peak_sequence=peak_sequence,
        )

    # =====================================================
    # SEGMENT ANALYSIS
    # =====================================================

    def _analyze_segment(
        self,
        segment: str,
        values: np.ndarray,
    ) -> MovementPhase:
        """
        Detect preparatory and propulsive landmarks.
        """

        settings = self.SEGMENT_SETTINGS[
            segment
        ]

        preparatory_start = (
            self.motion_start_frame
        )

        propulsion_start = (
            self._propulsion_search_start(
                settings["propulsion_start"]
            )
        )

        preparatory_end = max(
            preparatory_start,
            propulsion_start,
        )

        preparatory_onset = self._detect_sustained_onset(
            values=values,
            search_start=preparatory_start,
            search_end=preparatory_end,
            onset_ratio=float(
                settings["onset_ratio"]
            ),
            sustained_frames=int(
                settings["sustained_frames"]
            ),
        )

        propulsion_onset = self._detect_sustained_onset(
            values=values,
            search_start=propulsion_start,
            search_end=self.release_frame,
            onset_ratio=float(
                settings["onset_ratio"]
            ),
            sustained_frames=int(
                settings["sustained_frames"]
            ),
        )

        if propulsion_onset is None:
            return self._empty_phase(
                segment=segment,
                preparatory_onset=(
                    preparatory_onset
                ),
            )

        peak_frame, peak_value = self._find_peak(
            values=values,
            search_start=propulsion_onset,
            search_end=self.release_frame,
        )

        if (
            peak_frame is None
            or peak_value is None
        ):
            return self._empty_phase(
                segment=segment,
                preparatory_onset=(
                    preparatory_onset
                ),
                propulsion_onset=(
                    propulsion_onset
                ),
            )

        acceleration = self._smooth(
            self._derivative(values)
        )

        peak_acceleration_frame = (
            self._find_peak_acceleration(
                acceleration=acceleration,
                search_start=propulsion_onset,
                search_end=peak_frame,
            )
        )

        peak_acceleration_value = None

        if peak_acceleration_frame is not None:
            value = acceleration[
                peak_acceleration_frame
            ]

            if not np.isnan(value):
                peak_acceleration_value = float(
                    value
                )

        deceleration_frame = (
            self._find_deceleration_start(
                values=values,
                peak_frame=peak_frame,
                search_end=self.release_frame,
            )
        )

        end_frame = self._find_phase_end(
            values=values,
            peak_frame=peak_frame,
            search_end=self.release_frame,
            peak_value=peak_value,
        )

        return MovementPhase(
            segment=segment,
            preparatory_onset_frame=(
                preparatory_onset
            ),
            propulsion_onset_frame=(
                propulsion_onset
            ),
            peak_velocity_frame=peak_frame,
            peak_acceleration_frame=(
                peak_acceleration_frame
            ),
            deceleration_start_frame=(
                deceleration_frame
            ),
            end_frame=end_frame,
            peak_velocity_value=peak_value,
            peak_acceleration_value=(
                peak_acceleration_value
            ),
            preparatory_to_propulsion_ms=(
                self._frame_gap_ms(
                    preparatory_onset,
                    propulsion_onset,
                )
            ),
            propulsion_to_peak_ms=(
                self._frame_gap_ms(
                    propulsion_onset,
                    peak_frame,
                )
            ),
            peak_to_release_ms=(
                self._frame_gap_ms(
                    peak_frame,
                    self.release_frame,
                )
            ),
            propulsion_duration_ms=(
                self._frame_gap_ms(
                    propulsion_onset,
                    end_frame,
                )
            ),
        )

    # =====================================================
    # ONSET DETECTION
    # =====================================================

    def _detect_sustained_onset(
        self,
        values: np.ndarray,
        search_start: int,
        search_end: int,
        onset_ratio: float,
        sustained_frames: int,
    ) -> int | None:
        """
        Detect onset using both baseline noise and relative peak size.
        """

        if search_end <= search_start:
            return None

        search_values = np.asarray(
            values[
                search_start:
                search_end + 1
            ],
            dtype=float,
        )

        if (
            search_values.size == 0
            or np.all(np.isnan(search_values))
        ):
            return None

        peak_value = float(
            np.nanmax(search_values)
        )

        if peak_value <= 0:
            return None

        baseline_start = max(
            0,
            search_start - self.baseline_frame_count,
        )

        baseline = np.asarray(
            values[
                baseline_start:
                search_start
            ],
            dtype=float,
        )

        valid_baseline = baseline[
            ~np.isnan(baseline)
        ]

        if valid_baseline.size >= 3:
            baseline_median = float(
                np.median(valid_baseline)
            )

            median_absolute_deviation = float(
                np.median(
                    np.abs(
                        valid_baseline
                        - baseline_median
                    )
                )
            )

            robust_standard_deviation = (
                median_absolute_deviation
                * 1.4826
            )

            noise_threshold = (
                baseline_median
                + robust_standard_deviation
                * self.noise_multiplier
            )
        else:
            noise_threshold = 0.0

        relative_threshold = (
            peak_value * onset_ratio
        )

        # A baseline threshold above the observed propulsion peak can
        # never be crossed. Cap its influence while still requiring
        # meaningful movement relative to the segment's own peak.
        capped_noise_threshold = min(
            noise_threshold,
            peak_value * 0.80,
        )

        threshold = max(
            capped_noise_threshold,
            relative_threshold,
        )

        onset = self._search_sustained_threshold(
            values=values,
            search_start=search_start,
            search_end=search_end,
            sustained_frames=sustained_frames,
            threshold=threshold,
        )

        if onset is not None:
            return onset

        # Adaptive fallback: some valid propulsion signals are short
        # and do not remain above the first strict threshold for three
        # frames. Retry with a lower relative threshold while retaining
        # the baseline-noise floor.
        fallback_threshold = max(
            noise_threshold,
            peak_value * onset_ratio * 0.65,
        )

        fallback_onset = self._search_sustained_threshold(
            values=values,
            search_start=search_start,
            search_end=search_end,
            sustained_frames=max(
                2,
                sustained_frames - 1,
            ),
            threshold=fallback_threshold,
        )

        if fallback_onset is not None:
            return fallback_onset

        # Final evidence-based fallback for signals such as ball
        # vertical speed that rise steadily into release but only cross
        # the amplitude threshold briefly. Require consecutive positive
        # values and a sustained non-decreasing trend.
        return self._search_rising_burst(
            values=values,
            search_start=search_start,
            search_end=search_end,
            minimum_value=peak_value * onset_ratio * 0.45,
            sustained_frames=3,
        )

    @staticmethod
    def _search_sustained_threshold(
        values: np.ndarray,
        search_start: int,
        search_end: int,
        sustained_frames: int,
        threshold: float,
    ) -> int | None:
        """
        Search for consecutive valid frames above a threshold.
        """

        for frame in range(
            search_start,
            search_end - sustained_frames + 2,
        ):
            sample = np.asarray(
                values[
                    frame:
                    frame + sustained_frames
                ],
                dtype=float,
            )

            if (
                sample.size != sustained_frames
                or np.any(np.isnan(sample))
            ):
                continue

            if np.all(sample >= threshold):
                return frame

        return None

    @staticmethod
    def _search_rising_burst(
        values: np.ndarray,
        search_start: int,
        search_end: int,
        minimum_value: float,
        sustained_frames: int,
    ) -> int | None:
        """
        Detect a short positive burst that rises toward release.

        This is used only after both amplitude-based onset searches
        fail. Every sample must be valid, positive, above a modest
        player-relative floor, and non-decreasing.
        """

        for frame in range(
            search_start,
            search_end - sustained_frames + 2,
        ):
            sample = np.asarray(
                values[
                    frame:
                    frame + sustained_frames
                ],
                dtype=float,
            )

            if (
                sample.size != sustained_frames
                or np.any(np.isnan(sample))
            ):
                continue

            if np.any(sample < minimum_value):
                continue

            if np.all(
                np.diff(sample) >= 0
            ):
                return frame

        return None

    def _orient_propulsion_signal(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """
        Orient a signed joint-velocity signal toward propulsion.

        Joint-angle conventions can make extension positive for one
        joint and negative for another. The dominant signed movement
        between dip and release is treated as the propulsive direction.
        """

        numeric = np.asarray(
            values,
            dtype=float,
        )

        propulsion_sample = numeric[
            self.dip_frame:
            self.release_frame + 1
        ]

        valid = propulsion_sample[
            ~np.isnan(propulsion_sample)
        ]

        if valid.size == 0:
            return numeric.copy()

        positive_peak = float(
            np.max(valid)
        )

        negative_peak = abs(
            float(np.min(valid))
        )

        if negative_peak > positive_peak:
            return -numeric

        return numeric.copy()

    # =====================================================
    # LANDMARK DETECTION
    # =====================================================

    @staticmethod
    def _find_peak(
        values: np.ndarray,
        search_start: int,
        search_end: int,
    ) -> tuple[int | None, float | None]:
        sample = np.asarray(
            values[
                search_start:
                search_end + 1
            ],
            dtype=float,
        )

        if (
            sample.size == 0
            or np.all(np.isnan(sample))
        ):
            return None, None

        local_index = int(
            np.nanargmax(sample)
        )

        peak_value = float(
            sample[local_index]
        )

        if peak_value <= 0:
            return None, None

        return (
            search_start + local_index,
            peak_value,
        )

    @staticmethod
    def _find_peak_acceleration(
        acceleration: np.ndarray,
        search_start: int,
        search_end: int,
    ) -> int | None:
        sample = np.asarray(
            acceleration[
                search_start:
                search_end + 1
            ],
            dtype=float,
        )

        if (
            sample.size == 0
            or np.all(np.isnan(sample))
        ):
            return None

        return (
            search_start
            + int(np.nanargmax(sample))
        )

    @staticmethod
    def _find_deceleration_start(
        values: np.ndarray,
        peak_frame: int,
        search_end: int,
    ) -> int | None:
        for frame in range(
            peak_frame + 1,
            search_end + 1,
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

            if current < previous:
                return frame

        return None

    def _find_phase_end(
        self,
        values: np.ndarray,
        peak_frame: int,
        search_end: int,
        peak_value: float,
    ) -> int:
        threshold = peak_value * 0.10

        for frame in range(
            peak_frame + 1,
            search_end + 1,
        ):
            value = values[frame]

            if np.isnan(value):
                continue

            if value <= threshold:
                return frame

        return search_end

    # =====================================================
    # SEARCH WINDOWS
    # =====================================================

    def _propulsion_search_start(
        self,
        rule: str,
    ) -> int:
        if rule == "dip":
            return self.dip_frame

        if rule == "takeoff_minus":
            return max(
                self.dip_frame,
                self.takeoff_frame - 8,
            )

        return self.dip_frame

    # =====================================================
    # NUMERIC HELPERS
    # =====================================================

    def _derivative(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        numeric = np.asarray(
            values,
            dtype=float,
        )

        derivative = np.full_like(
            numeric,
            np.nan,
            dtype=float,
        )

        for frame in range(
            1,
            numeric.size,
        ):
            previous = numeric[
                frame - 1
            ]

            current = numeric[
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
        numeric = np.asarray(
            values,
            dtype=float,
        )

        if self.smoothing_window <= 1:
            return numeric.copy()

        half_window = (
            self.smoothing_window // 2
        )

        result = np.full_like(
            numeric,
            np.nan,
            dtype=float,
        )

        for frame in range(
            numeric.size
        ):
            start = max(
                0,
                frame - half_window,
            )

            end = min(
                numeric.size,
                frame + half_window + 1,
            )

            sample = numeric[
                start:end
            ]

            valid = sample[
                ~np.isnan(sample)
            ]

            if valid.size > 0:
                result[frame] = float(
                    np.mean(valid)
                )

        return result

    def _side_series(
        self,
        right: np.ndarray,
        left: np.ndarray,
    ) -> np.ndarray:
        if self.shooting_side == "RIGHT":
            return right

        return left

    def _frame_gap_ms(
        self,
        first_frame: int | None,
        second_frame: int | None,
    ) -> float | None:
        if (
            first_frame is None
            or second_frame is None
        ):
            return None

        return (
            (
                int(second_frame)
                - int(first_frame)
            )
            / self.sampling_rate
            * 1000.0
        )

    @staticmethod
    def _clamp_frame(
        frame: int,
        total_frames: int,
    ) -> int:
        return max(
            0,
            min(
                int(frame),
                total_frames - 1,
            ),
        )

    @staticmethod
    def _build_sequence(
        phases: tuple[MovementPhase, ...],
        field_name: str,
    ) -> tuple[str, ...]:
        detected = [
            phase
            for phase in phases
            if getattr(
                phase,
                field_name,
            ) is not None
        ]

        detected.sort(
            key=lambda phase: int(
                getattr(
                    phase,
                    field_name,
                )
            )
        )

        return tuple(
            phase.segment
            for phase in detected
        )

    @staticmethod
    def _empty_phase(
        segment: str,
        preparatory_onset: int | None = None,
        propulsion_onset: int | None = None,
    ) -> MovementPhase:
        return MovementPhase(
            segment=segment,
            preparatory_onset_frame=(
                preparatory_onset
            ),
            propulsion_onset_frame=(
                propulsion_onset
            ),
            peak_velocity_frame=None,
            peak_acceleration_frame=None,
            deceleration_start_frame=None,
            end_frame=None,
            peak_velocity_value=None,
            peak_acceleration_value=None,
            preparatory_to_propulsion_ms=None,
            propulsion_to_peak_ms=None,
            peak_to_release_ms=None,
            propulsion_duration_ms=None,
        )


def format_phase(
    phase: MovementPhase,
) -> str:
    """
    Format one phase for terminal review.
    """

    def frame_text(
        value: int | None,
    ) -> str:
        if value is None:
            return "N/A"

        return str(value)

    def duration_text(
        value: float | None,
    ) -> str:
        if value is None:
            return "N/A"

        return f"{value:.1f} ms"

    return (
        f"{phase.segment}: "
        f"prep onset {frame_text(phase.preparatory_onset_frame)}, "
        f"propulsion onset {frame_text(phase.propulsion_onset_frame)}, "
        f"peak velocity {frame_text(phase.peak_velocity_frame)}, "
        f"peak acceleration {frame_text(phase.peak_acceleration_frame)}, "
        f"deceleration {frame_text(phase.deceleration_start_frame)}, "
        f"end {frame_text(phase.end_frame)}, "
        f"propulsion duration "
        f"{duration_text(phase.propulsion_duration_ms)}"
    )


def _run_refined_phase_test() -> None:
    """
    Run refined phase analysis on the first trial.
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

    result = RefinedPhaseBiomechanicsAnalyzer(
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

    phases = (
        result.knee_phase,
        result.hip_phase,
        result.pelvis_phase,
        result.shoulder_phase,
        result.elbow_phase,
        result.wrist_phase,
        result.ball_phase,
    )

    assert any(
        phase.propulsion_onset_frame
        is not None
        for phase in phases
    )

    print(
        "REFINED PHASE BIOMECHANICS TEST PASSED"
    )

    print(
        f"Trial file: {trial_file.name}"
    )

    print(
        "Event window: "
        f"motion {result.motion_start_frame}, "
        f"dip {result.dip_frame}, "
        f"takeoff {result.takeoff_frame}, "
        f"release {result.release_frame}"
    )

    print()

    for phase in phases:
        print(
            format_phase(phase)
        )

    print()

    print(
        "Preparatory onset sequence:"
    )

    print(
        " -> ".join(
            result.preparatory_onset_sequence
        )
    )

    print()

    print(
        "Propulsion onset sequence:"
    )

    print(
        " -> ".join(
            result.propulsion_onset_sequence
        )
    )

    print()

    print(
        "Peak sequence:"
    )

    print(
        " -> ".join(
            result.peak_sequence
        )
    )


if __name__ == "__main__":
    _run_refined_phase_test()