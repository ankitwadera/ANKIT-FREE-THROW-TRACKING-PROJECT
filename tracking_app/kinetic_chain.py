from __future__ import annotations

from dataclasses import dataclass
from math import isnan

import numpy as np

from tracking_app.analyzer import ShotAnalyzer
from tracking_app.data_loader import load_first_trial
from tracking_app.timeseries import ShotTimeSeries, TimeSeriesAnalyzer


@dataclass(frozen=True)
class ChainPeak:
    """
    One detected kinetic-chain peak.

    frame:
        Tracking-frame index where the peak occurred.

    value:
        Peak measurement. Units depend on the segment:
        - degrees/second for joint angular velocity;
        - feet/second for pelvis and wrist speed.

    milliseconds_before_release:
        Positive means the peak occurred before release.
        Negative means the peak occurred after release.
    """

    segment: str
    frame: int | None
    value: float | None
    milliseconds_before_release: float | None

    @property
    def detected(self) -> bool:
        return (
            self.frame is not None
            and self.value is not None
        )


@dataclass(frozen=True)
class KineticChainAnalysis:
    """
    Complete kinetic-chain timing report for one shot.
    """

    motion_start_frame: int
    release_frame: int
    sampling_rate: float

    knee_extension_peak: ChainPeak
    hip_extension_peak: ChainPeak
    pelvis_vertical_velocity_peak: ChainPeak
    shoulder_extension_peak: ChainPeak
    elbow_extension_peak: ChainPeak
    wrist_speed_peak: ChainPeak

    detected_sequence: tuple[str, ...]
    expected_sequence: tuple[str, ...]

    sequence_violations: tuple[str, ...]
    detected_peak_count: int
    correctly_ordered_transition_count: int
    total_transition_count: int
    sequence_score: float

    knee_to_hip_ms: float | None
    hip_to_pelvis_ms: float | None
    pelvis_to_shoulder_ms: float | None
    shoulder_to_elbow_ms: float | None
    elbow_to_wrist_ms: float | None
    wrist_to_release_ms: float | None

    @property
    def all_peaks(self) -> tuple[ChainPeak, ...]:
        return (
            self.knee_extension_peak,
            self.hip_extension_peak,
            self.pelvis_vertical_velocity_peak,
            self.shoulder_extension_peak,
            self.elbow_extension_peak,
            self.wrist_speed_peak,
        )


class KineticChainAnalyzer:
    """
    Analyze lower-to-upper-body sequencing before ball release.

    Current expected sequence
    -------------------------
    1. Knee extension
    2. Hip extension
    3. Pelvis vertical velocity
    4. Shoulder extension
    5. Elbow extension
    6. Wrist speed
    7. Ball release

    Important interpretation note
    -----------------------------
    This is an engineering sequence model, not yet a clinically or
    universally validated basketball standard. It should be treated as
    a player-comparison and consistency tool until it is validated
    across more shots and participants.
    """

    EXPECTED_SEQUENCE = (
        "Knee extension",
        "Hip extension",
        "Pelvis vertical velocity",
        "Shoulder extension",
        "Elbow extension",
        "Wrist speed",
    )

    def __init__(
        self,
        time_series: ShotTimeSeries,
        sampling_rate: float,
        motion_start_frame: int | None,
        release_frame: int,
        shooting_side: str = "RIGHT",
        minimum_peak_prominence_ratio: float = 0.10,
    ) -> None:
        if sampling_rate <= 0:
            raise ValueError(
                "Kinetic-chain analysis requires a positive sampling rate."
            )

        if time_series.total_frames <= 0:
            raise ValueError(
                "Kinetic-chain analysis requires time-series frames."
            )

        if (
            release_frame < 0
            or release_frame >= time_series.total_frames
        ):
            raise ValueError(
                "Release frame is outside the available time series."
            )

        side = shooting_side.strip().upper()

        if side not in {"RIGHT", "LEFT"}:
            raise ValueError(
                "shooting_side must be RIGHT or LEFT."
            )

        self.time_series = time_series
        self.sampling_rate = float(sampling_rate)
        self.release_frame = int(release_frame)
        self.shooting_side = side

        if motion_start_frame is None:
            self.motion_start_frame = max(
                0,
                self.release_frame - int(
                    round(self.sampling_rate * 2.5)
                ),
            )
        else:
            self.motion_start_frame = max(
                0,
                min(
                    int(motion_start_frame),
                    self.release_frame,
                ),
            )

        self.minimum_peak_prominence_ratio = max(
            0.0,
            float(minimum_peak_prominence_ratio),
        )

    # =====================================================
    # PUBLIC ANALYSIS
    # =====================================================

    def analyze(self) -> KineticChainAnalysis:
        """
        Detect segment peaks and evaluate their ordering.
        """

        knee_angle = self._shooting_side_series(
            right=self.time_series.right_knee_angle,
            left=self.time_series.left_knee_angle,
        )

        hip_angle = self._shooting_side_series(
            right=self.time_series.right_hip_angle,
            left=self.time_series.left_hip_angle,
        )

        shoulder_angle = self._shooting_side_series(
            right=self.time_series.right_shoulder_angle,
            left=self.time_series.left_shoulder_angle,
        )

        elbow_angle = self._shooting_side_series(
            right=self.time_series.right_elbow_angle,
            left=self.time_series.left_elbow_angle,
        )

        wrist_speed = self._shooting_side_series(
            right=self.time_series.right_wrist_speed,
            left=self.time_series.left_wrist_speed,
        )

        knee_velocity = self._derivative(knee_angle)
        hip_velocity = self._derivative(hip_angle)
        shoulder_velocity = self._derivative(
            shoulder_angle
        )
        elbow_velocity = self._derivative(elbow_angle)
        pelvis_velocity = self._derivative(
            self.time_series.pelvis_height
        )

        knee_peak = self._find_peak(
            segment="Knee extension",
            values=knee_velocity,
        )

        hip_peak = self._find_peak(
            segment="Hip extension",
            values=hip_velocity,
        )

        pelvis_peak = self._find_peak(
            segment="Pelvis vertical velocity",
            values=pelvis_velocity,
        )

        shoulder_peak = self._find_peak(
            segment="Shoulder extension",
            values=shoulder_velocity,
        )

        elbow_peak = self._find_peak(
            segment="Elbow extension",
            values=elbow_velocity,
        )

        wrist_peak = self._find_peak(
            segment="Wrist speed",
            values=wrist_speed,
        )

        peaks = (
            knee_peak,
            hip_peak,
            pelvis_peak,
            shoulder_peak,
            elbow_peak,
            wrist_peak,
        )

        detected_peaks = [
            peak
            for peak in peaks
            if peak.detected
        ]

        detected_sequence = tuple(
            peak.segment
            for peak in sorted(
                detected_peaks,
                key=lambda peak: int(peak.frame),
            )
        )

        violations = self._find_sequence_violations(
            peaks
        )

        correctly_ordered_transitions = (
            self._count_correct_transitions(peaks)
        )

        total_transitions = max(
            0,
            len(detected_peaks) - 1,
        )

        sequence_score = self._calculate_sequence_score(
            detected_peak_count=len(detected_peaks),
            correctly_ordered_transitions=(
                correctly_ordered_transitions
            ),
            total_transitions=total_transitions,
        )

        return KineticChainAnalysis(
            motion_start_frame=self.motion_start_frame,
            release_frame=self.release_frame,
            sampling_rate=self.sampling_rate,
            knee_extension_peak=knee_peak,
            hip_extension_peak=hip_peak,
            pelvis_vertical_velocity_peak=pelvis_peak,
            shoulder_extension_peak=shoulder_peak,
            elbow_extension_peak=elbow_peak,
            wrist_speed_peak=wrist_peak,
            detected_sequence=detected_sequence,
            expected_sequence=self.EXPECTED_SEQUENCE,
            sequence_violations=tuple(violations),
            detected_peak_count=len(detected_peaks),
            correctly_ordered_transition_count=(
                correctly_ordered_transitions
            ),
            total_transition_count=total_transitions,
            sequence_score=sequence_score,
            knee_to_hip_ms=self._frame_gap_ms(
                knee_peak,
                hip_peak,
            ),
            hip_to_pelvis_ms=self._frame_gap_ms(
                hip_peak,
                pelvis_peak,
            ),
            pelvis_to_shoulder_ms=self._frame_gap_ms(
                pelvis_peak,
                shoulder_peak,
            ),
            shoulder_to_elbow_ms=self._frame_gap_ms(
                shoulder_peak,
                elbow_peak,
            ),
            elbow_to_wrist_ms=self._frame_gap_ms(
                elbow_peak,
                wrist_peak,
            ),
            wrist_to_release_ms=(
                self._peak_to_release_ms(
                    wrist_peak
                )
            ),
        )

    # =====================================================
    # PEAK DETECTION
    # =====================================================

    def _find_peak(
        self,
        segment: str,
        values: np.ndarray,
    ) -> ChainPeak:
        """
        Find the strongest positive peak from motion start to release.

        The selected peak must also exceed a small player-relative
        prominence threshold so nearly flat noise is not reported as
        a meaningful kinetic-chain event.
        """

        search_values = np.asarray(
            values[
                self.motion_start_frame:
                self.release_frame + 1
            ],
            dtype=float,
        )

        valid_mask = ~np.isnan(search_values)

        if not np.any(valid_mask):
            return ChainPeak(
                segment=segment,
                frame=None,
                value=None,
                milliseconds_before_release=None,
            )

        valid_values = search_values[valid_mask]

        peak_value = float(
            np.max(valid_values)
        )

        value_range = float(
            np.max(valid_values)
            - np.min(valid_values)
        )

        minimum_prominence = (
            value_range
            * self.minimum_peak_prominence_ratio
        )

        if (
            peak_value <= 0
            or peak_value < minimum_prominence
        ):
            return ChainPeak(
                segment=segment,
                frame=None,
                value=None,
                milliseconds_before_release=None,
            )

        local_peak_index = int(
            np.nanargmax(search_values)
        )

        peak_frame = (
            self.motion_start_frame
            + local_peak_index
        )

        return ChainPeak(
            segment=segment,
            frame=peak_frame,
            value=peak_value,
            milliseconds_before_release=(
                self._milliseconds_before_release(
                    peak_frame
                )
            ),
        )

    # =====================================================
    # DERIVATIVES
    # =====================================================

    def _derivative(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate a first derivative using valid adjacent samples.

        Angle input produces degrees/second.
        Height input produces feet/second.
        """

        numeric_values = np.asarray(
            values,
            dtype=float,
        )

        derivative = np.full_like(
            numeric_values,
            np.nan,
            dtype=float,
        )

        for index in range(
            1,
            numeric_values.size,
        ):
            previous_value = numeric_values[index - 1]
            current_value = numeric_values[index]

            if (
                np.isnan(previous_value)
                or np.isnan(current_value)
            ):
                continue

            derivative[index] = (
                current_value
                - previous_value
            ) * self.sampling_rate

        return self._smooth(
            derivative,
            window=5,
        )

    @staticmethod
    def _smooth(
        values: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """
        Apply a centered NaN-aware moving average.
        """

        numeric_values = np.asarray(
            values,
            dtype=float,
        )

        if window <= 1:
            return numeric_values.copy()

        half_window = window // 2

        result = np.full_like(
            numeric_values,
            np.nan,
            dtype=float,
        )

        for index in range(
            numeric_values.size
        ):
            start = max(
                0,
                index - half_window,
            )

            end = min(
                numeric_values.size,
                index + half_window + 1,
            )

            sample = numeric_values[start:end]
            valid_sample = sample[
                ~np.isnan(sample)
            ]

            if valid_sample.size > 0:
                result[index] = float(
                    np.mean(valid_sample)
                )

        return result

    # =====================================================
    # SEQUENCE EVALUATION
    # =====================================================

    def _find_sequence_violations(
        self,
        peaks: tuple[ChainPeak, ...],
    ) -> list[str]:
        """
        Identify adjacent expected segments that peak out of order.
        """

        violations: list[str] = []

        for first_peak, second_peak in zip(
            peaks,
            peaks[1:],
        ):
            if (
                not first_peak.detected
                or not second_peak.detected
            ):
                continue

            if int(first_peak.frame) > int(
                second_peak.frame
            ):
                violations.append(
                    f"{second_peak.segment} peaked before "
                    f"{first_peak.segment}."
                )

        return violations

    @staticmethod
    def _count_correct_transitions(
        peaks: tuple[ChainPeak, ...],
    ) -> int:
        """
        Count correctly ordered adjacent detected segment pairs.
        """

        correct = 0

        for first_peak, second_peak in zip(
            peaks,
            peaks[1:],
        ):
            if (
                not first_peak.detected
                or not second_peak.detected
            ):
                continue

            if int(first_peak.frame) <= int(
                second_peak.frame
            ):
                correct += 1

        return correct

    def _calculate_sequence_score(
        self,
        detected_peak_count: int,
        correctly_ordered_transitions: int,
        total_transitions: int,
    ) -> float:
        """
        Calculate a transparent 0-100 sequencing score.

        70% rewards correct ordering.
        30% rewards complete peak detection.
        """

        detection_component = (
            detected_peak_count
            / len(self.EXPECTED_SEQUENCE)
        )

        if total_transitions <= 0:
            ordering_component = 0.0
        else:
            ordering_component = (
                correctly_ordered_transitions
                / total_transitions
            )

        score = (
            ordering_component * 0.70
            + detection_component * 0.30
        ) * 100.0

        return float(
            np.clip(
                score,
                0.0,
                100.0,
            )
        )

    # =====================================================
    # TIMING HELPERS
    # =====================================================

    def _milliseconds_before_release(
        self,
        frame: int,
    ) -> float:
        return (
            (
                self.release_frame
                - int(frame)
            )
            / self.sampling_rate
            * 1000.0
        )

    def _peak_to_release_ms(
        self,
        peak: ChainPeak,
    ) -> float | None:
        if not peak.detected:
            return None

        return self._milliseconds_before_release(
            int(peak.frame)
        )

    def _frame_gap_ms(
        self,
        first_peak: ChainPeak,
        second_peak: ChainPeak,
    ) -> float | None:
        """
        Return positive milliseconds from first peak to second peak.

        A negative result means the second segment peaked earlier.
        """

        if (
            not first_peak.detected
            or not second_peak.detected
        ):
            return None

        return (
            (
                int(second_peak.frame)
                - int(first_peak.frame)
            )
            / self.sampling_rate
            * 1000.0
        )

    def _shooting_side_series(
        self,
        right: np.ndarray,
        left: np.ndarray,
    ) -> np.ndarray:
        if self.shooting_side == "RIGHT":
            return right

        return left


def format_chain_peak(
    peak: ChainPeak,
    value_suffix: str,
) -> str:
    """
    Format one peak for terminal and future report output.
    """

    if not peak.detected:
        return f"{peak.segment}: Not detected"

    timing = peak.milliseconds_before_release

    if timing is None or isnan(timing):
        timing_text = "timing unavailable"
    elif timing >= 0:
        timing_text = (
            f"{timing:.1f} ms before release"
        )
    else:
        timing_text = (
            f"{abs(timing):.1f} ms after release"
        )

    return (
        f"{peak.segment}: "
        f"Frame {peak.frame}, "
        f"{peak.value:.2f} {value_suffix}, "
        f"{timing_text}"
    )


def _format_gap(
    label: str,
    value: float | None,
) -> str:
    if value is None:
        return f"{label}: N/A"

    return f"{label}: {value:.1f} ms"


def _run_kinetic_chain_test() -> None:
    """
    Run standalone kinetic-chain analysis on the first trial.
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

    analyzer = KineticChainAnalyzer(
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
        release_frame=(
            shot_analysis.release_frame
        ),
        shooting_side="RIGHT",
    )

    result = analyzer.analyze()

    assert (
        result.motion_start_frame
        <= result.release_frame
    )

    assert (
        0.0
        <= result.sequence_score
        <= 100.0
    )

    assert result.detected_peak_count > 0

    print("KINETIC CHAIN TEST PASSED")
    print(f"Trial file: {trial_file.name}")
    print(
        f"Analysis window: "
        f"{result.motion_start_frame} "
        f"to {result.release_frame}"
    )
    print(
        f"Detected peaks: "
        f"{result.detected_peak_count} / "
        f"{len(result.expected_sequence)}"
    )
    print(
        f"Sequence score: "
        f"{result.sequence_score:.1f} / 100"
    )
    print()

    print(
        format_chain_peak(
            result.knee_extension_peak,
            "deg/s",
        )
    )
    print(
        format_chain_peak(
            result.hip_extension_peak,
            "deg/s",
        )
    )
    print(
        format_chain_peak(
            result.pelvis_vertical_velocity_peak,
            "ft/s",
        )
    )
    print(
        format_chain_peak(
            result.shoulder_extension_peak,
            "deg/s",
        )
    )
    print(
        format_chain_peak(
            result.elbow_extension_peak,
            "deg/s",
        )
    )
    print(
        format_chain_peak(
            result.wrist_speed_peak,
            "ft/s",
        )
    )
    print()

    print("Detected sequence:")
    print(" -> ".join(result.detected_sequence))
    print()

    print(
        _format_gap(
            "Knee to hip",
            result.knee_to_hip_ms,
        )
    )
    print(
        _format_gap(
            "Hip to pelvis",
            result.hip_to_pelvis_ms,
        )
    )
    print(
        _format_gap(
            "Pelvis to shoulder",
            result.pelvis_to_shoulder_ms,
        )
    )
    print(
        _format_gap(
            "Shoulder to elbow",
            result.shoulder_to_elbow_ms,
        )
    )
    print(
        _format_gap(
            "Elbow to wrist",
            result.elbow_to_wrist_ms,
        )
    )
    print(
        _format_gap(
            "Wrist to release",
            result.wrist_to_release_ms,
        )
    )

    if result.sequence_violations:
        print()
        print("Sequence warnings:")

        for violation in result.sequence_violations:
            print(f"- {violation}")


if __name__ == "__main__":
    _run_kinetic_chain_test()
