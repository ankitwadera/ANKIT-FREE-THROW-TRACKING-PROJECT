from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tracking_app.analyzer import ShotAnalyzer
from tracking_app.data_loader import load_first_trial
from tracking_app.timeseries import ShotTimeSeries, TimeSeriesAnalyzer


@dataclass(frozen=True)
class SignalDiagnostic:
    """
    Diagnostic summary for one propulsion signal.
    """

    signal_name: str
    search_start_frame: int
    search_end_frame: int

    valid_frame_count: int

    minimum_value: float | None
    maximum_value: float | None

    maximum_absolute_value: float | None

    positive_peak_frame: int | None
    positive_peak_value: float | None

    negative_peak_frame: int | None
    negative_peak_value: float | None

    baseline_mean: float | None
    baseline_standard_deviation: float | None

    strict_threshold: float | None
    fallback_threshold: float | None

    strict_onset_frame: int | None
    fallback_onset_frame: int | None


class PhaseSignalDiagnostics:
    """
    Inspect the exact shoulder and ball signals used for propulsion.

    This module does not change the detector. It explains why a signal
    is not being detected so the next threshold or signal definition
    can be based on evidence rather than guesswork.
    """

    def __init__(
        self,
        time_series: ShotTimeSeries,
        sampling_rate: float,
        motion_start_frame: int,
        dip_frame: int,
        takeoff_frame: int,
        release_frame: int,
        smoothing_window: int = 5,
        baseline_frame_count: int = 15,
        noise_multiplier: float = 3.0,
    ) -> None:
        self.time_series = time_series
        self.sampling_rate = float(sampling_rate)

        self.motion_start_frame = int(
            motion_start_frame
        )

        self.dip_frame = int(dip_frame)
        self.takeoff_frame = int(takeoff_frame)
        self.release_frame = int(release_frame)

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
    # PUBLIC DIAGNOSTICS
    # =====================================================

    def run(
        self,
    ) -> tuple[
        SignalDiagnostic,
        SignalDiagnostic,
    ]:
        """
        Diagnose shoulder angular velocity and ball vertical speed.
        """

        shoulder_velocity = self._smooth(
            self._derivative(
                self.time_series.right_shoulder_angle
            )
        )

        ball_vertical_speed = self._smooth(
            self.time_series.ball_vertical_speed
        )

        shoulder_diagnostic = self._diagnose_signal(
            signal_name="Right shoulder angular velocity",
            values=shoulder_velocity,
            search_start=self.dip_frame,
            search_end=self.release_frame,
            onset_ratio=0.22,
            sustained_frames=3,
        )

        ball_diagnostic = self._diagnose_signal(
            signal_name="Ball vertical speed",
            values=ball_vertical_speed,
            search_start=self.dip_frame,
            search_end=self.release_frame,
            onset_ratio=0.30,
            sustained_frames=3,
        )

        return (
            shoulder_diagnostic,
            ball_diagnostic,
        )

    # =====================================================
    # SIGNAL DIAGNOSTICS
    # =====================================================

    def _diagnose_signal(
        self,
        signal_name: str,
        values: np.ndarray,
        search_start: int,
        search_end: int,
        onset_ratio: float,
        sustained_frames: int,
    ) -> SignalDiagnostic:
        """
        Calculate peaks, baseline noise, thresholds, and onset results.
        """

        sample = np.asarray(
            values[
                search_start:
                search_end + 1
            ],
            dtype=float,
        )

        valid_mask = ~np.isnan(sample)
        valid_values = sample[
            valid_mask
        ]

        if valid_values.size == 0:
            return SignalDiagnostic(
                signal_name=signal_name,
                search_start_frame=search_start,
                search_end_frame=search_end,
                valid_frame_count=0,
                minimum_value=None,
                maximum_value=None,
                maximum_absolute_value=None,
                positive_peak_frame=None,
                positive_peak_value=None,
                negative_peak_frame=None,
                negative_peak_value=None,
                baseline_mean=None,
                baseline_standard_deviation=None,
                strict_threshold=None,
                fallback_threshold=None,
                strict_onset_frame=None,
                fallback_onset_frame=None,
            )

        positive_local = int(
            np.nanargmax(sample)
        )

        negative_local = int(
            np.nanargmin(sample)
        )

        positive_peak_value = float(
            sample[positive_local]
        )

        negative_peak_value = float(
            sample[negative_local]
        )

        dominant_peak_value = max(
            abs(positive_peak_value),
            abs(negative_peak_value),
        )

        oriented_values = (
            -values
            if abs(negative_peak_value)
            > abs(positive_peak_value)
            else values.copy()
        )

        oriented_sample = np.asarray(
            oriented_values[
                search_start:
                search_end + 1
            ],
            dtype=float,
        )

        oriented_peak = float(
            np.nanmax(oriented_sample)
        )

        baseline_start = max(
            0,
            search_start
            - self.baseline_frame_count,
        )

        baseline = np.asarray(
            oriented_values[
                baseline_start:
                search_start
            ],
            dtype=float,
        )

        valid_baseline = baseline[
            ~np.isnan(baseline)
        ]

        if valid_baseline.size >= 3:
            baseline_mean = float(
                np.mean(valid_baseline)
            )

            baseline_std = float(
                np.std(valid_baseline)
            )
        else:
            baseline_mean = 0.0
            baseline_std = 0.0

        noise_threshold = (
            baseline_mean
            + baseline_std
            * self.noise_multiplier
        )

        strict_threshold = max(
            noise_threshold,
            oriented_peak * onset_ratio,
        )

        fallback_threshold = max(
            noise_threshold,
            oriented_peak
            * onset_ratio
            * 0.65,
        )

        strict_onset = self._search_onset(
            values=oriented_values,
            search_start=search_start,
            search_end=search_end,
            threshold=strict_threshold,
            sustained_frames=sustained_frames,
        )

        fallback_onset = self._search_onset(
            values=oriented_values,
            search_start=search_start,
            search_end=search_end,
            threshold=fallback_threshold,
            sustained_frames=max(
                2,
                sustained_frames - 1,
            ),
        )

        return SignalDiagnostic(
            signal_name=signal_name,
            search_start_frame=search_start,
            search_end_frame=search_end,
            valid_frame_count=int(
                valid_values.size
            ),
            minimum_value=float(
                np.min(valid_values)
            ),
            maximum_value=float(
                np.max(valid_values)
            ),
            maximum_absolute_value=float(
                dominant_peak_value
            ),
            positive_peak_frame=(
                search_start
                + positive_local
            ),
            positive_peak_value=(
                positive_peak_value
            ),
            negative_peak_frame=(
                search_start
                + negative_local
            ),
            negative_peak_value=(
                negative_peak_value
            ),
            baseline_mean=baseline_mean,
            baseline_standard_deviation=(
                baseline_std
            ),
            strict_threshold=float(
                strict_threshold
            ),
            fallback_threshold=float(
                fallback_threshold
            ),
            strict_onset_frame=strict_onset,
            fallback_onset_frame=(
                fallback_onset
            ),
        )

    @staticmethod
    def _search_onset(
        values: np.ndarray,
        search_start: int,
        search_end: int,
        threshold: float,
        sustained_frames: int,
    ) -> int | None:
        """
        Search for consecutive valid frames above threshold.
        """

        for frame in range(
            search_start,
            search_end
            - sustained_frames
            + 2,
        ):
            sample = np.asarray(
                values[
                    frame:
                    frame + sustained_frames
                ],
                dtype=float,
            )

            if (
                sample.size
                != sustained_frames
                or np.any(np.isnan(sample))
            ):
                continue

            if np.all(
                sample >= threshold
            ):
                return frame

        return None

    # =====================================================
    # NUMERIC HELPERS
    # =====================================================

    def _derivative(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate first derivative.
        """

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
        """
        Centered NaN-aware moving average.
        """

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
                frame
                + half_window
                + 1,
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


def _format_value(
    value: float | None,
) -> str:
    if value is None:
        return "N/A"

    return f"{value:.4f}"


def _format_frame(
    value: int | None,
) -> str:
    if value is None:
        return "N/A"

    return str(value)


def print_diagnostic(
    diagnostic: SignalDiagnostic,
) -> None:
    """
    Print one signal diagnostic clearly.
    """

    print(diagnostic.signal_name)
    print("-" * 60)

    print(
        "Search window: "
        f"{diagnostic.search_start_frame} "
        f"to {diagnostic.search_end_frame}"
    )

    print(
        "Valid frames: "
        f"{diagnostic.valid_frame_count}"
    )

    print(
        "Minimum value: "
        f"{_format_value(diagnostic.minimum_value)}"
    )

    print(
        "Maximum value: "
        f"{_format_value(diagnostic.maximum_value)}"
    )

    print(
        "Maximum absolute value: "
        f"{_format_value(diagnostic.maximum_absolute_value)}"
    )

    print(
        "Positive peak: Frame "
        f"{_format_frame(diagnostic.positive_peak_frame)}, "
        f"value {_format_value(diagnostic.positive_peak_value)}"
    )

    print(
        "Negative peak: Frame "
        f"{_format_frame(diagnostic.negative_peak_frame)}, "
        f"value {_format_value(diagnostic.negative_peak_value)}"
    )

    print(
        "Baseline mean: "
        f"{_format_value(diagnostic.baseline_mean)}"
    )

    print(
        "Baseline standard deviation: "
        f"{_format_value(diagnostic.baseline_standard_deviation)}"
    )

    print(
        "Strict threshold: "
        f"{_format_value(diagnostic.strict_threshold)}"
    )

    print(
        "Fallback threshold: "
        f"{_format_value(diagnostic.fallback_threshold)}"
    )

    print(
        "Strict onset: "
        f"{_format_frame(diagnostic.strict_onset_frame)}"
    )

    print(
        "Fallback onset: "
        f"{_format_frame(diagnostic.fallback_onset_frame)}"
    )


def _run_phase_signal_diagnostics() -> None:
    """
    Diagnose the first trial's shoulder and ball signals.
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

    diagnostics = PhaseSignalDiagnostics(
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
    )

    shoulder_diagnostic, ball_diagnostic = (
        diagnostics.run()
    )

    print(
        "PHASE SIGNAL DIAGNOSTICS"
    )

    print(
        f"Trial file: {trial_file.name}"
    )

    print()

    print_diagnostic(
        shoulder_diagnostic
    )

    print()

    print_diagnostic(
        ball_diagnostic
    )


if __name__ == "__main__":
    _run_phase_signal_diagnostics()