from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tracking_app.analyzer import ShotAnalyzer
from tracking_app.basketball_event_mapper import BasketballEventMapper
from tracking_app.data_loader import load_first_trial
from tracking_app.phase_biomechanics import RefinedPhaseBiomechanicsAnalyzer
from tracking_app.shot_timeline_builder import ShotTimelineBuilder
from tracking_app.timeseries import ShotTimeSeries, TimeSeriesAnalyzer
from tracking_app.whole_body_synchronization import (
    WholeBodySynchronizationEngine,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "first_trial_features.csv"
)


@dataclass(frozen=True)
class ShotFeatureRecord:
    """
    Flat, objective feature record for one free throw.

    The record intentionally stores measurements rather than coaching
    judgments. Future statistical and machine-learning layers can use
    these features to learn which patterns relate to successful shots.
    """

    participant_id: str
    trial_id: str
    result: str
    trial_file: str

    sampling_rate: float
    total_frames: int

    motion_start_frame: int | None
    dip_frame: int | None
    takeoff_frame: int | None
    release_frame: int | None
    ball_apex_frame: int | None
    landing_frame: int | None

    motion_to_dip_ms: float | None
    dip_to_takeoff_ms: float | None
    takeoff_to_release_ms: float | None
    motion_to_release_ms: float | None
    release_to_apex_ms: float | None
    release_to_landing_ms: float | None

    release_height_ft: float | None
    release_angle_deg: float | None
    release_ball_speed_ft_s: float | None
    release_horizontal_speed_ft_s: float | None
    release_vertical_speed_ft_s: float | None
    release_wrist_speed_ft_s: float | None
    release_ball_wrist_separation_ft: float | None

    release_right_elbow_angle_deg: float | None
    release_left_elbow_angle_deg: float | None
    release_right_knee_angle_deg: float | None
    release_left_knee_angle_deg: float | None
    release_right_hip_angle_deg: float | None
    release_left_hip_angle_deg: float | None
    release_right_shoulder_angle_deg: float | None
    release_left_shoulder_angle_deg: float | None

    right_knee_range_of_motion_deg: float | None
    left_knee_range_of_motion_deg: float | None
    right_hip_range_of_motion_deg: float | None
    left_hip_range_of_motion_deg: float | None
    right_elbow_range_of_motion_deg: float | None
    left_elbow_range_of_motion_deg: float | None
    right_shoulder_range_of_motion_deg: float | None
    left_shoulder_range_of_motion_deg: float | None

    peak_right_wrist_speed_ft_s: float | None
    peak_left_wrist_speed_ft_s: float | None
    peak_ball_speed_ft_s: float | None
    peak_ball_vertical_speed_ft_s: float | None
    minimum_pelvis_height_ft: float | None
    maximum_pelvis_height_ft: float | None
    pelvis_vertical_displacement_ft: float | None

    knee_propulsion_onset_frame: int | None
    hip_propulsion_onset_frame: int | None
    pelvis_propulsion_onset_frame: int | None
    shoulder_propulsion_onset_frame: int | None
    elbow_propulsion_onset_frame: int | None
    wrist_propulsion_onset_frame: int | None
    ball_propulsion_onset_frame: int | None

    knee_peak_velocity_frame: int | None
    hip_peak_velocity_frame: int | None
    pelvis_peak_velocity_frame: int | None
    shoulder_peak_velocity_frame: int | None
    elbow_peak_velocity_frame: int | None
    wrist_peak_velocity_frame: int | None
    ball_peak_velocity_frame: int | None

    knee_propulsion_duration_ms: float | None
    hip_propulsion_duration_ms: float | None
    pelvis_propulsion_duration_ms: float | None
    shoulder_propulsion_duration_ms: float | None
    elbow_propulsion_duration_ms: float | None
    wrist_propulsion_duration_ms: float | None
    ball_propulsion_duration_ms: float | None

    hip_to_knee_gap_ms: float | None
    pelvis_to_knee_gap_ms: float | None
    shoulder_to_elbow_gap_ms: float | None
    knee_to_elbow_gap_ms: float | None
    wrist_to_release_gap_ms: float | None
    ball_upward_to_release_gap_ms: float | None
    pelvis_to_release_gap_ms: float | None
    shoulder_to_release_gap_ms: float | None
    elbow_to_release_gap_ms: float | None
    ball_lowest_to_release_gap_ms: float | None

    propulsion_onset_sequence: str
    peak_velocity_sequence: str
    authoritative_event_count: int
    average_event_confidence: float | None

    analysis_status: str
    analysis_error: str


class ShotFeatureExtractor:
    """
    Convert one tracked free throw into a structured feature record.

    This is the bridge between biomechanics analysis and future:

    - player fingerprints;
    - shot comparison;
    - clustering;
    - statistical testing;
    - machine learning;
    - evidence-based coaching reports.
    """

    def __init__(
        self,
        trial_file: Path,
        trial_data: dict,
        shooting_side: str = "RIGHT",
    ) -> None:
        self.trial_file = Path(trial_file)
        self.trial_data = trial_data

        self.shooting_side = (
            shooting_side
            .strip()
            .upper()
        )

        if self.shooting_side not in {
            "RIGHT",
            "LEFT",
        }:
            raise ValueError(
                "shooting_side must be RIGHT or LEFT."
            )

        self.sampling_rate = self._safe_float(
            trial_data.get(
                "sampling_rate",
                30,
            ),
            fallback=30.0,
        )

        if self.sampling_rate <= 0:
            self.sampling_rate = 30.0

    # =====================================================
    # PUBLIC WORKFLOW
    # =====================================================

    def extract(
        self,
    ) -> ShotFeatureRecord:
        """
        Run the complete analysis pipeline and flatten its outputs.
        """

        try:
            shot_analysis = ShotAnalyzer(
                trial_data=self.trial_data,
                trial_file=self.trial_file,
                shooting_wrist=(
                    f"{self.shooting_side}_WRIST"
                ),
            ).analyze()

            time_series = TimeSeriesAnalyzer(
                trial_data=self.trial_data,
                smoothing_window=5,
            ).analyze()

            phase_analysis = (
                RefinedPhaseBiomechanicsAnalyzer(
                    time_series=time_series,
                    sampling_rate=self.sampling_rate,
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
                    shooting_side=(
                        self.shooting_side
                    ),
                ).analyze()
            )

            basketball_timeline = (
                BasketballEventMapper(
                    time_series=time_series,
                    sampling_rate=self.sampling_rate,
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
                    shooting_side=(
                        self.shooting_side
                    ),
                ).build_timeline()
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

            synchronization = (
                WholeBodySynchronizationEngine(
                    timeline=timeline,
                    sampling_rate=self.sampling_rate,
                ).analyze()
            )

            return self._build_success_record(
                shot_analysis=shot_analysis,
                time_series=time_series,
                phase_analysis=phase_analysis,
                timeline=timeline,
                synchronization=(
                    synchronization
                ),
            )

        except Exception as error:
            return self._build_failure_record(
                error
            )

    # =====================================================
    # SUCCESS RECORD
    # =====================================================

    def _build_success_record(
        self,
        shot_analysis: object,
        time_series: ShotTimeSeries,
        phase_analysis: object,
        timeline: object,
        synchronization: object,
    ) -> ShotFeatureRecord:
        """
        Flatten every objective measurement into one row.
        """

        relationships = {
            relationship.relationship_name:
            relationship
            for relationship in (
                synchronization.relationships
            )
        }

        event_confidences = [
            event.confidence
            for event in timeline.events
        ]

        return ShotFeatureRecord(
            participant_id=str(
                self.trial_data.get(
                    "participant_id",
                    "Unknown",
                )
            ),
            trial_id=str(
                self.trial_data.get(
                    "trial_id",
                    self.trial_file.stem,
                )
            ),
            result=str(
                self.trial_data.get(
                    "result",
                    "unknown",
                )
            ),
            trial_file=self.trial_file.name,

            sampling_rate=(
                self.sampling_rate
            ),
            total_frames=(
                time_series.total_frames
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
            landing_frame=(
                shot_analysis.landing_frame
            ),

            motion_to_dip_ms=self._frame_gap_ms(
                shot_analysis.motion_start_frame,
                shot_analysis.dip_frame,
            ),
            dip_to_takeoff_ms=self._frame_gap_ms(
                shot_analysis.dip_frame,
                shot_analysis.takeoff_frame,
            ),
            takeoff_to_release_ms=self._frame_gap_ms(
                shot_analysis.takeoff_frame,
                shot_analysis.release_frame,
            ),
            motion_to_release_ms=self._frame_gap_ms(
                shot_analysis.motion_start_frame,
                shot_analysis.release_frame,
            ),
            release_to_apex_ms=self._frame_gap_ms(
                shot_analysis.release_frame,
                shot_analysis.ball_apex_frame,
            ),
            release_to_landing_ms=self._frame_gap_ms(
                shot_analysis.release_frame,
                shot_analysis.landing_frame,
            ),

            release_height_ft=self._safe_optional(
                getattr(
                    shot_analysis,
                    "release_height",
                    None,
                )
            ),
            release_angle_deg=self._safe_optional(
                getattr(
                    shot_analysis,
                    "release_angle",
                    None,
                )
            ),
            release_ball_speed_ft_s=self._safe_optional(
                getattr(
                    shot_analysis,
                    "ball_speed",
                    None,
                )
            ),
            release_horizontal_speed_ft_s=self._safe_optional(
                getattr(
                    shot_analysis,
                    "ball_horizontal_speed",
                    None,
                )
            ),
            release_vertical_speed_ft_s=self._safe_optional(
                getattr(
                    shot_analysis,
                    "ball_vertical_speed",
                    None,
                )
            ),
            release_wrist_speed_ft_s=self._safe_optional(
                getattr(
                    shot_analysis,
                    "wrist_speed",
                    None,
                )
            ),
            release_ball_wrist_separation_ft=self._safe_optional(
                getattr(
                    shot_analysis,
                    "ball_wrist_separation",
                    None,
                )
            ),

            release_right_elbow_angle_deg=self._safe_optional(
                getattr(
                    shot_analysis,
                    "right_elbow_angle",
                    None,
                )
            ),
            release_left_elbow_angle_deg=self._safe_optional(
                getattr(
                    shot_analysis,
                    "left_elbow_angle",
                    None,
                )
            ),
            release_right_knee_angle_deg=self._safe_optional(
                getattr(
                    shot_analysis,
                    "right_knee_angle",
                    None,
                )
            ),
            release_left_knee_angle_deg=self._safe_optional(
                getattr(
                    shot_analysis,
                    "left_knee_angle",
                    None,
                )
            ),
            release_right_hip_angle_deg=self._safe_optional(
                getattr(
                    shot_analysis,
                    "right_hip_angle",
                    None,
                )
            ),
            release_left_hip_angle_deg=self._safe_optional(
                getattr(
                    shot_analysis,
                    "left_hip_angle",
                    None,
                )
            ),
            release_right_shoulder_angle_deg=self._safe_optional(
                getattr(
                    shot_analysis,
                    "right_shoulder_angle",
                    None,
                )
            ),
            release_left_shoulder_angle_deg=self._safe_optional(
                getattr(
                    shot_analysis,
                    "left_shoulder_angle",
                    None,
                )
            ),

            right_knee_range_of_motion_deg=self._range_of_motion(
                time_series.right_knee_angle,
                shot_analysis.motion_start_frame,
                shot_analysis.release_frame,
            ),
            left_knee_range_of_motion_deg=self._range_of_motion(
                time_series.left_knee_angle,
                shot_analysis.motion_start_frame,
                shot_analysis.release_frame,
            ),
            right_hip_range_of_motion_deg=self._range_of_motion(
                time_series.right_hip_angle,
                shot_analysis.motion_start_frame,
                shot_analysis.release_frame,
            ),
            left_hip_range_of_motion_deg=self._range_of_motion(
                time_series.left_hip_angle,
                shot_analysis.motion_start_frame,
                shot_analysis.release_frame,
            ),
            right_elbow_range_of_motion_deg=self._range_of_motion(
                time_series.right_elbow_angle,
                shot_analysis.motion_start_frame,
                shot_analysis.release_frame,
            ),
            left_elbow_range_of_motion_deg=self._range_of_motion(
                time_series.left_elbow_angle,
                shot_analysis.motion_start_frame,
                shot_analysis.release_frame,
            ),
            right_shoulder_range_of_motion_deg=self._range_of_motion(
                time_series.right_shoulder_angle,
                shot_analysis.motion_start_frame,
                shot_analysis.release_frame,
            ),
            left_shoulder_range_of_motion_deg=self._range_of_motion(
                time_series.left_shoulder_angle,
                shot_analysis.motion_start_frame,
                shot_analysis.release_frame,
            ),

            peak_right_wrist_speed_ft_s=self._window_max(
                time_series.right_wrist_speed,
                shot_analysis.motion_start_frame,
                shot_analysis.release_frame,
            ),
            peak_left_wrist_speed_ft_s=self._window_max(
                time_series.left_wrist_speed,
                shot_analysis.motion_start_frame,
                shot_analysis.release_frame,
            ),
            peak_ball_speed_ft_s=self._window_max(
                time_series.ball_speed,
                shot_analysis.motion_start_frame,
                shot_analysis.ball_apex_frame,
            ),
            peak_ball_vertical_speed_ft_s=self._window_max(
                time_series.ball_vertical_speed,
                shot_analysis.motion_start_frame,
                shot_analysis.release_frame,
            ),
            minimum_pelvis_height_ft=self._window_min(
                time_series.pelvis_height,
                shot_analysis.motion_start_frame,
                shot_analysis.release_frame,
            ),
            maximum_pelvis_height_ft=self._window_max(
                time_series.pelvis_height,
                shot_analysis.motion_start_frame,
                shot_analysis.release_frame,
            ),
            pelvis_vertical_displacement_ft=self._window_range(
                time_series.pelvis_height,
                shot_analysis.motion_start_frame,
                shot_analysis.release_frame,
            ),

            knee_propulsion_onset_frame=(
                phase_analysis
                .knee_phase
                .propulsion_onset_frame
            ),
            hip_propulsion_onset_frame=(
                phase_analysis
                .hip_phase
                .propulsion_onset_frame
            ),
            pelvis_propulsion_onset_frame=(
                phase_analysis
                .pelvis_phase
                .propulsion_onset_frame
            ),
            shoulder_propulsion_onset_frame=(
                phase_analysis
                .shoulder_phase
                .propulsion_onset_frame
            ),
            elbow_propulsion_onset_frame=(
                phase_analysis
                .elbow_phase
                .propulsion_onset_frame
            ),
            wrist_propulsion_onset_frame=(
                phase_analysis
                .wrist_phase
                .propulsion_onset_frame
            ),
            ball_propulsion_onset_frame=(
                phase_analysis
                .ball_phase
                .propulsion_onset_frame
            ),

            knee_peak_velocity_frame=(
                phase_analysis
                .knee_phase
                .peak_velocity_frame
            ),
            hip_peak_velocity_frame=(
                phase_analysis
                .hip_phase
                .peak_velocity_frame
            ),
            pelvis_peak_velocity_frame=(
                phase_analysis
                .pelvis_phase
                .peak_velocity_frame
            ),
            shoulder_peak_velocity_frame=(
                phase_analysis
                .shoulder_phase
                .peak_velocity_frame
            ),
            elbow_peak_velocity_frame=(
                phase_analysis
                .elbow_phase
                .peak_velocity_frame
            ),
            wrist_peak_velocity_frame=(
                phase_analysis
                .wrist_phase
                .peak_velocity_frame
            ),
            ball_peak_velocity_frame=(
                phase_analysis
                .ball_phase
                .peak_velocity_frame
            ),

            knee_propulsion_duration_ms=(
                phase_analysis
                .knee_phase
                .propulsion_duration_ms
            ),
            hip_propulsion_duration_ms=(
                phase_analysis
                .hip_phase
                .propulsion_duration_ms
            ),
            pelvis_propulsion_duration_ms=(
                phase_analysis
                .pelvis_phase
                .propulsion_duration_ms
            ),
            shoulder_propulsion_duration_ms=(
                phase_analysis
                .shoulder_phase
                .propulsion_duration_ms
            ),
            elbow_propulsion_duration_ms=(
                phase_analysis
                .elbow_phase
                .propulsion_duration_ms
            ),
            wrist_propulsion_duration_ms=(
                phase_analysis
                .wrist_phase
                .propulsion_duration_ms
            ),
            ball_propulsion_duration_ms=(
                phase_analysis
                .ball_phase
                .propulsion_duration_ms
            ),

            hip_to_knee_gap_ms=self._relationship_gap(
                relationships,
                "Hip to knee extension",
            ),
            pelvis_to_knee_gap_ms=self._relationship_gap(
                relationships,
                "Pelvis to knee extension",
            ),
            shoulder_to_elbow_gap_ms=self._relationship_gap(
                relationships,
                "Shoulder to elbow propulsion",
            ),
            knee_to_elbow_gap_ms=self._relationship_gap(
                relationships,
                "Knee to elbow extension",
            ),
            wrist_to_release_gap_ms=self._relationship_gap(
                relationships,
                "Wrist propulsion to release",
            ),
            ball_upward_to_release_gap_ms=self._relationship_gap(
                relationships,
                "Ball upward motion to release",
            ),
            pelvis_to_release_gap_ms=self._relationship_gap(
                relationships,
                "Pelvis upward motion to release",
            ),
            shoulder_to_release_gap_ms=self._relationship_gap(
                relationships,
                "Shoulder propulsion to release",
            ),
            elbow_to_release_gap_ms=self._relationship_gap(
                relationships,
                "Elbow extension to release",
            ),
            ball_lowest_to_release_gap_ms=self._relationship_gap(
                relationships,
                "Ball lowest point to release",
            ),

            propulsion_onset_sequence=" -> ".join(
                phase_analysis
                .propulsion_onset_sequence
            ),
            peak_velocity_sequence=" -> ".join(
                phase_analysis
                .peak_sequence
            ),
            authoritative_event_count=len(
                timeline.events
            ),
            average_event_confidence=(
                float(
                    np.mean(
                        event_confidences
                    )
                )
                if event_confidences
                else None
            ),

            analysis_status="success",
            analysis_error="",
        )

    # =====================================================
    # FAILURE RECORD
    # =====================================================

    def _build_failure_record(
        self,
        error: Exception,
    ) -> ShotFeatureRecord:
        """
        Return a complete failed row without losing error context.
        """

        values: dict[str, object] = {
            field_name: None
            for field_name in (
                ShotFeatureRecord
                .__dataclass_fields__
            )
        }

        values.update(
            {
                "participant_id": str(
                    self.trial_data.get(
                        "participant_id",
                        "Unknown",
                    )
                ),
                "trial_id": str(
                    self.trial_data.get(
                        "trial_id",
                        self.trial_file.stem,
                    )
                ),
                "result": str(
                    self.trial_data.get(
                        "result",
                        "unknown",
                    )
                ),
                "trial_file": (
                    self.trial_file.name
                ),
                "sampling_rate": (
                    self.sampling_rate
                ),
                "total_frames": len(
                    self.trial_data.get(
                        "tracking",
                        [],
                    )
                ),
                "propulsion_onset_sequence": "",
                "peak_velocity_sequence": "",
                "authoritative_event_count": 0,
                "analysis_status": "failed",
                "analysis_error": (
                    f"{type(error).__name__}: {error}"
                ),
            }
        )

        return ShotFeatureRecord(
            **values
        )

    # =====================================================
    # EXPORT
    # =====================================================

    @staticmethod
    def export_csv(
        record: ShotFeatureRecord,
        output_file: Path = DEFAULT_OUTPUT_FILE,
    ) -> Path:
        """
        Export one feature record to CSV.
        """

        destination = Path(
            output_file
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dictionary = asdict(
            record
        )

        with destination.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    dictionary
                ),
            )

            writer.writeheader()
            writer.writerow(
                dictionary
            )

        return destination

    @staticmethod
    def export_json(
        record: ShotFeatureRecord,
        output_file: Path,
    ) -> Path:
        """
        Export one feature record to JSON.
        """

        destination = Path(
            output_file
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with destination.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                asdict(record),
                file,
                indent=2,
            )

        return destination

    # =====================================================
    # HELPERS
    # =====================================================

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
    def _window_values(
        values: np.ndarray,
        start_frame: int | None,
        end_frame: int | None,
    ) -> np.ndarray:
        if values.size == 0:
            return np.array(
                [],
                dtype=float,
            )

        start = (
            0
            if start_frame is None
            else max(
                0,
                int(start_frame),
            )
        )

        end = (
            len(values) - 1
            if end_frame is None
            else min(
                len(values) - 1,
                int(end_frame),
            )
        )

        if end < start:
            return np.array(
                [],
                dtype=float,
            )

        sample = np.asarray(
            values[
                start:
                end + 1
            ],
            dtype=float,
        )

        return sample[
            ~np.isnan(sample)
        ]

    @classmethod
    def _window_max(
        cls,
        values: np.ndarray,
        start_frame: int | None,
        end_frame: int | None,
    ) -> float | None:
        sample = cls._window_values(
            values,
            start_frame,
            end_frame,
        )

        if sample.size == 0:
            return None

        return float(
            np.max(sample)
        )

    @classmethod
    def _window_min(
        cls,
        values: np.ndarray,
        start_frame: int | None,
        end_frame: int | None,
    ) -> float | None:
        sample = cls._window_values(
            values,
            start_frame,
            end_frame,
        )

        if sample.size == 0:
            return None

        return float(
            np.min(sample)
        )

    @classmethod
    def _window_range(
        cls,
        values: np.ndarray,
        start_frame: int | None,
        end_frame: int | None,
    ) -> float | None:
        sample = cls._window_values(
            values,
            start_frame,
            end_frame,
        )

        if sample.size == 0:
            return None

        return float(
            np.max(sample)
            - np.min(sample)
        )

    @classmethod
    def _range_of_motion(
        cls,
        values: np.ndarray,
        start_frame: int | None,
        end_frame: int | None,
    ) -> float | None:
        return cls._window_range(
            values,
            start_frame,
            end_frame,
        )

    @staticmethod
    def _relationship_gap(
        relationships: dict,
        name: str,
    ) -> float | None:
        relationship = relationships.get(
            name
        )

        if relationship is None:
            return None

        return relationship.milliseconds_gap

    @staticmethod
    def _safe_optional(
        value: object,
    ) -> float | None:
        if value is None:
            return None

        try:
            numeric = float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            return None

        if np.isnan(numeric):
            return None

        return numeric

    @staticmethod
    def _safe_float(
        value: object,
        fallback: float,
    ) -> float:
        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return fallback


def _run_shot_feature_extractor_test() -> None:
    """
    Extract and export objective features for the first free throw.
    """

    trial_file, trial_data = load_first_trial()

    extractor = ShotFeatureExtractor(
        trial_file=trial_file,
        trial_data=trial_data,
        shooting_side="RIGHT",
    )

    record = extractor.extract()

    assert (
        record.analysis_status
        == "success"
    )

    assert (
        record.authoritative_event_count
        > 0
    )

    csv_file = extractor.export_csv(
        record
    )

    json_file = extractor.export_json(
        record,
        PROJECT_ROOT
        / "outputs"
        / "first_trial_features.json",
    )

    assert csv_file.exists()
    assert csv_file.stat().st_size > 0

    assert json_file.exists()
    assert json_file.stat().st_size > 0

    feature_dictionary = asdict(
        record
    )

    populated_feature_count = sum(
        value not in {
            None,
            "",
        }
        for value in (
            feature_dictionary.values()
        )
    )

    print(
        "SHOT FEATURE EXTRACTOR TEST PASSED"
    )

    print(
        f"Trial file: {trial_file.name}"
    )

    print(
        "Total feature fields: "
        f"{len(feature_dictionary)}"
    )

    print(
        "Populated fields: "
        f"{populated_feature_count}"
    )

    print(
        "Authoritative events: "
        f"{record.authoritative_event_count}"
    )

    print(
        "Average event confidence: "
        f"{record.average_event_confidence:.3f}"
    )

    print()

    print(
        "CSV output: "
        f"{csv_file.resolve()}"
    )

    print(
        "JSON output: "
        f"{json_file.resolve()}"
    )

    print()

    print(
        "Selected objective features:"
    )

    print(
        "Motion to release: "
        f"{record.motion_to_release_ms:.1f} ms"
    )

    print(
        "Release angle: "
        f"{record.release_angle_deg:.2f} degrees"
    )

    print(
        "Release ball speed: "
        f"{record.release_ball_speed_ft_s:.2f} ft/s"
    )

    print(
        "Right knee range of motion: "
        f"{record.right_knee_range_of_motion_deg:.2f} degrees"
    )

    print(
        "Hip to knee gap: "
        f"{record.hip_to_knee_gap_ms:.1f} ms"
    )

    print(
        "Wrist to release gap: "
        f"{record.wrist_to_release_gap_ms:.1f} ms"
    )


if __name__ == "__main__":
    _run_shot_feature_extractor_test()