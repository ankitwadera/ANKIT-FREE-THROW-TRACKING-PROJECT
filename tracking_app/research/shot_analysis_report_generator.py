from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from tracking_app.analyzer import ShotAnalyzer
from tracking_app.basketball_event_mapper import BasketballEventMapper
from tracking_app.data_loader import load_first_trial
from tracking_app.phase_biomechanics import RefinedPhaseBiomechanicsAnalyzer
from tracking_app.research.personal_shot_scoring_engine import (
    PersonalShotScoringEngine,
)
from tracking_app.shot_feature_extractor import ShotFeatureExtractor
from tracking_app.shot_timeline_builder import ShotTimelineBuilder
from tracking_app.timeseries import TimeSeriesAnalyzer
from tracking_app.whole_body_synchronization import (
    WholeBodySynchronizationEngine,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "shot_analysis_report"
)


@dataclass(frozen=True)
class ReportEventRow:
    """
    One event in the final chronological shot timeline.
    """

    frame: int
    time_ms: float
    event_name: str
    source: str
    signal: str
    confidence: float


@dataclass(frozen=True)
class ReportMetricRow:
    """
    One coach-readable metric used in the report.
    """

    category: str
    metric: str
    value: float | None
    unit: str
    explanation: str


@dataclass(frozen=True)
class ReportRecommendationRow:
    """
    One prioritized coaching observation.
    """

    rank: int
    category: str
    feature: str
    observation: str
    similarity_score: float
    deviation_standard_deviations: float | None
    evidence_weight: float
    confidence: str


@dataclass(frozen=True)
class ShotAnalysisReportSummary:
    """
    Top-level report metadata and results.
    """

    participant_id: str
    trial_id: str
    trial_file: str
    recorded_result: str

    total_frames: int
    sampling_rate: float

    motion_start_frame: int | None
    dip_frame: int | None
    takeoff_frame: int | None
    release_frame: int | None
    ball_apex_frame: int | None
    landing_frame: int | None

    authoritative_event_count: int

    overall_similarity_score: float
    confidence_adjusted_score: float
    score_confidence: str

    lower_body_score: float | None
    upper_body_score: float | None
    ball_release_score: float | None
    timing_coordination_score: float | None

    strongest_deviation: str
    overall_assessment: str

    report_status: str
    report_error: str


class ShotAnalysisReportGenerator:
    """
    Create the first end-to-end coaching report package for one JSON trial.

    The generated package combines:

    - trial metadata;
    - direct shot events;
    - authoritative basketball timeline;
    - objective release and biomechanics metrics;
    - synchronization relationships;
    - personal baseline score;
    - prioritized coaching observations;
    - dashboard image;
    - permanent CSV, JSON, and text outputs.

    The report compares the selected shot with the participant's own
    successful history. It does not assign a universal free-throw grade.
    """

    def __init__(
        self,
        output_folder: Path = DEFAULT_OUTPUT_FOLDER,
        shooting_side: str = "RIGHT",
        maximum_recommendations: int = 6,
    ) -> None:
        self.output_folder = Path(
            output_folder
        )

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

        self.maximum_recommendations = max(
            1,
            int(
                maximum_recommendations
            ),
        )

        self.summary_file = (
            self.output_folder
            / "report_summary.csv"
        )

        self.timeline_file = (
            self.output_folder
            / "shot_timeline.csv"
        )

        self.metrics_file = (
            self.output_folder
            / "shot_metrics.csv"
        )

        self.recommendations_file = (
            self.output_folder
            / "coaching_observations.csv"
        )

        self.json_file = (
            self.output_folder
            / "shot_analysis_report.json"
        )

        self.text_file = (
            self.output_folder
            / "shot_analysis_report.txt"
        )

        self.dashboard_file = (
            self.output_folder
            / "shot_analysis_dashboard.png"
        )

        self.source_json_copy = (
            self.output_folder
            / "source_trial.json"
        )

    # =====================================================
    # PUBLIC WORKFLOW
    # =====================================================

    def run(
        self,
        trial_file: Path | None = None,
        trial_data: dict | None = None,
    ) -> tuple[
        ShotAnalysisReportSummary,
        list[ReportEventRow],
        list[ReportMetricRow],
        list[ReportRecommendationRow],
    ]:
        """
        Generate the full report package.

        When no trial is supplied, the first dataset trial is used.
        """

        if (
            trial_file is None
            or trial_data is None
        ):
            trial_file, trial_data = (
                load_first_trial()
            )

        trial_file = Path(
            trial_file
        )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            shot_analysis = ShotAnalyzer(
                trial_data=trial_data,
                trial_file=trial_file,
                shooting_wrist=(
                    f"{self.shooting_side}_WRIST"
                ),
            ).analyze()

            time_series = TimeSeriesAnalyzer(
                trial_data=trial_data,
                smoothing_window=5,
            ).analyze()

            sampling_rate = float(
                trial_data.get(
                    "sampling_rate",
                    30,
                )
            )

            phase_analysis = (
                RefinedPhaseBiomechanicsAnalyzer(
                    time_series=time_series,
                    sampling_rate=sampling_rate,
                    motion_start_frame=(
                        shot_analysis
                        .motion_start_frame
                    ),
                    dip_frame=(
                        shot_analysis.dip_frame
                    ),
                    takeoff_frame=(
                        shot_analysis
                        .takeoff_frame
                    ),
                    release_frame=(
                        shot_analysis
                        .release_frame
                    ),
                    shooting_side=(
                        self.shooting_side
                    ),
                ).analyze()
            )

            basketball_timeline = (
                BasketballEventMapper(
                    time_series=time_series,
                    sampling_rate=sampling_rate,
                    motion_start_frame=(
                        shot_analysis
                        .motion_start_frame
                    ),
                    dip_frame=(
                        shot_analysis.dip_frame
                    ),
                    takeoff_frame=(
                        shot_analysis
                        .takeoff_frame
                    ),
                    release_frame=(
                        shot_analysis
                        .release_frame
                    ),
                    ball_apex_frame=(
                        shot_analysis
                        .ball_apex_frame
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
                    shot_analysis
                    .landing_frame
                ),
            ).build()

            synchronization = (
                WholeBodySynchronizationEngine(
                    timeline=timeline,
                    sampling_rate=sampling_rate,
                ).analyze()
            )

            feature_record = ShotFeatureExtractor(
                trial_file=trial_file,
                trial_data=trial_data,
                shooting_side=(
                    self.shooting_side
                ),
            ).extract()

            if (
                feature_record.analysis_status
                != "success"
            ):
                raise RuntimeError(
                    feature_record.analysis_error
                )

            scoring_engine = (
                PersonalShotScoringEngine()
            )

            (
                score_summary,
                scored_features,
                category_scores,
            ) = scoring_engine.run(
                participant_id=(
                    feature_record
                    .participant_id
                ),
                trial_id=(
                    feature_record
                    .trial_id
                ),
            )

            event_rows = self.build_event_rows(
                timeline=timeline,
                sampling_rate=sampling_rate,
            )

            metric_rows = self.build_metric_rows(
                feature_record=feature_record,
                synchronization=synchronization,
            )

            recommendation_rows = (
                self.build_recommendation_rows(
                    scored_features
                )
            )

            summary = ShotAnalysisReportSummary(
                participant_id=(
                    feature_record
                    .participant_id
                ),
                trial_id=(
                    feature_record
                    .trial_id
                ),
                trial_file=(
                    feature_record
                    .trial_file
                ),
                recorded_result=(
                    feature_record.result
                ),

                total_frames=(
                    feature_record
                    .total_frames
                ),
                sampling_rate=(
                    feature_record
                    .sampling_rate
                ),

                motion_start_frame=(
                    feature_record
                    .motion_start_frame
                ),
                dip_frame=(
                    feature_record
                    .dip_frame
                ),
                takeoff_frame=(
                    feature_record
                    .takeoff_frame
                ),
                release_frame=(
                    feature_record
                    .release_frame
                ),
                ball_apex_frame=(
                    feature_record
                    .ball_apex_frame
                ),
                landing_frame=(
                    feature_record
                    .landing_frame
                ),

                authoritative_event_count=(
                    feature_record
                    .authoritative_event_count
                ),

                overall_similarity_score=(
                    score_summary
                    .overall_similarity_score
                ),
                confidence_adjusted_score=(
                    score_summary
                    .confidence_adjusted_score
                ),
                score_confidence=(
                    score_summary
                    .score_confidence
                ),

                lower_body_score=(
                    score_summary
                    .lower_body_score
                ),
                upper_body_score=(
                    score_summary
                    .upper_body_score
                ),
                ball_release_score=(
                    score_summary
                    .ball_release_score
                ),
                timing_coordination_score=(
                    score_summary
                    .timing_coordination_score
                ),

                strongest_deviation=(
                    self._friendly_feature_name(
                        score_summary
                        .strongest_deviation_feature
                    )
                ),
                overall_assessment=(
                    score_summary
                    .overall_assessment
                ),

                report_status="success",
                report_error="",
            )

            self.export_outputs(
                summary=summary,
                event_rows=event_rows,
                metric_rows=metric_rows,
                recommendation_rows=(
                    recommendation_rows
                ),
            )

            self.create_dashboard(
                summary=summary,
                event_rows=event_rows,
                metric_rows=metric_rows,
                recommendation_rows=(
                    recommendation_rows
                ),
            )

            self.copy_source_trial(
                trial_file=trial_file,
                trial_data=trial_data,
            )

            return (
                summary,
                event_rows,
                metric_rows,
                recommendation_rows,
            )

        except Exception as error:
            failure_summary = (
                self.build_failure_summary(
                    trial_file=trial_file,
                    trial_data=trial_data,
                    error=error,
                )
            )

            self.export_outputs(
                summary=failure_summary,
                event_rows=[],
                metric_rows=[],
                recommendation_rows=[],
            )

            raise

    # =====================================================
    # TIMELINE
    # =====================================================

    @staticmethod
    def build_event_rows(
        timeline: object,
        sampling_rate: float,
    ) -> list[ReportEventRow]:
        rows: list[
            ReportEventRow
        ] = []

        for event in timeline.events:
            rows.append(
                ReportEventRow(
                    frame=int(
                        event.frame
                    ),
                    time_ms=float(
                        event.frame
                        / sampling_rate
                        * 1000.0
                    ),
                    event_name=(
                        event.event_type.value
                    ),
                    source=(
                        event.source.value
                    ),
                    signal=(
                        event.source_signal
                    ),
                    confidence=float(
                        event.confidence
                    ),
                )
            )

        return rows

    # =====================================================
    # METRICS
    # =====================================================

    def build_metric_rows(
        self,
        feature_record: object,
        synchronization: object,
    ) -> list[ReportMetricRow]:
        rows = [
            ReportMetricRow(
                category="Shot Timing",
                metric="Motion start to release",
                value=(
                    feature_record
                    .motion_to_release_ms
                ),
                unit="ms",
                explanation=(
                    "Total time from the beginning of the shooting motion "
                    "to ball release."
                ),
            ),
            ReportMetricRow(
                category="Shot Timing",
                metric="Dip to takeoff",
                value=(
                    feature_record
                    .dip_to_takeoff_ms
                ),
                unit="ms",
                explanation=(
                    "Time between the lowest loading point and takeoff."
                ),
            ),
            ReportMetricRow(
                category="Shot Timing",
                metric="Takeoff to release",
                value=(
                    feature_record
                    .takeoff_to_release_ms
                ),
                unit="ms",
                explanation=(
                    "Time between leaving the floor and releasing the ball."
                ),
            ),
            ReportMetricRow(
                category="Release",
                metric="Release angle",
                value=(
                    feature_record
                    .release_angle_deg
                ),
                unit="degrees",
                explanation=(
                    "Upward direction of the ball at release."
                ),
            ),
            ReportMetricRow(
                category="Release",
                metric="Ball speed at release",
                value=(
                    feature_record
                    .release_ball_speed_ft_s
                ),
                unit="ft/s",
                explanation=(
                    "Three-dimensional speed of the ball at release."
                ),
            ),
            ReportMetricRow(
                category="Release",
                metric="Release height",
                value=(
                    feature_record
                    .release_height_ft
                ),
                unit="ft",
                explanation=(
                    "Vertical height of the ball at release."
                ),
            ),
            ReportMetricRow(
                category="Lower Body",
                metric="Right knee range of motion",
                value=(
                    feature_record
                    .right_knee_range_of_motion_deg
                ),
                unit="degrees",
                explanation=(
                    "Total right-knee angle change from motion start "
                    "to release."
                ),
            ),
            ReportMetricRow(
                category="Lower Body",
                metric="Right hip range of motion",
                value=(
                    feature_record
                    .right_hip_range_of_motion_deg
                ),
                unit="degrees",
                explanation=(
                    "Total right-hip angle change from motion start "
                    "to release."
                ),
            ),
            ReportMetricRow(
                category="Upper Body",
                metric="Right elbow angle at release",
                value=(
                    feature_record
                    .release_right_elbow_angle_deg
                ),
                unit="degrees",
                explanation=(
                    "Right-elbow bend at the release frame."
                ),
            ),
            ReportMetricRow(
                category="Upper Body",
                metric="Peak right wrist speed",
                value=(
                    feature_record
                    .peak_right_wrist_speed_ft_s
                ),
                unit="ft/s",
                explanation=(
                    "Fastest tracked right-wrist movement before release."
                ),
            ),
            ReportMetricRow(
                category="Coordination",
                metric="Hip-to-knee timing gap",
                value=(
                    feature_record
                    .hip_to_knee_gap_ms
                ),
                unit="ms",
                explanation=(
                    "Timing difference between hip and knee propulsion."
                ),
            ),
            ReportMetricRow(
                category="Coordination",
                metric="Knee-to-elbow timing gap",
                value=(
                    feature_record
                    .knee_to_elbow_gap_ms
                ),
                unit="ms",
                explanation=(
                    "Timing difference between lower-body extension "
                    "and elbow propulsion."
                ),
            ),
            ReportMetricRow(
                category="Coordination",
                metric="Elbow-to-release timing gap",
                value=(
                    feature_record
                    .elbow_to_release_gap_ms
                ),
                unit="ms",
                explanation=(
                    "Time from elbow-extension onset to ball release."
                ),
            ),
            ReportMetricRow(
                category="Coordination",
                metric="Synchronization relationships evaluated",
                value=float(
                    len(
                        synchronization
                        .relationships
                    )
                ),
                unit="relationships",
                explanation=(
                    "Number of whole-body timing relationships evaluated."
                ),
            ),
        ]

        return rows

    # =====================================================
    # RECOMMENDATIONS
    # =====================================================

    def build_recommendation_rows(
        self,
        scored_features: list[object],
    ) -> list[ReportRecommendationRow]:
        rows: list[
            ReportRecommendationRow
        ] = []

        for rank, feature in enumerate(
            scored_features[
                :self.maximum_recommendations
            ],
            start=1,
        ):
            rows.append(
                ReportRecommendationRow(
                    rank=rank,
                    category=(
                        feature.category
                    ),
                    feature=(
                        feature.coach_label
                    ),
                    observation=(
                        feature.coach_observation
                    ),
                    similarity_score=float(
                        feature
                        .feature_similarity_score
                    ),
                    deviation_standard_deviations=(
                        feature.absolute_z_score
                    ),
                    evidence_weight=float(
                        feature.evidence_weight
                    ),
                    confidence=(
                        self._feature_confidence(
                            feature
                        )
                    ),
                )
            )

        return rows

    @staticmethod
    def _feature_confidence(
        feature: object,
    ) -> str:
        stability = (
            feature
            .direction_stability_percentage
            or 0.0
        )

        if (
            feature.diagnostic_score
            >= 70
            and stability >= 85
            and feature.evidence_weight
            >= 0.75
        ):
            return "moderate"

        if (
            feature.diagnostic_score
            >= 55
            and stability >= 70
        ):
            return "limited"

        return "exploratory"

    # =====================================================
    # EXPORTS
    # =====================================================

    def export_outputs(
        self,
        summary: ShotAnalysisReportSummary,
        event_rows: list[ReportEventRow],
        metric_rows: list[ReportMetricRow],
        recommendation_rows: list[
            ReportRecommendationRow
        ],
    ) -> None:
        self.export_dataclass_rows(
            self.summary_file,
            [
                summary
            ],
        )

        self.export_dataclass_rows(
            self.timeline_file,
            event_rows,
        )

        self.export_dataclass_rows(
            self.metrics_file,
            metric_rows,
        )

        self.export_dataclass_rows(
            self.recommendations_file,
            recommendation_rows,
        )

        report_dictionary = {
            "summary": asdict(
                summary
            ),
            "timeline": [
                asdict(
                    row
                )
                for row in event_rows
            ],
            "metrics": [
                asdict(
                    row
                )
                for row in metric_rows
            ],
            "recommendations": [
                asdict(
                    row
                )
                for row in recommendation_rows
            ],
        }

        with self.json_file.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                report_dictionary,
                file,
                indent=2,
            )

        self.export_text_report(
            summary=summary,
            event_rows=event_rows,
            metric_rows=metric_rows,
            recommendation_rows=(
                recommendation_rows
            ),
        )

    @staticmethod
    def export_dataclass_rows(
        output_file: Path,
        rows: list[object],
    ) -> Path:
        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not rows:
            output_file.write_text(
                "",
                encoding="utf-8",
            )

            return output_file

        dictionaries = [
            asdict(
                row
            )
            for row in rows
        ]

        fieldnames = list(
            dictionaries[0]
        )

        with output_file.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                dictionaries
            )

        return output_file

    def export_text_report(
        self,
        summary: ShotAnalysisReportSummary,
        event_rows: list[ReportEventRow],
        metric_rows: list[ReportMetricRow],
        recommendation_rows: list[
            ReportRecommendationRow
        ],
    ) -> Path:
        lines = [
            "=" * 92,
            "PERSONAL FREE-THROW SHOT ANALYSIS REPORT",
            "=" * 92,
            "",
            "SHOT INFORMATION",
            "-" * 92,
            (
                f"Participant: "
                f"{summary.participant_id}"
            ),
            (
                f"Trial: "
                f"{summary.trial_id}"
            ),
            (
                f"Source file: "
                f"{summary.trial_file}"
            ),
            (
                f"Recorded result: "
                f"{summary.recorded_result}"
            ),
            (
                f"Tracking frames: "
                f"{summary.total_frames}"
            ),
            (
                f"Sampling rate: "
                f"{summary.sampling_rate:.1f} Hz"
            ),
            "",
            "PERSONAL SUCCESSFUL-BASELINE SCORE",
            "-" * 92,
            (
                "Overall similarity: "
                f"{summary.overall_similarity_score:.1f} / 100"
            ),
            (
                "Confidence-adjusted score: "
                f"{summary.confidence_adjusted_score:.1f} / 100"
            ),
            (
                "Score confidence: "
                f"{summary.score_confidence}"
            ),
            (
                "Strongest deviation area: "
                f"{summary.strongest_deviation}"
            ),
            "",
            "CATEGORY SCORES",
            "-" * 92,
            self._format_optional_score(
                "Lower body",
                summary.lower_body_score,
            ),
            self._format_optional_score(
                "Upper body",
                summary.upper_body_score,
            ),
            self._format_optional_score(
                "Ball and release",
                summary.ball_release_score,
            ),
            self._format_optional_score(
                "Timing and coordination",
                summary.timing_coordination_score,
            ),
            "",
            "OVERALL ASSESSMENT",
            "-" * 92,
            summary.overall_assessment,
            "",
            "AUTHORITATIVE SHOT TIMELINE",
            "-" * 92,
        ]

        for event in event_rows:
            lines.append(
                (
                    f"Frame {event.frame} "
                    f"({event.time_ms:.1f} ms): "
                    f"{self._friendly_event_name(event.event_name)} "
                    f"[{event.source}, confidence "
                    f"{event.confidence:.2f}]"
                )
            )

        lines.extend(
            [
                "",
                "SELECTED OBJECTIVE METRICS",
                "-" * 92,
            ]
        )

        for metric in metric_rows:
            value_text = (
                "N/A"
                if metric.value is None
                else (
                    f"{metric.value:.2f} "
                    f"{metric.unit}"
                )
            )

            lines.append(
                (
                    f"{metric.category} — "
                    f"{metric.metric}: "
                    f"{value_text}"
                )
            )

        lines.extend(
            [
                "",
                "PRIORITIZED COACHING OBSERVATIONS",
                "-" * 92,
            ]
        )

        if not recommendation_rows:
            lines.append(
                "No personally supported observations were available."
            )

        for row in recommendation_rows:
            deviation_text = (
                "N/A"
                if row.deviation_standard_deviations
                is None
                else (
                    f"{row.deviation_standard_deviations:.2f} SD"
                )
            )

            lines.extend(
                [
                    (
                        f"{row.rank}. "
                        f"{row.feature}"
                    ),
                    (
                        f"   Category: "
                        f"{row.category}"
                    ),
                    (
                        f"   Observation: "
                        f"{row.observation}"
                    ),
                    (
                        f"   Personal similarity: "
                        f"{row.similarity_score:.1f} / 100"
                    ),
                    (
                        f"   Deviation: "
                        f"{deviation_text}"
                    ),
                    (
                        f"   Evidence confidence: "
                        f"{row.confidence}"
                    ),
                    "",
                ]
            )

        lines.extend(
            [
                "LIMITATIONS",
                "-" * 92,
                (
                    "This report compares one tracked shot with the same "
                    "participant's previous successful shots."
                ),
                (
                    "It identifies measurable deviations but does not prove "
                    "that any one difference caused the make or miss."
                ),
                (
                    "Finger forces, ball spin, precise aim direction, fatigue, "
                    "and other variables may not be fully represented."
                ),
            ]
        )

        self.text_file.write_text(
            "\n".join(
                lines
            ),
            encoding="utf-8",
        )

        return self.text_file

    # =====================================================
    # DASHBOARD
    # =====================================================

    def create_dashboard(
        self,
        summary: ShotAnalysisReportSummary,
        event_rows: list[ReportEventRow],
        metric_rows: list[ReportMetricRow],
        recommendation_rows: list[
            ReportRecommendationRow
        ],
    ) -> Path:
        figure = plt.figure(
            figsize=(
                15,
                13,
            )
        )

        grid = figure.add_gridspec(
            4,
            1,
            height_ratios=[
                0.9,
                1.2,
                1.4,
                2.2,
            ],
        )

        score_axis = figure.add_subplot(
            grid[0]
        )

        category_axis = figure.add_subplot(
            grid[1]
        )

        timeline_axis = figure.add_subplot(
            grid[2]
        )

        recommendation_axis = (
            figure.add_subplot(
                grid[3]
            )
        )

        score_axis.barh(
            [
                0
            ],
            [
                summary
                .overall_similarity_score
            ],
        )

        score_axis.set_xlim(
            0,
            100,
        )

        score_axis.set_yticks(
            [
                0
            ],
            labels=[
                "Personal similarity"
            ],
        )

        score_axis.set_title(
            (
                f"Shot Analysis — "
                f"{summary.participant_id} "
                f"{summary.trial_id} "
                f"({summary.recorded_result})"
            )
        )

        score_axis.set_xlabel(
            "Score"
        )

        score_axis.grid(
            visible=True,
            axis="x",
            alpha=0.25,
        )

        category_values = [
            (
                "Lower Body",
                summary.lower_body_score,
            ),
            (
                "Upper Body",
                summary.upper_body_score,
            ),
            (
                "Ball & Release",
                summary.ball_release_score,
            ),
            (
                "Timing & Coordination",
                summary.timing_coordination_score,
            ),
        ]

        category_values = [
            (
                label,
                value,
            )
            for label, value in category_values
            if value is not None
        ]

        category_labels = [
            label
            for label, _ in category_values
        ]

        category_scores = [
            value
            for _, value in category_values
        ]

        category_positions = list(
            range(
                len(
                    category_labels
                )
            )
        )

        category_axis.barh(
            category_positions,
            category_scores,
        )

        category_axis.set_yticks(
            category_positions,
            labels=category_labels,
        )

        category_axis.set_xlim(
            0,
            100,
        )

        category_axis.set_xlabel(
            "Similarity score"
        )

        category_axis.set_title(
            "Category Scores"
        )

        category_axis.grid(
            visible=True,
            axis="x",
            alpha=0.25,
        )

        if event_rows:
            event_frames = [
                event.frame
                for event in event_rows
            ]

            event_positions = list(
                range(
                    len(
                        event_rows
                    )
                )
            )

            timeline_axis.scatter(
                event_frames,
                event_positions,
                s=45,
            )

            timeline_axis.set_yticks(
                event_positions,
                labels=[
                    self._friendly_event_name(
                        event.event_name
                    )
                    for event in event_rows
                ],
                fontsize=8,
            )

            timeline_axis.set_xlabel(
                "Frame"
            )

            timeline_axis.set_title(
                "Authoritative Shot Timeline"
            )

            timeline_axis.grid(
                visible=True,
                axis="x",
                alpha=0.25,
            )
        else:
            timeline_axis.text(
                0.5,
                0.5,
                "No timeline available",
                ha="center",
                va="center",
            )

            timeline_axis.set_axis_off()

        if recommendation_rows:
            top_rows = recommendation_rows[
                :self.maximum_recommendations
            ]

            labels = [
                row.feature
                for row in reversed(
                    top_rows
                )
            ]

            values = [
                row.similarity_score
                for row in reversed(
                    top_rows
                )
            ]

            positions = list(
                range(
                    len(
                        labels
                    )
                )
            )

            recommendation_axis.barh(
                positions,
                values,
            )

            recommendation_axis.set_yticks(
                positions,
                labels=labels,
                fontsize=8,
            )

            recommendation_axis.set_xlim(
                0,
                100,
            )

            recommendation_axis.set_xlabel(
                "Similarity to personal made-shot baseline"
            )

            recommendation_axis.set_title(
                "Lowest-Similarity Coaching Observations"
            )

            recommendation_axis.grid(
                visible=True,
                axis="x",
                alpha=0.25,
            )
        else:
            recommendation_axis.text(
                0.5,
                0.5,
                "No recommendations available",
                ha="center",
                va="center",
            )

            recommendation_axis.set_axis_off()

        figure.tight_layout()

        figure.savefig(
            self.dashboard_file,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(
            figure
        )

        return self.dashboard_file

    # =====================================================
    # SOURCE COPY
    # =====================================================

    def copy_source_trial(
        self,
        trial_file: Path,
        trial_data: dict,
    ) -> Path:
        """
        Preserve the analyzed source trial inside the report package.
        """

        if trial_file.exists():
            shutil.copy2(
                trial_file,
                self.source_json_copy,
            )

            return self.source_json_copy

        with self.source_json_copy.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                trial_data,
                file,
                indent=2,
            )

        return self.source_json_copy

    # =====================================================
    # FAILURE SUMMARY
    # =====================================================

    @staticmethod
    def build_failure_summary(
        trial_file: Path,
        trial_data: dict,
        error: Exception,
    ) -> ShotAnalysisReportSummary:
        return ShotAnalysisReportSummary(
            participant_id=str(
                trial_data.get(
                    "participant_id",
                    "Unknown",
                )
            ),
            trial_id=str(
                trial_data.get(
                    "trial_id",
                    trial_file.stem,
                )
            ),
            trial_file=(
                trial_file.name
            ),
            recorded_result=str(
                trial_data.get(
                    "result",
                    "unknown",
                )
            ),

            total_frames=len(
                trial_data.get(
                    "tracking",
                    [],
                )
            ),
            sampling_rate=float(
                trial_data.get(
                    "sampling_rate",
                    30,
                )
            ),

            motion_start_frame=None,
            dip_frame=None,
            takeoff_frame=None,
            release_frame=None,
            ball_apex_frame=None,
            landing_frame=None,

            authoritative_event_count=0,

            overall_similarity_score=0.0,
            confidence_adjusted_score=0.0,
            score_confidence="unavailable",

            lower_body_score=None,
            upper_body_score=None,
            ball_release_score=None,
            timing_coordination_score=None,

            strongest_deviation="N/A",
            overall_assessment=(
                "The report could not be completed."
            ),

            report_status="failed",
            report_error=(
                f"{type(error).__name__}: "
                f"{error}"
            ),
        )

    # =====================================================
    # LANGUAGE HELPERS
    # =====================================================

    @staticmethod
    def _friendly_event_name(
        event_name: str,
    ) -> str:
        return event_name.replace(
            "_",
            " ",
        ).title()

    @staticmethod
    def _friendly_feature_name(
        feature: str,
    ) -> str:
        mapping = {
            "takeoff_to_release_ms":
                "time from takeoff to release",
            "release_to_landing_ms":
                "time from release to landing",
            "motion_to_release_ms":
                "time from motion start to release",
            "release_right_hip_angle_deg":
                "right hip angle at release",
            "release_left_hip_angle_deg":
                "left hip angle at release",
            "release_right_knee_angle_deg":
                "right knee angle at release",
            "release_left_knee_angle_deg":
                "left knee angle at release",
            "release_right_elbow_angle_deg":
                "right elbow angle at release",
            "peak_right_wrist_speed_ft_s":
                "peak right wrist speed",
            "peak_left_wrist_speed_ft_s":
                "peak left wrist speed",
            "knee_to_elbow_gap_ms":
                "knee-to-elbow timing gap",
            "elbow_to_release_gap_ms":
                "elbow-to-release timing gap",
            "release_angle_deg":
                "ball release angle",
        }

        return mapping.get(
            feature,
            feature.replace(
                "_",
                " ",
            ),
        )

    @staticmethod
    def _format_optional_score(
        label: str,
        value: float | None,
    ) -> str:
        if value is None:
            return (
                f"{label}: N/A"
            )

        return (
            f"{label}: "
            f"{value:.1f} / 100"
        )


def _run_shot_analysis_report_test() -> None:
    """
    Generate a full report package for the first free-throw JSON trial.
    """

    generator = (
        ShotAnalysisReportGenerator()
    )

    (
        summary,
        event_rows,
        metric_rows,
        recommendation_rows,
    ) = generator.run()

    assert (
        summary.report_status
        == "success"
    )

    assert len(
        event_rows
    ) > 0

    assert len(
        metric_rows
    ) > 0

    assert len(
        recommendation_rows
    ) > 0

    assert generator.summary_file.exists()
    assert generator.timeline_file.exists()
    assert generator.metrics_file.exists()
    assert generator.recommendations_file.exists()
    assert generator.json_file.exists()
    assert generator.text_file.exists()
    assert generator.dashboard_file.exists()
    assert generator.source_json_copy.exists()

    print(
        "SHOT ANALYSIS REPORT GENERATOR TEST PASSED"
    )

    print(
        f"Participant: "
        f"{summary.participant_id}"
    )

    print(
        f"Trial: "
        f"{summary.trial_id}"
    )

    print(
        f"Recorded result: "
        f"{summary.recorded_result}"
    )

    print(
        "Overall similarity score: "
        f"{summary.overall_similarity_score:.1f} / 100"
    )

    print(
        "Confidence-adjusted score: "
        f"{summary.confidence_adjusted_score:.1f} / 100"
    )

    print(
        f"Score confidence: "
        f"{summary.score_confidence}"
    )

    print(
        "Authoritative timeline events: "
        f"{len(event_rows)}"
    )

    print(
        "Objective metrics: "
        f"{len(metric_rows)}"
    )

    print(
        "Coaching observations: "
        f"{len(recommendation_rows)}"
    )

    print()

    print(
        "Report folder: "
        f"{generator.output_folder.resolve()}"
    )

    print(
        "Summary CSV: "
        f"{generator.summary_file.resolve()}"
    )

    print(
        "Timeline CSV: "
        f"{generator.timeline_file.resolve()}"
    )

    print(
        "Metrics CSV: "
        f"{generator.metrics_file.resolve()}"
    )

    print(
        "Observations CSV: "
        f"{generator.recommendations_file.resolve()}"
    )

    print(
        "JSON report: "
        f"{generator.json_file.resolve()}"
    )

    print(
        "Text report: "
        f"{generator.text_file.resolve()}"
    )

    print(
        "Dashboard: "
        f"{generator.dashboard_file.resolve()}"
    )

    print(
        "Source trial copy: "
        f"{generator.source_json_copy.resolve()}"
    )


if __name__ == "__main__":
    _run_shot_analysis_report_test()