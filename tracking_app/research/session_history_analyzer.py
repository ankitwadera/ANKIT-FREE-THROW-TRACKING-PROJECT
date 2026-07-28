from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, pstdev

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FEATURE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "all_shot_features.csv"
)

DEFAULT_OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "session_history"
)


@dataclass(frozen=True)
class SessionShotRow:
    participant_id: str
    trial_id: str
    trial_file: str
    result: str

    shot_number: int

    release_angle_deg: float | None
    release_ball_speed_ft_s: float | None
    release_height_ft: float | None

    right_knee_range_of_motion_deg: float | None
    right_hip_range_of_motion_deg: float | None
    release_right_elbow_angle_deg: float | None

    knee_to_elbow_gap_ms: float | None
    elbow_to_release_gap_ms: float | None
    takeoff_to_release_ms: float | None


@dataclass(frozen=True)
class SessionFeatureSummary:
    participant_id: str
    feature: str

    valid_shots: int
    made_shots: int
    missed_shots: int

    session_mean: float
    session_median: float
    session_standard_deviation: float

    first_10_mean: float | None
    last_10_mean: float | None
    first_to_last_change: float | None

    made_mean: float | None
    missed_mean: float | None
    made_missed_difference: float | None

    consistency_score: float
    trend_direction: str


@dataclass(frozen=True)
class SessionHistorySummary:
    participant_id: str

    total_shots: int
    made_shots: int
    missed_shots: int
    make_percentage: float

    features_analyzed: int

    most_consistent_feature: str
    least_consistent_feature: str

    largest_positive_trend_feature: str
    largest_negative_trend_feature: str

    session_status: str


class SessionHistoryAnalyzer:
    """
    Analyze a participant's complete tracked free-throw history.

    This module does not change the shot-analysis pipeline. It builds the
    foundation for a future practice-history dashboard by measuring:

    - shot-by-shot trends;
    - make percentage;
    - feature consistency;
    - first-ten versus last-ten changes;
    - made-versus-missed averages;
    - potential fatigue or adaptation signals.
    """

    TRACKED_FEATURES = (
        "release_angle_deg",
        "release_ball_speed_ft_s",
        "release_height_ft",
        "right_knee_range_of_motion_deg",
        "right_hip_range_of_motion_deg",
        "release_right_elbow_angle_deg",
        "knee_to_elbow_gap_ms",
        "elbow_to_release_gap_ms",
        "takeoff_to_release_ms",
    )

    FRIENDLY_NAMES = {
        "release_angle_deg":
            "Ball release angle",
        "release_ball_speed_ft_s":
            "Ball speed at release",
        "release_height_ft":
            "Ball release height",
        "right_knee_range_of_motion_deg":
            "Shooting-side knee movement range",
        "right_hip_range_of_motion_deg":
            "Shooting-side hip movement range",
        "release_right_elbow_angle_deg":
            "Shooting elbow angle at release",
        "knee_to_elbow_gap_ms":
            "Knee-to-elbow timing",
        "elbow_to_release_gap_ms":
            "Elbow-to-release timing",
        "takeoff_to_release_ms":
            "Time from takeoff to release",
    }

    def __init__(
        self,
        feature_file: Path = DEFAULT_FEATURE_FILE,
        output_folder: Path = DEFAULT_OUTPUT_FOLDER,
    ) -> None:
        self.feature_file = Path(
            feature_file
        )

        self.output_folder = Path(
            output_folder
        )

        self.shot_history_file = (
            self.output_folder
            / "session_shot_history.csv"
        )

        self.feature_summary_file = (
            self.output_folder
            / "session_feature_summary.csv"
        )

        self.session_summary_file = (
            self.output_folder
            / "session_history_summary.csv"
        )

        self.dashboard_file = (
            self.output_folder
            / "session_history_dashboard.png"
        )

        self.report_file = (
            self.output_folder
            / "session_history_report.txt"
        )

    # =====================================================
    # PUBLIC WORKFLOW
    # =====================================================

    def run(
        self,
        participant_id: str = "P0001",
    ) -> tuple[
        SessionHistorySummary,
        list[SessionShotRow],
        list[SessionFeatureSummary],
    ]:
        records = self.load_feature_records()

        participant_records = [
            record
            for record in records
            if str(
                record.get(
                    "participant_id",
                    "",
                )
            ).strip()
            == participant_id
            and str(
                record.get(
                    "analysis_status",
                    "",
                )
            ).strip().lower()
            == "success"
        ]

        if not participant_records:
            raise ValueError(
                f"No successful shots were found for {participant_id}."
            )

        participant_records = sorted(
            participant_records,
            key=self._trial_sort_key,
        )

        shot_rows = self.build_shot_rows(
            participant_id=participant_id,
            records=participant_records,
        )

        feature_summaries = self.build_feature_summaries(
            participant_id=participant_id,
            records=participant_records,
        )

        summary = self.build_session_summary(
            participant_id=participant_id,
            shot_rows=shot_rows,
            feature_summaries=feature_summaries,
        )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.export_rows(
            self.shot_history_file,
            shot_rows,
        )

        self.export_rows(
            self.feature_summary_file,
            feature_summaries,
        )

        self.export_rows(
            self.session_summary_file,
            [
                summary
            ],
        )

        self.create_dashboard(
            summary=summary,
            shot_rows=shot_rows,
            feature_summaries=feature_summaries,
        )

        self.export_text_report(
            summary=summary,
            feature_summaries=feature_summaries,
        )

        return (
            summary,
            shot_rows,
            feature_summaries,
        )

    # =====================================================
    # LOAD
    # =====================================================

    def load_feature_records(
        self,
    ) -> list[dict[str, str]]:
        if not self.feature_file.exists():
            raise FileNotFoundError(
                f"Feature file was not found: {self.feature_file}"
            )

        with self.feature_file.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            return list(
                csv.DictReader(
                    file
                )
            )

    # =====================================================
    # SHOT HISTORY
    # =====================================================

    def build_shot_rows(
        self,
        participant_id: str,
        records: list[dict[str, str]],
    ) -> list[SessionShotRow]:
        rows: list[
            SessionShotRow
        ] = []

        for index, record in enumerate(
            records,
            start=1,
        ):
            rows.append(
                SessionShotRow(
                    participant_id=participant_id,
                    trial_id=str(
                        record.get(
                            "trial_id",
                            "",
                        )
                    ),
                    trial_file=str(
                        record.get(
                            "trial_file",
                            "",
                        )
                    ),
                    result=self._normalize_result(
                        record.get(
                            "result",
                            "",
                        )
                    ),

                    shot_number=index,

                    release_angle_deg=self._to_float(
                        record.get(
                            "release_angle_deg"
                        )
                    ),
                    release_ball_speed_ft_s=self._to_float(
                        record.get(
                            "release_ball_speed_ft_s"
                        )
                    ),
                    release_height_ft=self._to_float(
                        record.get(
                            "release_height_ft"
                        )
                    ),

                    right_knee_range_of_motion_deg=self._to_float(
                        record.get(
                            "right_knee_range_of_motion_deg"
                        )
                    ),
                    right_hip_range_of_motion_deg=self._to_float(
                        record.get(
                            "right_hip_range_of_motion_deg"
                        )
                    ),
                    release_right_elbow_angle_deg=self._to_float(
                        record.get(
                            "release_right_elbow_angle_deg"
                        )
                    ),

                    knee_to_elbow_gap_ms=self._to_float(
                        record.get(
                            "knee_to_elbow_gap_ms"
                        )
                    ),
                    elbow_to_release_gap_ms=self._to_float(
                        record.get(
                            "elbow_to_release_gap_ms"
                        )
                    ),
                    takeoff_to_release_ms=self._to_float(
                        record.get(
                            "takeoff_to_release_ms"
                        )
                    ),
                )
            )

        return rows

    # =====================================================
    # FEATURE SUMMARIES
    # =====================================================

    def build_feature_summaries(
        self,
        participant_id: str,
        records: list[dict[str, str]],
    ) -> list[SessionFeatureSummary]:
        output: list[
            SessionFeatureSummary
        ] = []

        for feature in self.TRACKED_FEATURES:
            valid_rows = [
                (
                    self._to_float(
                        record.get(
                            feature
                        )
                    ),
                    self._normalize_result(
                        record.get(
                            "result",
                            "",
                        )
                    ),
                )
                for record in records
            ]

            valid_rows = [
                (
                    value,
                    result,
                )
                for value, result in valid_rows
                if value is not None
            ]

            if len(
                valid_rows
            ) < 5:
                continue

            values = [
                value
                for value, _ in valid_rows
            ]

            made_values = [
                value
                for value, result in valid_rows
                if result == "made"
            ]

            missed_values = [
                value
                for value, result in valid_rows
                if result == "missed"
            ]

            first_values = values[
                :min(
                    10,
                    len(
                        values
                    ),
                )
            ]

            last_values = values[
                -min(
                    10,
                    len(
                        values
                    ),
                ):
            ]

            first_mean = (
                float(
                    mean(
                        first_values
                    )
                )
                if first_values
                else None
            )

            last_mean = (
                float(
                    mean(
                        last_values
                    )
                )
                if last_values
                else None
            )

            first_to_last_change = (
                None
                if (
                    first_mean is None
                    or last_mean is None
                )
                else (
                    last_mean
                    - first_mean
                )
            )

            session_mean = float(
                mean(
                    values
                )
            )

            session_sd = (
                float(
                    pstdev(
                        values
                    )
                )
                if len(
                    values
                )
                > 1
                else 0.0
            )

            consistency_score = self._consistency_score(
                session_mean=session_mean,
                session_sd=session_sd,
            )

            made_mean = (
                float(
                    mean(
                        made_values
                    )
                )
                if made_values
                else None
            )

            missed_mean = (
                float(
                    mean(
                        missed_values
                    )
                )
                if missed_values
                else None
            )

            made_missed_difference = (
                None
                if (
                    made_mean is None
                    or missed_mean is None
                )
                else (
                    made_mean
                    - missed_mean
                )
            )

            output.append(
                SessionFeatureSummary(
                    participant_id=participant_id,
                    feature=feature,

                    valid_shots=len(
                        values
                    ),
                    made_shots=len(
                        made_values
                    ),
                    missed_shots=len(
                        missed_values
                    ),

                    session_mean=session_mean,
                    session_median=float(
                        median(
                            values
                        )
                    ),
                    session_standard_deviation=(
                        session_sd
                    ),

                    first_10_mean=first_mean,
                    last_10_mean=last_mean,
                    first_to_last_change=(
                        first_to_last_change
                    ),

                    made_mean=made_mean,
                    missed_mean=missed_mean,
                    made_missed_difference=(
                        made_missed_difference
                    ),

                    consistency_score=(
                        consistency_score
                    ),
                    trend_direction=(
                        self._trend_direction(
                            first_to_last_change,
                            session_sd,
                        )
                    ),
                )
            )

        output.sort(
            key=lambda row: (
                -row.consistency_score,
                row.feature,
            )
        )

        return output

    # =====================================================
    # SESSION SUMMARY
    # =====================================================

    def build_session_summary(
        self,
        participant_id: str,
        shot_rows: list[SessionShotRow],
        feature_summaries: list[SessionFeatureSummary],
    ) -> SessionHistorySummary:
        made_count = sum(
            row.result
            == "made"
            for row in shot_rows
        )

        missed_count = sum(
            row.result
            == "missed"
            for row in shot_rows
        )

        total_shots = len(
            shot_rows
        )

        make_percentage = (
            made_count
            / total_shots
            * 100.0
            if total_shots
            else 0.0
        )

        if feature_summaries:
            most_consistent = max(
                feature_summaries,
                key=lambda row: (
                    row.consistency_score
                ),
            )

            least_consistent = min(
                feature_summaries,
                key=lambda row: (
                    row.consistency_score
                ),
            )

            trend_rows = [
                row
                for row in feature_summaries
                if row.first_to_last_change
                is not None
            ]

            positive = max(
                trend_rows,
                key=lambda row: (
                    row.first_to_last_change
                ),
            )

            negative = min(
                trend_rows,
                key=lambda row: (
                    row.first_to_last_change
                ),
            )

            most_consistent_name = self.FRIENDLY_NAMES.get(
                most_consistent.feature,
                most_consistent.feature,
            )

            least_consistent_name = self.FRIENDLY_NAMES.get(
                least_consistent.feature,
                least_consistent.feature,
            )

            positive_name = self.FRIENDLY_NAMES.get(
                positive.feature,
                positive.feature,
            )

            negative_name = self.FRIENDLY_NAMES.get(
                negative.feature,
                negative.feature,
            )

        else:
            most_consistent_name = "N/A"
            least_consistent_name = "N/A"
            positive_name = "N/A"
            negative_name = "N/A"

        status = (
            "Sufficient history"
            if total_shots >= 50
            else "Limited history"
        )

        return SessionHistorySummary(
            participant_id=participant_id,

            total_shots=total_shots,
            made_shots=made_count,
            missed_shots=missed_count,
            make_percentage=float(
                make_percentage
            ),

            features_analyzed=len(
                feature_summaries
            ),

            most_consistent_feature=(
                most_consistent_name
            ),
            least_consistent_feature=(
                least_consistent_name
            ),

            largest_positive_trend_feature=(
                positive_name
            ),
            largest_negative_trend_feature=(
                negative_name
            ),

            session_status=status,
        )

    # =====================================================
    # DASHBOARD
    # =====================================================

    def create_dashboard(
        self,
        summary: SessionHistorySummary,
        shot_rows: list[SessionShotRow],
        feature_summaries: list[SessionFeatureSummary],
    ) -> Path:
        figure = plt.figure(
            figsize=(
                15,
                12,
            )
        )

        grid = figure.add_gridspec(
            3,
            1,
            height_ratios=[
                1.0,
                1.6,
                2.2,
            ],
        )

        make_axis = figure.add_subplot(
            grid[0]
        )

        trend_axis = figure.add_subplot(
            grid[1]
        )

        consistency_axis = figure.add_subplot(
            grid[2]
        )

        results = np.asarray(
            [
                1
                if row.result
                == "made"
                else 0
                for row in shot_rows
            ],
            dtype=float,
        )

        shot_numbers = np.arange(
            1,
            len(
                shot_rows
            )
            + 1,
        )

        rolling_window = min(
            10,
            max(
                3,
                len(
                    shot_rows
                )
                // 10,
            ),
        )

        rolling_make = np.full(
            len(
                results
            ),
            np.nan,
        )

        for index in range(
            len(
                results
            )
        ):
            start = max(
                0,
                index
                - rolling_window
                + 1,
            )

            rolling_make[
                index
            ] = float(
                mean(
                    results[
                        start:
                        index
                        + 1
                    ]
                )
                * 100.0
            )

        make_axis.plot(
            shot_numbers,
            rolling_make,
            linewidth=2.0,
        )

        make_axis.set_ylim(
            0,
            100,
        )

        make_axis.set_ylabel(
            "Rolling make %"
        )

        make_axis.set_xlabel(
            "Shot number"
        )

        make_axis.set_title(
            (
                f"{summary.participant_id} Practice History — "
                f"{summary.make_percentage:.1f}% overall"
            )
        )

        make_axis.grid(
            visible=True,
            alpha=0.25,
        )

        if shot_rows:
            release_angles = np.asarray(
                [
                    (
                        row.release_angle_deg
                        if row.release_angle_deg
                        is not None
                        else np.nan
                    )
                    for row in shot_rows
                ],
                dtype=float,
            )

            elbow_angles = np.asarray(
                [
                    (
                        row.release_right_elbow_angle_deg
                        if row.release_right_elbow_angle_deg
                        is not None
                        else np.nan
                    )
                    for row in shot_rows
                ],
                dtype=float,
            )

            trend_axis.plot(
                shot_numbers,
                release_angles,
                label="Release angle",
                linewidth=1.8,
            )

            trend_axis.plot(
                shot_numbers,
                elbow_angles,
                label="Elbow angle at release",
                linewidth=1.8,
            )

            trend_axis.set_ylabel(
                "Degrees"
            )

            trend_axis.set_xlabel(
                "Shot number"
            )

            trend_axis.set_title(
                "Selected Biomechanical Trends"
            )

            trend_axis.legend()

            trend_axis.grid(
                visible=True,
                alpha=0.25,
            )

        labels = [
            self.FRIENDLY_NAMES.get(
                row.feature,
                row.feature,
            )
            for row in reversed(
                feature_summaries
            )
        ]

        values = [
            row.consistency_score
            for row in reversed(
                feature_summaries
            )
        ]

        positions = np.arange(
            len(
                labels
            )
        )

        consistency_axis.barh(
            positions,
            values,
        )

        consistency_axis.set_yticks(
            positions,
            labels=labels,
            fontsize=8,
        )

        consistency_axis.set_xlim(
            0,
            100,
        )

        consistency_axis.set_xlabel(
            "Consistency score"
        )

        consistency_axis.set_title(
            "Feature Consistency Across All Shots"
        )

        consistency_axis.grid(
            visible=True,
            axis="x",
            alpha=0.25,
        )

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
    # REPORT
    # =====================================================

    def export_text_report(
        self,
        summary: SessionHistorySummary,
        feature_summaries: list[SessionFeatureSummary],
    ) -> Path:
        lines = [
            "=" * 88,
            "PRACTICE HISTORY ANALYSIS",
            "=" * 88,
            "",
            f"Participant: {summary.participant_id}",
            f"Total shots: {summary.total_shots}",
            f"Made shots: {summary.made_shots}",
            f"Missed shots: {summary.missed_shots}",
            f"Make percentage: {summary.make_percentage:.1f}%",
            f"History status: {summary.session_status}",
            "",
            "SESSION HIGHLIGHTS",
            "-" * 88,
            (
                "Most consistent feature: "
                f"{summary.most_consistent_feature}"
            ),
            (
                "Least consistent feature: "
                f"{summary.least_consistent_feature}"
            ),
            (
                "Largest positive numerical trend: "
                f"{summary.largest_positive_trend_feature}"
            ),
            (
                "Largest negative numerical trend: "
                f"{summary.largest_negative_trend_feature}"
            ),
            "",
            "FEATURE DETAILS",
            "-" * 88,
        ]

        for row in feature_summaries:
            feature_name = self.FRIENDLY_NAMES.get(
                row.feature,
                row.feature,
            )

            change_text = (
                "N/A"
                if row.first_to_last_change
                is None
                else f"{row.first_to_last_change:+.2f}"
            )

            lines.append(
                (
                    f"{feature_name}: consistency "
                    f"{row.consistency_score:.1f}/100, "
                    f"first-to-last change {change_text}, "
                    f"trend {row.trend_direction}"
                )
            )

        lines.extend(
            [
                "",
                "INTERPRETATION LIMITATION",
                "-" * 88,
                (
                    "A numerical increase or decrease is not automatically "
                    "an improvement or decline. The direction must be interpreted "
                    "using the player's personal successful baseline and video."
                ),
            ]
        )

        self.report_file.write_text(
            "\n".join(
                lines
            ),
            encoding="utf-8",
        )

        return self.report_file

    # =====================================================
    # EXPORT
    # =====================================================

    @staticmethod
    def export_rows(
        output_file: Path,
        rows: list[object],
    ) -> Path:
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

        with output_file.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    dictionaries[0]
                ),
            )

            writer.writeheader()

            writer.writerows(
                dictionaries
            )

        return output_file

    # =====================================================
    # HELPERS
    # =====================================================

    @staticmethod
    def _to_float(
        value: object,
    ) -> float | None:
        if value is None:
            return None

        normalized = str(
            value
        ).strip().lower()

        if normalized in {
            "",
            "none",
            "nan",
            "null",
        }:
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

        if not np.isfinite(
            numeric
        ):
            return None

        return numeric

    @staticmethod
    def _normalize_result(
        value: object,
    ) -> str:
        normalized = str(
            value
        ).strip().lower()

        if normalized in {
            "made",
            "make",
            "hit",
            "successful",
        }:
            return "made"

        if normalized in {
            "missed",
            "miss",
            "failed",
        }:
            return "missed"

        return "unknown"

    @staticmethod
    def _trial_sort_key(
        record: dict[str, str],
    ) -> tuple[
        int,
        str,
    ]:
        trial_id = str(
            record.get(
                "trial_id",
                "",
            )
        )

        digits = "".join(
            character
            for character in trial_id
            if character.isdigit()
        )

        numeric_value = (
            int(
                digits
            )
            if digits
            else 0
        )

        return (
            numeric_value,
            trial_id,
        )

    @staticmethod
    def _consistency_score(
        session_mean: float,
        session_sd: float,
    ) -> float:
        if abs(
            session_mean
        ) <= 1e-9:
            return 50.0

        coefficient_of_variation = abs(
            session_sd
            / session_mean
        )

        score = (
            1.0
            - min(
                1.0,
                coefficient_of_variation,
            )
        ) * 100.0

        return float(
            np.clip(
                score,
                0.0,
                100.0,
            )
        )

    @staticmethod
    def _trend_direction(
        change: float | None,
        session_sd: float,
    ) -> str:
        if change is None:
            return "unavailable"

        meaningful_threshold = max(
            1e-9,
            session_sd
            * 0.25,
        )

        if change > meaningful_threshold:
            return "increasing"

        if change < -meaningful_threshold:
            return "decreasing"

        return "stable"


def _run_session_history_test() -> None:
    analyzer = SessionHistoryAnalyzer()

    (
        summary,
        shot_rows,
        feature_summaries,
    ) = analyzer.run(
        participant_id="P0001",
    )

    assert summary.total_shots > 0
    assert len(
        shot_rows
    ) > 0
    assert len(
        feature_summaries
    ) > 0

    assert analyzer.shot_history_file.exists()
    assert analyzer.feature_summary_file.exists()
    assert analyzer.session_summary_file.exists()
    assert analyzer.dashboard_file.exists()
    assert analyzer.report_file.exists()

    print(
        "SESSION HISTORY ANALYZER TEST PASSED"
    )

    print(
        f"Participant: {summary.participant_id}"
    )

    print(
        f"Total shots: {summary.total_shots}"
    )

    print(
        f"Made shots: {summary.made_shots}"
    )

    print(
        f"Missed shots: {summary.missed_shots}"
    )

    print(
        f"Make percentage: {summary.make_percentage:.1f}%"
    )

    print(
        f"Features analyzed: {summary.features_analyzed}"
    )

    print(
        f"Most consistent: {summary.most_consistent_feature}"
    )

    print(
        f"Least consistent: {summary.least_consistent_feature}"
    )

    print()

    print(
        "Shot history: "
        f"{analyzer.shot_history_file.resolve()}"
    )

    print(
        "Feature summary: "
        f"{analyzer.feature_summary_file.resolve()}"
    )

    print(
        "Session summary: "
        f"{analyzer.session_summary_file.resolve()}"
    )

    print(
        "Dashboard: "
        f"{analyzer.dashboard_file.resolve()}"
    )

    print(
        "Text report: "
        f"{analyzer.report_file.resolve()}"
    )


if __name__ == "__main__":
    _run_session_history_test()