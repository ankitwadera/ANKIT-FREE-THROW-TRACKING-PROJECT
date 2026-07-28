from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, pstdev

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "all_shot_features.csv"
)

DEFAULT_OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "personal_signal_diagnostics"
)


NON_FEATURE_FIELDS = {
    "participant_id",
    "trial_id",
    "result",
    "trial_file",
    "propulsion_onset_sequence",
    "peak_velocity_sequence",
    "analysis_status",
    "analysis_error",
}


EXCLUDED_ABSOLUTE_FRAME_FEATURES = {
    "motion_start_frame",
    "dip_frame",
    "takeoff_frame",
    "release_frame",
    "ball_apex_frame",
    "landing_frame",
    "knee_propulsion_onset_frame",
    "hip_propulsion_onset_frame",
    "pelvis_propulsion_onset_frame",
    "shoulder_propulsion_onset_frame",
    "elbow_propulsion_onset_frame",
    "wrist_propulsion_onset_frame",
    "ball_propulsion_onset_frame",
    "knee_peak_velocity_frame",
    "hip_peak_velocity_frame",
    "pelvis_peak_velocity_frame",
    "shoulder_peak_velocity_frame",
    "elbow_peak_velocity_frame",
    "wrist_peak_velocity_frame",
    "ball_peak_velocity_frame",
}


@dataclass(frozen=True)
class PersonalFeatureDiagnosticRow:
    """
    One participant-specific make-versus-miss feature comparison.
    """

    participant_id: str
    feature: str

    made_count: int
    missed_count: int
    completeness_percentage: float

    made_mean: float | None
    missed_mean: float | None
    mean_difference_made_minus_missed: float | None

    made_median: float | None
    missed_median: float | None

    made_standard_deviation: float | None
    missed_standard_deviation: float | None

    absolute_standardized_difference: float | None
    signed_standardized_difference: float | None

    distribution_overlap_percentage: float | None

    made_consistency_score: float | None
    missed_consistency_score: float | None

    bootstrap_positive_effect_percentage: float | None
    bootstrap_negative_effect_percentage: float | None
    bootstrap_direction_stability_percentage: float | None
    bootstrap_effect_mean: float | None
    bootstrap_effect_standard_deviation: float | None

    diagnostic_score: float
    diagnostic_rank: int

    signal_strength: str
    interpretation: str


@dataclass(frozen=True)
class ParticipantSignalSummary:
    """
    Overall diagnostic summary for one participant.
    """

    participant_id: str

    total_shots: int
    made_shots: int
    missed_shots: int
    numeric_features_analyzed: int

    strong_signal_features: int
    moderate_signal_features: int
    weak_signal_features: int
    no_signal_features: int

    top_feature: str
    top_feature_score: float

    mean_top_10_score: float
    mean_top_10_direction_stability_percentage: float

    personal_signal_status: str


class PersonalSignalDiagnosticStudio:
    """
    Explain why some participants have learnable personal patterns.

    For every participant and numeric feature, this module measures:

    - made-versus-missed difference;
    - standardized effect size;
    - distribution overlap;
    - consistency within makes and misses;
    - bootstrap stability of the direction of the difference.

    It does not train a prediction model. It diagnoses whether the raw
    feature distributions contain a stable personal signal.
    """

    def __init__(
        self,
        input_file: Path = DEFAULT_INPUT_FILE,
        output_folder: Path = DEFAULT_OUTPUT_FOLDER,
        random_seed: int = 42,
        bootstrap_iterations: int = 500,
        minimum_group_count: int = 10,
        exclude_absolute_frames: bool = True,
    ) -> None:
        self.input_file = Path(input_file)
        self.output_folder = Path(output_folder)

        self.random_seed = int(random_seed)
        self.bootstrap_iterations = max(
            100,
            int(bootstrap_iterations),
        )
        self.minimum_group_count = max(
            5,
            int(minimum_group_count),
        )
        self.exclude_absolute_frames = bool(
            exclude_absolute_frames
        )

        self.diagnostics_file = (
            self.output_folder
            / "personal_feature_diagnostics.csv"
        )

        self.participant_summary_file = (
            self.output_folder
            / "participant_signal_summary.csv"
        )

        self.top_features_chart_file = (
            self.output_folder
            / "personal_signal_top_features.png"
        )

        self.participant_signal_chart_file = (
            self.output_folder
            / "participant_signal_strength.png"
        )

        self.report_file = (
            self.output_folder
            / "personal_signal_diagnostic_report.txt"
        )

    # =====================================================
    # PUBLIC WORKFLOW
    # =====================================================

    def run(
        self,
    ) -> tuple[
        list[PersonalFeatureDiagnosticRow],
        list[ParticipantSignalSummary],
    ]:
        records = self.load_records()

        successful = [
            record
            for record in records
            if self._normalize_text(
                record.get(
                    "analysis_status",
                    "",
                )
            )
            == "success"
            and self._normalize_result(
                record.get(
                    "result",
                    "",
                )
            )
            in {
                "made",
                "missed",
            }
        ]

        if not successful:
            raise RuntimeError(
                "No successful made/missed records were available."
            )

        participants = sorted(
            {
                str(
                    record.get(
                        "participant_id",
                        "",
                    )
                ).strip()
                for record in successful
                if str(
                    record.get(
                        "participant_id",
                        "",
                    )
                ).strip()
            }
        )

        numeric_features = self.identify_numeric_features(
            successful
        )

        diagnostic_rows: list[
            PersonalFeatureDiagnosticRow
        ] = []

        participant_summaries: list[
            ParticipantSignalSummary
        ] = []

        for participant_number, participant_id in enumerate(
            participants,
            start=1,
        ):
            participant_records = [
                record
                for record in successful
                if str(
                    record.get(
                        "participant_id",
                        "",
                    )
                ).strip()
                == participant_id
            ]

            rows = self.analyze_participant(
                participant_id=participant_id,
                records=participant_records,
                numeric_features=numeric_features,
                participant_seed=(
                    self.random_seed
                    + participant_number
                    * 1000
                ),
            )

            diagnostic_rows.extend(
                rows
            )

            participant_summaries.append(
                self.build_participant_summary(
                    participant_id=participant_id,
                    records=participant_records,
                    rows=rows,
                )
            )

            top_feature = (
                rows[0].feature
                if rows
                else "N/A"
            )

            top_score = (
                rows[0].diagnostic_score
                if rows
                else 0.0
            )

            print(
                f"{participant_id}: "
                f"top feature {top_feature}, "
                f"score {top_score:.1f}"
            )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.export_dataclass_rows(
            self.diagnostics_file,
            diagnostic_rows,
        )

        self.export_dataclass_rows(
            self.participant_summary_file,
            participant_summaries,
        )

        self.create_top_features_chart(
            diagnostic_rows
        )

        self.create_participant_signal_chart(
            participant_summaries
        )

        self.export_text_report(
            diagnostic_rows=diagnostic_rows,
            participant_summaries=participant_summaries,
        )

        return (
            diagnostic_rows,
            participant_summaries,
        )

    # =====================================================
    # LOAD DATA
    # =====================================================

    def load_records(
        self,
    ) -> list[dict[str, str]]:
        if not self.input_file.exists():
            raise FileNotFoundError(
                "Master feature CSV was not found: "
                f"{self.input_file}"
            )

        with self.input_file.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            return list(
                csv.DictReader(file)
            )

    # =====================================================
    # FEATURE DISCOVERY
    # =====================================================

    def identify_numeric_features(
        self,
        records: list[dict[str, str]],
    ) -> list[str]:
        fieldnames = list(
            records[0]
        )

        features: list[
            str
        ] = []

        for field_name in fieldnames:
            if field_name in NON_FEATURE_FIELDS:
                continue

            if (
                self.exclude_absolute_frames
                and field_name
                in EXCLUDED_ABSOLUTE_FRAME_FEATURES
            ):
                continue

            values = [
                self._to_finite_float(
                    record.get(
                        field_name,
                        "",
                    )
                )
                for record in records
            ]

            finite_values = [
                value
                for value in values
                if value is not None
            ]

            if len(
                finite_values
            ) < 20:
                continue

            if len(
                set(finite_values)
            ) < 2:
                continue

            features.append(
                field_name
            )

        return features

    # =====================================================
    # PARTICIPANT ANALYSIS
    # =====================================================

    def analyze_participant(
        self,
        participant_id: str,
        records: list[dict[str, str]],
        numeric_features: list[str],
        participant_seed: int,
    ) -> list[PersonalFeatureDiagnosticRow]:
        made_records = [
            record
            for record in records
            if self._normalize_result(
                record.get(
                    "result",
                    "",
                )
            )
            == "made"
        ]

        missed_records = [
            record
            for record in records
            if self._normalize_result(
                record.get(
                    "result",
                    "",
                )
            )
            == "missed"
        ]

        raw_rows: list[
            dict[str, object]
        ] = []

        for feature_index, feature in enumerate(
            numeric_features,
            start=1,
        ):
            made_values = self._numeric_values(
                made_records,
                feature,
            )

            missed_values = self._numeric_values(
                missed_records,
                feature,
            )

            if (
                len(made_values)
                < self.minimum_group_count
                or len(missed_values)
                < self.minimum_group_count
            ):
                continue

            made_mean = float(
                mean(made_values)
            )

            missed_mean = float(
                mean(missed_values)
            )

            made_sd = self._safe_pstdev(
                made_values
            )

            missed_sd = self._safe_pstdev(
                missed_values
            )

            effect = self._standardized_difference(
                made_values,
                missed_values,
            )

            overlap = self._distribution_overlap_percentage(
                made_values,
                missed_values,
            )

            made_consistency = self._consistency_score(
                made_values
            )

            missed_consistency = self._consistency_score(
                missed_values
            )

            (
                bootstrap_positive,
                bootstrap_negative,
                bootstrap_direction_stability,
                bootstrap_effect_mean,
                bootstrap_effect_sd,
            ) = self._bootstrap_effect_stability(
                made_values=made_values,
                missed_values=missed_values,
                random_seed=(
                    participant_seed
                    + feature_index
                ),
            )

            total_available = (
                len(made_values)
                + len(missed_values)
            )

            completeness = (
                total_available
                / len(records)
                * 100.0
                if records
                else 0.0
            )

            diagnostic_score = self._diagnostic_score(
                absolute_effect=(
                    None
                    if effect is None
                    else abs(effect)
                ),
                overlap_percentage=overlap,
                direction_stability_percentage=(
                    bootstrap_direction_stability
                ),
                completeness_percentage=(
                    completeness
                ),
            )

            raw_rows.append(
                {
                    "participant_id": participant_id,
                    "feature": feature,

                    "made_count": len(
                        made_values
                    ),
                    "missed_count": len(
                        missed_values
                    ),
                    "completeness_percentage": (
                        completeness
                    ),

                    "made_mean": made_mean,
                    "missed_mean": missed_mean,
                    "mean_difference_made_minus_missed": (
                        made_mean
                        - missed_mean
                    ),

                    "made_median": float(
                        median(
                            made_values
                        )
                    ),
                    "missed_median": float(
                        median(
                            missed_values
                        )
                    ),

                    "made_standard_deviation": (
                        made_sd
                    ),
                    "missed_standard_deviation": (
                        missed_sd
                    ),

                    "absolute_standardized_difference": (
                        None
                        if effect is None
                        else abs(effect)
                    ),
                    "signed_standardized_difference": (
                        effect
                    ),

                    "distribution_overlap_percentage": (
                        overlap
                    ),

                    "made_consistency_score": (
                        made_consistency
                    ),
                    "missed_consistency_score": (
                        missed_consistency
                    ),

                    "bootstrap_positive_effect_percentage": (
                        bootstrap_positive
                    ),
                    "bootstrap_negative_effect_percentage": (
                        bootstrap_negative
                    ),
                    "bootstrap_direction_stability_percentage": (
                        bootstrap_direction_stability
                    ),
                    "bootstrap_effect_mean": (
                        bootstrap_effect_mean
                    ),
                    "bootstrap_effect_standard_deviation": (
                        bootstrap_effect_sd
                    ),

                    "diagnostic_score": (
                        diagnostic_score
                    ),
                }
            )

        raw_rows.sort(
            key=lambda row: (
                -float(
                    row[
                        "diagnostic_score"
                    ]
                ),
                str(
                    row[
                        "feature"
                    ]
                ),
            )
        )

        output: list[
            PersonalFeatureDiagnosticRow
        ] = []

        for rank, row in enumerate(
            raw_rows,
            start=1,
        ):
            score = float(
                row[
                    "diagnostic_score"
                ]
            )

            signal_strength = (
                self._signal_strength(
                    score
                )
            )

            interpretation = self._interpretation(
                feature=str(
                    row[
                        "feature"
                    ]
                ),
                signed_effect=row[
                    "signed_standardized_difference"
                ],
                overlap_percentage=row[
                    "distribution_overlap_percentage"
                ],
                stability_percentage=row[
                    "bootstrap_direction_stability_percentage"
                ],
            )

            output.append(
                PersonalFeatureDiagnosticRow(
                    participant_id=str(
                        row[
                            "participant_id"
                        ]
                    ),
                    feature=str(
                        row[
                            "feature"
                        ]
                    ),

                    made_count=int(
                        row[
                            "made_count"
                        ]
                    ),
                    missed_count=int(
                        row[
                            "missed_count"
                        ]
                    ),
                    completeness_percentage=float(
                        row[
                            "completeness_percentage"
                        ]
                    ),

                    made_mean=self._optional_float(
                        row[
                            "made_mean"
                        ]
                    ),
                    missed_mean=self._optional_float(
                        row[
                            "missed_mean"
                        ]
                    ),
                    mean_difference_made_minus_missed=self._optional_float(
                        row[
                            "mean_difference_made_minus_missed"
                        ]
                    ),

                    made_median=self._optional_float(
                        row[
                            "made_median"
                        ]
                    ),
                    missed_median=self._optional_float(
                        row[
                            "missed_median"
                        ]
                    ),

                    made_standard_deviation=self._optional_float(
                        row[
                            "made_standard_deviation"
                        ]
                    ),
                    missed_standard_deviation=self._optional_float(
                        row[
                            "missed_standard_deviation"
                        ]
                    ),

                    absolute_standardized_difference=self._optional_float(
                        row[
                            "absolute_standardized_difference"
                        ]
                    ),
                    signed_standardized_difference=self._optional_float(
                        row[
                            "signed_standardized_difference"
                        ]
                    ),

                    distribution_overlap_percentage=self._optional_float(
                        row[
                            "distribution_overlap_percentage"
                        ]
                    ),

                    made_consistency_score=self._optional_float(
                        row[
                            "made_consistency_score"
                        ]
                    ),
                    missed_consistency_score=self._optional_float(
                        row[
                            "missed_consistency_score"
                        ]
                    ),

                    bootstrap_positive_effect_percentage=self._optional_float(
                        row[
                            "bootstrap_positive_effect_percentage"
                        ]
                    ),
                    bootstrap_negative_effect_percentage=self._optional_float(
                        row[
                            "bootstrap_negative_effect_percentage"
                        ]
                    ),
                    bootstrap_direction_stability_percentage=self._optional_float(
                        row[
                            "bootstrap_direction_stability_percentage"
                        ]
                    ),
                    bootstrap_effect_mean=self._optional_float(
                        row[
                            "bootstrap_effect_mean"
                        ]
                    ),
                    bootstrap_effect_standard_deviation=self._optional_float(
                        row[
                            "bootstrap_effect_standard_deviation"
                        ]
                    ),

                    diagnostic_score=score,
                    diagnostic_rank=rank,

                    signal_strength=(
                        signal_strength
                    ),
                    interpretation=(
                        interpretation
                    ),
                )
            )

        return output

    # =====================================================
    # PARTICIPANT SUMMARY
    # =====================================================

    def build_participant_summary(
        self,
        participant_id: str,
        records: list[dict[str, str]],
        rows: list[PersonalFeatureDiagnosticRow],
    ) -> ParticipantSignalSummary:
        made_shots = sum(
            self._normalize_result(
                record.get(
                    "result",
                    "",
                )
            )
            == "made"
            for record in records
        )

        missed_shots = (
            len(records)
            - made_shots
        )

        strong_count = sum(
            row.signal_strength
            == "strong"
            for row in rows
        )

        moderate_count = sum(
            row.signal_strength
            == "moderate"
            for row in rows
        )

        weak_count = sum(
            row.signal_strength
            == "weak"
            for row in rows
        )

        no_signal_count = sum(
            row.signal_strength
            == "minimal"
            for row in rows
        )

        top_rows = rows[
            :10
        ]

        top_feature = (
            rows[0].feature
            if rows
            else "N/A"
        )

        top_feature_score = (
            rows[0].diagnostic_score
            if rows
            else 0.0
        )

        mean_top_10_score = (
            float(
                mean(
                    row.diagnostic_score
                    for row in top_rows
                )
            )
            if top_rows
            else 0.0
        )

        stability_values = [
            row.bootstrap_direction_stability_percentage
            for row in top_rows
            if row.bootstrap_direction_stability_percentage
            is not None
        ]

        mean_top_10_stability = (
            float(
                mean(
                    stability_values
                )
            )
            if stability_values
            else 0.0
        )

        status = self._participant_status(
            top_feature_score=top_feature_score,
            mean_top_10_score=mean_top_10_score,
            mean_top_10_stability=(
                mean_top_10_stability
            ),
        )

        return ParticipantSignalSummary(
            participant_id=participant_id,

            total_shots=len(
                records
            ),
            made_shots=made_shots,
            missed_shots=missed_shots,
            numeric_features_analyzed=len(
                rows
            ),

            strong_signal_features=(
                strong_count
            ),
            moderate_signal_features=(
                moderate_count
            ),
            weak_signal_features=(
                weak_count
            ),
            no_signal_features=(
                no_signal_count
            ),

            top_feature=top_feature,
            top_feature_score=(
                top_feature_score
            ),

            mean_top_10_score=(
                mean_top_10_score
            ),
            mean_top_10_direction_stability_percentage=(
                mean_top_10_stability
            ),

            personal_signal_status=status,
        )

    # =====================================================
    # STATISTICS
    # =====================================================

    @staticmethod
    def _standardized_difference(
        made_values: list[float],
        missed_values: list[float],
    ) -> float | None:
        if (
            len(made_values) < 2
            or len(missed_values) < 2
        ):
            return None

        made_variance = float(
            np.var(
                made_values,
                ddof=1,
            )
        )

        missed_variance = float(
            np.var(
                missed_values,
                ddof=1,
            )
        )

        degrees_of_freedom = (
            len(made_values)
            + len(missed_values)
            - 2
        )

        if degrees_of_freedom <= 0:
            return None

        pooled_variance = (
            (
                len(made_values)
                - 1
            )
            * made_variance
            + (
                len(missed_values)
                - 1
            )
            * missed_variance
        ) / degrees_of_freedom

        if pooled_variance <= 0:
            return None

        return float(
            (
                mean(made_values)
                - mean(missed_values)
            )
            / np.sqrt(
                pooled_variance
            )
        )

    @staticmethod
    def _distribution_overlap_percentage(
        made_values: list[float],
        missed_values: list[float],
        bins: int = 30,
    ) -> float | None:
        combined = np.asarray(
            [
                *made_values,
                *missed_values,
            ],
            dtype=float,
        )

        if combined.size < 4:
            return None

        minimum = float(
            np.min(
                combined
            )
        )

        maximum = float(
            np.max(
                combined
            )
        )

        if maximum == minimum:
            return 100.0

        made_histogram, edges = np.histogram(
            made_values,
            bins=bins,
            range=(
                minimum,
                maximum,
            ),
            density=True,
        )

        missed_histogram, _ = np.histogram(
            missed_values,
            bins=edges,
            density=True,
        )

        bin_widths = np.diff(
            edges
        )

        overlap = float(
            np.sum(
                np.minimum(
                    made_histogram,
                    missed_histogram,
                )
                * bin_widths
            )
        )

        return float(
            np.clip(
                overlap
                * 100.0,
                0.0,
                100.0,
            )
        )

    @staticmethod
    def _consistency_score(
        values: list[float],
    ) -> float | None:
        if not values:
            return None

        average = float(
            mean(
                values
            )
        )

        deviation = (
            float(
                pstdev(
                    values
                )
            )
            if len(values) > 1
            else 0.0
        )

        scale = (
            abs(
                average
            )
            + 1e-9
        )

        coefficient = (
            deviation
            / scale
        )

        return float(
            np.clip(
                100.0
                / (
                    1.0
                    + coefficient
                ),
                0.0,
                100.0,
            )
        )

    def _bootstrap_effect_stability(
        self,
        made_values: list[float],
        missed_values: list[float],
        random_seed: int,
    ) -> tuple[
        float,
        float,
        float,
        float | None,
        float | None,
    ]:
        generator = np.random.default_rng(
            random_seed
        )

        made_array = np.asarray(
            made_values,
            dtype=float,
        )

        missed_array = np.asarray(
            missed_values,
            dtype=float,
        )

        effects: list[
            float
        ] = []

        for _ in range(
            self.bootstrap_iterations
        ):
            made_sample = generator.choice(
                made_array,
                size=len(
                    made_array
                ),
                replace=True,
            )

            missed_sample = generator.choice(
                missed_array,
                size=len(
                    missed_array
                ),
                replace=True,
            )

            effect = self._standardized_difference(
                made_sample.tolist(),
                missed_sample.tolist(),
            )

            if effect is not None:
                effects.append(
                    effect
                )

        if not effects:
            return (
                0.0,
                0.0,
                0.0,
                None,
                None,
            )

        positive_percentage = (
            sum(
                effect > 0
                for effect in effects
            )
            / len(effects)
            * 100.0
        )

        negative_percentage = (
            sum(
                effect < 0
                for effect in effects
            )
            / len(effects)
            * 100.0
        )

        direction_stability = max(
            positive_percentage,
            negative_percentage,
        )

        return (
            float(
                positive_percentage
            ),
            float(
                negative_percentage
            ),
            float(
                direction_stability
            ),
            float(
                mean(
                    effects
                )
            ),
            float(
                pstdev(
                    effects
                )
            )
            if len(effects) > 1
            else 0.0,
        )

    @staticmethod
    def _diagnostic_score(
        absolute_effect: float | None,
        overlap_percentage: float | None,
        direction_stability_percentage: float | None,
        completeness_percentage: float,
    ) -> float:
        if absolute_effect is None:
            effect_component = 0.0
        else:
            effect_component = min(
                1.0,
                absolute_effect
                / 1.0,
            )

        overlap_component = (
            0.0
            if overlap_percentage is None
            else 1.0
            - overlap_percentage
            / 100.0
        )

        stability_component = (
            0.0
            if direction_stability_percentage
            is None
            else direction_stability_percentage
            / 100.0
        )

        completeness_component = (
            completeness_percentage
            / 100.0
        )

        score = (
            effect_component
            * 0.40
            + overlap_component
            * 0.25
            + stability_component
            * 0.25
            + completeness_component
            * 0.10
        ) * 100.0

        return float(
            np.clip(
                score,
                0.0,
                100.0,
            )
        )

    # =====================================================
    # INTERPRETATION
    # =====================================================

    @staticmethod
    def _signal_strength(
        score: float,
    ) -> str:
        if score >= 75:
            return "strong"

        if score >= 60:
            return "moderate"

        if score >= 45:
            return "weak"

        return "minimal"

    @staticmethod
    def _participant_status(
        top_feature_score: float,
        mean_top_10_score: float,
        mean_top_10_stability: float,
    ) -> str:
        if (
            top_feature_score >= 75
            and mean_top_10_score >= 60
            and mean_top_10_stability >= 80
        ):
            return "Clear personal signal"

        if (
            top_feature_score >= 65
            and mean_top_10_score >= 52
            and mean_top_10_stability >= 70
        ):
            return "Moderate personal signal"

        if top_feature_score >= 55:
            return "Limited personal signal"

        return "No stable personal signal"

    @staticmethod
    def _interpretation(
        feature: str,
        signed_effect: object,
        overlap_percentage: object,
        stability_percentage: object,
    ) -> str:
        effect = (
            None
            if signed_effect is None
            else float(
                signed_effect
            )
        )

        overlap = (
            None
            if overlap_percentage is None
            else float(
                overlap_percentage
            )
        )

        stability = (
            None
            if stability_percentage is None
            else float(
                stability_percentage
            )
        )

        if effect is None:
            direction_text = (
                "No reliable made-versus-missed direction was available."
            )

        elif effect > 0:
            direction_text = (
                f"Made shots tended to have higher {feature} values."
            )

        elif effect < 0:
            direction_text = (
                f"Made shots tended to have lower {feature} values."
            )

        else:
            direction_text = (
                f"Made and missed {feature} averages were nearly identical."
            )

        overlap_text = (
            ""
            if overlap is None
            else (
                f" Distribution overlap was {overlap:.1f}%."
            )
        )

        stability_text = (
            ""
            if stability is None
            else (
                f" Bootstrap direction stability was {stability:.1f}%."
            )
        )

        return (
            direction_text
            + overlap_text
            + stability_text
        )

    # =====================================================
    # CHARTS
    # =====================================================

    def create_top_features_chart(
        self,
        rows: list[
            PersonalFeatureDiagnosticRow
        ],
    ) -> Path:
        participants = sorted(
            {
                row.participant_id
                for row in rows
            }
        )

        selected: list[
            PersonalFeatureDiagnosticRow
        ] = []

        for participant_id in participants:
            participant_rows = [
                row
                for row in rows
                if row.participant_id
                == participant_id
            ][
                :5
            ]

            selected.extend(
                participant_rows
            )

        labels = [
            (
                f"{row.participant_id}: "
                f"{row.feature}"
            )
            for row in reversed(
                selected
            )
        ]

        values = [
            row.diagnostic_score
            for row in reversed(
                selected
            )
        ]

        figure, axis = plt.subplots(
            figsize=(
                14,
                max(
                    10,
                    len(labels)
                    * 0.34,
                ),
            )
        )

        positions = np.arange(
            len(labels)
        )

        axis.barh(
            positions,
            values,
        )

        axis.set_yticks(
            positions,
            labels=labels,
            fontsize=8,
        )

        axis.set_xlim(
            0,
            100,
        )

        axis.set_xlabel(
            "Personal diagnostic score"
        )

        axis.set_title(
            "Top Personal Make-vs-Miss Signals"
        )

        axis.grid(
            visible=True,
            axis="x",
            alpha=0.25,
        )

        figure.tight_layout()

        figure.savefig(
            self.top_features_chart_file,
            dpi=170,
            bbox_inches="tight",
        )

        plt.close(
            figure
        )

        return self.top_features_chart_file

    def create_participant_signal_chart(
        self,
        summaries: list[
            ParticipantSignalSummary
        ],
    ) -> Path:
        labels = [
            summary.participant_id
            for summary in summaries
        ]

        top_scores = [
            summary.top_feature_score
            for summary in summaries
        ]

        mean_top_scores = [
            summary.mean_top_10_score
            for summary in summaries
        ]

        positions = np.arange(
            len(labels)
        )

        width = 0.38

        figure, axis = plt.subplots(
            figsize=(
                11,
                7,
            )
        )

        axis.bar(
            positions
            - width / 2,
            top_scores,
            width,
            label="Top feature",
        )

        axis.bar(
            positions
            + width / 2,
            mean_top_scores,
            width,
            label="Top-10 average",
        )

        axis.set_xticks(
            positions,
            labels=labels,
        )

        axis.set_ylim(
            0,
            100,
        )

        axis.set_ylabel(
            "Diagnostic score"
        )

        axis.set_title(
            "Participant Personal-Signal Strength"
        )

        axis.legend()

        axis.grid(
            visible=True,
            axis="y",
            alpha=0.25,
        )

        figure.tight_layout()

        figure.savefig(
            self.participant_signal_chart_file,
            dpi=170,
            bbox_inches="tight",
        )

        plt.close(
            figure
        )

        return self.participant_signal_chart_file

    # =====================================================
    # REPORT
    # =====================================================

    def export_text_report(
        self,
        diagnostic_rows: list[
            PersonalFeatureDiagnosticRow
        ],
        participant_summaries: list[
            ParticipantSignalSummary
        ],
    ) -> Path:
        lines = [
            "=" * 88,
            "BASKETBALL BIOMECHANICS RESEARCH STUDIO",
            "PERSONAL SIGNAL DIAGNOSTIC REPORT",
            "=" * 88,
            "",
            "PURPOSE",
            "-" * 88,
            (
                "This report examines whether each participant has stable "
                "feature differences between their own made and missed shots."
            ),
            (
                "It describes personal signals. It does not define universal "
                "free-throw mechanics or prove that changing a feature will "
                "cause better shooting."
            ),
            "",
            "PARTICIPANT SUMMARY",
            "-" * 88,
        ]

        for summary in participant_summaries:
            lines.extend(
                [
                    (
                        f"{summary.participant_id}: "
                        f"{summary.personal_signal_status}"
                    ),
                    (
                        f"  Shots: {summary.total_shots} "
                        f"({summary.made_shots} made, "
                        f"{summary.missed_shots} missed)"
                    ),
                    (
                        f"  Top feature: {summary.top_feature} "
                        f"({summary.top_feature_score:.1f})"
                    ),
                    (
                        f"  Top-10 average score: "
                        f"{summary.mean_top_10_score:.1f}"
                    ),
                    (
                        f"  Top-10 direction stability: "
                        f"{summary.mean_top_10_direction_stability_percentage:.1f}%"
                    ),
                    "",
                ]
            )

        lines.extend(
            [
                "TOP FEATURES BY PARTICIPANT",
                "-" * 88,
            ]
        )

        for summary in participant_summaries:
            lines.append(
                ""
            )

            lines.append(
                summary.participant_id
            )

            participant_rows = [
                row
                for row in diagnostic_rows
                if row.participant_id
                == summary.participant_id
            ][
                :15
            ]

            for row in participant_rows:
                effect_text = (
                    "N/A"
                    if row.signed_standardized_difference
                    is None
                    else (
                        f"{row.signed_standardized_difference:.3f}"
                    )
                )

                overlap_text = (
                    "N/A"
                    if row.distribution_overlap_percentage
                    is None
                    else (
                        f"{row.distribution_overlap_percentage:.1f}%"
                    )
                )

                stability_text = (
                    "N/A"
                    if row.bootstrap_direction_stability_percentage
                    is None
                    else (
                        f"{row.bootstrap_direction_stability_percentage:.1f}%"
                    )
                )

                lines.append(
                    (
                        f"  {row.diagnostic_rank}. {row.feature}: "
                        f"score={row.diagnostic_score:.1f}, "
                        f"effect={effect_text}, "
                        f"overlap={overlap_text}, "
                        f"stability={stability_text}"
                    )
                )

        lines.extend(
            [
                "",
                "HOW TO READ THIS REPORT",
                "-" * 88,
                (
                    "Higher absolute effect size means the made and missed "
                    "averages are farther apart relative to their variability."
                ),
                (
                    "Lower distribution overlap means the two groups are more "
                    "separable."
                ),
                (
                    "Higher bootstrap direction stability means repeated "
                    "resampling usually preserves the same made-versus-missed "
                    "direction."
                ),
                (
                    "A useful coaching candidate should ideally show all three: "
                    "meaningful separation, low overlap, and stable direction."
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
    # EXPORT HELPERS
    # =====================================================

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

    # =====================================================
    # SMALL HELPERS
    # =====================================================

    @staticmethod
    def _numeric_values(
        records: list[dict[str, str]],
        feature: str,
    ) -> list[float]:
        output: list[
            float
        ] = []

        for record in records:
            value = (
                PersonalSignalDiagnosticStudio
                ._to_finite_float(
                    record.get(
                        feature,
                        "",
                    )
                )
            )

            if value is not None:
                output.append(
                    value
                )

        return output

    @staticmethod
    def _safe_pstdev(
        values: list[float],
    ) -> float | None:
        if not values:
            return None

        if len(values) == 1:
            return 0.0

        return float(
            pstdev(
                values
            )
        )

    @staticmethod
    def _to_finite_float(
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
    def _optional_float(
        value: object,
    ) -> float | None:
        if value is None:
            return None

        numeric = float(
            value
        )

        if not np.isfinite(
            numeric
        ):
            return None

        return numeric

    @staticmethod
    def _normalize_text(
        value: object,
    ) -> str:
        return str(
            value
        ).strip().lower()

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


def _run_personal_signal_diagnostic_test() -> None:
    studio = PersonalSignalDiagnosticStudio(
        bootstrap_iterations=500,
        exclude_absolute_frames=True,
    )

    (
        diagnostic_rows,
        participant_summaries,
    ) = studio.run()

    assert len(
        diagnostic_rows
    ) > 0

    assert len(
        participant_summaries
    ) > 0

    assert studio.diagnostics_file.exists()
    assert studio.participant_summary_file.exists()
    assert studio.top_features_chart_file.exists()
    assert studio.participant_signal_chart_file.exists()
    assert studio.report_file.exists()

    print()
    print(
        "PERSONAL SIGNAL DIAGNOSTIC TEST PASSED"
    )

    print(
        "Participants analyzed: "
        f"{len(participant_summaries)}"
    )

    print(
        "Participant-feature rows: "
        f"{len(diagnostic_rows)}"
    )

    print()

    print(
        "Feature diagnostics: "
        f"{studio.diagnostics_file.resolve()}"
    )

    print(
        "Participant summary: "
        f"{studio.participant_summary_file.resolve()}"
    )

    print(
        "Top-features chart: "
        f"{studio.top_features_chart_file.resolve()}"
    )

    print(
        "Participant-signal chart: "
        f"{studio.participant_signal_chart_file.resolve()}"
    )

    print(
        "Text report: "
        f"{studio.report_file.resolve()}"
    )

    print()
    print(
        "Participant diagnostic results:"
    )

    for summary in participant_summaries:
        print(
            f"{summary.participant_id}: "
            f"{summary.personal_signal_status}, "
            f"top feature "
            f"{summary.top_feature}, "
            f"score "
            f"{summary.top_feature_score:.1f}, "
            f"top-10 average "
            f"{summary.mean_top_10_score:.1f}"
        )


if __name__ == "__main__":
    _run_personal_signal_diagnostic_test()