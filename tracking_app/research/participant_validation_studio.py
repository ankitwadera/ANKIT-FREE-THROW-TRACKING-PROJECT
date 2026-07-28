from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ImportError as error:
    raise ImportError(
        "Participant Validation Studio requires scikit-learn. "
        "Install it inside the project virtual environment."
    ) from error


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
    / "participant_validation"
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


# Absolute frame-number features can encode trial timing rather than
# basketball mechanics. They are excluded from the main generalization
# test so the model focuses more on comparable shot characteristics.
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
class ParticipantValidationRow:
    """
    One leave-one-participant-out validation fold.
    """

    held_out_participant: str

    training_rows: int
    validation_rows: int

    validation_made_shots: int
    validation_missed_shots: int

    balanced_accuracy: float
    roc_auc: float | None
    precision: float
    recall: float
    f1_score: float

    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    predicted_make_count: int
    predicted_miss_count: int


@dataclass(frozen=True)
class ParticipantPredictionRow:
    """
    One model prediction for one held-out shot.
    """

    held_out_participant: str
    trial_id: str
    trial_file: str

    actual_result: str
    predicted_result: str

    actual_value: int
    predicted_value: int
    predicted_make_probability: float

    correct_prediction: bool
    absolute_probability_error: float


@dataclass(frozen=True)
class ParticipantFeatureImportanceRow:
    """
    One model feature importance within one held-out fold.
    """

    held_out_participant: str
    feature: str
    importance: float
    fold_rank: int


@dataclass(frozen=True)
class GeneralizationFeatureSummaryRow:
    """
    Aggregate feature importance across held-out participants.
    """

    feature: str
    mean_importance: float
    standard_deviation: float
    minimum_importance: float
    maximum_importance: float
    mean_rank: float
    top_10_fold_count: int
    participant_fold_count: int


@dataclass(frozen=True)
class ParticipantValidationSummary:
    """
    Aggregate leave-one-participant-out model result.
    """

    participant_count: int
    total_predictions: int
    feature_count: int

    mean_balanced_accuracy: float
    standard_deviation_balanced_accuracy: float

    mean_roc_auc: float | None
    standard_deviation_roc_auc: float | None

    mean_precision: float
    mean_recall: float
    mean_f1_score: float

    pooled_balanced_accuracy: float
    pooled_roc_auc: float | None

    total_true_negatives: int
    total_false_positives: int
    total_false_negatives: int
    total_true_positives: int

    generalization_status: str


class ParticipantValidationStudio:
    """
    Leave-one-participant-out validation for free-throw biomechanics.

    For each participant:

    - train on every other participant;
    - test only on the held-out participant;
    - export predictions, metrics, confusion counts, and feature rankings.

    This is substantially stricter than random shot-level validation.
    It tests whether the measured patterns transfer to a shooter the
    model did not see during training.
    """

    def __init__(
        self,
        input_file: Path = DEFAULT_INPUT_FILE,
        output_folder: Path = DEFAULT_OUTPUT_FOLDER,
        random_seed: int = 42,
        forest_trees: int = 500,
        minimum_numeric_count: int = 30,
        exclude_absolute_frames: bool = True,
    ) -> None:
        self.input_file = Path(input_file)
        self.output_folder = Path(output_folder)

        self.random_seed = int(random_seed)
        self.forest_trees = max(
            100,
            int(forest_trees),
        )
        self.minimum_numeric_count = max(
            10,
            int(minimum_numeric_count),
        )
        self.exclude_absolute_frames = bool(
            exclude_absolute_frames
        )

        self.validation_file = (
            self.output_folder
            / "participant_validation.csv"
        )

        self.predictions_file = (
            self.output_folder
            / "participant_predictions.csv"
        )

        self.fold_importance_file = (
            self.output_folder
            / "participant_feature_importance.csv"
        )

        self.feature_summary_file = (
            self.output_folder
            / "generalization_feature_summary.csv"
        )

        self.summary_file = (
            self.output_folder
            / "participant_validation_summary.csv"
        )

        self.accuracy_chart_file = (
            self.output_folder
            / "participant_accuracy.png"
        )

        self.confusion_chart_file = (
            self.output_folder
            / "participant_confusion_matrices.png"
        )

        self.report_file = (
            self.output_folder
            / "participant_validation_report.txt"
        )

    # =====================================================
    # PUBLIC WORKFLOW
    # =====================================================

    def run(
        self,
    ) -> tuple[
        ParticipantValidationSummary,
        list[ParticipantValidationRow],
        list[ParticipantPredictionRow],
        list[GeneralizationFeatureSummaryRow],
    ]:
        """
        Run leave-one-participant-out validation and export all outputs.
        """

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
                "No successful made/missed shot records were available."
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

        if len(participants) < 3:
            raise RuntimeError(
                "Participant-held-out validation requires at least "
                "three participants."
            )

        feature_names = self.identify_numeric_features(
            successful
        )

        if not feature_names:
            raise RuntimeError(
                "No usable numeric feature fields were found."
            )

        (
            validation_rows,
            prediction_rows,
            importance_rows,
        ) = self.run_participant_folds(
            records=successful,
            participants=participants,
            feature_names=feature_names,
        )

        feature_summary_rows = (
            self.build_generalization_feature_summary(
                importance_rows=importance_rows,
                feature_names=feature_names,
                participant_count=len(
                    participants
                ),
            )
        )

        summary = self.build_summary(
            validation_rows=validation_rows,
            prediction_rows=prediction_rows,
            feature_count=len(
                feature_names
            ),
        )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.export_dataclass_rows(
            self.validation_file,
            validation_rows,
        )

        self.export_dataclass_rows(
            self.predictions_file,
            prediction_rows,
        )

        self.export_dataclass_rows(
            self.fold_importance_file,
            importance_rows,
        )

        self.export_dataclass_rows(
            self.feature_summary_file,
            feature_summary_rows,
        )

        self.export_dataclass_rows(
            self.summary_file,
            [summary],
        )

        self.create_accuracy_chart(
            validation_rows
        )

        self.create_confusion_chart(
            validation_rows
        )

        self.export_text_report(
            summary=summary,
            validation_rows=validation_rows,
            feature_summary_rows=(
                feature_summary_rows
            ),
        )

        return (
            summary,
            validation_rows,
            prediction_rows,
            feature_summary_rows,
        )

    # =====================================================
    # LOAD AND FEATURE DISCOVERY
    # =====================================================

    def load_records(
        self,
    ) -> list[dict[str, str]]:
        """
        Read the master shot-feature dataset.
        """

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

    def identify_numeric_features(
        self,
        records: list[dict[str, str]],
    ) -> list[str]:
        """
        Select usable numeric predictors.

        Absolute frame-number features are excluded by default because
        they may encode trial timing rather than transferable mechanics.
        """

        fieldnames = list(
            records[0]
        )

        output: list[
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

            if (
                len(finite_values)
                < self.minimum_numeric_count
            ):
                continue

            if len(
                set(finite_values)
            ) < 2:
                continue

            output.append(
                field_name
            )

        return output

    # =====================================================
    # PARTICIPANT FOLDS
    # =====================================================

    def run_participant_folds(
        self,
        records: list[dict[str, str]],
        participants: list[str],
        feature_names: list[str],
    ) -> tuple[
        list[ParticipantValidationRow],
        list[ParticipantPredictionRow],
        list[ParticipantFeatureImportanceRow],
    ]:
        """
        Train on all other participants and test on one held-out person.
        """

        validation_rows: list[
            ParticipantValidationRow
        ] = []

        prediction_rows: list[
            ParticipantPredictionRow
        ] = []

        importance_rows: list[
            ParticipantFeatureImportanceRow
        ] = []

        for fold_number, held_out in enumerate(
            participants,
            start=1,
        ):
            training_records = [
                record
                for record in records
                if str(
                    record.get(
                        "participant_id",
                        "",
                    )
                ).strip()
                != held_out
            ]

            validation_records = [
                record
                for record in records
                if str(
                    record.get(
                        "participant_id",
                        "",
                    )
                ).strip()
                == held_out
            ]

            (
                training_matrix,
                training_target,
            ) = self.build_matrix(
                training_records,
                feature_names,
            )

            (
                validation_matrix,
                validation_target,
            ) = self.build_matrix(
                validation_records,
                feature_names,
            )

            if len(
                np.unique(
                    training_target
                )
            ) < 2:
                raise RuntimeError(
                    "Training data for held-out participant "
                    f"{held_out} did not contain both classes."
                )

            imputer = SimpleImputer(
                strategy="median"
            )

            training_imputed = (
                imputer.fit_transform(
                    training_matrix
                )
            )

            validation_imputed = (
                imputer.transform(
                    validation_matrix
                )
            )

            model = RandomForestClassifier(
                n_estimators=(
                    self.forest_trees
                ),
                random_state=(
                    self.random_seed
                    + fold_number
                ),
                class_weight="balanced",
                min_samples_leaf=4,
                max_features="sqrt",
                n_jobs=-1,
            )

            model.fit(
                training_imputed,
                training_target,
            )

            predicted_values = model.predict(
                validation_imputed
            )

            predicted_probabilities = (
                model.predict_proba(
                    validation_imputed
                )[:, 1]
            )

            balanced_accuracy = float(
                balanced_accuracy_score(
                    validation_target,
                    predicted_values,
                )
            )

            precision = float(
                precision_score(
                    validation_target,
                    predicted_values,
                    zero_division=0,
                )
            )

            recall = float(
                recall_score(
                    validation_target,
                    predicted_values,
                    zero_division=0,
                )
            )

            fold_f1 = float(
                f1_score(
                    validation_target,
                    predicted_values,
                    zero_division=0,
                )
            )

            fold_auc = None

            if len(
                np.unique(
                    validation_target
                )
            ) == 2:
                fold_auc = float(
                    roc_auc_score(
                        validation_target,
                        predicted_probabilities,
                    )
                )

            (
                true_negatives,
                false_positives,
                false_negatives,
                true_positives,
            ) = confusion_matrix(
                validation_target,
                predicted_values,
                labels=[
                    0,
                    1,
                ],
            ).ravel()

            validation_rows.append(
                ParticipantValidationRow(
                    held_out_participant=(
                        held_out
                    ),
                    training_rows=len(
                        training_records
                    ),
                    validation_rows=len(
                        validation_records
                    ),
                    validation_made_shots=int(
                        np.sum(
                            validation_target
                            == 1
                        )
                    ),
                    validation_missed_shots=int(
                        np.sum(
                            validation_target
                            == 0
                        )
                    ),
                    balanced_accuracy=(
                        balanced_accuracy
                    ),
                    roc_auc=fold_auc,
                    precision=precision,
                    recall=recall,
                    f1_score=fold_f1,
                    true_negatives=int(
                        true_negatives
                    ),
                    false_positives=int(
                        false_positives
                    ),
                    false_negatives=int(
                        false_negatives
                    ),
                    true_positives=int(
                        true_positives
                    ),
                    predicted_make_count=int(
                        np.sum(
                            predicted_values
                            == 1
                        )
                    ),
                    predicted_miss_count=int(
                        np.sum(
                            predicted_values
                            == 0
                        )
                    ),
                )
            )

            for (
                record,
                actual_value,
                predicted_value,
                probability,
            ) in zip(
                validation_records,
                validation_target,
                predicted_values,
                predicted_probabilities,
            ):
                actual_result = (
                    "made"
                    if int(actual_value)
                    == 1
                    else "missed"
                )

                predicted_result = (
                    "made"
                    if int(predicted_value)
                    == 1
                    else "missed"
                )

                prediction_rows.append(
                    ParticipantPredictionRow(
                        held_out_participant=(
                            held_out
                        ),
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
                        actual_result=(
                            actual_result
                        ),
                        predicted_result=(
                            predicted_result
                        ),
                        actual_value=int(
                            actual_value
                        ),
                        predicted_value=int(
                            predicted_value
                        ),
                        predicted_make_probability=float(
                            probability
                        ),
                        correct_prediction=(
                            int(actual_value)
                            == int(
                                predicted_value
                            )
                        ),
                        absolute_probability_error=float(
                            abs(
                                int(actual_value)
                                - float(
                                    probability
                                )
                            )
                        ),
                    )
                )

            fold_importances = np.asarray(
                model.feature_importances_,
                dtype=float,
            )

            rank_order = np.argsort(
                -fold_importances
            )

            fold_ranks = np.empty_like(
                rank_order
            )

            fold_ranks[
                rank_order
            ] = np.arange(
                1,
                len(
                    rank_order
                )
                + 1,
            )

            for feature_index, feature in enumerate(
                feature_names
            ):
                importance_rows.append(
                    ParticipantFeatureImportanceRow(
                        held_out_participant=(
                            held_out
                        ),
                        feature=feature,
                        importance=float(
                            fold_importances[
                                feature_index
                            ]
                        ),
                        fold_rank=int(
                            fold_ranks[
                                feature_index
                            ]
                        ),
                    )
                )

            print(
                "Held out "
                f"{held_out}: "
                f"balanced accuracy "
                f"{balanced_accuracy:.3f}, "
                f"ROC AUC "
                f"{fold_auc:.3f}"
                if fold_auc is not None
                else (
                    "Held out "
                    f"{held_out}: "
                    f"balanced accuracy "
                    f"{balanced_accuracy:.3f}, "
                    "ROC AUC N/A"
                )
            )

        return (
            validation_rows,
            prediction_rows,
            importance_rows,
        )

    def build_matrix(
        self,
        records: list[dict[str, str]],
        feature_names: list[str],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        """
        Convert records to a numeric matrix and binary outcome.
        """

        rows: list[
            list[float]
        ] = []

        target: list[
            int
        ] = []

        for record in records:
            result = self._normalize_result(
                record.get(
                    "result",
                    "",
                )
            )

            if result not in {
                "made",
                "missed",
            }:
                continue

            row: list[
                float
            ] = []

            for feature in feature_names:
                value = self._to_finite_float(
                    record.get(
                        feature,
                        "",
                    )
                )

                row.append(
                    np.nan
                    if value is None
                    else value
                )

            rows.append(
                row
            )

            target.append(
                1
                if result == "made"
                else 0
            )

        return (
            np.asarray(
                rows,
                dtype=float,
            ),
            np.asarray(
                target,
                dtype=int,
            ),
        )

    # =====================================================
    # FEATURE SUMMARY
    # =====================================================

    @staticmethod
    def build_generalization_feature_summary(
        importance_rows: list[
            ParticipantFeatureImportanceRow
        ],
        feature_names: list[str],
        participant_count: int,
    ) -> list[
        GeneralizationFeatureSummaryRow
    ]:
        """
        Aggregate random-forest importance across held-out participants.
        """

        grouped: dict[
            str,
            list[
                ParticipantFeatureImportanceRow
            ],
        ] = {
            feature: []
            for feature in feature_names
        }

        for row in importance_rows:
            grouped.setdefault(
                row.feature,
                [],
            ).append(row)

        output: list[
            GeneralizationFeatureSummaryRow
        ] = []

        for feature in feature_names:
            rows = grouped.get(
                feature,
                [],
            )

            importance_values = [
                row.importance
                for row in rows
            ]

            ranks = [
                row.fold_rank
                for row in rows
            ]

            if not importance_values:
                continue

            output.append(
                GeneralizationFeatureSummaryRow(
                    feature=feature,
                    mean_importance=float(
                        mean(
                            importance_values
                        )
                    ),
                    standard_deviation=(
                        float(
                            pstdev(
                                importance_values
                            )
                        )
                        if len(
                            importance_values
                        )
                        > 1
                        else 0.0
                    ),
                    minimum_importance=float(
                        min(
                            importance_values
                        )
                    ),
                    maximum_importance=float(
                        max(
                            importance_values
                        )
                    ),
                    mean_rank=float(
                        mean(
                            ranks
                        )
                    ),
                    top_10_fold_count=sum(
                        rank <= 10
                        for rank in ranks
                    ),
                    participant_fold_count=(
                        participant_count
                    ),
                )
            )

        output.sort(
            key=lambda row: (
                -row.mean_importance,
                row.mean_rank,
                row.feature,
            )
        )

        return output

    # =====================================================
    # SUMMARY
    # =====================================================

    def build_summary(
        self,
        validation_rows: list[
            ParticipantValidationRow
        ],
        prediction_rows: list[
            ParticipantPredictionRow
        ],
        feature_count: int,
    ) -> ParticipantValidationSummary:
        """
        Build the overall generalization summary.
        """

        balanced_accuracies = [
            row.balanced_accuracy
            for row in validation_rows
        ]

        auc_values = [
            row.roc_auc
            for row in validation_rows
            if row.roc_auc is not None
        ]

        precisions = [
            row.precision
            for row in validation_rows
        ]

        recalls = [
            row.recall
            for row in validation_rows
        ]

        f1_values = [
            row.f1_score
            for row in validation_rows
        ]

        actual = np.asarray(
            [
                row.actual_value
                for row in prediction_rows
            ],
            dtype=int,
        )

        predicted = np.asarray(
            [
                row.predicted_value
                for row in prediction_rows
            ],
            dtype=int,
        )

        probabilities = np.asarray(
            [
                row.predicted_make_probability
                for row in prediction_rows
            ],
            dtype=float,
        )

        pooled_balanced_accuracy = float(
            balanced_accuracy_score(
                actual,
                predicted,
            )
        )

        pooled_auc = None

        if len(
            np.unique(
                actual
            )
        ) == 2:
            pooled_auc = float(
                roc_auc_score(
                    actual,
                    probabilities,
                )
            )

        (
            true_negatives,
            false_positives,
            false_negatives,
            true_positives,
        ) = confusion_matrix(
            actual,
            predicted,
            labels=[
                0,
                1,
            ],
        ).ravel()

        mean_balanced_accuracy = float(
            mean(
                balanced_accuracies
            )
        )

        return ParticipantValidationSummary(
            participant_count=len(
                validation_rows
            ),
            total_predictions=len(
                prediction_rows
            ),
            feature_count=feature_count,

            mean_balanced_accuracy=(
                mean_balanced_accuracy
            ),
            standard_deviation_balanced_accuracy=(
                float(
                    pstdev(
                        balanced_accuracies
                    )
                )
                if len(
                    balanced_accuracies
                )
                > 1
                else 0.0
            ),

            mean_roc_auc=(
                float(
                    mean(
                        auc_values
                    )
                )
                if auc_values
                else None
            ),
            standard_deviation_roc_auc=(
                float(
                    pstdev(
                        auc_values
                    )
                )
                if len(
                    auc_values
                )
                > 1
                else (
                    0.0
                    if auc_values
                    else None
                )
            ),

            mean_precision=float(
                mean(
                    precisions
                )
            ),
            mean_recall=float(
                mean(
                    recalls
                )
            ),
            mean_f1_score=float(
                mean(
                    f1_values
                )
            ),

            pooled_balanced_accuracy=(
                pooled_balanced_accuracy
            ),
            pooled_roc_auc=(
                pooled_auc
            ),

            total_true_negatives=int(
                true_negatives
            ),
            total_false_positives=int(
                false_positives
            ),
            total_false_negatives=int(
                false_negatives
            ),
            total_true_positives=int(
                true_positives
            ),

            generalization_status=(
                self._generalization_status(
                    mean_balanced_accuracy
                )
            ),
        )

    @staticmethod
    def _generalization_status(
        balanced_accuracy: float,
    ) -> str:
        """
        Provide a cautious research interpretation.
        """

        if balanced_accuracy >= 0.70:
            return "Promising"

        if balanced_accuracy >= 0.60:
            return "Moderate signal"

        if balanced_accuracy >= 0.55:
            return "Weak signal"

        return "Near chance"

    # =====================================================
    # CHARTS
    # =====================================================

    def create_accuracy_chart(
        self,
        rows: list[
            ParticipantValidationRow
        ],
    ) -> Path:
        """
        Plot held-out balanced accuracy and ROC AUC.
        """

        labels = [
            row.held_out_participant
            for row in rows
        ]

        balanced = [
            row.balanced_accuracy
            for row in rows
        ]

        auc_values = [
            (
                np.nan
                if row.roc_auc is None
                else row.roc_auc
            )
            for row in rows
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
            balanced,
            width,
            label="Balanced accuracy",
        )

        axis.bar(
            positions
            + width / 2,
            auc_values,
            width,
            label="ROC AUC",
        )

        axis.axhline(
            y=0.5,
            linestyle="--",
            linewidth=1,
            label="Chance reference",
        )

        axis.set_xticks(
            positions,
            labels=labels,
        )

        axis.set_ylim(
            0.0,
            1.0,
        )

        axis.set_ylabel(
            "Score"
        )

        axis.set_title(
            "Leave-One-Participant-Out Performance"
        )

        axis.legend()

        axis.grid(
            visible=True,
            axis="y",
            alpha=0.25,
        )

        figure.tight_layout()

        figure.savefig(
            self.accuracy_chart_file,
            dpi=170,
            bbox_inches="tight",
        )

        plt.close(
            figure
        )

        return self.accuracy_chart_file

    def create_confusion_chart(
        self,
        rows: list[
            ParticipantValidationRow
        ],
    ) -> Path:
        """
        Plot confusion counts for each held-out participant.
        """

        labels = [
            row.held_out_participant
            for row in rows
        ]

        true_negatives = [
            row.true_negatives
            for row in rows
        ]

        false_positives = [
            row.false_positives
            for row in rows
        ]

        false_negatives = [
            row.false_negatives
            for row in rows
        ]

        true_positives = [
            row.true_positives
            for row in rows
        ]

        positions = np.arange(
            len(labels)
        )

        width = 0.20

        figure, axis = plt.subplots(
            figsize=(
                12,
                7,
            )
        )

        axis.bar(
            positions
            - 1.5 * width,
            true_negatives,
            width,
            label="True misses",
        )

        axis.bar(
            positions
            - 0.5 * width,
            false_positives,
            width,
            label="Misses predicted made",
        )

        axis.bar(
            positions
            + 0.5 * width,
            false_negatives,
            width,
            label="Makes predicted missed",
        )

        axis.bar(
            positions
            + 1.5 * width,
            true_positives,
            width,
            label="True makes",
        )

        axis.set_xticks(
            positions,
            labels=labels,
        )

        axis.set_ylabel(
            "Shot count"
        )

        axis.set_title(
            "Participant-Held-Out Confusion Counts"
        )

        axis.legend()

        axis.grid(
            visible=True,
            axis="y",
            alpha=0.25,
        )

        figure.tight_layout()

        figure.savefig(
            self.confusion_chart_file,
            dpi=170,
            bbox_inches="tight",
        )

        plt.close(
            figure
        )

        return self.confusion_chart_file

    # =====================================================
    # REPORT
    # =====================================================

    def export_text_report(
        self,
        summary: ParticipantValidationSummary,
        validation_rows: list[
            ParticipantValidationRow
        ],
        feature_summary_rows: list[
            GeneralizationFeatureSummaryRow
        ],
    ) -> Path:
        """
        Write a cautious, readable generalization report.
        """

        lines = [
            "=" * 84,
            "BASKETBALL BIOMECHANICS RESEARCH STUDIO",
            "PARTICIPANT HOLD-OUT VALIDATION REPORT",
            "=" * 84,
            "",
            "VALIDATION DESIGN",
            "-" * 84,
            (
                "Each participant was held out completely while the model "
                "trained on every other participant."
            ),
            (
                "Absolute frame-number features were excluded: "
                f"{self.exclude_absolute_frames}"
            ),
            (
                "This test evaluates transfer to an unseen shooter and is "
                "stricter than random shot-level cross-validation."
            ),
            "",
            "OVERALL PERFORMANCE",
            "-" * 84,
            (
                "Participants held out: "
                f"{summary.participant_count}"
            ),
            (
                "Total held-out predictions: "
                f"{summary.total_predictions}"
            ),
            (
                "Features used: "
                f"{summary.feature_count}"
            ),
            (
                "Mean balanced accuracy: "
                f"{summary.mean_balanced_accuracy:.3f}"
            ),
            (
                "Balanced accuracy SD: "
                f"{summary.standard_deviation_balanced_accuracy:.3f}"
            ),
            (
                "Mean ROC AUC: "
                f"{summary.mean_roc_auc:.3f}"
                if summary.mean_roc_auc
                is not None
                else "Mean ROC AUC: N/A"
            ),
            (
                "Pooled balanced accuracy: "
                f"{summary.pooled_balanced_accuracy:.3f}"
            ),
            (
                "Pooled ROC AUC: "
                f"{summary.pooled_roc_auc:.3f}"
                if summary.pooled_roc_auc
                is not None
                else "Pooled ROC AUC: N/A"
            ),
            (
                "Generalization status: "
                f"{summary.generalization_status}"
            ),
            "",
            "PARTICIPANT RESULTS",
            "-" * 84,
        ]

        for row in validation_rows:
            lines.append(
                (
                    f"{row.held_out_participant}: "
                    f"balanced accuracy={row.balanced_accuracy:.3f}, "
                    f"ROC AUC="
                    f"{row.roc_auc:.3f}"
                    if row.roc_auc is not None
                    else (
                        f"{row.held_out_participant}: "
                        f"balanced accuracy={row.balanced_accuracy:.3f}, "
                        "ROC AUC=N/A"
                    )
                )
            )

        lines.extend(
            [
                "",
                "MOST CONSISTENT GENERALIZATION FEATURES",
                "-" * 84,
            ]
        )

        for rank, row in enumerate(
            feature_summary_rows[
                :25
            ],
            start=1,
        ):
            lines.append(
                (
                    f"{rank}. {row.feature}: "
                    f"mean importance={row.mean_importance:.5f}, "
                    f"mean rank={row.mean_rank:.1f}, "
                    f"top-10 folds={row.top_10_fold_count}/"
                    f"{row.participant_fold_count}"
                )
            )

        lines.extend(
            [
                "",
                "IMPORTANT LIMITATIONS",
                "-" * 84,
                (
                    "Only five participants are represented, so one unusual "
                    "participant can strongly affect the average."
                ),
                (
                    "Feature importance indicates predictive usefulness in "
                    "this dataset, not causation or a universal ideal."
                ),
                (
                    "A future coaching product requires more participants, "
                    "external validation, and player-specific baselines."
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
        """
        Export dataclass records to CSV.
        """

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
            asdict(row)
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
    def _to_finite_float(
        value: object,
    ) -> float | None:
        """
        Convert a value to a finite float when possible.
        """

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


def _run_participant_validation_test() -> None:
    """
    Run the participant-held-out validation studio.
    """

    studio = ParticipantValidationStudio(
        exclude_absolute_frames=True,
    )

    (
        summary,
        validation_rows,
        prediction_rows,
        feature_summary_rows,
    ) = studio.run()

    assert len(
        validation_rows
    ) >= 3

    assert len(
        prediction_rows
    ) > 0

    assert len(
        feature_summary_rows
    ) > 0

    assert studio.validation_file.exists()
    assert studio.predictions_file.exists()
    assert studio.fold_importance_file.exists()
    assert studio.feature_summary_file.exists()
    assert studio.summary_file.exists()
    assert studio.accuracy_chart_file.exists()
    assert studio.confusion_chart_file.exists()
    assert studio.report_file.exists()

    print()
    print(
        "PARTICIPANT HOLD-OUT VALIDATION TEST PASSED"
    )

    print(
        "Participants held out: "
        f"{summary.participant_count}"
    )

    print(
        "Features used: "
        f"{summary.feature_count}"
    )

    print(
        "Total held-out predictions: "
        f"{summary.total_predictions}"
    )

    print(
        "Mean balanced accuracy: "
        f"{summary.mean_balanced_accuracy:.3f}"
    )

    print(
        "Mean ROC AUC: "
        f"{summary.mean_roc_auc:.3f}"
        if summary.mean_roc_auc
        is not None
        else "Mean ROC AUC: N/A"
    )

    print(
        "Pooled balanced accuracy: "
        f"{summary.pooled_balanced_accuracy:.3f}"
    )

    print(
        "Pooled ROC AUC: "
        f"{summary.pooled_roc_auc:.3f}"
        if summary.pooled_roc_auc
        is not None
        else "Pooled ROC AUC: N/A"
    )

    print(
        "Generalization status: "
        f"{summary.generalization_status}"
    )

    print()

    print(
        "Validation folds: "
        f"{studio.validation_file.resolve()}"
    )

    print(
        "Predictions: "
        f"{studio.predictions_file.resolve()}"
    )

    print(
        "Fold feature importance: "
        f"{studio.fold_importance_file.resolve()}"
    )

    print(
        "Generalization feature summary: "
        f"{studio.feature_summary_file.resolve()}"
    )

    print(
        "Overall summary: "
        f"{studio.summary_file.resolve()}"
    )

    print(
        "Accuracy chart: "
        f"{studio.accuracy_chart_file.resolve()}"
    )

    print(
        "Confusion chart: "
        f"{studio.confusion_chart_file.resolve()}"
    )

    print(
        "Text report: "
        f"{studio.report_file.resolve()}"
    )

    print()

    print(
        "Top 15 generalization features:"
    )

    for rank, row in enumerate(
        feature_summary_rows[
            :15
        ],
        start=1,
    ):
        print(
            f"{rank}. "
            f"{row.feature}: "
            f"importance "
            f"{row.mean_importance:.5f}, "
            f"mean rank "
            f"{row.mean_rank:.1f}"
        )


if __name__ == "__main__":
    _run_participant_validation_test()