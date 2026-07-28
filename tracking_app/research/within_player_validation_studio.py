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
    from sklearn.model_selection import RepeatedStratifiedKFold
except ImportError as error:
    raise ImportError(
        "Within-Player Validation Studio requires scikit-learn."
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
    / "within_player_validation"
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
class WithinPlayerFoldRow:
    participant_id: str
    repeat_index: int
    fold_index: int

    training_rows: int
    validation_rows: int

    training_makes: int
    training_misses: int
    validation_makes: int
    validation_misses: int

    balanced_accuracy: float
    roc_auc: float | None
    precision: float
    recall: float
    f1_score: float

    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int


@dataclass(frozen=True)
class WithinPlayerPredictionRow:
    participant_id: str
    repeat_index: int
    fold_index: int

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
class WithinPlayerFeatureImportanceRow:
    participant_id: str
    repeat_index: int
    fold_index: int

    feature: str
    importance: float
    fold_rank: int


@dataclass(frozen=True)
class ParticipantWithinPlayerSummary:
    participant_id: str

    shot_count: int
    made_shots: int
    missed_shots: int
    feature_count: int
    fold_count: int

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

    within_player_status: str


@dataclass(frozen=True)
class WithinPlayerFeatureSummaryRow:
    participant_id: str
    feature: str

    mean_importance: float
    standard_deviation: float
    minimum_importance: float
    maximum_importance: float

    mean_rank: float
    top_10_fold_count: int
    fold_count: int


@dataclass(frozen=True)
class OverallWithinPlayerSummary:
    participant_count: int
    total_shots: int
    total_predictions: int
    feature_count: int
    total_folds: int

    mean_participant_balanced_accuracy: float
    standard_deviation_participant_balanced_accuracy: float

    mean_participant_roc_auc: float | None
    standard_deviation_participant_roc_auc: float | None

    pooled_balanced_accuracy: float
    pooled_roc_auc: float | None

    mean_precision: float
    mean_recall: float
    mean_f1_score: float

    general_status: str


class WithinPlayerValidationStudio:
    """
    Evaluate whether each shooter has learnable personal make/miss patterns.

    Each participant is analyzed separately:

        one participant's shots
                ↓
        repeated stratified cross-validation
                ↓
        train on some of that player's shots
                ↓
        test on different shots from the same player

    This evaluates personalization potential. It does not test transfer
    to a new shooter.
    """

    def __init__(
        self,
        input_file: Path = DEFAULT_INPUT_FILE,
        output_folder: Path = DEFAULT_OUTPUT_FOLDER,
        random_seed: int = 42,
        cross_validation_splits: int = 5,
        cross_validation_repeats: int = 5,
        forest_trees: int = 400,
        minimum_numeric_count: int = 20,
        exclude_absolute_frames: bool = True,
    ) -> None:
        self.input_file = Path(input_file)
        self.output_folder = Path(output_folder)

        self.random_seed = int(random_seed)

        self.cross_validation_splits = max(
            3,
            int(cross_validation_splits),
        )

        self.cross_validation_repeats = max(
            1,
            int(cross_validation_repeats),
        )

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

        self.fold_file = (
            self.output_folder
            / "within_player_folds.csv"
        )

        self.predictions_file = (
            self.output_folder
            / "within_player_predictions.csv"
        )

        self.fold_importance_file = (
            self.output_folder
            / "within_player_feature_importance.csv"
        )

        self.participant_summary_file = (
            self.output_folder
            / "participant_within_player_summary.csv"
        )

        self.feature_summary_file = (
            self.output_folder
            / "within_player_feature_summary.csv"
        )

        self.overall_summary_file = (
            self.output_folder
            / "overall_within_player_summary.csv"
        )

        self.performance_chart_file = (
            self.output_folder
            / "within_player_performance.png"
        )

        self.confusion_chart_file = (
            self.output_folder
            / "within_player_confusion.png"
        )

        self.report_file = (
            self.output_folder
            / "within_player_validation_report.txt"
        )

    # =====================================================
    # PUBLIC WORKFLOW
    # =====================================================

    def run(
        self,
    ) -> tuple[
        OverallWithinPlayerSummary,
        list[ParticipantWithinPlayerSummary],
        list[WithinPlayerFoldRow],
        list[WithinPlayerPredictionRow],
        list[WithinPlayerFeatureSummaryRow],
    ]:
        records = self.load_records()

        successful = [
            record
            for record in records
            if self._normalize_text(
                record.get("analysis_status", "")
            )
            == "success"
            and self._normalize_result(
                record.get("result", "")
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

        if not participants:
            raise RuntimeError(
                "No participants were found."
            )

        global_feature_names = self.identify_numeric_features(
            successful
        )

        (
            fold_rows,
            prediction_rows,
            importance_rows,
            participant_summaries,
        ) = self.run_all_participants(
            records=successful,
            participants=participants,
            global_feature_names=global_feature_names,
        )

        feature_summary_rows = (
            self.build_feature_summaries(
                importance_rows
            )
        )

        overall_summary = self.build_overall_summary(
            participant_summaries=participant_summaries,
            prediction_rows=prediction_rows,
            feature_count=len(
                global_feature_names
            ),
            total_shots=len(
                successful
            ),
            total_folds=len(
                fold_rows
            ),
        )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.export_dataclass_rows(
            self.fold_file,
            fold_rows,
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
            self.participant_summary_file,
            participant_summaries,
        )

        self.export_dataclass_rows(
            self.feature_summary_file,
            feature_summary_rows,
        )

        self.export_dataclass_rows(
            self.overall_summary_file,
            [overall_summary],
        )

        self.create_performance_chart(
            participant_summaries
        )

        self.create_confusion_chart(
            participant_summaries
        )

        self.export_text_report(
            overall_summary=overall_summary,
            participant_summaries=participant_summaries,
            feature_summary_rows=feature_summary_rows,
        )

        return (
            overall_summary,
            participant_summaries,
            fold_rows,
            prediction_rows,
            feature_summary_rows,
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
    # FEATURES
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

            finite_values = [
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
                for value in finite_values
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

            features.append(
                field_name
            )

        return features

    def select_participant_features(
        self,
        participant_records: list[
            dict[str, str]
        ],
        global_feature_names: list[str],
    ) -> list[str]:
        """
        Remove features that are constant or nearly empty for one player.
        """

        selected: list[
            str
        ] = []

        minimum_count = max(
            10,
            int(
                round(
                    len(participant_records)
                    * 0.50
                )
            ),
        )

        for feature in global_feature_names:
            values = [
                self._to_finite_float(
                    record.get(
                        feature,
                        "",
                    )
                )
                for record in participant_records
            ]

            finite_values = [
                value
                for value in values
                if value is not None
            ]

            if len(
                finite_values
            ) < minimum_count:
                continue

            if len(
                set(finite_values)
            ) < 2:
                continue

            selected.append(
                feature
            )

        return selected

    # =====================================================
    # PARTICIPANT WORKFLOW
    # =====================================================

    def run_all_participants(
        self,
        records: list[dict[str, str]],
        participants: list[str],
        global_feature_names: list[str],
    ) -> tuple[
        list[WithinPlayerFoldRow],
        list[WithinPlayerPredictionRow],
        list[WithinPlayerFeatureImportanceRow],
        list[ParticipantWithinPlayerSummary],
    ]:
        fold_rows: list[
            WithinPlayerFoldRow
        ] = []

        prediction_rows: list[
            WithinPlayerPredictionRow
        ] = []

        importance_rows: list[
            WithinPlayerFeatureImportanceRow
        ] = []

        participant_summaries: list[
            ParticipantWithinPlayerSummary
        ] = []

        for participant_number, participant_id in enumerate(
            participants,
            start=1,
        ):
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
            ]

            participant_features = (
                self.select_participant_features(
                    participant_records,
                    global_feature_names,
                )
            )

            if not participant_features:
                raise RuntimeError(
                    f"No usable features remained for {participant_id}."
                )

            matrix, target = self.build_matrix(
                participant_records,
                participant_features,
            )

            class_counts = np.bincount(
                target,
                minlength=2,
            )

            smallest_class_count = int(
                np.min(
                    class_counts
                )
            )

            splits = min(
                self.cross_validation_splits,
                smallest_class_count,
            )

            if splits < 2:
                raise RuntimeError(
                    f"{participant_id} does not have enough made and "
                    "missed shots for stratified validation."
                )

            splitter = RepeatedStratifiedKFold(
                n_splits=splits,
                n_repeats=self.cross_validation_repeats,
                random_state=(
                    self.random_seed
                    + participant_number
                ),
            )

            participant_fold_rows: list[
                WithinPlayerFoldRow
            ] = []

            participant_prediction_rows: list[
                WithinPlayerPredictionRow
            ] = []

            total_fold_index = 0

            for (
                training_indices,
                validation_indices,
            ) in splitter.split(
                matrix,
                target,
            ):
                repeat_index = (
                    total_fold_index
                    // splits
                    + 1
                )

                fold_index = (
                    total_fold_index
                    % splits
                    + 1
                )

                total_fold_index += 1

                training_matrix = matrix[
                    training_indices
                ]

                validation_matrix = matrix[
                    validation_indices
                ]

                training_target = target[
                    training_indices
                ]

                validation_target = target[
                    validation_indices
                ]

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
                    n_estimators=self.forest_trees,
                    random_state=(
                        self.random_seed
                        + participant_number
                        * 100
                        + total_fold_index
                    ),
                    class_weight="balanced",
                    min_samples_leaf=3,
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

                probabilities = model.predict_proba(
                    validation_imputed
                )[:, 1]

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
                            probabilities,
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

                fold_row = WithinPlayerFoldRow(
                    participant_id=participant_id,
                    repeat_index=repeat_index,
                    fold_index=fold_index,

                    training_rows=len(
                        training_indices
                    ),
                    validation_rows=len(
                        validation_indices
                    ),

                    training_makes=int(
                        np.sum(
                            training_target
                            == 1
                        )
                    ),
                    training_misses=int(
                        np.sum(
                            training_target
                            == 0
                        )
                    ),
                    validation_makes=int(
                        np.sum(
                            validation_target
                            == 1
                        )
                    ),
                    validation_misses=int(
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
                )

                fold_rows.append(
                    fold_row
                )

                participant_fold_rows.append(
                    fold_row
                )

                validation_records = [
                    participant_records[index]
                    for index in validation_indices
                ]

                for (
                    record,
                    actual_value,
                    predicted_value,
                    probability,
                ) in zip(
                    validation_records,
                    validation_target,
                    predicted_values,
                    probabilities,
                ):
                    prediction_row = (
                        WithinPlayerPredictionRow(
                            participant_id=participant_id,
                            repeat_index=repeat_index,
                            fold_index=fold_index,

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
                                "made"
                                if int(
                                    actual_value
                                )
                                == 1
                                else "missed"
                            ),
                            predicted_result=(
                                "made"
                                if int(
                                    predicted_value
                                )
                                == 1
                                else "missed"
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
                                int(
                                    actual_value
                                )
                                == int(
                                    predicted_value
                                )
                            ),
                            absolute_probability_error=float(
                                abs(
                                    int(
                                        actual_value
                                    )
                                    - float(
                                        probability
                                    )
                                )
                            ),
                        )
                    )

                    prediction_rows.append(
                        prediction_row
                    )

                    participant_prediction_rows.append(
                        prediction_row
                    )

                feature_importances = np.asarray(
                    model.feature_importances_,
                    dtype=float,
                )

                rank_order = np.argsort(
                    -feature_importances
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
                    participant_features
                ):
                    importance_rows.append(
                        WithinPlayerFeatureImportanceRow(
                            participant_id=participant_id,
                            repeat_index=repeat_index,
                            fold_index=fold_index,
                            feature=feature,
                            importance=float(
                                feature_importances[
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

            participant_summary = (
                self.build_participant_summary(
                    participant_id=participant_id,
                    participant_records=participant_records,
                    feature_count=len(
                        participant_features
                    ),
                    fold_rows=participant_fold_rows,
                    prediction_rows=participant_prediction_rows,
                )
            )

            participant_summaries.append(
                participant_summary
            )

            print(
                f"{participant_id}: "
                f"balanced accuracy "
                f"{participant_summary.mean_balanced_accuracy:.3f}, "
                "ROC AUC "
                f"{participant_summary.mean_roc_auc:.3f}"
                if participant_summary.mean_roc_auc
                is not None
                else (
                    f"{participant_id}: "
                    f"balanced accuracy "
                    f"{participant_summary.mean_balanced_accuracy:.3f}, "
                    "ROC AUC N/A"
                )
            )

        return (
            fold_rows,
            prediction_rows,
            importance_rows,
            participant_summaries,
        )

    def build_matrix(
        self,
        records: list[dict[str, str]],
        feature_names: list[str],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
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

            row = []

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
    # PARTICIPANT SUMMARY
    # =====================================================

    def build_participant_summary(
        self,
        participant_id: str,
        participant_records: list[
            dict[str, str]
        ],
        feature_count: int,
        fold_rows: list[
            WithinPlayerFoldRow
        ],
        prediction_rows: list[
            WithinPlayerPredictionRow
        ],
    ) -> ParticipantWithinPlayerSummary:
        balanced_accuracies = [
            row.balanced_accuracy
            for row in fold_rows
        ]

        auc_values = [
            row.roc_auc
            for row in fold_rows
            if row.roc_auc is not None
        ]

        precisions = [
            row.precision
            for row in fold_rows
        ]

        recalls = [
            row.recall
            for row in fold_rows
        ]

        f1_values = [
            row.f1_score
            for row in fold_rows
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

        made_shots = sum(
            self._normalize_result(
                record.get(
                    "result",
                    "",
                )
            )
            == "made"
            for record in participant_records
        )

        missed_shots = (
            len(
                participant_records
            )
            - made_shots
        )

        average_balanced_accuracy = float(
            mean(
                balanced_accuracies
            )
        )

        return ParticipantWithinPlayerSummary(
            participant_id=participant_id,

            shot_count=len(
                participant_records
            ),
            made_shots=made_shots,
            missed_shots=missed_shots,
            feature_count=feature_count,
            fold_count=len(
                fold_rows
            ),

            mean_balanced_accuracy=(
                average_balanced_accuracy
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

            within_player_status=(
                self._status_label(
                    average_balanced_accuracy
                )
            ),
        )

    # =====================================================
    # FEATURE SUMMARY
    # =====================================================

    @staticmethod
    def build_feature_summaries(
        importance_rows: list[
            WithinPlayerFeatureImportanceRow
        ],
    ) -> list[
        WithinPlayerFeatureSummaryRow
    ]:
        grouped: dict[
            tuple[
                str,
                str,
            ],
            list[
                WithinPlayerFeatureImportanceRow
            ],
        ] = {}

        for row in importance_rows:
            grouped.setdefault(
                (
                    row.participant_id,
                    row.feature,
                ),
                [],
            ).append(
                row
            )

        output: list[
            WithinPlayerFeatureSummaryRow
        ] = []

        for (
            participant_id,
            feature,
        ), rows in grouped.items():
            importances = [
                row.importance
                for row in rows
            ]

            ranks = [
                row.fold_rank
                for row in rows
            ]

            output.append(
                WithinPlayerFeatureSummaryRow(
                    participant_id=participant_id,
                    feature=feature,

                    mean_importance=float(
                        mean(
                            importances
                        )
                    ),
                    standard_deviation=(
                        float(
                            pstdev(
                                importances
                            )
                        )
                        if len(
                            importances
                        )
                        > 1
                        else 0.0
                    ),
                    minimum_importance=float(
                        min(
                            importances
                        )
                    ),
                    maximum_importance=float(
                        max(
                            importances
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
                    fold_count=len(
                        rows
                    ),
                )
            )

        output.sort(
            key=lambda row: (
                row.participant_id,
                -row.mean_importance,
                row.mean_rank,
                row.feature,
            )
        )

        return output

    # =====================================================
    # OVERALL SUMMARY
    # =====================================================

    def build_overall_summary(
        self,
        participant_summaries: list[
            ParticipantWithinPlayerSummary
        ],
        prediction_rows: list[
            WithinPlayerPredictionRow
        ],
        feature_count: int,
        total_shots: int,
        total_folds: int,
    ) -> OverallWithinPlayerSummary:
        balanced_accuracies = [
            row.mean_balanced_accuracy
            for row in participant_summaries
        ]

        auc_values = [
            row.mean_roc_auc
            for row in participant_summaries
            if row.mean_roc_auc is not None
        ]

        precisions = [
            row.mean_precision
            for row in participant_summaries
        ]

        recalls = [
            row.mean_recall
            for row in participant_summaries
        ]

        f1_values = [
            row.mean_f1_score
            for row in participant_summaries
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

        mean_balanced_accuracy = float(
            mean(
                balanced_accuracies
            )
        )

        return OverallWithinPlayerSummary(
            participant_count=len(
                participant_summaries
            ),
            total_shots=total_shots,
            total_predictions=len(
                prediction_rows
            ),
            feature_count=feature_count,
            total_folds=total_folds,

            mean_participant_balanced_accuracy=(
                mean_balanced_accuracy
            ),
            standard_deviation_participant_balanced_accuracy=(
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

            mean_participant_roc_auc=(
                float(
                    mean(
                        auc_values
                    )
                )
                if auc_values
                else None
            ),
            standard_deviation_participant_roc_auc=(
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

            pooled_balanced_accuracy=(
                pooled_balanced_accuracy
            ),
            pooled_roc_auc=(
                pooled_auc
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

            general_status=(
                self._status_label(
                    mean_balanced_accuracy
                )
            ),
        )

    @staticmethod
    def _status_label(
        balanced_accuracy: float,
    ) -> str:
        if balanced_accuracy >= 0.70:
            return "Promising personal signal"

        if balanced_accuracy >= 0.60:
            return "Moderate personal signal"

        if balanced_accuracy >= 0.55:
            return "Weak personal signal"

        return "Near chance"

    # =====================================================
    # CHARTS
    # =====================================================

    def create_performance_chart(
        self,
        summaries: list[
            ParticipantWithinPlayerSummary
        ],
    ) -> Path:
        labels = [
            row.participant_id
            for row in summaries
        ]

        balanced = [
            row.mean_balanced_accuracy
            for row in summaries
        ]

        auc_values = [
            (
                np.nan
                if row.mean_roc_auc is None
                else row.mean_roc_auc
            )
            for row in summaries
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
            "Within-Player Prediction Performance"
        )

        axis.legend()

        axis.grid(
            visible=True,
            axis="y",
            alpha=0.25,
        )

        figure.tight_layout()

        figure.savefig(
            self.performance_chart_file,
            dpi=170,
            bbox_inches="tight",
        )

        plt.close(
            figure
        )

        return self.performance_chart_file

    def create_confusion_chart(
        self,
        summaries: list[
            ParticipantWithinPlayerSummary
        ],
    ) -> Path:
        labels = [
            row.participant_id
            for row in summaries
        ]

        true_negatives = [
            row.total_true_negatives
            for row in summaries
        ]

        false_positives = [
            row.total_false_positives
            for row in summaries
        ]

        false_negatives = [
            row.total_false_negatives
            for row in summaries
        ]

        true_positives = [
            row.total_true_positives
            for row in summaries
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
            "Prediction count across repeats"
        )

        axis.set_title(
            "Within-Player Confusion Counts"
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
        overall_summary: OverallWithinPlayerSummary,
        participant_summaries: list[
            ParticipantWithinPlayerSummary
        ],
        feature_summary_rows: list[
            WithinPlayerFeatureSummaryRow
        ],
    ) -> Path:
        lines = [
            "=" * 84,
            "BASKETBALL BIOMECHANICS RESEARCH STUDIO",
            "WITHIN-PLAYER VALIDATION REPORT",
            "=" * 84,
            "",
            "VALIDATION DESIGN",
            "-" * 84,
            (
                "Each participant was modeled separately using repeated "
                "stratified cross-validation."
            ),
            (
                "Training and validation shots came from the same player, "
                "but no shot appeared in both sets within a fold."
            ),
            (
                "Absolute frame-number features were excluded: "
                f"{self.exclude_absolute_frames}"
            ),
            "",
            "OVERALL RESULTS",
            "-" * 84,
            (
                "Participants: "
                f"{overall_summary.participant_count}"
            ),
            (
                "Shots: "
                f"{overall_summary.total_shots}"
            ),
            (
                "Repeated held-out predictions: "
                f"{overall_summary.total_predictions}"
            ),
            (
                "Total folds: "
                f"{overall_summary.total_folds}"
            ),
            (
                "Mean participant balanced accuracy: "
                f"{overall_summary.mean_participant_balanced_accuracy:.3f}"
            ),
            (
                "Mean participant ROC AUC: "
                f"{overall_summary.mean_participant_roc_auc:.3f}"
                if overall_summary.mean_participant_roc_auc
                is not None
                else "Mean participant ROC AUC: N/A"
            ),
            (
                "Pooled balanced accuracy: "
                f"{overall_summary.pooled_balanced_accuracy:.3f}"
            ),
            (
                "Pooled ROC AUC: "
                f"{overall_summary.pooled_roc_auc:.3f}"
                if overall_summary.pooled_roc_auc
                is not None
                else "Pooled ROC AUC: N/A"
            ),
            (
                "Status: "
                f"{overall_summary.general_status}"
            ),
            "",
            "PARTICIPANT RESULTS",
            "-" * 84,
        ]

        for summary in participant_summaries:
            lines.append(
                (
                    f"{summary.participant_id}: "
                    f"shots={summary.shot_count}, "
                    f"makes={summary.made_shots}, "
                    f"misses={summary.missed_shots}, "
                    f"balanced accuracy="
                    f"{summary.mean_balanced_accuracy:.3f}, "
                    f"ROC AUC="
                    f"{summary.mean_roc_auc:.3f}"
                    if summary.mean_roc_auc
                    is not None
                    else (
                        f"{summary.participant_id}: "
                        f"shots={summary.shot_count}, "
                        f"makes={summary.made_shots}, "
                        f"misses={summary.missed_shots}, "
                        f"balanced accuracy="
                        f"{summary.mean_balanced_accuracy:.3f}, "
                        "ROC AUC=N/A"
                    )
                )
            )

        lines.extend(
            [
                "",
                "TOP PERSONAL FEATURES BY PARTICIPANT",
                "-" * 84,
            ]
        )

        for summary in participant_summaries:
            lines.append(
                ""
            )

            lines.append(
                summary.participant_id
            )

            participant_features = [
                row
                for row in feature_summary_rows
                if row.participant_id
                == summary.participant_id
            ]

            for rank, row in enumerate(
                participant_features[
                    :15
                ],
                start=1,
            ):
                lines.append(
                    (
                        f"  {rank}. {row.feature}: "
                        f"importance={row.mean_importance:.5f}, "
                        f"mean rank={row.mean_rank:.1f}, "
                        f"top-10 folds="
                        f"{row.top_10_fold_count}/"
                        f"{row.fold_count}"
                    )
                )

        lines.extend(
            [
                "",
                "IMPORTANT INTERPRETATION",
                "-" * 84,
                (
                    "Strong within-player results would support building a "
                    "personal baseline from that player's own shots."
                ),
                (
                    "Weak results would mean the current features still do "
                    "not capture the small differences separating that "
                    "player's makes and misses."
                ),
                (
                    "Feature importance is player-specific and must not be "
                    "treated as a universal free-throw rule."
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
    # HELPERS
    # =====================================================

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


def _run_within_player_validation_test() -> None:
    studio = WithinPlayerValidationStudio(
        exclude_absolute_frames=True,
    )

    (
        overall_summary,
        participant_summaries,
        fold_rows,
        prediction_rows,
        feature_summary_rows,
    ) = studio.run()

    assert len(
        participant_summaries
    ) > 0

    assert len(
        fold_rows
    ) > 0

    assert len(
        prediction_rows
    ) > 0

    assert len(
        feature_summary_rows
    ) > 0

    assert studio.fold_file.exists()
    assert studio.predictions_file.exists()
    assert studio.fold_importance_file.exists()
    assert studio.participant_summary_file.exists()
    assert studio.feature_summary_file.exists()
    assert studio.overall_summary_file.exists()
    assert studio.performance_chart_file.exists()
    assert studio.confusion_chart_file.exists()
    assert studio.report_file.exists()

    print()
    print(
        "WITHIN-PLAYER VALIDATION TEST PASSED"
    )

    print(
        "Participants analyzed: "
        f"{overall_summary.participant_count}"
    )

    print(
        "Total shots: "
        f"{overall_summary.total_shots}"
    )

    print(
        "Total folds: "
        f"{overall_summary.total_folds}"
    )

    print(
        "Repeated predictions: "
        f"{overall_summary.total_predictions}"
    )

    print(
        "Mean participant balanced accuracy: "
        f"{overall_summary.mean_participant_balanced_accuracy:.3f}"
    )

    print(
        "Mean participant ROC AUC: "
        f"{overall_summary.mean_participant_roc_auc:.3f}"
        if overall_summary.mean_participant_roc_auc
        is not None
        else "Mean participant ROC AUC: N/A"
    )

    print(
        "Pooled balanced accuracy: "
        f"{overall_summary.pooled_balanced_accuracy:.3f}"
    )

    print(
        "Pooled ROC AUC: "
        f"{overall_summary.pooled_roc_auc:.3f}"
        if overall_summary.pooled_roc_auc
        is not None
        else "Pooled ROC AUC: N/A"
    )

    print(
        "Status: "
        f"{overall_summary.general_status}"
    )

    print()

    print(
        "Fold metrics: "
        f"{studio.fold_file.resolve()}"
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
        "Participant summary: "
        f"{studio.participant_summary_file.resolve()}"
    )

    print(
        "Feature summary: "
        f"{studio.feature_summary_file.resolve()}"
    )

    print(
        "Overall summary: "
        f"{studio.overall_summary_file.resolve()}"
    )

    print(
        "Performance chart: "
        f"{studio.performance_chart_file.resolve()}"
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
        "Participant results:"
    )

    for summary in participant_summaries:
        auc_text = (
            "N/A"
            if summary.mean_roc_auc
            is None
            else f"{summary.mean_roc_auc:.3f}"
        )

        print(
            f"{summary.participant_id}: "
            f"balanced accuracy "
            f"{summary.mean_balanced_accuracy:.3f}, "
            f"ROC AUC {auc_text}, "
            f"status "
            f"{summary.within_player_status}"
        )


if __name__ == "__main__":
    _run_within_player_validation_test()