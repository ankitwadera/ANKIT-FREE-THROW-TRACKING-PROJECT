from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as error:
    raise ImportError(
        "Feature Importance Studio requires scikit-learn. "
        "Install it with: "
        ".venv\\Scripts\\python.exe -m pip install scikit-learn"
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
    / "feature_importance"
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


@dataclass(frozen=True)
class FeatureImportanceRow:
    feature: str
    valid_count: int
    completeness_percentage: float

    standardized_difference: float | None
    absolute_standardized_difference: float | None

    point_biserial_correlation: float | None
    absolute_point_biserial_correlation: float | None

    mutual_information_mean: float | None
    mutual_information_standard_deviation: float | None

    random_forest_importance_mean: float | None
    random_forest_importance_standard_deviation: float | None

    permutation_importance_mean: float | None
    permutation_importance_standard_deviation: float | None

    consensus_score: float
    consensus_rank: int

    method_count: int
    stability_score: float


@dataclass(frozen=True)
class ModelPerformanceRow:
    repeat_index: int
    fold_index: int
    training_rows: int
    validation_rows: int
    balanced_accuracy: float
    roc_auc: float | None


class FeatureImportanceStudio:
    """
    Rank objective free-throw features using multiple methods.

    Methods
    -------
    1. Standardized made-versus-missed difference
    2. Point-biserial correlation with outcome
    3. Mutual information
    4. Random-forest impurity importance
    5. Permutation importance

    The final consensus score combines normalized ranks across methods.
    This identifies useful candidate features in this dataset. It does
    not prove causation or establish ideal free-throw mechanics.
    """

    def __init__(
        self,
        input_file: Path = DEFAULT_INPUT_FILE,
        output_folder: Path = DEFAULT_OUTPUT_FOLDER,
        random_seed: int = 42,
        cross_validation_splits: int = 5,
        cross_validation_repeats: int = 4,
        forest_trees: int = 400,
        permutation_repeats: int = 12,
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
        self.permutation_repeats = max(
            5,
            int(permutation_repeats),
        )

        self.importance_file = (
            self.output_folder
            / "feature_importance.csv"
        )
        self.performance_file = (
            self.output_folder
            / "model_performance.csv"
        )
        self.bar_chart_file = (
            self.output_folder
            / "feature_importance_bar_chart.png"
        )
        self.report_file = (
            self.output_folder
            / "feature_importance_report.txt"
        )

    # =====================================================
    # PUBLIC WORKFLOW
    # =====================================================

    def run(
        self,
    ) -> tuple[
        list[FeatureImportanceRow],
        list[ModelPerformanceRow],
    ]:
        records = self.load_records()

        successful = [
            record
            for record in records
            if self._normalize_text(
                record.get("analysis_status", "")
            )
            == "success"
        ]

        if not successful:
            raise RuntimeError(
                "No successful shot-feature rows were found."
            )

        feature_names = self.identify_numeric_features(
            successful
        )

        if not feature_names:
            raise RuntimeError(
                "No usable numeric features were found."
            )

        matrix, target = self.build_feature_matrix(
            successful,
            feature_names,
        )

        (
            mutual_information_runs,
            forest_importance_runs,
            permutation_importance_runs,
            performance_rows,
        ) = self.run_cross_validation(
            matrix=matrix,
            target=target,
            feature_names=feature_names,
        )

        effect_sizes = self.calculate_effect_sizes(
            matrix,
            target,
        )

        point_biserial = self.calculate_point_biserial(
            matrix,
            target,
        )

        completeness = self.calculate_feature_completeness(
            successful,
            feature_names,
        )

        rows = self.build_consensus_rows(
            feature_names=feature_names,
            completeness=completeness,
            effect_sizes=effect_sizes,
            point_biserial=point_biserial,
            mutual_information_runs=mutual_information_runs,
            forest_importance_runs=forest_importance_runs,
            permutation_importance_runs=permutation_importance_runs,
        )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.export_dataclass_rows(
            self.importance_file,
            rows,
        )

        self.export_dataclass_rows(
            self.performance_file,
            performance_rows,
        )

        self.create_bar_chart(rows)
        self.export_text_report(
            rows=rows,
            performance_rows=performance_rows,
            total_rows=len(successful),
            feature_count=len(feature_names),
        )

        return rows, performance_rows

    # =====================================================
    # DATA
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
            return list(csv.DictReader(file))

    def identify_numeric_features(
        self,
        records: list[dict[str, str]],
    ) -> list[str]:
        fieldnames = list(records[0])
        output: list[str] = []

        for field_name in fieldnames:
            if field_name in NON_FEATURE_FIELDS:
                continue

            values = [
                self._to_finite_float(
                    record.get(field_name, "")
                )
                for record in records
            ]

            finite_values = [
                value
                for value in values
                if value is not None
            ]

            if len(finite_values) < 30:
                continue

            if len(set(finite_values)) < 2:
                continue

            output.append(field_name)

        return output

    def build_feature_matrix(
        self,
        records: list[dict[str, str]],
        feature_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        rows: list[list[float]] = []
        target: list[int] = []

        for record in records:
            result = self._normalize_result(
                record.get("result", "")
            )

            if result not in {"made", "missed"}:
                continue

            row = []

            for feature in feature_names:
                value = self._to_finite_float(
                    record.get(feature, "")
                )

                row.append(
                    np.nan
                    if value is None
                    else value
                )

            rows.append(row)
            target.append(
                1 if result == "made" else 0
            )

        matrix = np.asarray(
            rows,
            dtype=float,
        )

        target_array = np.asarray(
            target,
            dtype=int,
        )

        if matrix.shape[0] < 50:
            raise RuntimeError(
                "At least 50 made/missed rows are required."
            )

        if len(np.unique(target_array)) < 2:
            raise RuntimeError(
                "Both made and missed shots are required."
            )

        return matrix, target_array

    # =====================================================
    # SIMPLE METHODS
    # =====================================================

    @staticmethod
    def calculate_effect_sizes(
        matrix: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        values = []

        for column_index in range(
            matrix.shape[1]
        ):
            column = matrix[:, column_index]

            made = column[
                (target == 1)
                & np.isfinite(column)
            ]
            missed = column[
                (target == 0)
                & np.isfinite(column)
            ]

            if (
                len(made) < 2
                or len(missed) < 2
            ):
                values.append(np.nan)
                continue

            made_variance = np.var(
                made,
                ddof=1,
            )
            missed_variance = np.var(
                missed,
                ddof=1,
            )

            degrees_of_freedom = (
                len(made)
                + len(missed)
                - 2
            )

            pooled_variance = (
                (
                    (len(made) - 1)
                    * made_variance
                )
                + (
                    (len(missed) - 1)
                    * missed_variance
                )
            ) / degrees_of_freedom

            if pooled_variance <= 0:
                values.append(np.nan)
                continue

            values.append(
                (
                    np.mean(made)
                    - np.mean(missed)
                )
                / np.sqrt(pooled_variance)
            )

        return np.asarray(
            values,
            dtype=float,
        )

    @staticmethod
    def calculate_point_biserial(
        matrix: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        values = []

        for column_index in range(
            matrix.shape[1]
        ):
            column = matrix[:, column_index]
            valid = np.isfinite(column)

            if (
                np.sum(valid) < 3
                or np.std(column[valid]) == 0
            ):
                values.append(np.nan)
                continue

            correlation = np.corrcoef(
                column[valid],
                target[valid],
            )[0, 1]

            values.append(
                correlation
                if np.isfinite(correlation)
                else np.nan
            )

        return np.asarray(
            values,
            dtype=float,
        )

    # =====================================================
    # CROSS-VALIDATION METHODS
    # =====================================================

    def run_cross_validation(
        self,
        matrix: np.ndarray,
        target: np.ndarray,
        feature_names: list[str],
    ) -> tuple[
        list[np.ndarray],
        list[np.ndarray],
        list[np.ndarray],
        list[ModelPerformanceRow],
    ]:
        del feature_names

        splitter = RepeatedStratifiedKFold(
            n_splits=self.cross_validation_splits,
            n_repeats=self.cross_validation_repeats,
            random_state=self.random_seed,
        )

        mutual_information_runs: list[
            np.ndarray
        ] = []
        forest_importance_runs: list[
            np.ndarray
        ] = []
        permutation_importance_runs: list[
            np.ndarray
        ] = []
        performance_rows: list[
            ModelPerformanceRow
        ] = []

        total_fold_index = 0

        for train_indices, validation_indices in splitter.split(
            matrix,
            target,
        ):
            repeat_index = (
                total_fold_index
                // self.cross_validation_splits
                + 1
            )
            fold_index = (
                total_fold_index
                % self.cross_validation_splits
                + 1
            )
            total_fold_index += 1

            training_matrix = matrix[
                train_indices
            ]
            validation_matrix = matrix[
                validation_indices
            ]
            training_target = target[
                train_indices
            ]
            validation_target = target[
                validation_indices
            ]

            imputer = SimpleImputer(
                strategy="median"
            )

            training_imputed = imputer.fit_transform(
                training_matrix
            )
            validation_imputed = imputer.transform(
                validation_matrix
            )

            mutual_information = mutual_info_classif(
                training_imputed,
                training_target,
                discrete_features=False,
                random_state=(
                    self.random_seed
                    + total_fold_index
                ),
            )

            model = RandomForestClassifier(
                n_estimators=self.forest_trees,
                random_state=(
                    self.random_seed
                    + total_fold_index
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

            predictions = model.predict(
                validation_imputed
            )

            probabilities = model.predict_proba(
                validation_imputed
            )[:, 1]

            balanced_accuracy = balanced_accuracy_score(
                validation_target,
                predictions,
            )

            roc_auc = None

            if len(
                np.unique(validation_target)
            ) == 2:
                roc_auc = float(
                    roc_auc_score(
                        validation_target,
                        probabilities,
                    )
                )

            permutation = permutation_importance(
                model,
                validation_imputed,
                validation_target,
                n_repeats=self.permutation_repeats,
                random_state=(
                    self.random_seed
                    + total_fold_index
                ),
                scoring="balanced_accuracy",
                n_jobs=-1,
            )

            mutual_information_runs.append(
                np.asarray(
                    mutual_information,
                    dtype=float,
                )
            )

            forest_importance_runs.append(
                np.asarray(
                    model.feature_importances_,
                    dtype=float,
                )
            )

            permutation_importance_runs.append(
                np.asarray(
                    permutation.importances_mean,
                    dtype=float,
                )
            )

            performance_rows.append(
                ModelPerformanceRow(
                    repeat_index=repeat_index,
                    fold_index=fold_index,
                    training_rows=len(
                        train_indices
                    ),
                    validation_rows=len(
                        validation_indices
                    ),
                    balanced_accuracy=float(
                        balanced_accuracy
                    ),
                    roc_auc=roc_auc,
                )
            )

        return (
            mutual_information_runs,
            forest_importance_runs,
            permutation_importance_runs,
            performance_rows,
        )

    # =====================================================
    # CONSENSUS
    # =====================================================

    def build_consensus_rows(
        self,
        feature_names: list[str],
        completeness: np.ndarray,
        effect_sizes: np.ndarray,
        point_biserial: np.ndarray,
        mutual_information_runs: list[np.ndarray],
        forest_importance_runs: list[np.ndarray],
        permutation_importance_runs: list[np.ndarray],
    ) -> list[FeatureImportanceRow]:
        mutual_matrix = np.vstack(
            mutual_information_runs
        )
        forest_matrix = np.vstack(
            forest_importance_runs
        )
        permutation_matrix = np.vstack(
            permutation_importance_runs
        )

        mutual_mean = np.mean(
            mutual_matrix,
            axis=0,
        )
        mutual_std = np.std(
            mutual_matrix,
            axis=0,
        )

        forest_mean = np.mean(
            forest_matrix,
            axis=0,
        )
        forest_std = np.std(
            forest_matrix,
            axis=0,
        )

        permutation_mean = np.mean(
            permutation_matrix,
            axis=0,
        )
        permutation_std = np.std(
            permutation_matrix,
            axis=0,
        )

        methods = [
            np.abs(effect_sizes),
            np.abs(point_biserial),
            mutual_mean,
            forest_mean,
            np.maximum(
                permutation_mean,
                0.0,
            ),
        ]

        normalized_methods = [
            self._normalize_scores(
                method
            )
            for method in methods
        ]

        consensus = np.nanmean(
            np.vstack(
                normalized_methods
            ),
            axis=0,
        ) * 100.0

        stability = self.calculate_stability(
            mutual_matrix,
            forest_matrix,
            permutation_matrix,
        )

        order = np.argsort(
            -consensus
        )

        ranks = np.empty_like(
            order
        )
        ranks[
            order
        ] = np.arange(
            1,
            len(order) + 1,
        )

        rows = []

        for index, feature in enumerate(
            feature_names
        ):
            method_count = sum(
                np.isfinite(method[index])
                for method in methods
            )

            rows.append(
                FeatureImportanceRow(
                    feature=feature,
                    valid_count=int(
                        round(
                            completeness[index, 0]
                        )
                    ),
                    completeness_percentage=float(
                        completeness[index, 1]
                    ),

                    standardized_difference=self._optional_float(
                        effect_sizes[index]
                    ),
                    absolute_standardized_difference=self._optional_float(
                        abs(effect_sizes[index])
                    ),

                    point_biserial_correlation=self._optional_float(
                        point_biserial[index]
                    ),
                    absolute_point_biserial_correlation=self._optional_float(
                        abs(point_biserial[index])
                    ),

                    mutual_information_mean=self._optional_float(
                        mutual_mean[index]
                    ),
                    mutual_information_standard_deviation=self._optional_float(
                        mutual_std[index]
                    ),

                    random_forest_importance_mean=self._optional_float(
                        forest_mean[index]
                    ),
                    random_forest_importance_standard_deviation=self._optional_float(
                        forest_std[index]
                    ),

                    permutation_importance_mean=self._optional_float(
                        permutation_mean[index]
                    ),
                    permutation_importance_standard_deviation=self._optional_float(
                        permutation_std[index]
                    ),

                    consensus_score=float(
                        consensus[index]
                    ),
                    consensus_rank=int(
                        ranks[index]
                    ),

                    method_count=method_count,
                    stability_score=float(
                        stability[index]
                    ),
                )
            )

        rows.sort(
            key=lambda row: (
                row.consensus_rank,
                row.feature,
            )
        )

        return rows

    @staticmethod
    def calculate_stability(
        mutual_matrix: np.ndarray,
        forest_matrix: np.ndarray,
        permutation_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Higher means a feature's importance is more stable across folds.
        """

        stability_components = []

        for matrix in (
            mutual_matrix,
            forest_matrix,
            permutation_matrix,
        ):
            average = np.mean(
                np.abs(matrix),
                axis=0,
            )
            deviation = np.std(
                matrix,
                axis=0,
            )

            component = 1.0 / (
                1.0
                + (
                    deviation
                    / (
                        average
                        + 1e-9
                    )
                )
            )

            stability_components.append(
                component
            )

        return np.mean(
            np.vstack(
                stability_components
            ),
            axis=0,
        ) * 100.0

    @staticmethod
    def _normalize_scores(
        values: np.ndarray,
    ) -> np.ndarray:
        output = np.full_like(
            values,
            np.nan,
            dtype=float,
        )

        valid = np.isfinite(
            values
        )

        if not np.any(valid):
            return output

        valid_values = values[
            valid
        ]

        minimum = np.min(
            valid_values
        )
        maximum = np.max(
            valid_values
        )

        if maximum == minimum:
            output[valid] = 0.0
            return output

        output[valid] = (
            valid_values
            - minimum
        ) / (
            maximum
            - minimum
        )

        return output

    # =====================================================
    # COMPLETENESS
    # =====================================================

    def calculate_feature_completeness(
        self,
        records: list[dict[str, str]],
        feature_names: list[str],
    ) -> np.ndarray:
        rows = []

        for feature in feature_names:
            count = sum(
                self._to_finite_float(
                    record.get(feature, "")
                )
                is not None
                for record in records
            )

            percentage = (
                count
                / len(records)
                * 100.0
            )

            rows.append(
                (
                    count,
                    percentage,
                )
            )

        return np.asarray(
            rows,
            dtype=float,
        )

    # =====================================================
    # OUTPUTS
    # =====================================================

    def create_bar_chart(
        self,
        rows: list[FeatureImportanceRow],
    ) -> Path:
        top_rows = rows[
            :25
        ]

        labels = [
            row.feature
            for row in reversed(
                top_rows
            )
        ]

        values = [
            row.consensus_score
            for row in reversed(
                top_rows
            )
        ]

        figure, axis = plt.subplots(
            figsize=(
                13,
                max(
                    8,
                    len(labels)
                    * 0.42,
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
        )

        axis.set_xlabel(
            "Consensus importance score"
        )

        axis.set_title(
            "Top Free-Throw Feature Candidates"
        )

        axis.grid(
            visible=True,
            axis="x",
            alpha=0.25,
        )

        figure.tight_layout()

        figure.savefig(
            self.bar_chart_file,
            dpi=170,
            bbox_inches="tight",
        )

        plt.close(
            figure
        )

        return self.bar_chart_file

    def export_text_report(
        self,
        rows: list[FeatureImportanceRow],
        performance_rows: list[ModelPerformanceRow],
        total_rows: int,
        feature_count: int,
    ) -> Path:
        balanced_accuracies = [
            row.balanced_accuracy
            for row in performance_rows
        ]

        roc_auc_values = [
            row.roc_auc
            for row in performance_rows
            if row.roc_auc is not None
        ]

        lines = [
            "=" * 82,
            "BASKETBALL BIOMECHANICS RESEARCH STUDIO",
            "FEATURE IMPORTANCE REPORT",
            "=" * 82,
            "",
            "DATASET",
            "-" * 82,
            f"Successful made/missed rows: {total_rows}",
            f"Numeric candidate features: {feature_count}",
            (
                "Cross-validation folds completed: "
                f"{len(performance_rows)}"
            ),
            "",
            "MODEL PERFORMANCE",
            "-" * 82,
            (
                "Mean balanced accuracy: "
                f"{mean(balanced_accuracies):.3f}"
            ),
            (
                "Mean ROC AUC: "
                f"{mean(roc_auc_values):.3f}"
                if roc_auc_values
                else "Mean ROC AUC: N/A"
            ),
            "",
            "IMPORTANT INTERPRETATION",
            "-" * 82,
            (
                "This ranking identifies features that appear useful for "
                "distinguishing makes and misses in this dataset."
            ),
            (
                "It does not prove causation, define ideal mechanics, or "
                "guarantee performance on new players."
            ),
            (
                "Raw frame-number features can rank highly because many "
                "events move together with trial timing. Release-relative "
                "timing features are often more interpretable for coaching."
            ),
            "",
            "TOP 30 CONSENSUS FEATURES",
            "-" * 82,
        ]

        for row in rows[
            :30
        ]:
            lines.append(
                (
                    f"{row.consensus_rank}. {row.feature}: "
                    f"score={row.consensus_score:.2f}, "
                    f"stability={row.stability_score:.1f}, "
                    f"effect={self._format_optional(row.standardized_difference)}, "
                    f"point-biserial={self._format_optional(row.point_biserial_correlation)}, "
                    f"MI={self._format_optional(row.mutual_information_mean)}, "
                    f"RF={self._format_optional(row.random_forest_importance_mean)}, "
                    f"permutation={self._format_optional(row.permutation_importance_mean)}"
                )
            )

        self.report_file.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        return self.report_file

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

    @staticmethod
    def _optional_float(
        value: float,
    ) -> float | None:
        if not np.isfinite(
            value
        ):
            return None

        return float(
            value
        )

    @staticmethod
    def _format_optional(
        value: float | None,
    ) -> str:
        if value is None:
            return "N/A"

        return f"{value:.4f}"


def _run_feature_importance_test() -> None:
    studio = FeatureImportanceStudio()

    rows, performance_rows = studio.run()

    assert len(rows) > 0
    assert len(performance_rows) > 0

    assert studio.importance_file.exists()
    assert studio.performance_file.exists()
    assert studio.bar_chart_file.exists()
    assert studio.report_file.exists()

    balanced_accuracies = [
        row.balanced_accuracy
        for row in performance_rows
    ]

    roc_auc_values = [
        row.roc_auc
        for row in performance_rows
        if row.roc_auc is not None
    ]

    print(
        "FEATURE IMPORTANCE STUDIO TEST PASSED"
    )

    print(
        f"Features ranked: {len(rows)}"
    )

    print(
        "Cross-validation folds: "
        f"{len(performance_rows)}"
    )

    print(
        "Mean balanced accuracy: "
        f"{mean(balanced_accuracies):.3f}"
    )

    print(
        "Mean ROC AUC: "
        f"{mean(roc_auc_values):.3f}"
        if roc_auc_values
        else "Mean ROC AUC: N/A"
    )

    print()

    print(
        "Feature importance: "
        f"{studio.importance_file.resolve()}"
    )

    print(
        "Model performance: "
        f"{studio.performance_file.resolve()}"
    )

    print(
        "Bar chart: "
        f"{studio.bar_chart_file.resolve()}"
    )

    print(
        "Text report: "
        f"{studio.report_file.resolve()}"
    )

    print()

    print(
        "Top 15 consensus features:"
    )

    for row in rows[
        :15
    ]:
        print(
            f"{row.consensus_rank}. "
            f"{row.feature}: "
            f"{row.consensus_score:.2f} "
            f"(stability "
            f"{row.stability_score:.1f})"
        )


if __name__ == "__main__":
    _run_feature_importance_test()