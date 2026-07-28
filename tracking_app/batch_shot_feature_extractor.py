from __future__ import annotations

import csv
import json
from dataclasses import asdict, fields
from pathlib import Path
from statistics import mean

from tracking_app.shot_feature_extractor import (
    ShotFeatureExtractor,
    ShotFeatureRecord,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_FOLDER = (
    PROJECT_ROOT
    / "SPL-Open-Data-main"
    / "basketball"
    / "freethrow"
    / "data"
)

DEFAULT_OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "all_shot_features.csv"
)

DEFAULT_COMPLETENESS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "shot_feature_completeness.csv"
)


class BatchShotFeatureExtractor:
    """
    Run the single-shot feature extractor across the full dataset.

    Every trial becomes one row. Failed trials are preserved with their
    error messages so the batch can finish without silently losing data.
    """

    def __init__(
        self,
        data_folder: Path = DEFAULT_DATA_FOLDER,
        output_file: Path = DEFAULT_OUTPUT_FILE,
        completeness_file: Path = DEFAULT_COMPLETENESS_FILE,
        shooting_side: str = "RIGHT",
        maximum_trials: int | None = None,
    ) -> None:
        self.data_folder = Path(data_folder)
        self.output_file = Path(output_file)
        self.completeness_file = Path(completeness_file)

        self.shooting_side = shooting_side.strip().upper()

        if self.shooting_side not in {
            "RIGHT",
            "LEFT",
        }:
            raise ValueError(
                "shooting_side must be RIGHT or LEFT."
            )

        self.maximum_trials = (
            None
            if maximum_trials is None
            else max(
                1,
                int(maximum_trials),
            )
        )

    # =====================================================
    # PUBLIC WORKFLOW
    # =====================================================

    def run(
        self,
    ) -> list[ShotFeatureRecord]:
        """
        Discover trials, extract features, and export outputs.
        """

        trial_files = self.discover_trials()

        if not trial_files:
            searched_text = "\n".join(
                f"- {folder}"
                for folder in self._candidate_data_folders()
            )

            raise FileNotFoundError(
                "No BB_FT_*.json files were found. "
                "Searched recursively in:\n"
                f"{searched_text}"
            )

        if self.maximum_trials is not None:
            trial_files = trial_files[
                :self.maximum_trials
            ]

        records: list[
            ShotFeatureRecord
        ] = []

        for index, trial_file in enumerate(
            trial_files,
            start=1,
        ):
            try:
                with trial_file.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    trial_data = json.load(
                        file
                    )

                record = ShotFeatureExtractor(
                    trial_file=trial_file,
                    trial_data=trial_data,
                    shooting_side=self.shooting_side,
                ).extract()

            except Exception as error:
                record = self._emergency_failure_record(
                    trial_file=trial_file,
                    error=error,
                )

            records.append(record)

            if (
                index == 1
                or index % 25 == 0
                or index == len(trial_files)
            ):
                print(
                    f"Processed {index} / "
                    f"{len(trial_files)} trials"
                )

        self.export_master_csv(
            records
        )

        self.export_completeness_csv(
            records
        )

        return records

    # =====================================================
    # DISCOVERY
    # =====================================================

    def discover_trials(
        self,
    ) -> list[Path]:
        """
        Find all free-throw JSON files recursively.
        """

        seen: set[Path] = set()
        trial_files: list[Path] = []

        for folder in self._candidate_data_folders():
            if not folder.exists():
                continue

            for trial_file in sorted(
                folder.rglob(
                    "BB_FT_*.json"
                )
            ):
                resolved = trial_file.resolve()

                if resolved in seen:
                    continue

                seen.add(resolved)
                trial_files.append(
                    trial_file
                )

        trial_files.sort(
            key=lambda path: (
                path.name,
                str(path),
            )
        )

        return trial_files

    def _candidate_data_folders(
        self,
    ) -> list[Path]:
        """
        Return likely dataset locations.
        """

        repository_root = (
            PROJECT_ROOT
            / "SPL-Open-Data-main"
        )

        candidates = [
            self.data_folder,
            repository_root
            / "basketball"
            / "freethrow"
            / "data",
            repository_root
            / "SPL-Open-Data-main"
            / "basketball"
            / "freethrow"
            / "data",
            PROJECT_ROOT,
        ]

        unique: list[Path] = []
        seen: set[Path] = set()

        for candidate in candidates:
            resolved = candidate.resolve()

            if resolved in seen:
                continue

            seen.add(resolved)
            unique.append(candidate)

        return unique

    # =====================================================
    # MASTER CSV
    # =====================================================

    def export_master_csv(
        self,
        records: list[ShotFeatureRecord],
    ) -> Path:
        """
        Export one row per shot with a stable field order.
        """

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        field_names = [
            field.name
            for field in fields(
                ShotFeatureRecord
            )
        ]

        with self.output_file.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=field_names,
            )

            writer.writeheader()

            for record in records:
                writer.writerow(
                    asdict(record)
                )

        return self.output_file

    # =====================================================
    # FEATURE COMPLETENESS
    # =====================================================

    def export_completeness_csv(
        self,
        records: list[ShotFeatureRecord],
    ) -> Path:
        """
        Report how often each feature is populated.
        """

        self.completeness_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        field_names = [
            field.name
            for field in fields(
                ShotFeatureRecord
            )
        ]

        total_records = len(records)

        with self.completeness_file.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "feature",
                    "populated_count",
                    "missing_count",
                    "completeness_percentage",
                ],
            )

            writer.writeheader()

            for field_name in field_names:
                populated = sum(
                    self._is_populated(
                        getattr(
                            record,
                            field_name,
                        )
                    )
                    for record in records
                )

                missing = (
                    total_records
                    - populated
                )

                completeness = (
                    populated
                    / total_records
                    * 100.0
                    if total_records > 0
                    else 0.0
                )

                writer.writerow(
                    {
                        "feature": field_name,
                        "populated_count": populated,
                        "missing_count": missing,
                        "completeness_percentage": completeness,
                    }
                )

        return self.completeness_file

    # =====================================================
    # FAILURE FALLBACK
    # =====================================================

    def _emergency_failure_record(
        self,
        trial_file: Path,
        error: Exception,
    ) -> ShotFeatureRecord:
        """
        Build a valid failure row if loading fails before extraction.
        """

        values: dict[
            str,
            object,
        ] = {
            field.name: None
            for field in fields(
                ShotFeatureRecord
            )
        }

        values.update(
            {
                "participant_id": "Unknown",
                "trial_id": trial_file.stem,
                "result": "unknown",
                "trial_file": trial_file.name,
                "sampling_rate": 0.0,
                "total_frames": 0,
                "propulsion_onset_sequence": "",
                "peak_velocity_sequence": "",
                "authoritative_event_count": 0,
                "analysis_status": "failed",
                "analysis_error": (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            }
        )

        return ShotFeatureRecord(
            **values
        )

    # =====================================================
    # HELPERS
    # =====================================================

    @staticmethod
    def _is_populated(
        value: object,
    ) -> bool:
        """
        Treat None and empty strings as missing.
        """

        return value not in {
            None,
            "",
        }


def _average_populated_fields(
    records: list[ShotFeatureRecord],
) -> float:
    """
    Calculate average populated fields per record.
    """

    if not records:
        return 0.0

    field_names = [
        field.name
        for field in fields(
            ShotFeatureRecord
        )
    ]

    counts = []

    for record in records:
        counts.append(
            sum(
                BatchShotFeatureExtractor
                ._is_populated(
                    getattr(
                        record,
                        field_name,
                    )
                )
                for field_name in field_names
            )
        )

    return float(
        mean(counts)
    )


def _run_batch_feature_test() -> None:
    """
    Run the full dataset feature extraction.
    """

    extractor = BatchShotFeatureExtractor(
        maximum_trials=None,
        shooting_side="RIGHT",
    )

    records = extractor.run()

    successful = [
        record
        for record in records
        if record.analysis_status
        == "success"
    ]

    failed = [
        record
        for record in records
        if record.analysis_status
        == "failed"
    ]

    participants = {
        record.participant_id
        for record in successful
    }

    made_count = sum(
        str(record.result).lower()
        == "made"
        for record in successful
    )

    missed_count = sum(
        str(record.result).lower()
        == "missed"
        for record in successful
    )

    assert len(records) > 0
    assert extractor.output_file.exists()
    assert extractor.output_file.stat().st_size > 0
    assert extractor.completeness_file.exists()
    assert extractor.completeness_file.stat().st_size > 0

    print()
    print(
        "BATCH SHOT FEATURE EXTRACTION TEST PASSED"
    )

    print(
        f"Trials discovered: {len(records)}"
    )

    print(
        f"Successful: {len(successful)}"
    )

    print(
        f"Failed: {len(failed)}"
    )

    print(
        "Participants represented: "
        f"{len(participants)}"
    )

    print(
        f"Made shots: {made_count}"
    )

    print(
        f"Missed shots: {missed_count}"
    )

    print(
        "Average populated fields: "
        f"{_average_populated_fields(records):.1f} / "
        f"{len(fields(ShotFeatureRecord))}"
    )

    print()

    print(
        "Master feature CSV: "
        f"{extractor.output_file.resolve()}"
    )

    print(
        "Completeness CSV: "
        f"{extractor.completeness_file.resolve()}"
    )

    if failed:
        print()
        print("First 10 failed trials:")

        for record in failed[
            :10
        ]:
            print(
                f"- {record.trial_file}: "
                f"{record.analysis_error}"
            )


if __name__ == "__main__":
    _run_batch_feature_test()