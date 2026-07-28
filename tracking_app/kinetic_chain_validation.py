from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tracking_app.analyzer import ShotAnalyzer
from tracking_app.kinetic_chain import KineticChainAnalyzer
from tracking_app.timeseries import TimeSeriesAnalyzer


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
    / "kinetic_chain_validation.csv"
)


@dataclass(frozen=True)
class TrialCandidate:
    """
    Lightweight metadata used before full shot analysis.
    """

    trial_file: Path
    participant_id: str
    trial_id: str
    result: str


@dataclass(frozen=True)
class ValidationRecord:
    """
    One CSV-ready kinetic-chain validation result.
    """

    participant_id: str
    trial_id: str
    result: str
    trial_file: str

    sampling_rate: float
    motion_start_frame: int | None
    release_frame: int | None

    knee_peak_frame: int | None
    hip_peak_frame: int | None
    pelvis_peak_frame: int | None
    shoulder_peak_frame: int | None
    elbow_peak_frame: int | None
    wrist_peak_frame: int | None

    knee_peak_value: float | None
    hip_peak_value: float | None
    pelvis_peak_value: float | None
    shoulder_peak_value: float | None
    elbow_peak_value: float | None
    wrist_peak_value: float | None

    knee_to_hip_ms: float | None
    hip_to_pelvis_ms: float | None
    pelvis_to_shoulder_ms: float | None
    shoulder_to_elbow_ms: float | None
    elbow_to_wrist_ms: float | None
    wrist_to_release_ms: float | None

    detected_sequence: str
    sequence_violations: str
    detected_peak_count: int
    sequence_score: float

    analysis_status: str
    analysis_error: str


class KineticChainValidator:
    """
    Run kinetic-chain analysis across a deterministic validation sample.

    The default sampler aims for:
    - multiple participants;
    - both made and missed shots;
    - no repeated trial files;
    - a reproducible sample order.

    This is a calibration tool. It does not assume the current sequence
    model is already correct.
    """

    def __init__(
        self,
        data_folder: Path = DEFAULT_DATA_FOLDER,
        output_file: Path = DEFAULT_OUTPUT_FILE,
        sample_size: int = 12,
        shooting_side: str = "RIGHT",
    ) -> None:
        self.data_folder = Path(data_folder)
        self.output_file = Path(output_file)
        self.sample_size = max(1, int(sample_size))

        side = shooting_side.strip().upper()

        if side not in {"RIGHT", "LEFT"}:
            raise ValueError(
                "shooting_side must be RIGHT or LEFT."
            )

        self.shooting_side = side

    # =====================================================
    # PUBLIC WORKFLOW
    # =====================================================

    def run(self) -> list[ValidationRecord]:
        """
        Select trials, analyze them, and export a validation CSV.
        """

        candidates = self.discover_trials()

        if not candidates:
            searched_text = "\n".join(
                f"- {folder}"
                for folder in self._candidate_data_folders()
            )

            raise FileNotFoundError(
                "No BB_FT_*.json free-throw trials were found. "
                "Searched recursively in:\n"
                f"{searched_text}"
            )

        selected = self.select_validation_sample(
            candidates
        )

        records = [
            self.analyze_candidate(candidate)
            for candidate in selected
        ]

        self.export_csv(records)

        return records

    # =====================================================
    # TRIAL DISCOVERY
    # =====================================================

    def discover_trials(self) -> list[TrialCandidate]:
        """
        Read lightweight metadata from every JSON trial.

        The search is recursive so it also works when the dataset
        stores trials inside participant folders or when the extracted
        repository contains an additional nested repository folder.
        """

        search_roots = self._candidate_data_folders()

        existing_roots = [
            folder
            for folder in search_roots
            if folder.exists()
        ]

        if not existing_roots:
            searched_text = "\n".join(
                f"- {folder}"
                for folder in search_roots
            )

            raise FileNotFoundError(
                "No candidate free-throw data folder exists. "
                "Searched:\n"
                f"{searched_text}"
            )

        trial_files: list[Path] = []
        seen_files: set[Path] = set()

        for folder in existing_roots:
            for trial_file in sorted(
                folder.rglob("BB_FT_*.json")
            ):
                resolved = trial_file.resolve()

                if resolved in seen_files:
                    continue

                seen_files.add(resolved)
                trial_files.append(trial_file)

        candidates: list[TrialCandidate] = []

        for trial_file in trial_files:
            try:
                with trial_file.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    trial_data = json.load(file)

            except (
                OSError,
                json.JSONDecodeError,
            ):
                continue

            participant_id = str(
                trial_data.get(
                    "participant_id",
                    "Unknown",
                )
            )

            trial_id = str(
                trial_data.get(
                    "trial_id",
                    trial_file.stem,
                )
            )

            result = self._normalize_result(
                trial_data.get(
                    "result",
                    "unknown",
                )
            )

            candidates.append(
                TrialCandidate(
                    trial_file=trial_file,
                    participant_id=participant_id,
                    trial_id=trial_id,
                    result=result,
                )
            )

        return candidates

    def _candidate_data_folders(self) -> list[Path]:
        """
        Return likely dataset locations in priority order.

        The first entry is the configured folder. Additional entries
        handle common ZIP extraction layouts without requiring the user
        to edit source code.
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

        unique_candidates: list[Path] = []
        seen: set[Path] = set()

        for candidate in candidates:
            resolved = candidate.resolve()

            if resolved in seen:
                continue

            seen.add(resolved)
            unique_candidates.append(candidate)

        return unique_candidates

    # =====================================================
    # DETERMINISTIC SAMPLE SELECTION
    # =====================================================

    def select_validation_sample(
        self,
        candidates: list[TrialCandidate],
    ) -> list[TrialCandidate]:
        """
        Build a reproducible participant-diverse make/miss sample.

        Selection proceeds in rounds:
        1. One make per participant where available.
        2. One miss per participant where available.
        3. Remaining trials in sorted order.

        This prevents the first participant from dominating the sample.
        """

        grouped: dict[
            str,
            dict[str, list[TrialCandidate]],
        ] = {}

        for candidate in candidates:
            participant_group = grouped.setdefault(
                candidate.participant_id,
                {
                    "made": [],
                    "missed": [],
                    "unknown": [],
                },
            )

            participant_group.setdefault(
                candidate.result,
                [],
            ).append(candidate)

        for result_groups in grouped.values():
            for result_trials in result_groups.values():
                result_trials.sort(
                    key=lambda item: (
                        item.trial_id,
                        item.trial_file.name,
                    )
                )

        selected: list[TrialCandidate] = []
        selected_paths: set[Path] = set()

        participant_ids = sorted(grouped)

        for desired_result in (
            "made",
            "missed",
        ):
            for participant_id in participant_ids:
                available = grouped[
                    participant_id
                ].get(
                    desired_result,
                    [],
                )

                candidate = self._first_unselected(
                    available,
                    selected_paths,
                )

                if candidate is None:
                    continue

                selected.append(candidate)
                selected_paths.add(
                    candidate.trial_file
                )

                if len(selected) >= self.sample_size:
                    return selected

        for candidate in candidates:
            if candidate.trial_file in selected_paths:
                continue

            selected.append(candidate)
            selected_paths.add(
                candidate.trial_file
            )

            if len(selected) >= self.sample_size:
                break

        return selected

    @staticmethod
    def _first_unselected(
        candidates: Iterable[TrialCandidate],
        selected_paths: set[Path],
    ) -> TrialCandidate | None:
        for candidate in candidates:
            if candidate.trial_file not in selected_paths:
                return candidate

        return None

    # =====================================================
    # ANALYSIS
    # =====================================================

    def analyze_candidate(
        self,
        candidate: TrialCandidate,
    ) -> ValidationRecord:
        """
        Run the complete shot and kinetic-chain pipeline for one trial.
        """

        try:
            with candidate.trial_file.open(
                "r",
                encoding="utf-8",
            ) as file:
                trial_data = json.load(file)

            sampling_rate = self._safe_sampling_rate(
                trial_data.get(
                    "sampling_rate",
                    30,
                )
            )

            shot_analysis = ShotAnalyzer(
                trial_data=trial_data,
                trial_file=candidate.trial_file,
                shooting_wrist=(
                    f"{self.shooting_side}_WRIST"
                ),
            ).analyze()

            time_series = TimeSeriesAnalyzer(
                trial_data=trial_data,
                smoothing_window=5,
            ).analyze()

            kinetic_chain = KineticChainAnalyzer(
                time_series=time_series,
                sampling_rate=sampling_rate,
                motion_start_frame=(
                    shot_analysis.motion_start_frame
                ),
                release_frame=(
                    shot_analysis.release_frame
                ),
                shooting_side=self.shooting_side,
            ).analyze()

            return ValidationRecord(
                participant_id=candidate.participant_id,
                trial_id=candidate.trial_id,
                result=candidate.result,
                trial_file=candidate.trial_file.name,
                sampling_rate=sampling_rate,
                motion_start_frame=(
                    shot_analysis.motion_start_frame
                ),
                release_frame=(
                    shot_analysis.release_frame
                ),
                knee_peak_frame=(
                    kinetic_chain
                    .knee_extension_peak
                    .frame
                ),
                hip_peak_frame=(
                    kinetic_chain
                    .hip_extension_peak
                    .frame
                ),
                pelvis_peak_frame=(
                    kinetic_chain
                    .pelvis_vertical_velocity_peak
                    .frame
                ),
                shoulder_peak_frame=(
                    kinetic_chain
                    .shoulder_extension_peak
                    .frame
                ),
                elbow_peak_frame=(
                    kinetic_chain
                    .elbow_extension_peak
                    .frame
                ),
                wrist_peak_frame=(
                    kinetic_chain
                    .wrist_speed_peak
                    .frame
                ),
                knee_peak_value=(
                    kinetic_chain
                    .knee_extension_peak
                    .value
                ),
                hip_peak_value=(
                    kinetic_chain
                    .hip_extension_peak
                    .value
                ),
                pelvis_peak_value=(
                    kinetic_chain
                    .pelvis_vertical_velocity_peak
                    .value
                ),
                shoulder_peak_value=(
                    kinetic_chain
                    .shoulder_extension_peak
                    .value
                ),
                elbow_peak_value=(
                    kinetic_chain
                    .elbow_extension_peak
                    .value
                ),
                wrist_peak_value=(
                    kinetic_chain
                    .wrist_speed_peak
                    .value
                ),
                knee_to_hip_ms=(
                    kinetic_chain.knee_to_hip_ms
                ),
                hip_to_pelvis_ms=(
                    kinetic_chain.hip_to_pelvis_ms
                ),
                pelvis_to_shoulder_ms=(
                    kinetic_chain
                    .pelvis_to_shoulder_ms
                ),
                shoulder_to_elbow_ms=(
                    kinetic_chain
                    .shoulder_to_elbow_ms
                ),
                elbow_to_wrist_ms=(
                    kinetic_chain
                    .elbow_to_wrist_ms
                ),
                wrist_to_release_ms=(
                    kinetic_chain
                    .wrist_to_release_ms
                ),
                detected_sequence=" -> ".join(
                    kinetic_chain.detected_sequence
                ),
                sequence_violations=" | ".join(
                    kinetic_chain.sequence_violations
                ),
                detected_peak_count=(
                    kinetic_chain
                    .detected_peak_count
                ),
                sequence_score=(
                    kinetic_chain.sequence_score
                ),
                analysis_status="success",
                analysis_error="",
            )

        except Exception as error:
            return ValidationRecord(
                participant_id=candidate.participant_id,
                trial_id=candidate.trial_id,
                result=candidate.result,
                trial_file=candidate.trial_file.name,
                sampling_rate=0.0,
                motion_start_frame=None,
                release_frame=None,
                knee_peak_frame=None,
                hip_peak_frame=None,
                pelvis_peak_frame=None,
                shoulder_peak_frame=None,
                elbow_peak_frame=None,
                wrist_peak_frame=None,
                knee_peak_value=None,
                hip_peak_value=None,
                pelvis_peak_value=None,
                shoulder_peak_value=None,
                elbow_peak_value=None,
                wrist_peak_value=None,
                knee_to_hip_ms=None,
                hip_to_pelvis_ms=None,
                pelvis_to_shoulder_ms=None,
                shoulder_to_elbow_ms=None,
                elbow_to_wrist_ms=None,
                wrist_to_release_ms=None,
                detected_sequence="",
                sequence_violations="",
                detected_peak_count=0,
                sequence_score=0.0,
                analysis_status="failed",
                analysis_error=(
                    f"{type(error).__name__}: {error}"
                ),
            )

    # =====================================================
    # CSV EXPORT
    # =====================================================

    def export_csv(
        self,
        records: list[ValidationRecord],
    ) -> Path:
        """
        Export all validation records to CSV.
        """

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = list(
            ValidationRecord.__dataclass_fields__
        )

        with self.output_file.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for record in records:
                writer.writerow(
                    {
                        field_name: getattr(
                            record,
                            field_name,
                        )
                        for field_name in fieldnames
                    }
                )

        return self.output_file

    # =====================================================
    # HELPERS
    # =====================================================

    @staticmethod
    def _normalize_result(
        value: object,
    ) -> str:
        normalized = (
            str(value)
            .strip()
            .lower()
        )

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
    def _safe_sampling_rate(
        value: object,
    ) -> float:
        try:
            sampling_rate = float(value)
        except (TypeError, ValueError):
            sampling_rate = 30.0

        if sampling_rate <= 0:
            sampling_rate = 30.0

        return sampling_rate


def _average_success_score(
    records: list[ValidationRecord],
) -> float | None:
    successful_scores = [
        record.sequence_score
        for record in records
        if record.analysis_status == "success"
    ]

    if not successful_scores:
        return None

    return sum(successful_scores) / len(
        successful_scores
    )


def _run_validation_test() -> None:
    """
    Run the default deterministic validation sample.
    """

    validator = KineticChainValidator(
        sample_size=12,
        shooting_side="RIGHT",
    )

    records = validator.run()

    successful = [
        record
        for record in records
        if record.analysis_status == "success"
    ]

    failed = [
        record
        for record in records
        if record.analysis_status == "failed"
    ]

    participants = sorted(
        {
            record.participant_id
            for record in records
        }
    )

    made_count = sum(
        record.result == "made"
        for record in records
    )

    missed_count = sum(
        record.result == "missed"
        for record in records
    )

    average_score = _average_success_score(
        records
    )

    assert len(records) > 0
    assert validator.output_file.exists()
    assert validator.output_file.stat().st_size > 0

    print("KINETIC CHAIN VALIDATION TEST PASSED")
    print(
        f"Trials analyzed: {len(records)}"
    )
    print(
        f"Successful: {len(successful)}"
    )
    print(
        f"Failed: {len(failed)}"
    )
    print(
        f"Participants represented: "
        f"{len(participants)}"
    )
    print(
        f"Made shots: {made_count}"
    )
    print(
        f"Missed shots: {missed_count}"
    )

    if average_score is None:
        print(
            "Average sequence score: N/A"
        )
    else:
        print(
            "Average sequence score: "
            f"{average_score:.1f} / 100"
        )

    print(
        "Output file: "
        f"{validator.output_file.resolve()}"
    )

    if failed:
        print()
        print("Failed trials:")

        for record in failed:
            print(
                f"- {record.trial_file}: "
                f"{record.analysis_error}"
            )


if __name__ == "__main__":
    _run_validation_test()