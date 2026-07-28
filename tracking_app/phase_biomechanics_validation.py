from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from tracking_app.analyzer import ShotAnalyzer
from tracking_app.phase_biomechanics import (
    MovementPhase,
    RefinedPhaseBiomechanicsAnalyzer,
)
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
    / "phase_biomechanics_validation.csv"
)


@dataclass(frozen=True)
class TrialCandidate:
    """
    Lightweight metadata for sample selection.
    """

    trial_file: Path
    participant_id: str
    trial_id: str
    result: str


@dataclass(frozen=True)
class PhaseValidationRecord:
    """
    One CSV-ready refined phase-analysis result.
    """

    participant_id: str
    trial_id: str
    result: str
    trial_file: str

    sampling_rate: float
    motion_start_frame: int | None
    dip_frame: int | None
    takeoff_frame: int | None
    release_frame: int | None

    knee_prep_onset: int | None
    knee_propulsion_onset: int | None
    knee_peak_velocity: int | None
    knee_peak_acceleration: int | None
    knee_deceleration: int | None
    knee_end: int | None
    knee_propulsion_duration_ms: float | None

    hip_prep_onset: int | None
    hip_propulsion_onset: int | None
    hip_peak_velocity: int | None
    hip_peak_acceleration: int | None
    hip_deceleration: int | None
    hip_end: int | None
    hip_propulsion_duration_ms: float | None

    pelvis_prep_onset: int | None
    pelvis_propulsion_onset: int | None
    pelvis_peak_velocity: int | None
    pelvis_peak_acceleration: int | None
    pelvis_deceleration: int | None
    pelvis_end: int | None
    pelvis_propulsion_duration_ms: float | None

    shoulder_prep_onset: int | None
    shoulder_propulsion_onset: int | None
    shoulder_peak_velocity: int | None
    shoulder_peak_acceleration: int | None
    shoulder_deceleration: int | None
    shoulder_end: int | None
    shoulder_propulsion_duration_ms: float | None

    elbow_prep_onset: int | None
    elbow_propulsion_onset: int | None
    elbow_peak_velocity: int | None
    elbow_peak_acceleration: int | None
    elbow_deceleration: int | None
    elbow_end: int | None
    elbow_propulsion_duration_ms: float | None

    wrist_prep_onset: int | None
    wrist_propulsion_onset: int | None
    wrist_peak_velocity: int | None
    wrist_peak_acceleration: int | None
    wrist_deceleration: int | None
    wrist_end: int | None
    wrist_propulsion_duration_ms: float | None

    ball_prep_onset: int | None
    ball_propulsion_onset: int | None
    ball_peak_velocity: int | None
    ball_peak_acceleration: int | None
    ball_deceleration: int | None
    ball_end: int | None
    ball_propulsion_duration_ms: float | None

    preparatory_onset_sequence: str
    propulsion_onset_sequence: str
    peak_sequence: str

    detected_prep_count: int
    detected_propulsion_count: int
    detected_peak_count: int

    analysis_status: str
    analysis_error: str


class PhaseBiomechanicsValidator:
    """
    Validate the refined phase detector across a balanced sample.

    The sampler aims for an equal number of made and missed shots and
    rotates through participants to reduce participant dominance.
    """

    def __init__(
        self,
        sample_size: int = 60,
        data_folder: Path = DEFAULT_DATA_FOLDER,
        output_file: Path = DEFAULT_OUTPUT_FILE,
        shooting_side: str = "RIGHT",
    ) -> None:
        self.sample_size = max(
            2,
            int(sample_size),
        )

        self.data_folder = Path(
            data_folder
        )

        self.output_file = Path(
            output_file
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

    # =====================================================
    # PUBLIC WORKFLOW
    # =====================================================

    def run(
        self,
    ) -> list[PhaseValidationRecord]:
        """
        Select a balanced sample, analyze it, and export CSV.
        """

        candidates = self.discover_trials()

        if not candidates:
            searched_text = "\n".join(
                f"- {folder}"
                for folder in self._candidate_data_folders()
            )

            raise FileNotFoundError(
                "No BB_FT_*.json files were found. "
                "Searched recursively in:\n"
                f"{searched_text}"
            )

        selected = self.select_balanced_sample(
            candidates
        )

        records = [
            self.analyze_candidate(
                candidate
            )
            for candidate in selected
        ]

        self.export_csv(records)

        return records

    # =====================================================
    # DISCOVERY
    # =====================================================

    def discover_trials(
        self,
    ) -> list[TrialCandidate]:
        """
        Find all free-throw JSON trials recursively.
        """

        trial_files: list[Path] = []
        seen: set[Path] = set()

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

        candidates: list[TrialCandidate] = []

        for trial_file in trial_files:
            try:
                with trial_file.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    trial_data = json.load(
                        file
                    )

            except (
                OSError,
                json.JSONDecodeError,
            ):
                continue

            candidates.append(
                TrialCandidate(
                    trial_file=trial_file,
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
                    result=self._normalize_result(
                        trial_data.get(
                            "result",
                            "unknown",
                        )
                    ),
                )
            )

        return candidates

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
    # BALANCED SAMPLE
    # =====================================================

    def select_balanced_sample(
        self,
        candidates: list[TrialCandidate],
    ) -> list[TrialCandidate]:
        """
        Select made and missed shots as evenly as possible.
        """

        made_target = (
            self.sample_size // 2
        )

        missed_target = (
            self.sample_size
            - made_target
        )

        made = self._participant_round_robin(
            [
                candidate
                for candidate in candidates
                if candidate.result == "made"
            ],
            made_target,
        )

        missed = self._participant_round_robin(
            [
                candidate
                for candidate in candidates
                if candidate.result == "missed"
            ],
            missed_target,
        )

        selected = made + missed

        selected_paths = {
            candidate.trial_file
            for candidate in selected
        }

        if len(selected) < self.sample_size:
            for candidate in sorted(
                candidates,
                key=lambda item: (
                    item.participant_id,
                    item.trial_id,
                    item.trial_file.name,
                ),
            ):
                if candidate.trial_file in selected_paths:
                    continue

                selected.append(
                    candidate
                )

                selected_paths.add(
                    candidate.trial_file
                )

                if (
                    len(selected)
                    >= self.sample_size
                ):
                    break

        return selected[
            :self.sample_size
        ]

    @staticmethod
    def _participant_round_robin(
        candidates: list[TrialCandidate],
        target_count: int,
    ) -> list[TrialCandidate]:
        """
        Rotate through participants until the target count is reached.
        """

        grouped: dict[
            str,
            list[TrialCandidate],
        ] = {}

        for candidate in candidates:
            grouped.setdefault(
                candidate.participant_id,
                [],
            ).append(candidate)

        for trials in grouped.values():
            trials.sort(
                key=lambda item: (
                    item.trial_id,
                    item.trial_file.name,
                )
            )

        participant_ids = sorted(
            grouped
        )

        selected: list[TrialCandidate] = []
        index = 0

        while (
            len(selected) < target_count
            and participant_ids
        ):
            remaining_participants: list[
                str
            ] = []

            for participant_id in participant_ids:
                trials = grouped[
                    participant_id
                ]

                if index < len(trials):
                    selected.append(
                        trials[index]
                    )

                    if (
                        len(selected)
                        >= target_count
                    ):
                        break

                if (
                    index + 1
                    < len(trials)
                ):
                    remaining_participants.append(
                        participant_id
                    )

            index += 1

            if len(selected) >= target_count:
                break

            participant_ids = (
                remaining_participants
            )

        return selected

    # =====================================================
    # ANALYSIS
    # =====================================================

    def analyze_candidate(
        self,
        candidate: TrialCandidate,
    ) -> PhaseValidationRecord:
        """
        Run the complete refined phase pipeline for one trial.
        """

        try:
            with candidate.trial_file.open(
                "r",
                encoding="utf-8",
            ) as file:
                trial_data = json.load(
                    file
                )

            sampling_rate = (
                self._safe_sampling_rate(
                    trial_data.get(
                        "sampling_rate",
                        30,
                    )
                )
            )

            shot_analysis = ShotAnalyzer(
                trial_data=trial_data,
                trial_file=(
                    candidate.trial_file
                ),
                shooting_wrist=(
                    f"{self.shooting_side}_WRIST"
                ),
            ).analyze()

            time_series = TimeSeriesAnalyzer(
                trial_data=trial_data,
                smoothing_window=5,
            ).analyze()

            phase_analysis = (
                RefinedPhaseBiomechanicsAnalyzer(
                    time_series=time_series,
                    sampling_rate=sampling_rate,
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

            return self._success_record(
                candidate=candidate,
                sampling_rate=sampling_rate,
                phase_analysis=phase_analysis,
            )

        except Exception as error:
            return self._failed_record(
                candidate=candidate,
                error=error,
            )

    def _success_record(
        self,
        candidate: TrialCandidate,
        sampling_rate: float,
        phase_analysis: object,
    ) -> PhaseValidationRecord:
        """
        Convert a successful analysis to a flat CSV record.
        """

        phases: dict[
            str,
            MovementPhase,
        ] = {
            "knee": phase_analysis.knee_phase,
            "hip": phase_analysis.hip_phase,
            "pelvis": phase_analysis.pelvis_phase,
            "shoulder": phase_analysis.shoulder_phase,
            "elbow": phase_analysis.elbow_phase,
            "wrist": phase_analysis.wrist_phase,
            "ball": phase_analysis.ball_phase,
        }

        flat: dict[str, object] = {
            "participant_id": candidate.participant_id,
            "trial_id": candidate.trial_id,
            "result": candidate.result,
            "trial_file": candidate.trial_file.name,
            "sampling_rate": sampling_rate,
            "motion_start_frame": phase_analysis.motion_start_frame,
            "dip_frame": phase_analysis.dip_frame,
            "takeoff_frame": phase_analysis.takeoff_frame,
            "release_frame": phase_analysis.release_frame,
        }

        for prefix, phase in phases.items():
            flat[
                f"{prefix}_prep_onset"
            ] = (
                phase.preparatory_onset_frame
            )

            flat[
                f"{prefix}_propulsion_onset"
            ] = (
                phase.propulsion_onset_frame
            )

            flat[
                f"{prefix}_peak_velocity"
            ] = (
                phase.peak_velocity_frame
            )

            flat[
                f"{prefix}_peak_acceleration"
            ] = (
                phase.peak_acceleration_frame
            )

            flat[
                f"{prefix}_deceleration"
            ] = (
                phase.deceleration_start_frame
            )

            flat[
                f"{prefix}_end"
            ] = phase.end_frame

            flat[
                f"{prefix}_propulsion_duration_ms"
            ] = (
                phase.propulsion_duration_ms
            )

        flat[
            "preparatory_onset_sequence"
        ] = " -> ".join(
            phase_analysis.preparatory_onset_sequence
        )

        flat[
            "propulsion_onset_sequence"
        ] = " -> ".join(
            phase_analysis.propulsion_onset_sequence
        )

        flat[
            "peak_sequence"
        ] = " -> ".join(
            phase_analysis.peak_sequence
        )

        all_phases = tuple(
            phases.values()
        )

        flat[
            "detected_prep_count"
        ] = sum(
            phase.preparatory_onset_frame
            is not None
            for phase in all_phases
        )

        flat[
            "detected_propulsion_count"
        ] = sum(
            phase.propulsion_onset_frame
            is not None
            for phase in all_phases
        )

        flat[
            "detected_peak_count"
        ] = sum(
            phase.peak_velocity_frame
            is not None
            for phase in all_phases
        )

        flat[
            "analysis_status"
        ] = "success"

        flat[
            "analysis_error"
        ] = ""

        return PhaseValidationRecord(
            **flat
        )

    @staticmethod
    def _failed_record(
        candidate: TrialCandidate,
        error: Exception,
    ) -> PhaseValidationRecord:
        """
        Return a complete failed record without stopping the batch.
        """

        empty: dict[str, object] = {
            field_name: None
            for field_name in (
                PhaseValidationRecord
                .__dataclass_fields__
            )
        }

        empty.update(
            {
                "participant_id": candidate.participant_id,
                "trial_id": candidate.trial_id,
                "result": candidate.result,
                "trial_file": candidate.trial_file.name,
                "sampling_rate": 0.0,
                "preparatory_onset_sequence": "",
                "propulsion_onset_sequence": "",
                "peak_sequence": "",
                "detected_prep_count": 0,
                "detected_propulsion_count": 0,
                "detected_peak_count": 0,
                "analysis_status": "failed",
                "analysis_error": (
                    f"{type(error).__name__}: {error}"
                ),
            }
        )

        return PhaseValidationRecord(
            **empty
        )

    # =====================================================
    # EXPORT
    # =====================================================

    def export_csv(
        self,
        records: list[PhaseValidationRecord],
    ) -> Path:
        """
        Export all phase-validation records.
        """

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = list(
            PhaseValidationRecord
            .__dataclass_fields__
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
                    asdict(record)
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
            sampling_rate = float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            sampling_rate = 30.0

        if sampling_rate <= 0:
            sampling_rate = 30.0

        return sampling_rate


def _average_detection_count(
    records: list[PhaseValidationRecord],
    field_name: str,
) -> float:
    values = [
        float(
            getattr(
                record,
                field_name,
            )
        )
        for record in records
        if record.analysis_status
        == "success"
    ]

    if not values:
        return 0.0

    return float(
        mean(values)
    )


def _run_phase_validation_test() -> None:
    """
    Run the balanced 60-shot phase validation.
    """

    validator = PhaseBiomechanicsValidator(
        sample_size=60,
        shooting_side="RIGHT",
    )

    records = validator.run()

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

    made_count = sum(
        record.result == "made"
        for record in successful
    )

    missed_count = sum(
        record.result == "missed"
        for record in successful
    )

    participants = {
        record.participant_id
        for record in successful
    }

    assert len(records) > 0
    assert validator.output_file.exists()
    assert validator.output_file.stat().st_size > 0

    print(
        "PHASE BIOMECHANICS VALIDATION TEST PASSED"
    )

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
        "Average preparatory phases detected: "
        f"{_average_detection_count(records, 'detected_prep_count'):.2f} / 7"
    )

    print(
        "Average propulsion phases detected: "
        f"{_average_detection_count(records, 'detected_propulsion_count'):.2f} / 7"
    )

    print(
        "Average peaks detected: "
        f"{_average_detection_count(records, 'detected_peak_count'):.2f} / 7"
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
    _run_phase_validation_test()