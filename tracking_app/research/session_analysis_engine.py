from __future__ import annotations

import csv
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import numpy as np

from tracking_app.research.feature_metadata import (
    get_feature_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "player_history.sqlite3"
)

DEFAULT_FEATURE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "all_shot_features.csv"
)

DEFAULT_OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "session_analysis"
)


@dataclass(frozen=True)
class SessionShotAnalysisRow:
    session_id: int
    shot_file_id: int

    participant_id: str
    player_display_name: str

    session_name: str
    session_date: str

    shot_number: int
    trial_id: str
    original_filename: str
    result: str

    matched_feature_row: bool

    overall_similarity_score: float | None
    confidence_adjusted_score: float | None

    release_angle_deg: float | None
    release_ball_speed_ft_s: float | None
    release_height_ft: float | None

    knee_to_elbow_gap_ms: float | None
    elbow_to_release_gap_ms: float | None
    takeoff_to_release_ms: float | None

    shot_attention_score: float
    shot_status: str


@dataclass(frozen=True)
class SessionCategorySummary:
    session_id: int
    category: str

    features_analyzed: int
    valid_shots: int

    average_consistency_score: float
    strongest_feature: str
    weakest_feature: str


@dataclass(frozen=True)
class SessionAnalysisSummary:
    session_id: int
    player_id: int

    participant_id: str
    player_display_name: str

    session_name: str
    session_date: str
    session_type: str
    location: str

    assigned_shots: int
    matched_shots: int
    unmatched_shots: int

    made_shots: int
    missed_shots: int
    make_percentage: float

    average_similarity_score: float | None
    average_confidence_adjusted_score: float | None

    overall_consistency_score: float | None
    strongest_category: str
    weakest_category: str

    highest_attention_trial: str
    most_consistent_trial: str

    data_status: str


class SessionAnalysisEngine:
    """
    Build a coach-facing analysis for one practice session.

    The engine joins:

        practice session
            → assigned shot files
            → trial IDs
            → extracted feature rows

    It produces:
    - session summary;
    - ordered shot browser rows;
    - make/miss totals;
    - mechanics and timing summaries;
    - strongest and weakest categories;
    - highest-attention shots for review.

    It does not modify player, session, or shot-history data.
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

    def __init__(
        self,
        database_file: Path = DEFAULT_DATABASE_FILE,
        feature_file: Path = DEFAULT_FEATURE_FILE,
        output_folder: Path = DEFAULT_OUTPUT_FOLDER,
    ) -> None:
        self.database_file = Path(
            database_file
        )

        self.feature_file = Path(
            feature_file
        )

        self.output_folder = Path(
            output_folder
        )

        self.shot_rows_file = (
            self.output_folder
            / "session_shot_rows.csv"
        )

        self.category_summary_file = (
            self.output_folder
            / "session_category_summary.csv"
        )

        self.session_summary_file = (
            self.output_folder
            / "session_analysis_summary.csv"
        )

        self.report_file = (
            self.output_folder
            / "session_analysis_report.txt"
        )

    # =====================================================
    # CONNECTION
    # =====================================================

    def connect(
        self,
    ) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_file
        )

        connection.row_factory = sqlite3.Row

        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        return connection

    @contextmanager
    def open_connection(
        self,
    ):
        connection = self.connect()

        try:
            yield connection

        finally:
            connection.close()

    # =====================================================
    # PUBLIC WORKFLOW
    # =====================================================

    def run(
        self,
        session_id: int,
    ) -> tuple[
        SessionAnalysisSummary,
        list[SessionShotAnalysisRow],
        list[SessionCategorySummary],
    ]:
        session = self._load_session(
            session_id
        )

        assigned_shots = self._load_session_shots(
            session_id
        )

        feature_records = self._load_feature_records(
            participant_id=session[
                "participant_id"
            ]
        )

        feature_lookup = {
            str(
                record.get(
                    "trial_id",
                    ""
                )
            ).strip(): record
            for record in feature_records
        }

        shot_rows = self._build_shot_rows(
            session=session,
            assigned_shots=assigned_shots,
            feature_lookup=feature_lookup,
        )

        category_summaries = self._build_category_summaries(
            session_id=session_id,
            shot_rows=shot_rows,
            feature_lookup=feature_lookup,
        )

        summary = self._build_summary(
            session=session,
            shot_rows=shot_rows,
            category_summaries=category_summaries,
        )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._export_rows(
            self.shot_rows_file,
            shot_rows,
        )

        self._export_rows(
            self.category_summary_file,
            category_summaries,
        )

        self._export_rows(
            self.session_summary_file,
            [
                summary
            ],
        )

        self._export_report(
            summary=summary,
            shot_rows=shot_rows,
            category_summaries=category_summaries,
        )

        return (
            summary,
            shot_rows,
            category_summaries,
        )

    # =====================================================
    # LOAD
    # =====================================================

    def _load_session(
        self,
        session_id: int,
    ) -> sqlite3.Row:
        with self.open_connection() as connection:
            row = connection.execute(
                """
                SELECT
                    ps.session_id,
                    ps.player_id,

                    p.participant_id,
                    p.display_name,

                    ps.session_name,
                    ps.session_date,
                    ps.session_type,
                    ps.location,
                    ps.notes

                FROM practice_sessions AS ps

                INNER JOIN players AS p
                    ON p.player_id = ps.player_id

                WHERE ps.session_id = ?
                """,
                (
                    int(
                        session_id
                    ),
                ),
            ).fetchone()

        if row is None:
            raise ValueError(
                f"No practice session was found with session_id {session_id}."
            )

        return row

    def _load_session_shots(
        self,
        session_id: int,
    ) -> list[sqlite3.Row]:
        with self.open_connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    pss.shot_file_id,
                    pss.display_order,

                    sf.trial_id,
                    sf.original_filename,
                    sf.recorded_result

                FROM practice_session_shots AS pss

                INNER JOIN shot_files AS sf
                    ON sf.shot_file_id = pss.shot_file_id

                WHERE
                    pss.session_id = ?
                    AND sf.active = 1

                ORDER BY
                    pss.display_order,
                    sf.trial_id
                """,
                (
                    int(
                        session_id
                    ),
                ),
            ).fetchall()

        return list(
            rows
        )

    def _load_feature_records(
        self,
        participant_id: str,
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
            rows = list(
                csv.DictReader(
                    file
                )
            )

        return [
            row
            for row in rows
            if (
                str(
                    row.get(
                        "participant_id",
                        "",
                    )
                ).strip()
                == participant_id
                and str(
                    row.get(
                        "analysis_status",
                        "",
                    )
                ).strip().lower()
                == "success"
            )
        ]

    # =====================================================
    # SHOT ROWS
    # =====================================================

    def _build_shot_rows(
        self,
        session: sqlite3.Row,
        assigned_shots: list[sqlite3.Row],
        feature_lookup: dict[
            str,
            dict[str, str],
        ],
    ) -> list[SessionShotAnalysisRow]:
        rows: list[
            SessionShotAnalysisRow
        ] = []

        for index, assigned_shot in enumerate(
            assigned_shots,
            start=1,
        ):
            trial_id = str(
                assigned_shot[
                    "trial_id"
                ]
            )

            feature_record = feature_lookup.get(
                trial_id
            )

            matched = (
                feature_record is not None
            )

            overall_similarity_score = (
                self._first_numeric(
                    feature_record,
                    (
                        "overall_similarity_score",
                        "shot_similarity_score",
                        "overall_score",
                    ),
                )
                if matched
                else None
            )

            confidence_adjusted_score = (
                self._first_numeric(
                    feature_record,
                    (
                        "confidence_adjusted_score",
                        "adjusted_score",
                    ),
                )
                if matched
                else None
            )

            result = (
                self._normalize_result(
                    feature_record.get(
                        "result"
                    )
                )
                if matched
                else self._normalize_result(
                    assigned_shot[
                        "recorded_result"
                    ]
                )
            )

            attention_score = self._shot_attention_score(
                matched_feature_row=matched,
                overall_similarity_score=(
                    overall_similarity_score
                ),
                confidence_adjusted_score=(
                    confidence_adjusted_score
                ),
                result=result,
            )

            shot_status = self._shot_status(
                matched_feature_row=matched,
                attention_score=attention_score,
            )

            rows.append(
                SessionShotAnalysisRow(
                    session_id=int(
                        session[
                            "session_id"
                        ]
                    ),
                    shot_file_id=int(
                        assigned_shot[
                            "shot_file_id"
                        ]
                    ),

                    participant_id=str(
                        session[
                            "participant_id"
                        ]
                    ),
                    player_display_name=str(
                        session[
                            "display_name"
                        ]
                    ),

                    session_name=str(
                        session[
                            "session_name"
                        ]
                    ),
                    session_date=str(
                        session[
                            "session_date"
                        ]
                    ),

                    shot_number=index,
                    trial_id=trial_id,
                    original_filename=str(
                        assigned_shot[
                            "original_filename"
                        ]
                    ),
                    result=result,

                    matched_feature_row=matched,

                    overall_similarity_score=(
                        overall_similarity_score
                    ),
                    confidence_adjusted_score=(
                        confidence_adjusted_score
                    ),

                    release_angle_deg=(
                        self._to_float(
                            feature_record.get(
                                "release_angle_deg"
                            )
                        )
                        if matched
                        else None
                    ),
                    release_ball_speed_ft_s=(
                        self._to_float(
                            feature_record.get(
                                "release_ball_speed_ft_s"
                            )
                        )
                        if matched
                        else None
                    ),
                    release_height_ft=(
                        self._to_float(
                            feature_record.get(
                                "release_height_ft"
                            )
                        )
                        if matched
                        else None
                    ),

                    knee_to_elbow_gap_ms=(
                        self._to_float(
                            feature_record.get(
                                "knee_to_elbow_gap_ms"
                            )
                        )
                        if matched
                        else None
                    ),
                    elbow_to_release_gap_ms=(
                        self._to_float(
                            feature_record.get(
                                "elbow_to_release_gap_ms"
                            )
                        )
                        if matched
                        else None
                    ),
                    takeoff_to_release_ms=(
                        self._to_float(
                            feature_record.get(
                                "takeoff_to_release_ms"
                            )
                        )
                        if matched
                        else None
                    ),

                    shot_attention_score=(
                        attention_score
                    ),
                    shot_status=(
                        shot_status
                    ),
                )
            )

        return rows

    @staticmethod
    def _shot_attention_score(
        matched_feature_row: bool,
        overall_similarity_score: float | None,
        confidence_adjusted_score: float | None,
        result: str,
    ) -> float:
        if not matched_feature_row:
            return 100.0

        available_scores = [
            value
            for value in (
                overall_similarity_score,
                confidence_adjusted_score,
            )
            if value is not None
        ]

        if available_scores:
            score_component = (
                100.0
                - float(
                    mean(
                        available_scores
                    )
                )
            )

        else:
            score_component = 45.0

        result_component = (
            15.0
            if result == "missed"
            else 0.0
        )

        return float(
            np.clip(
                score_component
                + result_component,
                0.0,
                100.0,
            )
        )

    @staticmethod
    def _shot_status(
        matched_feature_row: bool,
        attention_score: float,
    ) -> str:
        if not matched_feature_row:
            return "Needs feature extraction"

        if attention_score >= 65:
            return "High review priority"

        if attention_score >= 40:
            return "Review"

        return "Stable"

    # =====================================================
    # CATEGORY SUMMARY
    # =====================================================

    def _build_category_summaries(
        self,
        session_id: int,
        shot_rows: list[
            SessionShotAnalysisRow
        ],
        feature_lookup: dict[
            str,
            dict[str, str],
        ],
    ) -> list[SessionCategorySummary]:
        category_feature_scores: dict[
            str,
            list[
                tuple[
                    str,
                    float,
                ]
            ],
        ] = {}

        for feature in self.TRACKED_FEATURES:
            metadata = get_feature_metadata(
                feature
            )

            values = []

            for shot in shot_rows:
                feature_record = feature_lookup.get(
                    shot.trial_id
                )

                if feature_record is None:
                    continue

                value = self._to_float(
                    feature_record.get(
                        feature
                    )
                )

                if value is not None:
                    values.append(
                        value
                    )

            if len(
                values
            ) < 2:
                continue

            feature_mean = float(
                mean(
                    values
                )
            )

            feature_sd = float(
                np.std(
                    values,
                    ddof=0,
                )
            )

            consistency_score = self._consistency_score(
                session_mean=feature_mean,
                session_sd=feature_sd,
            )

            category_feature_scores.setdefault(
                metadata.category,
                [],
            ).append(
                (
                    metadata.display_name,
                    consistency_score,
                )
            )

        matched_shots = sum(
            row.matched_feature_row
            for row in shot_rows
        )

        summaries = []

        for category, feature_scores in sorted(
            category_feature_scores.items()
        ):
            strongest_feature = max(
                feature_scores,
                key=lambda item: item[1],
            )[0]

            weakest_feature = min(
                feature_scores,
                key=lambda item: item[1],
            )[0]

            summaries.append(
                SessionCategorySummary(
                    session_id=int(
                        session_id
                    ),
                    category=category,

                    features_analyzed=len(
                        feature_scores
                    ),
                    valid_shots=int(
                        matched_shots
                    ),

                    average_consistency_score=float(
                        mean(
                            score
                            for _, score
                            in feature_scores
                        )
                    ),
                    strongest_feature=(
                        strongest_feature
                    ),
                    weakest_feature=(
                        weakest_feature
                    ),
                )
            )

        return summaries

    # =====================================================
    # SESSION SUMMARY
    # =====================================================

    def _build_summary(
        self,
        session: sqlite3.Row,
        shot_rows: list[
            SessionShotAnalysisRow
        ],
        category_summaries: list[
            SessionCategorySummary
        ],
    ) -> SessionAnalysisSummary:
        matched_shots = sum(
            row.matched_feature_row
            for row in shot_rows
        )

        made_shots = sum(
            row.result == "made"
            for row in shot_rows
        )

        missed_shots = sum(
            row.result == "missed"
            for row in shot_rows
        )

        classified_shots = (
            made_shots
            + missed_shots
        )

        make_percentage = (
            made_shots
            / classified_shots
            * 100.0
            if classified_shots
            else 0.0
        )

        similarity_scores = [
            row.overall_similarity_score
            for row in shot_rows
            if row.overall_similarity_score
            is not None
        ]

        adjusted_scores = [
            row.confidence_adjusted_score
            for row in shot_rows
            if row.confidence_adjusted_score
            is not None
        ]

        category_scores = [
            row.average_consistency_score
            for row in category_summaries
        ]

        strongest_category = (
            max(
                category_summaries,
                key=lambda row: (
                    row.average_consistency_score
                ),
            ).category
            if category_summaries
            else "N/A"
        )

        weakest_category = (
            min(
                category_summaries,
                key=lambda row: (
                    row.average_consistency_score
                ),
            ).category
            if category_summaries
            else "N/A"
        )

        highest_attention_trial = (
            max(
                shot_rows,
                key=lambda row: (
                    row.shot_attention_score
                ),
            ).trial_id
            if shot_rows
            else "N/A"
        )

        scored_rows = [
            row
            for row in shot_rows
            if row.overall_similarity_score
            is not None
        ]

        most_consistent_trial = (
            max(
                scored_rows,
                key=lambda row: (
                    row.overall_similarity_score
                ),
            ).trial_id
            if scored_rows
            else "N/A"
        )

        if not shot_rows:
            data_status = "No shots assigned"

        elif matched_shots == 0:
            data_status = "No matched feature rows"

        elif matched_shots < len(
            shot_rows
        ):
            data_status = "Partially matched"

        else:
            data_status = "Ready"

        return SessionAnalysisSummary(
            session_id=int(
                session[
                    "session_id"
                ]
            ),
            player_id=int(
                session[
                    "player_id"
                ]
            ),

            participant_id=str(
                session[
                    "participant_id"
                ]
            ),
            player_display_name=str(
                session[
                    "display_name"
                ]
            ),

            session_name=str(
                session[
                    "session_name"
                ]
            ),
            session_date=str(
                session[
                    "session_date"
                ]
            ),
            session_type=str(
                session[
                    "session_type"
                ]
            ),
            location=str(
                session[
                    "location"
                ]
            ),

            assigned_shots=len(
                shot_rows
            ),
            matched_shots=int(
                matched_shots
            ),
            unmatched_shots=(
                len(
                    shot_rows
                )
                - matched_shots
            ),

            made_shots=int(
                made_shots
            ),
            missed_shots=int(
                missed_shots
            ),
            make_percentage=float(
                make_percentage
            ),

            average_similarity_score=(
                float(
                    mean(
                        similarity_scores
                    )
                )
                if similarity_scores
                else None
            ),
            average_confidence_adjusted_score=(
                float(
                    mean(
                        adjusted_scores
                    )
                )
                if adjusted_scores
                else None
            ),

            overall_consistency_score=(
                float(
                    mean(
                        category_scores
                    )
                )
                if category_scores
                else None
            ),
            strongest_category=(
                strongest_category
            ),
            weakest_category=(
                weakest_category
            ),

            highest_attention_trial=(
                highest_attention_trial
            ),
            most_consistent_trial=(
                most_consistent_trial
            ),

            data_status=(
                data_status
            ),
        )

    # =====================================================
    # EXPORT
    # =====================================================

    @staticmethod
    def _export_rows(
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

    def _export_report(
        self,
        summary: SessionAnalysisSummary,
        shot_rows: list[
            SessionShotAnalysisRow
        ],
        category_summaries: list[
            SessionCategorySummary
        ],
    ) -> Path:
        lines = [
            "=" * 96,
            "PRACTICE SESSION ANALYSIS",
            "=" * 96,
            "",
            (
                f"Player: "
                f"{summary.player_display_name} "
                f"({summary.participant_id})"
            ),
            (
                f"Session: "
                f"{summary.session_name}"
            ),
            (
                f"Date: "
                f"{summary.session_date}"
            ),
            (
                f"Type: "
                f"{summary.session_type}"
            ),
            (
                f"Location: "
                f"{summary.location or 'N/A'}"
            ),
            "",
            (
                f"Assigned shots: "
                f"{summary.assigned_shots}"
            ),
            (
                f"Matched shots: "
                f"{summary.matched_shots}"
            ),
            (
                f"Make percentage: "
                f"{summary.make_percentage:.1f}%"
            ),
            (
                "Overall consistency: "
                + (
                    "N/A"
                    if summary.overall_consistency_score
                    is None
                    else (
                        f"{summary.overall_consistency_score:.1f}/100"
                    )
                )
            ),
            (
                f"Strongest category: "
                f"{summary.strongest_category}"
            ),
            (
                f"Weakest category: "
                f"{summary.weakest_category}"
            ),
            (
                f"Highest-attention trial: "
                f"{summary.highest_attention_trial}"
            ),
            (
                f"Data status: "
                f"{summary.data_status}"
            ),
            "",
            "CATEGORY SUMMARY",
            "-" * 96,
        ]

        for category in category_summaries:
            lines.append(
                (
                    f"{category.category}: "
                    f"{category.average_consistency_score:.1f}/100 · "
                    f"strongest {category.strongest_feature} · "
                    f"weakest {category.weakest_feature}"
                )
            )

        lines.extend(
            [
                "",
                "SHOT BROWSER",
                "-" * 96,
            ]
        )

        for shot in shot_rows:
            lines.append(
                (
                    f"{shot.shot_number}. "
                    f"{shot.trial_id} · "
                    f"{shot.result} · "
                    f"attention {shot.shot_attention_score:.1f} · "
                    f"{shot.shot_status}"
                )
            )

        self.report_file.write_text(
            "\n".join(
                lines
            ),
            encoding="utf-8",
        )

        return self.report_file

    # =====================================================
    # HELPERS
    # =====================================================

    @staticmethod
    def _first_numeric(
        record: dict[str, str] | None,
        fields: tuple[str, ...],
    ) -> float | None:
        if record is None:
            return None

        for field in fields:
            value = SessionAnalysisEngine._to_float(
                record.get(
                    field
                )
            )

            if value is not None:
                return value

        return None

    @staticmethod
    def _to_float(
        value: object,
    ) -> float | None:
        if value is None:
            return None

        cleaned = str(
            value
        ).strip().lower()

        if cleaned in {
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
        cleaned = str(
            value
            if value is not None
            else ""
        ).strip().lower()

        if cleaned in {
            "made",
            "make",
            "successful",
            "hit",
        }:
            return "made"

        if cleaned in {
            "missed",
            "miss",
            "failed",
        }:
            return "missed"

        return "unknown"

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

        return float(
            np.clip(
                (
                    1.0
                    - min(
                        1.0,
                        coefficient_of_variation,
                    )
                )
                * 100.0,
                0.0,
                100.0,
            )
        )


def _run_session_analysis_engine_test() -> None:
    """
    Isolated integration test with temporary database and feature CSV.
    """

    import json
    import tempfile

    from tracking_app.research.player_history_repository import (
        PlayerHistoryRepository,
    )
    from tracking_app.research.practice_session_repository import (
        PracticeSessionRepository,
    )

    with tempfile.TemporaryDirectory() as temporary_folder:
        root = Path(
            temporary_folder
        )

        database_file = (
            root
            / "history.sqlite3"
        )

        feature_file = (
            root
            / "features.csv"
        )

        history_repository = (
            PlayerHistoryRepository(
                database_file=database_file,
                storage_folder=(
                    root
                    / "files"
                ),
            )
        )

        player = history_repository.create_or_get_player(
            participant_id="TEST_P1",
            display_name="Session Player",
        )

        uploads = []

        for trial_number in range(
            1,
            6,
        ):
            result = (
                "made"
                if trial_number
                in {
                    1,
                    2,
                    4,
                }
                else "missed"
            )

            payload = {
                "participant_id": "TEST_P1",
                "trial_id": (
                    f"T{trial_number:04d}"
                ),
                "result": result,
                "sampling_rate": 30,
                "tracking": [
                    {
                        "frame": 0,
                        "time": 0,
                        "data": {},
                    }
                ],
            }

            uploads.append(
                (
                    f"trial_{trial_number}.json",
                    json.dumps(
                        payload
                    ).encode(
                        "utf-8"
                    ),
                )
            )

        history_repository.import_json_files(
            uploads,
            selected_player_id=player.player_id,
            enforce_selected_player=True,
        )

        session_repository = PracticeSessionRepository(
            database_file=database_file
        )

        session = session_repository.create_session(
            player_id=player.player_id,
            session_name="Test Session",
            session_date="2026-07-27",
            session_type="Free Throws",
            location="Test Gym",
        )

        shot_files = history_repository.list_shot_files(
            player_id=player.player_id
        )

        session_repository.assign_shots(
            session_id=session.session_id,
            shot_file_ids=[
                shot.shot_file_id
                for shot in shot_files
            ],
        )

        fieldnames = [
            "participant_id",
            "trial_id",
            "analysis_status",
            "result",
            "overall_similarity_score",
            "confidence_adjusted_score",
            *SessionAnalysisEngine.TRACKED_FEATURES,
        ]

        with feature_file.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for trial_number in range(
                1,
                6,
            ):
                result = (
                    "made"
                    if trial_number
                    in {
                        1,
                        2,
                        4,
                    }
                    else "missed"
                )

                writer.writerow(
                    {
                        "participant_id": "TEST_P1",
                        "trial_id": (
                            f"T{trial_number:04d}"
                        ),
                        "analysis_status": "success",
                        "result": result,
                        "overall_similarity_score": (
                            85
                            - trial_number
                            * 3
                        ),
                        "confidence_adjusted_score": (
                            80
                            - trial_number
                            * 2
                        ),
                        "release_angle_deg": (
                            50
                            + trial_number
                        ),
                        "release_ball_speed_ft_s": (
                            10
                            + trial_number
                            * 0.1
                        ),
                        "release_height_ft": (
                            7.0
                            + trial_number
                            * 0.02
                        ),
                        "right_knee_range_of_motion_deg": (
                            35
                            + trial_number
                        ),
                        "right_hip_range_of_motion_deg": (
                            25
                            + trial_number
                            * 0.5
                        ),
                        "release_right_elbow_angle_deg": (
                            130
                            + trial_number
                        ),
                        "knee_to_elbow_gap_ms": (
                            100
                            - trial_number
                            * 3
                        ),
                        "elbow_to_release_gap_ms": (
                            250
                            - trial_number
                            * 2
                        ),
                        "takeoff_to_release_ms": (
                            150
                            + trial_number
                        ),
                    }
                )

        engine = SessionAnalysisEngine(
            database_file=database_file,
            feature_file=feature_file,
            output_folder=(
                root
                / "outputs"
            ),
        )

        (
            summary,
            shot_rows,
            category_summaries,
        ) = engine.run(
            session_id=session.session_id
        )

        assert summary.assigned_shots == 5
        assert summary.matched_shots == 5
        assert summary.made_shots == 3
        assert summary.missed_shots == 2
        assert summary.make_percentage == 60.0
        assert len(
            shot_rows
        ) == 5
        assert len(
            category_summaries
        ) > 0

        assert engine.shot_rows_file.exists()
        assert engine.category_summary_file.exists()
        assert engine.session_summary_file.exists()
        assert engine.report_file.exists()

        print(
            "SESSION ANALYSIS ENGINE TEST PASSED"
        )

        print(
            f"Player: {summary.player_display_name}"
        )

        print(
            f"Session: {summary.session_name}"
        )

        print(
            f"Assigned shots: {summary.assigned_shots}"
        )

        print(
            f"Matched shots: {summary.matched_shots}"
        )

        print(
            f"Make percentage: "
            f"{summary.make_percentage:.1f}%"
        )

        print(
            "Overall consistency: "
            + (
                "N/A"
                if summary.overall_consistency_score
                is None
                else (
                    f"{summary.overall_consistency_score:.1f}/100"
                )
            )
        )

        print(
            f"Strongest category: "
            f"{summary.strongest_category}"
        )

        print(
            f"Highest-attention trial: "
            f"{summary.highest_attention_trial}"
        )

        print(
            "Session-to-shot matching: PASSED"
        )

        print(
            "Session analysis exports: PASSED"
        )


if __name__ == "__main__":
    _run_session_analysis_engine_test()