from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from tracking_app.research.feature_metadata import (
    get_feature_metadata,
)


@dataclass(frozen=True)
class MovementPattern:
    """
    One grouped coaching pattern built from several related features.
    """

    pattern_name: str
    category: str

    feature_count: int
    supporting_features: tuple[str, ...]

    average_similarity_score: float
    strongest_deviation_feature: str
    strongest_deviation_similarity: float

    pattern_priority_score: float
    pattern_level: str

    headline: str
    summary: str
    evidence_summary: str

    review_event: str


class MovementPatternInterpreter:
    """
    Combine isolated feature deviations into coach-readable movement themes.

    Instead of showing:

        elbow angle
        elbow range of motion
        shoulder angle
        wrist speed

    as four disconnected alerts, this module can group related features into:

        Upper-Body Extension

    This is easier for a coach to understand and review on animation.
    """

    def __init__(
        self,
        minimum_feature_count: int = 1,
        maximum_patterns: int = 5,
    ) -> None:
        self.minimum_feature_count = max(
            1,
            int(
                minimum_feature_count
            ),
        )

        self.maximum_patterns = max(
            1,
            int(
                maximum_patterns
            ),
        )

    def build_patterns(
        self,
        scored_features: list[object],
    ) -> list[MovementPattern]:
        """
        Group scored features by configured pattern_group.
        """

        grouped: dict[
            str,
            list[object],
        ] = {}

        for feature in scored_features:
            metadata = get_feature_metadata(
                feature.feature
            )

            grouped.setdefault(
                metadata.pattern_group,
                [],
            ).append(
                feature
            )

        patterns: list[
            MovementPattern
        ] = []

        for pattern_name, rows in grouped.items():
            if len(
                rows
            ) < self.minimum_feature_count:
                continue

            rows = sorted(
                rows,
                key=lambda row: (
                    row.feature_similarity_score,
                    -row.evidence_weight,
                    row.feature,
                ),
            )

            strongest = rows[0]

            average_similarity = float(
                mean(
                    row.feature_similarity_score
                    for row in rows
                )
            )

            average_evidence = float(
                mean(
                    row.evidence_weight
                    for row in rows
                )
            )

            low_similarity_component = (
                100.0
                - average_similarity
            )

            priority_score = float(
                low_similarity_component
                * 0.70
                + average_evidence
                * 100.0
                * 0.30
            )

            level = self._pattern_level(
                priority_score
            )

            category = get_feature_metadata(
                strongest.feature
            ).category

            supporting_names = tuple(
                get_feature_metadata(
                    row.feature
                ).display_name
                for row in rows
            )

            headline = self._headline(
                pattern_name=pattern_name,
                level=level,
            )

            summary = self._pattern_summary(
                pattern_name=pattern_name,
                rows=rows,
                average_similarity=average_similarity,
            )

            evidence_summary = self._evidence_summary(
                rows
            )

            review_event = self._review_event(
                rows
            )

            patterns.append(
                MovementPattern(
                    pattern_name=pattern_name,
                    category=category,

                    feature_count=len(
                        rows
                    ),
                    supporting_features=(
                        supporting_names
                    ),

                    average_similarity_score=(
                        average_similarity
                    ),
                    strongest_deviation_feature=(
                        get_feature_metadata(
                            strongest.feature
                        ).display_name
                    ),
                    strongest_deviation_similarity=float(
                        strongest
                        .feature_similarity_score
                    ),

                    pattern_priority_score=(
                        priority_score
                    ),
                    pattern_level=level,

                    headline=headline,
                    summary=summary,
                    evidence_summary=(
                        evidence_summary
                    ),

                    review_event=review_event,
                )
            )

        patterns.sort(
            key=lambda pattern: (
                -pattern.pattern_priority_score,
                pattern.average_similarity_score,
                pattern.pattern_name,
            )
        )

        return patterns[
            :self.maximum_patterns
        ]

    @staticmethod
    def _pattern_level(
        priority_score: float,
    ) -> str:
        if priority_score >= 65:
            return "high"

        if priority_score >= 45:
            return "moderate"

        return "low"

    @staticmethod
    def _headline(
        pattern_name: str,
        level: str,
    ) -> str:
        if level == "high":
            return (
                f"Large difference in {pattern_name.lower()}"
            )

        if level == "moderate":
            return (
                f"Meaningful difference in {pattern_name.lower()}"
            )

        return (
            f"{pattern_name} generally matched baseline"
        )

    @staticmethod
    def _pattern_summary(
        pattern_name: str,
        rows: list[object],
        average_similarity: float,
    ) -> str:
        strongest = rows[0]

        strongest_metadata = get_feature_metadata(
            strongest.feature
        )

        direction_text = (
            strongest_metadata.higher_meaning
            if strongest.shot_value
            > strongest.successful_mean
            else strongest_metadata.lower_meaning
        )

        if len(
            rows
        ) == 1:
            return (
                f"The main {pattern_name.lower()} difference was "
                f"{direction_text}. The feature matched the successful "
                f"baseline at {average_similarity:.1f}/100."
            )

        return (
            f"Several related measurements in {pattern_name.lower()} "
            f"differed from the player's successful pattern. The strongest "
            f"difference showed {direction_text}. The grouped similarity "
            f"score was {average_similarity:.1f}/100."
        )

    @staticmethod
    def _evidence_summary(
        rows: list[object],
    ) -> str:
        evidence_values = [
            float(
                row.evidence_weight
            )
            for row in rows
        ]

        average_evidence = float(
            mean(
                evidence_values
            )
        )

        strongest_rows = sorted(
            rows,
            key=lambda row: (
                row.feature_similarity_score,
                -row.evidence_weight,
            ),
        )[
            :3
        ]

        names = [
            get_feature_metadata(
                row.feature
            ).display_name
            for row in strongest_rows
        ]

        return (
            f"Supported by {len(rows)} related feature"
            f"{'' if len(rows) == 1 else 's'}; "
            f"average evidence weight {average_evidence:.3f}. "
            f"Primary supporting measurements: "
            f"{', '.join(names)}."
        )

    @staticmethod
    def _review_event(
        rows: list[object],
    ) -> str:
        """
        Choose the most common configured review event.
        """

        event_counts: dict[
            str,
            int,
        ] = {}

        for row in rows:
            event_name = get_feature_metadata(
                row.feature
            ).review_event

            event_counts[
                event_name
            ] = (
                event_counts.get(
                    event_name,
                    0,
                )
                + 1
            )

        return max(
            event_counts,
            key=event_counts.get,
        )


def _run_movement_pattern_interpreter_test() -> None:
    """
    Small structural test using fake scored features.
    """

    @dataclass(frozen=True)
    class FakeFeature:
        feature: str
        shot_value: float
        successful_mean: float
        feature_similarity_score: float
        evidence_weight: float

    rows = [
        FakeFeature(
            feature=(
                "release_right_elbow_angle_deg"
            ),
            shot_value=118.0,
            successful_mean=137.0,
            feature_similarity_score=35.0,
            evidence_weight=0.72,
        ),
        FakeFeature(
            feature=(
                "right_elbow_range_of_motion_deg"
            ),
            shot_value=39.0,
            successful_mean=60.0,
            feature_similarity_score=20.0,
            evidence_weight=0.78,
        ),
        FakeFeature(
            feature=(
                "release_right_knee_angle_deg"
            ),
            shot_value=150.0,
            successful_mean=163.0,
            feature_similarity_score=55.0,
            evidence_weight=0.68,
        ),
    ]

    interpreter = (
        MovementPatternInterpreter()
    )

    patterns = interpreter.build_patterns(
        rows
    )

    assert len(
        patterns
    ) >= 2

    assert patterns[
        0
    ].pattern_name in {
        "Upper-Body Extension",
        "Lower-Body Extension",
    }

    print(
        "MOVEMENT PATTERN INTERPRETER TEST PASSED"
    )

    print(
        f"Patterns created: {len(patterns)}"
    )

    for pattern in patterns:
        print(
            f"- {pattern.pattern_name}: "
            f"{pattern.pattern_priority_score:.1f}, "
            f"{pattern.pattern_level}"
        )


if __name__ == "__main__":
    _run_movement_pattern_interpreter_test()