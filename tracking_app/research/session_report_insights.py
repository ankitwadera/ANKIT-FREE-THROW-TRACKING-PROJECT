from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class GradeContributor:
    rank: int
    direction: str
    label: str
    impact_points: float
    explanation: str


@dataclass(frozen=True)
class ShotReviewReason:
    trial_id: str
    shot_number: int
    reason: str
    detail: str


@dataclass(frozen=True)
class SessionReportInsights:
    session_story: str
    grade_explanation: str
    what_went_well: tuple[str, ...]
    primary_focus: str
    next_practice_summary: str
    contributors: tuple[GradeContributor, ...]
    shot_reasons: tuple[ShotReviewReason, ...]


def build_session_report_insights(
    session_summary: object,
    shot_rows: list[object],
    category_summaries: list[object],
    grade_summary: object,
    grade_components: list[object],
    action_items: list[object],
) -> SessionReportInsights:
    """
    Convert session-analysis outputs into concise coach-facing communication.

    This module does not generate new biomechanical measurements. It organizes
    the existing evidence into:
    - one session story;
    - a plain-language grade explanation;
    - strengths;
    - the primary focus;
    - the next-practice summary;
    - biggest grade contributors;
    - a reason to review each priority shot.
    """

    category_lookup = {
        row.category: row
        for row in category_summaries
    }

    strongest = category_lookup.get(
        session_summary.strongest_category
    )

    weakest = category_lookup.get(
        session_summary.weakest_category
    )

    early_late_text = _early_late_story(
        shot_rows
    )

    structural_text = _structural_story(
        strongest_category=(
            session_summary.strongest_category
        ),
        weakest_category=(
            session_summary.weakest_category
        ),
    )

    session_story = (
        f"{session_summary.player_display_name} completed "
        f"{session_summary.assigned_shots} tracked attempts and made "
        f"{session_summary.make_percentage:.1f}% of classified shots. "
        f"{structural_text} {early_late_text}"
    ).strip()

    component_lookup = {
        component.component: component
        for component in grade_components
    }

    lowest_component = (
        min(
            grade_components,
            key=lambda component: component.score,
        )
        if grade_components
        else None
    )

    highest_component = (
        max(
            grade_components,
            key=lambda component: component.score,
        )
        if grade_components
        else None
    )

    grade_explanation = (
        f"The {grade_summary.overall_grade:.1f} session grade was driven most "
        f"positively by "
        f"{highest_component.component if highest_component else 'the strongest measured area'} "
        f"and was limited most by "
        f"{lowest_component.component if lowest_component else 'the main review area'}. "
        f"The score should be read as a session review summary, not a player "
        f"talent rating."
    )

    what_went_well = []

    if strongest is not None:
        what_went_well.append(
            (
                f"{strongest.category} was highly repeatable "
                f"({strongest.average_consistency_score:.1f}/100)."
            )
        )

    completeness = component_lookup.get(
        "Data Completeness"
    )

    if (
        completeness is not None
        and completeness.score >= 90.0
    ):
        what_went_well.append(
            "The session had complete or near-complete matched tracking data."
        )

    consistency = component_lookup.get(
        "Consistency"
    )

    if (
        consistency is not None
        and consistency.score >= 75.0
    ):
        what_went_well.append(
            (
                f"Overall mechanics were repeatable across the session "
                f"({consistency.score:.1f}/100)."
            )
        )

    execution = component_lookup.get(
        "Execution"
    )

    if (
        execution is not None
        and execution.score >= 75.0
    ):
        what_went_well.append(
            (
                f"Shot conversion was strong at "
                f"{session_summary.make_percentage:.1f}%."
            )
        )

    if not what_went_well:
        what_went_well.append(
            "The session established a measurable baseline for future review."
        )

    primary_focus = (
        f"{session_summary.weakest_category}: "
        f"{weakest.weakest_feature}"
        if weakest is not None
        else grade_summary.primary_focus
    )

    next_practice_summary = (
        f"{grade_summary.recommended_drill}. "
        f"Estimated review and drill time: "
        f"{grade_summary.estimated_review_minutes} minutes."
    )

    contributors = _build_contributors(
        grade_components=grade_components,
        category_summaries=category_summaries,
    )

    shot_reasons = _build_shot_reasons(
        shot_rows=shot_rows,
        weakest_category=(
            session_summary.weakest_category
        ),
    )

    return SessionReportInsights(
        session_story=session_story,
        grade_explanation=grade_explanation,
        what_went_well=tuple(
            what_went_well[
                :3
            ]
        ),
        primary_focus=primary_focus,
        next_practice_summary=next_practice_summary,
        contributors=tuple(
            contributors
        ),
        shot_reasons=tuple(
            shot_reasons
        ),
    )


def _early_late_story(
    shot_rows: list[object],
) -> str:
    matched = [
        row
        for row in shot_rows
        if row.matched_feature_row
    ]

    if len(
        matched
    ) < 6:
        return (
            "The available sample is too small to make a strong early-versus-"
            "late session statement."
        )

    split = max(
        1,
        len(
            matched
        )
        // 3,
    )

    early_attention = mean(
        row.shot_attention_score
        for row in matched[
            :split
        ]
    )

    late_attention = mean(
        row.shot_attention_score
        for row in matched[
            -split:
        ]
    )

    change = (
        late_attention
        - early_attention
    )

    if change >= 10.0:
        return (
            "Review priority increased meaningfully during the final third of "
            "the workout, so pace, fatigue, and changing cues should be checked."
        )

    if change >= 5.0:
        return (
            "The final third became somewhat less stable than the opening "
            "portion of the workout."
        )

    if change <= -5.0:
        return (
            "The final third was more stable than the opening portion, "
            "suggesting the player settled into a more repeatable rhythm."
        )

    return (
        "The opening and closing portions showed similar overall review "
        "priority, indicating no broad late-session decline."
    )


def _structural_story(
    strongest_category: str,
    weakest_category: str,
) -> str:
    if (
        strongest_category == "Upper Body"
        and weakest_category == "Timing & Coordination"
    ):
        return (
            "Upper-body positioning remained stable, while the largest "
            "variation was temporal rather than structural."
        )

    if weakest_category == "Timing & Coordination":
        return (
            "The main variation occurred in movement timing and sequencing "
            "rather than one clear positional breakdown."
        )

    if weakest_category == "Lower Body":
        return (
            "The main variation was concentrated in lower-body movement and "
            "force-transfer sequencing."
        )

    if weakest_category == "Upper Body":
        return (
            "The primary variability occurred in shooting-arm positioning and "
            "extension."
        )

    if weakest_category == "Ball & Release":
        return (
            "The largest variation appeared in release conditions and ball "
            "delivery."
        )

    return (
        f"{strongest_category} was the most repeatable area, while "
        f"{weakest_category} deserves the closest review."
    )


def _build_contributors(
    grade_components: list[object],
    category_summaries: list[object],
) -> list[GradeContributor]:
    candidates = []

    for component in grade_components:
        impact = (
            component.weighted_score
            - component.weight
            * 70.0
        )

        candidates.append(
            (
                abs(
                    impact
                ),
                "up"
                if impact >= 0
                else "down",
                component.component,
                impact,
                (
                    f"{component.component} scored "
                    f"{component.score:.1f}/100 with a "
                    f"{component.weight:.0%} grade weight."
                ),
            )
        )

    for category in category_summaries:
        impact = (
            category.average_consistency_score
            - 70.0
        ) * 0.15

        candidates.append(
            (
                abs(
                    impact
                ),
                "up"
                if impact >= 0
                else "down",
                category.category,
                impact,
                (
                    f"{category.category} consistency was "
                    f"{category.average_consistency_score:.1f}/100."
                ),
            )
        )

    candidates.sort(
        key=lambda item: (
            -item[
                0
            ],
            item[
                2
            ],
        )
    )

    contributors = []

    seen = set()

    for (
        _,
        direction,
        label,
        impact,
        explanation,
    ) in candidates:
        if label in seen:
            continue

        seen.add(
            label
        )

        contributors.append(
            GradeContributor(
                rank=len(
                    contributors
                )
                + 1,
                direction=direction,
                label=label,
                impact_points=float(
                    impact
                ),
                explanation=explanation,
            )
        )

        if len(
            contributors
        ) >= 5:
            break

    return contributors


def _build_shot_reasons(
    shot_rows: list[object],
    weakest_category: str,
) -> list[ShotReviewReason]:
    ordered = sorted(
        shot_rows,
        key=lambda row: (
            -row.shot_attention_score,
            row.shot_number,
        )
    )

    reasons = []

    for shot in ordered:
        if not shot.matched_feature_row:
            reason = "Feature data missing"
            detail = (
                "Run feature extraction before completing biomechanical review."
            )

        elif shot.result == "missed":
            reason = (
                f"{weakest_category} review"
            )
            detail = (
                "This missed attempt carries elevated review priority and "
                "should be compared with a stable made attempt."
            )

        elif shot.shot_attention_score >= 55.0:
            reason = "High deviation from session pattern"
            detail = (
                "This attempt differs more from the session's stable pattern "
                "than most other shots."
            )

        else:
            reason = "Stable reference attempt"
            detail = (
                "Use this attempt as a comparison reference for higher-"
                "attention shots."
            )

        reasons.append(
            ShotReviewReason(
                trial_id=shot.trial_id,
                shot_number=int(
                    shot.shot_number
                ),
                reason=reason,
                detail=detail,
            )
        )

    return reasons


def _run_session_report_insights_test() -> None:
    """
    Lightweight structural test with fake analysis objects.
    """

    class Fake:
        def __init__(
            self,
            **kwargs,
        ):
            self.__dict__.update(
                kwargs
            )

    session_summary = Fake(
        player_display_name="Test Player",
        assigned_shots=10,
        make_percentage=60.0,
        strongest_category="Upper Body",
        weakest_category="Timing & Coordination",
    )

    shots = [
        Fake(
            matched_feature_row=True,
            shot_attention_score=(
                35.0
                + index
                * 3.0
            ),
            shot_number=index,
            trial_id=f"T{index:04d}",
            result=(
                "missed"
                if index
                in {
                    3,
                    7,
                }
                else "made"
            ),
        )
        for index in range(
            1,
            11,
        )
    ]

    categories = [
        Fake(
            category="Upper Body",
            average_consistency_score=92.0,
            weakest_feature="Elbow angle",
        ),
        Fake(
            category="Timing & Coordination",
            average_consistency_score=58.0,
            weakest_feature="Elbow-to-release timing",
        ),
    ]

    components = [
        Fake(
            component="Execution",
            score=60.0,
            weight=0.30,
            weighted_score=18.0,
        ),
        Fake(
            component="Consistency",
            score=78.0,
            weight=0.35,
            weighted_score=27.3,
        ),
        Fake(
            component="Shot Quality",
            score=76.0,
            weight=0.25,
            weighted_score=19.0,
        ),
        Fake(
            component="Data Completeness",
            score=100.0,
            weight=0.10,
            weighted_score=10.0,
        ),
    ]

    grade_summary = Fake(
        overall_grade=74.3,
        primary_focus="Timing",
        recommended_drill="Tempo free throws",
        estimated_review_minutes=24,
    )

    insights = build_session_report_insights(
        session_summary=session_summary,
        shot_rows=shots,
        category_summaries=categories,
        grade_summary=grade_summary,
        grade_components=components,
        action_items=[],
    )

    assert insights.session_story
    assert insights.grade_explanation
    assert insights.what_went_well
    assert insights.contributors
    assert insights.shot_reasons

    print(
        "SESSION REPORT INSIGHTS TEST PASSED"
    )

    print(
        f"Story: {insights.session_story}"
    )

    print(
        f"Primary focus: {insights.primary_focus}"
    )

    print(
        f"Contributors: {len(insights.contributors)}"
    )

    print(
        f"Shot reasons: {len(insights.shot_reasons)}"
    )


if __name__ == "__main__":
    _run_session_report_insights_test()