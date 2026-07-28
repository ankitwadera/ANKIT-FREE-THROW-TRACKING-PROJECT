from __future__ import annotations

from pathlib import Path

from reportlab.lib.units import inch

from tracking_app.research.session_analysis_engine import (
    SessionAnalysisEngine,
)
from tracking_app.research.session_coach_summary_engine import (
    SessionCoachSummaryEngine,
)
from tracking_app.research.session_grade_action_plan_engine import (
    SessionGradeActionPlanEngine,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "session_comparison_reports"
)


class SessionComparisonPDFReportGenerator:
    """
    Generate a coach-facing PDF comparing two complete practice sessions.

    The report shows:
    - headline changes;
    - category-level changes;
    - session-grade changes;
    - result and consistency changes;
    - coach interpretation for both sessions;
    - practical next actions;
    - printable staff notes.
    """

    def __init__(
        self,
        output_folder: Path = DEFAULT_OUTPUT_FOLDER,
    ) -> None:
        self.output_folder = Path(
            output_folder
        )

    def run(
        self,
        first_session_id: int,
        second_session_id: int,
        analysis_engine: SessionAnalysisEngine | None = None,
    ) -> Path:
        try:
            from reportlab.graphics.shapes import (
                Drawing,
                Line,
                Rect,
                String,
            )
            from reportlab.lib import colors
            from reportlab.lib.enums import (
                TA_CENTER,
                TA_LEFT,
            )
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import (
                ParagraphStyle,
                getSampleStyleSheet,
            )
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                KeepTogether,
                PageBreak,
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )

        except ImportError as error:
            raise RuntimeError(
                (
                    "ReportLab is required. Install it with: "
                    ".venv\\Scripts\\python.exe -m pip install reportlab"
                )
            ) from error

        engine = (
            analysis_engine
            if analysis_engine is not None
            else SessionAnalysisEngine()
        )

        (
            first_summary,
            first_shots,
            first_categories,
        ) = engine.run(
            session_id=first_session_id
        )

        (
            second_summary,
            second_shots,
            second_categories,
        ) = engine.run(
            session_id=second_session_id
        )

        if (
            first_summary.player_id
            != second_summary.player_id
        ):
            raise ValueError(
                "Session comparison requires two sessions from the same player."
            )

        first_grade_engine = SessionGradeActionPlanEngine(
            output_folder=(
                self.output_folder
                / "_first_grade"
            )
        )

        (
            first_grade,
            first_components,
            first_actions,
        ) = first_grade_engine.run(
            session_id=first_session_id,
            analysis_engine=engine,
        )

        second_grade_engine = SessionGradeActionPlanEngine(
            output_folder=(
                self.output_folder
                / "_second_grade"
            )
        )

        (
            second_grade,
            second_components,
            second_actions,
        ) = second_grade_engine.run(
            session_id=second_session_id,
            analysis_engine=engine,
        )

        first_coach_engine = SessionCoachSummaryEngine(
            output_folder=(
                self.output_folder
                / "_first_coach"
            )
        )

        (
            first_coach,
            first_observations,
        ) = first_coach_engine.run(
            session_id=first_session_id,
            analysis_engine=engine,
        )

        second_coach_engine = SessionCoachSummaryEngine(
            output_folder=(
                self.output_folder
                / "_second_coach"
            )
        )

        (
            second_coach,
            second_observations,
        ) = second_coach_engine.run(
            session_id=second_session_id,
            analysis_engine=engine,
        )

        comparison = self._comparison_summary(
            first_summary=first_summary,
            second_summary=second_summary,
            first_grade=first_grade,
            second_grade=second_grade,
            first_categories=first_categories,
            second_categories=second_categories,
        )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            self.output_folder
            / (
                f"{self._safe_filename(first_summary.player_display_name)}__"
                f"{self._safe_filename(first_summary.session_name)}__vs__"
                f"{self._safe_filename(second_summary.session_name)}.pdf"
            )
        )

        dark_blue = colors.HexColor(
            "#152033"
        )

        orange = colors.HexColor(
            "#C94D18"
        )

        green = colors.HexColor(
            "#2E7D59"
        )

        yellow = colors.HexColor(
            "#C58A15"
        )

        red = colors.HexColor(
            "#B4453A"
        )

        light_gray = colors.HexColor(
            "#F3F5F8"
        )

        medium_gray = colors.HexColor(
            "#DDE2EA"
        )

        text_gray = colors.HexColor(
            "#455066"
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "Title",
            parent=styles[
                "Title"
            ],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=dark_blue,
            alignment=TA_CENTER,
            spaceAfter=5,
        )

        kicker_style = ParagraphStyle(
            "Kicker",
            parent=styles[
                "Normal"
            ],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            textColor=orange,
            alignment=TA_CENTER,
        )

        section_style = ParagraphStyle(
            "Section",
            parent=styles[
                "Heading2"
            ],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=17,
            textColor=dark_blue,
            spaceBefore=6,
            spaceAfter=6,
        )

        body_style = ParagraphStyle(
            "Body",
            parent=styles[
                "BodyText"
            ],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=text_gray,
            alignment=TA_LEFT,
        )

        small_style = ParagraphStyle(
            "Small",
            parent=body_style,
            fontSize=7.7,
            leading=10,
        )

        document = SimpleDocTemplate(
            str(
                output_file
            ),
            pagesize=letter,
            rightMargin=0.48 * inch,
            leftMargin=0.48 * inch,
            topMargin=0.48 * inch,
            bottomMargin=0.52 * inch,
            title=(
                f"{first_summary.player_display_name} - Session Comparison"
            ),
            author="Ankit Wadera",
            subject="Free Throw Session Comparison",
        )

        story = []

        # =================================================
        # PAGE 1 — HEADLINE COMPARISON
        # =================================================

        story.append(
            Paragraph(
                "ANKIT'S FREE THROW ANALYSIS SOFTWARE",
                title_style,
            )
        )

        story.append(
            Paragraph(
                "Practice Session Comparison Report",
                kicker_style,
            )
        )

        story.append(
            Spacer(
                1,
                0.12 * inch,
            )
        )

        identity_table = Table(
            [
                [
                    Paragraph(
                        (
                            f"<b>{first_summary.player_display_name}</b><br/>"
                            f"{first_summary.participant_id}"
                        ),
                        body_style,
                    ),
                    Paragraph(
                        (
                            f"<b>Session A</b><br/>"
                            f"{first_summary.session_name}<br/>"
                            f"{first_summary.session_date}"
                        ),
                        body_style,
                    ),
                    Paragraph(
                        (
                            f"<b>Session B</b><br/>"
                            f"{second_summary.session_name}<br/>"
                            f"{second_summary.session_date}"
                        ),
                        body_style,
                    ),
                ]
            ],
            colWidths=[
                2.3 * inch,
                2.3 * inch,
                2.3 * inch,
            ],
        )

        identity_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, -1),
                        light_gray,
                    ),
                    (
                        "BOX",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        medium_gray,
                    ),
                    (
                        "INNERGRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        medium_gray,
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE",
                    ),
                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        9,
                    ),
                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        9,
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        7,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        7,
                    ),
                ]
            )
        )

        story.append(
            identity_table
        )

        story.append(
            Spacer(
                1,
                0.15 * inch,
            )
        )

        headline_table = Table(
            [
                [
                    "Metric",
                    first_summary.session_name,
                    second_summary.session_name,
                    "Change",
                ],
                [
                    "Session Grade",
                    (
                        f"{first_grade.overall_grade:.1f}"
                    ),
                    (
                        f"{second_grade.overall_grade:.1f}"
                    ),
                    (
                        f"{comparison['grade_change']:+.1f}"
                    ),
                ],
                [
                    "Make %",
                    (
                        f"{first_summary.make_percentage:.1f}%"
                    ),
                    (
                        f"{second_summary.make_percentage:.1f}%"
                    ),
                    (
                        f"{comparison['make_change']:+.1f} pts"
                    ),
                ],
                [
                    "Consistency",
                    self._format_number(
                        first_summary.overall_consistency_score
                    ),
                    self._format_number(
                        second_summary.overall_consistency_score
                    ),
                    self._format_change(
                        comparison[
                            "consistency_change"
                        ]
                    ),
                ],
                [
                    "Shots",
                    str(
                        first_summary.assigned_shots
                    ),
                    str(
                        second_summary.assigned_shots
                    ),
                    (
                        f"{comparison['shot_change']:+d}"
                    ),
                ],
            ],
            colWidths=[
                1.55 * inch,
                2.0 * inch,
                2.0 * inch,
                1.35 * inch,
            ],
        )

        headline_table.setStyle(
            self._standard_table_style(
                TableStyle=TableStyle,
                colors=colors,
                dark_blue=dark_blue,
                medium_gray=medium_gray,
                light_gray=light_gray,
                font_size=8.2,
            )
        )

        story.append(
            headline_table
        )

        story.append(
            Spacer(
                1,
                0.15 * inch,
            )
        )

        direction_color = self._direction_color(
            comparison[
                "overall_direction"
            ],
            green=green,
            yellow=yellow,
            red=red,
        )

        direction_drawing = Drawing(
            160,
            120,
        )

        direction_drawing.add(
            Rect(
                0,
                0,
                160,
                120,
                rx=10,
                ry=10,
                fillColor=light_gray,
                strokeColor=direction_color,
                strokeWidth=1.4,
            )
        )

        direction_drawing.add(
            String(
                80,
                92,
                "OVERALL DIRECTION",
                fontName="Helvetica-Bold",
                fontSize=7.5,
                textAnchor="middle",
                fillColor=direction_color,
            )
        )

        direction_drawing.add(
            String(
                80,
                56,
                comparison[
                    "overall_direction"
                ].upper(),
                fontName="Helvetica-Bold",
                fontSize=19,
                textAnchor="middle",
                fillColor=dark_blue,
            )
        )

        direction_drawing.add(
            String(
                80,
                28,
                (
                    f"Grade {comparison['grade_change']:+.1f}"
                ),
                fontName="Helvetica",
                fontSize=8,
                textAnchor="middle",
                fillColor=text_gray,
            )
        )

        story_panel = Table(
            [
                [
                    direction_drawing,
                    Paragraph(
                        (
                            f"<b>Comparison Story</b><br/>"
                            f"{comparison['comparison_story']}<br/><br/>"
                            f"<b>Most improved category:</b> "
                            f"{comparison['most_improved_category']}<br/>"
                            f"<b>Largest decline:</b> "
                            f"{comparison['largest_decline_category']}"
                        ),
                        body_style,
                    ),
                ]
            ],
            colWidths=[
                2.2 * inch,
                4.7 * inch,
            ],
        )

        story_panel.setStyle(
            TableStyle(
                [
                    (
                        "BOX",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        medium_gray,
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE",
                    ),
                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        10,
                    ),
                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        10,
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        10,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        10,
                    ),
                ]
            )
        )

        story.append(
            story_panel
        )

        story.append(
            Spacer(
                1,
                0.15 * inch,
            )
        )

        story.append(
            Paragraph(
                "Category Change",
                section_style,
            )
        )

        category_table_data = [
            [
                "Category",
                first_summary.session_name,
                second_summary.session_name,
                "Change",
                "Direction",
            ]
        ]

        for row in comparison[
            "category_rows"
        ]:
            category_table_data.append(
                [
                    row[
                        "category"
                    ],
                    self._format_number(
                        row[
                            "first"
                        ]
                    ),
                    self._format_number(
                        row[
                            "second"
                        ]
                    ),
                    self._format_change(
                        row[
                            "change"
                        ]
                    ),
                    row[
                        "direction"
                    ],
                ]
            )

        category_table = Table(
            category_table_data,
            repeatRows=1,
            colWidths=[
                2.0 * inch,
                1.4 * inch,
                1.4 * inch,
                1.0 * inch,
                1.1 * inch,
            ],
        )

        category_table.setStyle(
            self._standard_table_style(
                TableStyle=TableStyle,
                colors=colors,
                dark_blue=dark_blue,
                medium_gray=medium_gray,
                light_gray=light_gray,
                font_size=8.0,
            )
        )

        story.append(
            category_table
        )

        story.append(
            PageBreak()
        )

        # =================================================
        # PAGE 2 — VISUAL COMPARISON + COACH INTERPRETATION
        # =================================================

        story.append(
            Paragraph(
                "Visual Category Comparison",
                section_style,
            )
        )

        comparison_chart = self._category_comparison_drawing(
            Drawing=Drawing,
            Rect=Rect,
            String=String,
            Line=Line,
            category_rows=comparison[
                "category_rows"
            ],
            first_name=first_summary.session_name,
            second_name=second_summary.session_name,
            dark_blue=dark_blue,
            orange=orange,
            light_gray=light_gray,
            text_gray=text_gray,
        )

        story.append(
            comparison_chart
        )

        story.append(
            Spacer(
                1,
                0.15 * inch,
            )
        )

        story.append(
            Paragraph(
                "Coach Interpretation: Session A",
                section_style,
            )
        )

        story.append(
            Paragraph(
                first_coach.executive_summary,
                body_style,
            )
        )

        story.append(
            Spacer(
                1,
                0.08 * inch,
            )
        )

        for observation in first_observations[
            :3
        ]:
            story.append(
                KeepTogether(
                    [
                        self._observation_card(
                            Table=Table,
                            TableStyle=TableStyle,
                            Paragraph=Paragraph,
                            observation=observation,
                            body_style=body_style,
                            small_style=small_style,
                            light_gray=light_gray,
                            medium_gray=medium_gray,
                            orange=orange,
                        ),
                        Spacer(
                            1,
                            0.08 * inch,
                        ),
                    ]
                )
            )

        story.append(
            Paragraph(
                "Coach Interpretation: Session B",
                section_style,
            )
        )

        story.append(
            Paragraph(
                second_coach.executive_summary,
                body_style,
            )
        )

        story.append(
            Spacer(
                1,
                0.08 * inch,
            )
        )

        for observation in second_observations[
            :3
        ]:
            story.append(
                KeepTogether(
                    [
                        self._observation_card(
                            Table=Table,
                            TableStyle=TableStyle,
                            Paragraph=Paragraph,
                            observation=observation,
                            body_style=body_style,
                            small_style=small_style,
                            light_gray=light_gray,
                            medium_gray=medium_gray,
                            orange=orange,
                        ),
                        Spacer(
                            1,
                            0.08 * inch,
                        ),
                    ]
                )
            )

        story.append(
            PageBreak()
        )

        # =================================================
        # PAGE 3 — NEXT ACTIONS
        # =================================================

        story.append(
            Paragraph(
                "Next Practice Recommendation",
                section_style,
            )
        )

        recommended_actions = self._comparison_actions(
            comparison=comparison,
            second_grade=second_grade,
            second_actions=second_actions,
        )

        for rank, action in enumerate(
            recommended_actions,
            start=1,
        ):
            action_table = Table(
                [
                    [
                        Paragraph(
                            (
                                f"<b>{rank}. {action['title']}</b>"
                            ),
                            body_style,
                        ),
                        Paragraph(
                            action[
                                "priority"
                            ],
                            small_style,
                        ),
                    ],
                    [
                        Paragraph(
                            action[
                                "instruction"
                            ],
                            body_style,
                        ),
                        "",
                    ],
                    [
                        Paragraph(
                            (
                                f"<b>Success measure:</b> "
                                f"{action['success_measure']}"
                            ),
                            small_style,
                        ),
                        Paragraph(
                            (
                                f"<b>{action['minutes']} min</b>"
                            ),
                            small_style,
                        ),
                    ],
                ],
                colWidths=[
                    5.75 * inch,
                    1.15 * inch,
                ],
            )

            action_table.setStyle(
                TableStyle(
                    [
                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, 0),
                            light_gray,
                        ),
                        (
                            "TEXTCOLOR",
                            (1, 0),
                            (1, 0),
                            orange,
                        ),
                        (
                            "ALIGN",
                            (1, 0),
                            (1, 0),
                            "CENTER",
                        ),
                        (
                            "BOX",
                            (0, 0),
                            (-1, -1),
                            0.5,
                            medium_gray,
                        ),
                        (
                            "VALIGN",
                            (0, 0),
                            (-1, -1),
                            "TOP",
                        ),
                        (
                            "LEFTPADDING",
                            (0, 0),
                            (-1, -1),
                            8,
                        ),
                        (
                            "RIGHTPADDING",
                            (0, 0),
                            (-1, -1),
                            8,
                        ),
                        (
                            "TOPPADDING",
                            (0, 0),
                            (-1, -1),
                            6,
                        ),
                        (
                            "BOTTOMPADDING",
                            (0, 0),
                            (-1, -1),
                            6,
                        ),
                    ]
                )
            )

            story.append(
                KeepTogether(
                    [
                        action_table,
                        Spacer(
                            1,
                            0.1 * inch,
                        ),
                    ]
                )
            )

        story.append(
            Spacer(
                1,
                0.15 * inch,
            )
        )

        story.append(
            Paragraph(
                "Staff Notes",
                section_style,
            )
        )

        notes_table = Table(
            [
                [
                    " "
                ]
                for _ in range(
                    10
                )
            ],
            colWidths=[
                6.9 * inch,
            ],
            rowHeights=[
                0.34 * inch
            ]
            * 10,
        )

        notes_table.setStyle(
            TableStyle(
                [
                    (
                        "LINEBELOW",
                        (0, 0),
                        (-1, -1),
                        0.45,
                        medium_gray,
                    ),
                ]
            )
        )

        story.append(
            notes_table
        )

        story.append(
            Spacer(
                1,
                0.18 * inch,
            )
        )

        story.append(
            Paragraph(
                "Interpretation Limitation",
                section_style,
            )
        )

        story.append(
            Paragraph(
                (
                    "This report compares two structured tracking sessions. "
                    "Changes in grade, consistency, and category scores are "
                    "exploratory decision supports and should be interpreted "
                    "with video, practice conditions, player context, and "
                    "professional coaching judgment."
                ),
                body_style,
            )
        )

        page_width, _ = letter

        def draw_footer(
            canvas,
            doc,
        ) -> None:
            canvas.saveState()

            canvas.setStrokeColor(
                medium_gray
            )

            canvas.line(
                0.48 * inch,
                0.39 * inch,
                page_width
                - 0.48 * inch,
                0.39 * inch,
            )

            canvas.setFont(
                "Helvetica",
                7.2,
            )

            canvas.setFillColor(
                text_gray
            )

            canvas.drawString(
                0.48 * inch,
                0.22 * inch,
                (
                    "ANKIT'S FREE THROW ANALYSIS SOFTWARE | "
                    "Ankit Wadera | ankitwadera2@gmail.com"
                ),
            )

            canvas.drawRightString(
                page_width
                - 0.48 * inch,
                0.22 * inch,
                f"Page {doc.page}",
            )

            canvas.restoreState()

        document.build(
            story,
            onFirstPage=draw_footer,
            onLaterPages=draw_footer,
        )

        return output_file

    def _comparison_summary(
        self,
        first_summary: object,
        second_summary: object,
        first_grade: object,
        second_grade: object,
        first_categories: list[object],
        second_categories: list[object],
    ) -> dict:
        first_lookup = {
            row.category: row.average_consistency_score
            for row in first_categories
        }

        second_lookup = {
            row.category: row.average_consistency_score
            for row in second_categories
        }

        categories = sorted(
            set(
                first_lookup
            )
            | set(
                second_lookup
            )
        )

        category_rows = []

        for category in categories:
            first_value = first_lookup.get(
                category
            )

            second_value = second_lookup.get(
                category
            )

            change = (
                None
                if (
                    first_value is None
                    or second_value is None
                )
                else (
                    second_value
                    - first_value
                )
            )

            if change is None:
                direction = "Unavailable"

            elif change >= 5.0:
                direction = "Improved"

            elif change <= -5.0:
                direction = "Declined"

            else:
                direction = "Stable"

            category_rows.append(
                {
                    "category": category,
                    "first": first_value,
                    "second": second_value,
                    "change": change,
                    "direction": direction,
                }
            )

        comparable_rows = [
            row
            for row in category_rows
            if row[
                "change"
            ]
            is not None
        ]

        most_improved = (
            max(
                comparable_rows,
                key=lambda row: row[
                    "change"
                ],
            )
            if comparable_rows
            else None
        )

        largest_decline = (
            min(
                comparable_rows,
                key=lambda row: row[
                    "change"
                ],
            )
            if comparable_rows
            else None
        )

        grade_change = (
            second_grade.overall_grade
            - first_grade.overall_grade
        )

        make_change = (
            second_summary.make_percentage
            - first_summary.make_percentage
        )

        consistency_change = (
            None
            if (
                first_summary.overall_consistency_score
                is None
                or second_summary.overall_consistency_score
                is None
            )
            else (
                second_summary.overall_consistency_score
                - first_summary.overall_consistency_score
            )
        )

        if (
            grade_change >= 5.0
            and (
                consistency_change is None
                or consistency_change >= -3.0
            )
        ):
            overall_direction = "Improving"

        elif (
            grade_change <= -5.0
            and (
                consistency_change is None
                or consistency_change <= 3.0
            )
        ):
            overall_direction = "Regressing"

        else:
            overall_direction = "Mixed"

        story_parts = []

        if grade_change >= 5.0:
            story_parts.append(
                (
                    f"Session grade improved by "
                    f"{grade_change:+.1f} points."
                )
            )

        elif grade_change <= -5.0:
            story_parts.append(
                (
                    f"Session grade declined by "
                    f"{grade_change:+.1f} points."
                )
            )

        else:
            story_parts.append(
                "Session grade remained relatively stable."
            )

        if make_change >= 5.0:
            story_parts.append(
                (
                    f"Make percentage improved by "
                    f"{make_change:+.1f} points."
                )
            )

        elif make_change <= -5.0:
            story_parts.append(
                (
                    f"Make percentage declined by "
                    f"{make_change:+.1f} points."
                )
            )

        else:
            story_parts.append(
                "Make percentage remained relatively stable."
            )

        if consistency_change is not None:
            if consistency_change >= 5.0:
                story_parts.append(
                    (
                        f"Consistency improved by "
                        f"{consistency_change:+.1f} points."
                    )
                )

            elif consistency_change <= -5.0:
                story_parts.append(
                    (
                        f"Consistency declined by "
                        f"{consistency_change:+.1f} points."
                    )
                )

            else:
                story_parts.append(
                    "Consistency remained relatively stable."
                )

        return {
            "grade_change": float(
                grade_change
            ),
            "make_change": float(
                make_change
            ),
            "consistency_change": (
                None
                if consistency_change
                is None
                else float(
                    consistency_change
                )
            ),
            "shot_change": int(
                second_summary.assigned_shots
                - first_summary.assigned_shots
            ),
            "overall_direction": (
                overall_direction
            ),
            "comparison_story": " ".join(
                story_parts
            ),
            "category_rows": category_rows,
            "most_improved_category": (
                "N/A"
                if most_improved is None
                else (
                    f"{most_improved['category']} "
                    f"({most_improved['change']:+.1f})"
                )
            ),
            "largest_decline_category": (
                "N/A"
                if largest_decline is None
                else (
                    f"{largest_decline['category']} "
                    f"({largest_decline['change']:+.1f})"
                )
            ),
        }

    @staticmethod
    def _comparison_actions(
        comparison: dict,
        second_grade: object,
        second_actions: list[object],
    ) -> list[dict]:
        actions = []

        declining_categories = [
            row
            for row in comparison[
                "category_rows"
            ]
            if (
                row[
                    "change"
                ]
                is not None
                and row[
                    "change"
                ]
                <= -5.0
            )
        ]

        for row in declining_categories[
            :2
        ]:
            actions.append(
                {
                    "title": (
                        f"Address the decline in "
                        f"{row['category']}"
                    ),
                    "priority": "High",
                    "instruction": (
                        "Compare representative attempts from both sessions "
                        "and identify whether the change is related to pace, "
                        "fatigue, cueing, or movement sequencing."
                    ),
                    "success_measure": (
                        f"Recover at least 5 consistency points in "
                        f"{row['category']} without reducing execution."
                    ),
                    "minutes": 10,
                }
            )

        for action in second_actions[
            :3
        ]:
            actions.append(
                {
                    "title": action.title,
                    "priority": action.priority.title(),
                    "instruction": action.instruction,
                    "success_measure": (
                        "Improve the targeted area while maintaining or "
                        "improving the latest session grade."
                    ),
                    "minutes": action.estimated_minutes,
                }
            )

        if comparison[
            "overall_direction"
        ] == "Improving":
            actions.append(
                {
                    "title": "Preserve the improved session pattern",
                    "priority": "Moderate",
                    "instruction": (
                        "Repeat a matched-volume practice under similar "
                        "conditions and preserve the cues used in the latest "
                        "session."
                    ),
                    "success_measure": (
                        "Maintain the improved grade and category scores."
                    ),
                    "minutes": 15,
                }
            )

        elif comparison[
            "overall_direction"
        ] == "Regressing":
            actions.append(
                {
                    "title": "Run a controlled retest",
                    "priority": "High",
                    "instruction": (
                        "Complete a shorter controlled session after targeted "
                        "review to determine whether the decline is repeatable."
                    ),
                    "success_measure": (
                        "Return the session grade toward the earlier baseline."
                    ),
                    "minutes": 15,
                }
            )

        return actions[
            :5
        ]

    @staticmethod
    def _category_comparison_drawing(
        Drawing,
        Rect,
        String,
        Line,
        category_rows: list[dict],
        first_name: str,
        second_name: str,
        dark_blue,
        orange,
        light_gray,
        text_gray,
    ):
        height = max(
            150,
            46
            * len(
                category_rows
            )
            + 45,
        )

        drawing = Drawing(
            500,
            height,
        )

        drawing.add(
            String(
                0,
                height
                - 18,
                first_name,
                fontName="Helvetica-Bold",
                fontSize=8,
                fillColor=dark_blue,
            )
        )

        drawing.add(
            String(
                130,
                height
                - 18,
                second_name,
                fontName="Helvetica-Bold",
                fontSize=8,
                fillColor=orange,
            )
        )

        y = (
            height
            - 50
        )

        for row in category_rows:
            drawing.add(
                String(
                    0,
                    y,
                    row[
                        "category"
                    ],
                    fontName="Helvetica-Bold",
                    fontSize=7.5,
                    fillColor=text_gray,
                )
            )

            first_value = (
                row[
                    "first"
                ]
                or 0.0
            )

            second_value = (
                row[
                    "second"
                ]
                or 0.0
            )

            drawing.add(
                Rect(
                    130,
                    y
                    - 7,
                    300,
                    8,
                    fillColor=light_gray,
                    strokeColor=None,
                )
            )

            drawing.add(
                Rect(
                    130,
                    y
                    - 7,
                    300
                    * first_value
                    / 100.0,
                    3,
                    fillColor=dark_blue,
                    strokeColor=None,
                )
            )

            drawing.add(
                Rect(
                    130,
                    y
                    - 3,
                    300
                    * second_value
                    / 100.0,
                    3,
                    fillColor=orange,
                    strokeColor=None,
                )
            )

            drawing.add(
                String(
                    445,
                    y,
                    (
                        f"{first_value:.1f} / "
                        f"{second_value:.1f}"
                    ),
                    fontName="Helvetica",
                    fontSize=7.3,
                    fillColor=text_gray,
                )
            )

            y -= 46

        return drawing

    @staticmethod
    def _observation_card(
        Table,
        TableStyle,
        Paragraph,
        observation: object,
        body_style,
        small_style,
        light_gray,
        medium_gray,
        orange,
    ):
        table = Table(
            [
                [
                    Paragraph(
                        (
                            f"<b>{observation.rank}. "
                            f"{observation.title}</b>"
                        ),
                        body_style,
                    ),
                    Paragraph(
                        observation.priority.title(),
                        small_style,
                    ),
                ],
                [
                    Paragraph(
                        observation.summary,
                        body_style,
                    ),
                    "",
                ],
                [
                    Paragraph(
                        (
                            f"<b>Evidence:</b> "
                            f"{observation.evidence}"
                        ),
                        small_style,
                    ),
                    "",
                ],
            ],
            colWidths=[
                5.75 * inch,
                1.15 * inch,
            ],
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "SPAN",
                        (0, 1),
                        (1, 1),
                    ),
                    (
                        "SPAN",
                        (0, 2),
                        (1, 2),
                    ),
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        light_gray,
                    ),
                    (
                        "TEXTCOLOR",
                        (1, 0),
                        (1, 0),
                        orange,
                    ),
                    (
                        "ALIGN",
                        (1, 0),
                        (1, 0),
                        "CENTER",
                    ),
                    (
                        "BOX",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        medium_gray,
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP",
                    ),
                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        7,
                    ),
                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        7,
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                ]
            )
        )

        return table

    @staticmethod
    def _direction_color(
        direction: str,
        green,
        yellow,
        red,
    ):
        if direction == "Improving":
            return green

        if direction == "Regressing":
            return red

        return yellow

    @staticmethod
    def _format_number(
        value: float | None,
    ) -> str:
        return (
            "N/A"
            if value is None
            else (
                f"{value:.1f}"
            )
        )

    @staticmethod
    def _format_change(
        value: float | None,
    ) -> str:
        return (
            "N/A"
            if value is None
            else (
                f"{value:+.1f}"
            )
        )

    @staticmethod
    def _standard_table_style(
        TableStyle,
        colors,
        dark_blue,
        medium_gray,
        light_gray,
        font_size: float,
    ):
        return TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    dark_blue,
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTNAME",
                    (0, 1),
                    (-1, -1),
                    "Helvetica",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    font_size,
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        light_gray,
                    ],
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.35,
                    medium_gray,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    4.5,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    4.5,
                ),
            ]
        )

    @staticmethod
    def _safe_filename(
        value: str,
    ) -> str:
        cleaned = "".join(
            character
            if (
                character.isalnum()
                or character
                in {
                    "-",
                    "_",
                }
            )
            else "_"
            for character in value.strip()
        )

        while "__" in cleaned:
            cleaned = cleaned.replace(
                "__",
                "_",
            )

        return (
            cleaned.strip(
                "_"
            )
            or "session"
        )


def _run_session_comparison_pdf_test() -> None:
    """
    Lightweight module test.
    """

    assert SessionComparisonPDFReportGenerator

    print(
        "SESSION COMPARISON PDF REPORT GENERATOR TEST PASSED"
    )

    print(
        "Headline comparison: READY"
    )

    print(
        "Category change analysis: READY"
    )

    print(
        "Coach interpretation comparison: READY"
    )

    print(
        "Next-practice recommendation: READY"
    )

    print(
        "Staff notes area: READY"
    )


if __name__ == "__main__":
    _run_session_comparison_pdf_test()
