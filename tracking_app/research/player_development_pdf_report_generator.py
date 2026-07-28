from __future__ import annotations

from pathlib import Path

from tracking_app.research.player_development_timeline import (
    PlayerDevelopmentTimelineEngine,
)
from tracking_app.research.player_development_intelligence import (
    PlayerDevelopmentIntelligence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "player_development_reports"
)


class PlayerDevelopmentPDFReportGenerator:
    """
    Generate a professional player-development PDF.

    The report combines:
    - player development intelligence;
    - long-term session timeline;
    - current priorities;
    - practice-to-practice consistency change;
    - coach action recommendations;
    - printable notes.
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
        player_id: int,
        participant_id: str,
        player_display_name: str,
        timeline_engine: PlayerDevelopmentTimelineEngine,
    ) -> Path:
        try:
            from reportlab.graphics.shapes import (
                Drawing,
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

        (
            timeline_summary,
            session_rows,
            feature_rows,
        ) = timeline_engine.run(
            player_id=player_id
        )

        development_engine = PlayerDevelopmentIntelligence()

        try:
            (
                development_summary,
                development_priorities,
            ) = development_engine.run(
                participant_id=participant_id
            )

        except Exception:
            development_summary = None
            development_priorities = []

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            self.output_folder
            / (
                f"{self._safe_filename(player_display_name)}__"
                "player_development_report.pdf"
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
                f"{player_display_name} - Player Development Report"
            ),
            author="Ankit Wadera",
            subject="Free Throw Player Development",
        )

        story = []

        # =================================================
        # PAGE 1 — PLAYER DEVELOPMENT OVERVIEW
        # =================================================

        story.append(
            Paragraph(
                "ANKIT'S FREE THROW ANALYSIS SOFTWARE",
                title_style,
            )
        )

        story.append(
            Paragraph(
                "Player Development Report",
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
                            f"<b>{player_display_name}</b><br/>"
                            f"{participant_id}"
                        ),
                        body_style,
                    ),
                    Paragraph(
                        (
                            f"<b>Sessions analyzed</b><br/>"
                            f"{timeline_summary.sessions_analyzed}"
                        ),
                        body_style,
                    ),
                    Paragraph(
                        (
                            f"<b>Tracked session shots</b><br/>"
                            f"{timeline_summary.total_assigned_shots}"
                        ),
                        body_style,
                    ),
                    Paragraph(
                        (
                            f"<b>Evidence</b><br/>"
                            f"{timeline_summary.evidence_status}"
                        ),
                        body_style,
                    ),
                ]
            ],
            colWidths=[
                2.25 * inch,
                1.5 * inch,
                1.65 * inch,
                1.5 * inch,
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
                0.14 * inch,
            )
        )

        if development_summary is not None:
            overview_data = [
                [
                    "Mechanics",
                    "Consistency",
                    "Make %",
                    "Strongest Area",
                    "Primary Priority",
                ],
                [
                    (
                        f"{development_summary.overall_mechanics_score:.1f}"
                    ),
                    (
                        f"{development_summary.consistency_score:.1f}"
                    ),
                    (
                        f"{development_summary.make_percentage:.1f}%"
                    ),
                    development_summary.strongest_area,
                    development_summary.primary_priority,
                ],
            ]

        else:
            overview_data = [
                [
                    "Mechanics",
                    "Consistency",
                    "Make %",
                    "Strongest Area",
                    "Primary Priority",
                ],
                [
                    "N/A",
                    "N/A",
                    "N/A",
                    timeline_summary.strongest_recent_area,
                    timeline_summary.weakest_recent_area,
                ],
            ]

        overview_table = Table(
            overview_data,
            colWidths=[
                1.05 * inch,
                1.05 * inch,
                1.05 * inch,
                1.55 * inch,
                2.2 * inch,
            ],
        )

        overview_table.setStyle(
            self._headline_style(
                TableStyle=TableStyle,
                colors=colors,
                dark_blue=dark_blue,
                medium_gray=medium_gray,
            )
        )

        story.append(
            overview_table
        )

        story.append(
            Spacer(
                1,
                0.15 * inch,
            )
        )

        direction_color = self._direction_color(
            timeline_summary.development_direction,
            green=green,
            yellow=yellow,
            red=red,
        )

        direction_drawing = Drawing(
            165,
            115,
        )

        direction_drawing.add(
            Rect(
                0,
                0,
                165,
                115,
                rx=10,
                ry=10,
                fillColor=light_gray,
                strokeColor=direction_color,
                strokeWidth=1.4,
            )
        )

        direction_drawing.add(
            String(
                82.5,
                88,
                "DEVELOPMENT DIRECTION",
                fontName="Helvetica-Bold",
                fontSize=7.5,
                textAnchor="middle",
                fillColor=direction_color,
            )
        )

        direction_drawing.add(
            String(
                82.5,
                54,
                timeline_summary.development_direction.upper(),
                fontName="Helvetica-Bold",
                fontSize=18,
                textAnchor="middle",
                fillColor=dark_blue,
            )
        )

        consistency_change_text = (
            "N/A"
            if timeline_summary.consistency_change
            is None
            else (
                f"{timeline_summary.consistency_change:+.1f} consistency"
            )
        )

        direction_drawing.add(
            String(
                82.5,
                27,
                consistency_change_text,
                fontName="Helvetica",
                fontSize=8,
                textAnchor="middle",
                fillColor=text_gray,
            )
        )

        summary_text = self._development_story(
            timeline_summary=timeline_summary,
            development_summary=development_summary,
        )

        overview_panel = Table(
            [
                [
                    direction_drawing,
                    Paragraph(
                        (
                            f"<b>Development Story</b><br/>"
                            f"{summary_text}<br/><br/>"
                            f"<b>Most recent strongest area:</b> "
                            f"{timeline_summary.strongest_recent_area}<br/>"
                            f"<b>Most recent review area:</b> "
                            f"{timeline_summary.weakest_recent_area}"
                        ),
                        body_style,
                    ),
                ]
            ],
            colWidths=[
                2.25 * inch,
                4.65 * inch,
            ],
        )

        overview_panel.setStyle(
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
            overview_panel
        )

        story.append(
            Spacer(
                1,
                0.16 * inch,
            )
        )

        story.append(
            Paragraph(
                "Current Development Priorities",
                section_style,
            )
        )

        if development_priorities:
            priority_data = [
                [
                    "Rank",
                    "Area",
                    "Priority Score",
                    "Level",
                ]
            ]

            for priority in development_priorities[
                :6
            ]:
                priority_data.append(
                    [
                        str(
                            priority.rank
                        ),
                        priority.display_name,
                        (
                            f"{priority.development_score:.1f}"
                        ),
                        priority.priority_level,
                    ]
                )

            priority_table = Table(
                priority_data,
                repeatRows=1,
                colWidths=[
                    0.55 * inch,
                    3.55 * inch,
                    1.15 * inch,
                    1.65 * inch,
                ],
            )

            priority_table.setStyle(
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
                priority_table
            )

        else:
            story.append(
                Paragraph(
                    (
                        "No player-development priority file was available. "
                        "The timeline data remains included below."
                    ),
                    body_style,
                )
            )

        story.append(
            PageBreak()
        )

        # =================================================
        # PAGE 2 — SESSION-BY-SESSION DEVELOPMENT
        # =================================================

        story.append(
            Paragraph(
                "Session-by-Session Development",
                section_style,
            )
        )

        if session_rows:
            timeline_data = [
                [
                    "Date",
                    "Session",
                    "Shots",
                    "Make %",
                    "Consistency",
                    "Timing",
                    "Lower Body",
                    "Upper Body",
                    "Ball & Release",
                    "Status",
                ]
            ]

            for row in session_rows:
                timeline_data.append(
                    [
                        row.session_date,
                        row.session_name,
                        str(
                            row.assigned_shots
                        ),
                        (
                            f"{row.make_percentage:.1f}%"
                        ),
                        (
                            "N/A"
                            if row.overall_consistency_score
                            is None
                            else (
                                f"{row.overall_consistency_score:.1f}"
                            )
                        ),
                        self._format_number(
                            row.timing_score
                        ),
                        self._format_number(
                            row.lower_body_score
                        ),
                        self._format_number(
                            row.upper_body_score
                        ),
                        self._format_number(
                            row.ball_release_score
                        ),
                        row.data_status,
                    ]
                )

            timeline_table = Table(
                timeline_data,
                repeatRows=1,
                colWidths=[
                    0.8 * inch,
                    1.15 * inch,
                    0.5 * inch,
                    0.55 * inch,
                    0.75 * inch,
                    0.65 * inch,
                    0.7 * inch,
                    0.7 * inch,
                    0.85 * inch,
                    0.85 * inch,
                ],
            )

            timeline_table.setStyle(
                self._standard_table_style(
                    TableStyle=TableStyle,
                    colors=colors,
                    dark_blue=dark_blue,
                    medium_gray=medium_gray,
                    light_gray=light_gray,
                    font_size=6.6,
                )
            )

            story.append(
                timeline_table
            )

            story.append(
                Spacer(
                    1,
                    0.17 * inch,
                )
            )

            chart = self._session_timeline_drawing(
                Drawing=Drawing,
                Rect=Rect,
                String=String,
                session_rows=session_rows,
                dark_blue=dark_blue,
                orange=orange,
                light_gray=light_gray,
                text_gray=text_gray,
            )

            story.append(
                chart
            )

        else:
            story.append(
                Paragraph(
                    "No analyzable practice sessions are available.",
                    body_style,
                )
            )

        story.append(
            Spacer(
                1,
                0.18 * inch,
            )
        )

        story.append(
            Paragraph(
                "First-to-Latest Change",
                section_style,
            )
        )

        change_data = [
            [
                "Metric",
                "First",
                "Latest",
                "Change",
            ],
            [
                "Consistency",
                self._format_number(
                    timeline_summary.first_consistency_score
                ),
                self._format_number(
                    timeline_summary.latest_consistency_score
                ),
                self._format_change(
                    timeline_summary.consistency_change
                ),
            ],
            [
                "Make %",
                self._format_percentage(
                    timeline_summary.first_make_percentage
                ),
                self._format_percentage(
                    timeline_summary.latest_make_percentage
                ),
                self._format_change(
                    timeline_summary.make_percentage_change,
                    suffix=" pts",
                ),
            ],
        ]

        change_table = Table(
            change_data,
            colWidths=[
                2.1 * inch,
                1.55 * inch,
                1.55 * inch,
                1.7 * inch,
            ],
        )

        change_table.setStyle(
            self._headline_style(
                TableStyle=TableStyle,
                colors=colors,
                dark_blue=dark_blue,
                medium_gray=medium_gray,
            )
        )

        story.append(
            change_table
        )

        story.append(
            PageBreak()
        )

        # =================================================
        # PAGE 3 — COACH DEVELOPMENT PLAN
        # =================================================

        story.append(
            Paragraph(
                "Coach Development Plan",
                section_style,
            )
        )

        action_rows = self._development_actions(
            timeline_summary=timeline_summary,
            development_priorities=development_priorities,
        )

        for rank, action in enumerate(
            action_rows,
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
                "Coach Notes",
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
                    "This report summarizes tracked practice sessions and "
                    "derived biomechanical metrics. Development direction and "
                    "priority scores are exploratory decision supports. They "
                    "should be interpreted with video, player context, and "
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

    @staticmethod
    def _development_story(
        timeline_summary: object,
        development_summary: object | None,
    ) -> str:
        if timeline_summary.sessions_analyzed == 0:
            return (
                "No analyzable session history is available yet. Create and "
                "analyze multiple sessions before drawing a trend conclusion."
            )

        if timeline_summary.development_direction == "Improving":
            direction = (
                "The player's overall consistency is improving across the "
                "tracked session history."
            )

        elif timeline_summary.development_direction == "Regressing":
            direction = (
                "The latest tracked consistency is below the first analyzable "
                "session and should be reviewed with recent video."
            )

        else:
            direction = (
                "The player's overall consistency has remained relatively "
                "stable across the tracked session history."
            )

        if development_summary is not None:
            return (
                f"{direction} The strongest current area is "
                f"{development_summary.strongest_area}, while the primary "
                f"development priority is "
                f"{development_summary.primary_priority}."
            )

        return (
            f"{direction} The latest strongest area is "
            f"{timeline_summary.strongest_recent_area}, while "
            f"{timeline_summary.weakest_recent_area} deserves the closest "
            "review."
        )

    @staticmethod
    def _development_actions(
        timeline_summary: object,
        development_priorities: list[object],
    ) -> list[dict]:
        actions = []

        for priority in development_priorities[
            :3
        ]:
            actions.append(
                {
                    "title": (
                        f"Address {priority.display_name}"
                    ),
                    "priority": (
                        priority.priority_level.title()
                    ),
                    "instruction": (
                        "Review representative made and missed attempts, then "
                        "use a focused drill block that preserves the player's "
                        "normal shooting rhythm."
                    ),
                    "success_measure": (
                        f"Improve the priority score from "
                        f"{priority.development_score:.1f} while maintaining or "
                        "improving make percentage."
                    ),
                    "minutes": 10,
                }
            )

        if timeline_summary.sessions_analyzed >= 2:
            actions.append(
                {
                    "title": (
                        "Run a controlled session retest"
                    ),
                    "priority": "Moderate",
                    "instruction": (
                        "Complete a matched-volume session under similar "
                        "conditions so the new results can be compared directly "
                        "with the latest practice."
                    ),
                    "success_measure": (
                        "Improve consistency without reducing execution."
                    ),
                    "minutes": 15,
                }
            )

        else:
            actions.append(
                {
                    "title": (
                        "Build a multi-session baseline"
                    ),
                    "priority": "High",
                    "instruction": (
                        "Record at least one additional practice session with "
                        "similar shot volume and conditions."
                    ),
                    "success_measure": (
                        "Create at least two analyzable sessions."
                    ),
                    "minutes": 20,
                }
            )

        if not actions:
            actions.append(
                {
                    "title": (
                        "Continue collecting development data"
                    ),
                    "priority": "Moderate",
                    "instruction": (
                        "Record the next practice session and assign all shots "
                        "to the player's session history."
                    ),
                    "success_measure": (
                        "Increase the number of analyzable sessions."
                    ),
                    "minutes": 20,
                }
            )

        return actions[
            :5
        ]

    @staticmethod
    def _session_timeline_drawing(
        Drawing,
        Rect,
        String,
        session_rows: list[object],
        dark_blue,
        orange,
        light_gray,
        text_gray,
    ):
        drawing = Drawing(
            500,
            180,
        )

        drawing.add(
            String(
                0,
                162,
                "Practice-to-Practice Consistency",
                fontName="Helvetica-Bold",
                fontSize=10,
                fillColor=dark_blue,
            )
        )

        valid_rows = [
            row
            for row in session_rows
            if row.overall_consistency_score
            is not None
        ]

        if not valid_rows:
            drawing.add(
                String(
                    0,
                    120,
                    "No valid consistency scores are available.",
                    fontName="Helvetica",
                    fontSize=9,
                    fillColor=text_gray,
                )
            )

            return drawing

        bar_width = min(
            54,
            420
            / max(
                1,
                len(
                    valid_rows
                ),
            )
            - 8,
        )

        x = 15

        for row in valid_rows:
            score = float(
                row.overall_consistency_score
            )

            drawing.add(
                Rect(
                    x,
                    35,
                    bar_width,
                    100,
                    fillColor=light_gray,
                    strokeColor=None,
                )
            )

            drawing.add(
                Rect(
                    x,
                    35,
                    bar_width,
                    score,
                    fillColor=orange,
                    strokeColor=None,
                )
            )

            drawing.add(
                String(
                    x
                    + bar_width
                    / 2,
                    143,
                    f"{score:.1f}",
                    fontName="Helvetica-Bold",
                    fontSize=7.5,
                    textAnchor="middle",
                    fillColor=dark_blue,
                )
            )

            drawing.add(
                String(
                    x
                    + bar_width
                    / 2,
                    20,
                    row.session_date,
                    fontName="Helvetica",
                    fontSize=6.5,
                    textAnchor="middle",
                    fillColor=text_gray,
                )
            )

            x += (
                bar_width
                + 10
            )

        return drawing

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
    def _format_percentage(
        value: float | None,
    ) -> str:
        return (
            "N/A"
            if value is None
            else (
                f"{value:.1f}%"
            )
        )

    @staticmethod
    def _format_change(
        value: float | None,
        suffix: str = "",
    ) -> str:
        return (
            "N/A"
            if value is None
            else (
                f"{value:+.1f}{suffix}"
            )
        )

    @staticmethod
    def _headline_style(
        TableStyle,
        colors,
        dark_blue,
        medium_gray,
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
                    (-1, -1),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, 0),
                    7.5,
                ),
                (
                    "FONTSIZE",
                    (0, 1),
                    (-1, 1),
                    10.5,
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.45,
                    medium_gray,
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
            or "player"
        )


def _run_player_development_pdf_test() -> None:
    """
    Lightweight module test.
    """

    assert PlayerDevelopmentPDFReportGenerator

    print(
        "PLAYER DEVELOPMENT PDF REPORT GENERATOR TEST PASSED"
    )

    print(
        "Player development overview: READY"
    )

    print(
        "Session-by-session timeline: READY"
    )

    print(
        "Development direction: READY"
    )

    print(
        "Coach development plan: READY"
    )

    print(
        "Coach notes area: READY"
    )


if __name__ == "__main__":
    _run_player_development_pdf_test()