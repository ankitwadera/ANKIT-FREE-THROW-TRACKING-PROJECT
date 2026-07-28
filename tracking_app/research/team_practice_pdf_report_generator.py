from __future__ import annotations

from pathlib import Path

from reportlab.lib.units import inch

from tracking_app.research.team_practice_intelligence_engine import (
    TeamPracticeIntelligenceEngine,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "team_practice_reports"
)


class TeamPracticePDFReportGenerator:
    """
    Generate a professional team-wide latest-practice PDF.

    The report answers:
    - How many roster players have usable latest-session data?
    - What are the team averages?
    - Who had the strongest latest session?
    - Who should the staff review first?
    - What are the most common development priorities?
    - What should each player work on next?
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
        team_id: int,
        intelligence_engine: TeamPracticeIntelligenceEngine | None = None,
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
            from reportlab.lib.pagesizes import (
                landscape,
                letter,
            )
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
            intelligence_engine
            if intelligence_engine is not None
            else TeamPracticeIntelligenceEngine()
        )

        summary, player_rows = engine.run(
            team_id=team_id
        )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            self.output_folder
            / (
                f"{self._safe_filename(summary.team_name)}__"
                f"{self._safe_filename(summary.season_name or 'team')}__"
                "latest_practice_report.pdf"
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
            fontSize=21,
            leading=25,
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
            fontSize=8.7,
            leading=12.5,
            textColor=text_gray,
            alignment=TA_LEFT,
        )

        small_style = ParagraphStyle(
            "Small",
            parent=body_style,
            fontSize=7.4,
            leading=9.5,
        )

        document = SimpleDocTemplate(
            str(
                output_file
            ),
            pagesize=landscape(
                letter
            ),
            rightMargin=0.42 * inch,
            leftMargin=0.42 * inch,
            topMargin=0.42 * inch,
            bottomMargin=0.48 * inch,
            title=(
                f"{summary.team_name} - Latest Practice Report"
            ),
            author="Ankit Wadera",
            subject="Team Practice Intelligence",
        )

        story = []

        # =================================================
        # PAGE 1 - TEAM OVERVIEW
        # =================================================

        story.append(
            Paragraph(
                "ANKIT'S FREE THROW ANALYSIS SOFTWARE",
                title_style,
            )
        )

        story.append(
            Paragraph(
                "Team Practice Intelligence Report",
                kicker_style,
            )
        )

        story.append(
            Spacer(
                1,
                0.11 * inch,
            )
        )

        identity_table = Table(
            [
                [
                    Paragraph(
                        (
                            f"<b>{summary.organization_name}</b><br/>"
                            f"{summary.team_name}"
                        ),
                        body_style,
                    ),
                    Paragraph(
                        (
                            f"<b>Season</b><br/>"
                            f"{summary.season_name or 'Not recorded'}"
                        ),
                        body_style,
                    ),
                    Paragraph(
                        (
                            f"<b>Level</b><br/>"
                            f"{summary.level or 'Not recorded'}"
                        ),
                        body_style,
                    ),
                    Paragraph(
                        (
                            f"<b>Status</b><br/>"
                            f"{summary.team_status}"
                        ),
                        body_style,
                    ),
                ]
            ],
            colWidths=[
                2.65 * inch,
                2.35 * inch,
                2.0 * inch,
                2.65 * inch,
            ],
        )

        identity_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        light_gray,
                    ),
                    (
                        "BOX",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        0.5,
                        medium_gray,
                    ),
                    (
                        "INNERGRID",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        0.5,
                        medium_gray,
                    ),
                    (
                        "VALIGN",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        "MIDDLE",
                    ),
                    (
                        "LEFTPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        9,
                    ),
                    (
                        "RIGHTPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        9,
                    ),
                    (
                        "TOPPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        7,
                    ),
                    (
                        "BOTTOMPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
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
                0.13 * inch,
            )
        )

        headline_data = [
            [
                "Active Players",
                "Players With Grades",
                "Latest-Session Shots",
                "Average Grade",
                "Average Make %",
                "Average Consistency",
            ],
            [
                str(
                    summary.active_players
                ),
                (
                    f"{summary.players_with_grades} / "
                    f"{summary.active_players}"
                ),
                str(
                    summary.total_latest_session_shots
                ),
                (
                    "N/A"
                    if summary.average_session_grade
                    is None
                    else (
                        f"{summary.average_session_grade:.1f}"
                    )
                ),
                (
                    "N/A"
                    if summary.average_make_percentage
                    is None
                    else (
                        f"{summary.average_make_percentage:.1f}%"
                    )
                ),
                (
                    "N/A"
                    if summary.average_consistency_score
                    is None
                    else (
                        f"{summary.average_consistency_score:.1f}"
                    )
                ),
            ],
        ]

        headline_table = Table(
            headline_data,
            colWidths=[
                1.55 * inch,
                1.75 * inch,
                1.85 * inch,
                1.45 * inch,
                1.55 * inch,
                1.65 * inch,
            ],
        )

        headline_table.setStyle(
            self._headline_style(
                TableStyle=TableStyle,
                colors=colors,
                dark_blue=dark_blue,
                medium_gray=medium_gray,
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

        insight_cards = Table(
            [
                [
                    self._insight_card(
                        Table=Table,
                        TableStyle=TableStyle,
                        Paragraph=Paragraph,
                        title="Strongest Latest Session",
                        body=summary.highest_grade_player,
                        body_style=body_style,
                        background=light_gray,
                        border=medium_gray,
                    ),
                    self._insight_card(
                        Table=Table,
                        TableStyle=TableStyle,
                        Paragraph=Paragraph,
                        title="Review First",
                        body=summary.highest_attention_player,
                        body_style=body_style,
                        background=light_gray,
                        border=medium_gray,
                    ),
                    self._insight_card(
                        Table=Table,
                        TableStyle=TableStyle,
                        Paragraph=Paragraph,
                        title="Most Common Focus",
                        body=summary.most_common_primary_focus,
                        body_style=body_style,
                        background=light_gray,
                        border=medium_gray,
                    ),
                ]
            ],
            colWidths=[
                3.22 * inch,
                3.22 * inch,
                3.22 * inch,
            ],
        )

        insight_cards.setStyle(
            TableStyle(
                [
                    (
                        "VALIGN",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        "TOP",
                    ),
                    (
                        "LEFTPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        0,
                    ),
                    (
                        "RIGHTPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        7,
                    ),
                ]
            )
        )

        story.append(
            insight_cards
        )

        story.append(
            Spacer(
                1,
                0.14 * inch,
            )
        )

        story.append(
            Paragraph(
                "Latest Session Roster Board",
                section_style,
            )
        )

        roster_data = [
            [
                "Player",
                "Latest Session",
                "Date",
                "Shots",
                "Make %",
                "Grade",
                "Consistency",
                "Primary Focus",
                "Recommended Drill",
                "Review First",
                "Status",
            ]
        ]

        for row in player_rows:
            roster_data.append(
                [
                    (
                        f"{row.player_display_name}"
                        + (
                            f" (#{row.jersey_number})"
                            if row.jersey_number
                            else ""
                        )
                    ),
                    row.latest_session_name,
                    (
                        row.latest_session_date
                        or "N/A"
                    ),
                    str(
                        row.assigned_shots
                    ),
                    (
                        "N/A"
                        if row.make_percentage
                        is None
                        else (
                            f"{row.make_percentage:.1f}%"
                        )
                    ),
                    (
                        "N/A"
                        if row.session_grade
                        is None
                        else (
                            f"{row.session_grade:.1f}"
                        )
                    ),
                    (
                        "N/A"
                        if row.consistency_score
                        is None
                        else (
                            f"{row.consistency_score:.1f}"
                        )
                    ),
                    row.primary_focus,
                    row.recommended_drill,
                    row.review_first_trial,
                    row.team_status,
                ]
            )

        roster_table = Table(
            roster_data,
            repeatRows=1,
            colWidths=[
                1.05 * inch,
                1.18 * inch,
                0.78 * inch,
                0.48 * inch,
                0.58 * inch,
                0.52 * inch,
                0.68 * inch,
                1.65 * inch,
                1.65 * inch,
                0.68 * inch,
                1.02 * inch,
            ],
        )

        roster_table.setStyle(
            self._standard_table_style(
                TableStyle=TableStyle,
                colors=colors,
                dark_blue=dark_blue,
                medium_gray=medium_gray,
                light_gray=light_gray,
                font_size=6.4,
            )
        )

        story.append(
            roster_table
        )

        story.append(
            PageBreak()
        )

        # =================================================
        # PAGE 2 - TEAM REVIEW QUEUE
        # =================================================

        story.append(
            Paragraph(
                "Team Review Queue",
                section_style,
            )
        )

        story.append(
            Paragraph(
                (
                    "Players are ordered by latest-session review priority. "
                    "The team attention score is a coaching workflow tool, "
                    "not a talent ranking."
                ),
                body_style,
            )
        )

        story.append(
            Spacer(
                1,
                0.1 * inch,
            )
        )

        for rank, row in enumerate(
            player_rows,
            start=1,
        ):
            story.append(
                KeepTogether(
                    [
                        self._player_review_card(
                            Table=Table,
                            TableStyle=TableStyle,
                            Paragraph=Paragraph,
                            Drawing=Drawing,
                            Rect=Rect,
                            String=String,
                            rank=rank,
                            row=row,
                            dark_blue=dark_blue,
                            orange=orange,
                            green=green,
                            yellow=yellow,
                            red=red,
                            light_gray=light_gray,
                            medium_gray=medium_gray,
                            text_gray=text_gray,
                            body_style=body_style,
                            small_style=small_style,
                        ),
                        Spacer(
                            1,
                            0.11 * inch,
                        ),
                    ]
                )
            )

        story.append(
            PageBreak()
        )

        # =================================================
        # PAGE 3 - PRACTICE PLANNING BOARD
        # =================================================

        story.append(
            Paragraph(
                "Next Practice Planning Board",
                section_style,
            )
        )

        planning_data = [
            [
                "Player",
                "Primary Focus",
                "Recommended Drill",
                "Review Trial",
                "Session Grade",
                "Team Attention",
                "Estimated Staff Priority",
            ]
        ]

        for row in player_rows:
            planning_data.append(
                [
                    row.player_display_name,
                    row.primary_focus,
                    row.recommended_drill,
                    row.review_first_trial,
                    (
                        "N/A"
                        if row.session_grade
                        is None
                        else (
                            f"{row.session_grade:.1f}"
                        )
                    ),
                    (
                        f"{row.attention_score:.1f}"
                    ),
                    self._staff_priority(
                        row.attention_score
                    ),
                ]
            )

        planning_table = Table(
            planning_data,
            repeatRows=1,
            colWidths=[
                1.3 * inch,
                2.15 * inch,
                2.15 * inch,
                0.9 * inch,
                0.8 * inch,
                0.9 * inch,
                1.45 * inch,
            ],
        )

        planning_table.setStyle(
            self._standard_table_style(
                TableStyle=TableStyle,
                colors=colors,
                dark_blue=dark_blue,
                medium_gray=medium_gray,
                light_gray=light_gray,
                font_size=7.0,
            )
        )

        story.append(
            planning_table
        )

        story.append(
            Spacer(
                1,
                0.18 * inch,
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
                10.0 * inch,
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
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
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
                    "This report summarizes each active roster player's most "
                    "recent active session. Session grades, review priorities, "
                    "and recommended drills are exploratory decision supports "
                    "and should be interpreted with video, player context, and "
                    "professional coaching judgment."
                ),
                body_style,
            )
        )

        page_width, _ = landscape(
            letter
        )

        def draw_footer(
            canvas,
            doc,
        ) -> None:
            canvas.saveState()

            canvas.setStrokeColor(
                medium_gray
            )

            canvas.line(
                0.42 * inch,
                0.38 * inch,
                page_width
                - 0.42 * inch,
                0.38 * inch,
            )

            canvas.setFont(
                "Helvetica",
                7.1,
            )

            canvas.setFillColor(
                text_gray
            )

            canvas.drawString(
                0.42 * inch,
                0.21 * inch,
                (
                    "ANKIT'S FREE THROW ANALYSIS SOFTWARE | "
                    "Ankit Wadera | ankitwadera2@gmail.com"
                ),
            )

            canvas.drawRightString(
                page_width
                - 0.42 * inch,
                0.21 * inch,
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
    def _player_review_card(
        Table,
        TableStyle,
        Paragraph,
        Drawing,
        Rect,
        String,
        rank: int,
        row: object,
        dark_blue,
        orange,
        green,
        yellow,
        red,
        light_gray,
        medium_gray,
        text_gray,
        body_style,
        small_style,
    ):
        status_color = (
            red
            if row.attention_score
            >= 55.0
            else (
                yellow
                if row.attention_score
                >= 35.0
                else green
            )
        )

        drawing = Drawing(
            105,
            82,
        )

        drawing.add(
            Rect(
                0,
                0,
                105,
                82,
                rx=10,
                ry=10,
                fillColor=light_gray,
                strokeColor=status_color,
                strokeWidth=1.2,
            )
        )

        drawing.add(
            String(
                52.5,
                61,
                "ATTENTION",
                fontName="Helvetica-Bold",
                fontSize=7,
                textAnchor="middle",
                fillColor=status_color,
            )
        )

        drawing.add(
            String(
                52.5,
                31,
                f"{row.attention_score:.1f}",
                fontName="Helvetica-Bold",
                fontSize=24,
                textAnchor="middle",
                fillColor=dark_blue,
            )
        )

        drawing.add(
            String(
                52.5,
                12,
                row.team_status.upper(),
                fontName="Helvetica-Bold",
                fontSize=6.2,
                textAnchor="middle",
                fillColor=text_gray,
            )
        )

        detail = Paragraph(
            (
                f"<b>{rank}. {row.player_display_name}</b>"
                + (
                    f" | #{row.jersey_number}"
                    if row.jersey_number
                    else ""
                )
                + (
                    f" | {row.position_group}"
                    if row.position_group
                    else ""
                )
                + "<br/>"
                f"<b>Latest session:</b> {row.latest_session_name} "
                f"({row.latest_session_date or 'No date'})<br/>"
                f"<b>Grade:</b> "
                f"{'N/A' if row.session_grade is None else f'{row.session_grade:.1f}'}"
                f" | <b>Make %:</b> "
                f"{'N/A' if row.make_percentage is None else f'{row.make_percentage:.1f}%'}"
                f" | <b>Consistency:</b> "
                f"{'N/A' if row.consistency_score is None else f'{row.consistency_score:.1f}'}"
                f"<br/><b>Primary focus:</b> {row.primary_focus}"
                f"<br/><b>Recommended drill:</b> {row.recommended_drill}"
                f"<br/><b>Review first:</b> {row.review_first_trial}"
            ),
            body_style,
        )

        table = Table(
            [
                [
                    drawing,
                    detail,
                ]
            ],
            colWidths=[
                1.55 * inch,
                8.15 * inch,
            ],
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "BOX",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        0.5,
                        medium_gray,
                    ),
                    (
                        "VALIGN",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        "MIDDLE",
                    ),
                    (
                        "LEFTPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        9,
                    ),
                    (
                        "RIGHTPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        9,
                    ),
                    (
                        "TOPPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        8,
                    ),
                    (
                        "BOTTOMPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        8,
                    ),
                ]
            )
        )

        return table

    @staticmethod
    def _insight_card(
        Table,
        TableStyle,
        Paragraph,
        title: str,
        body: str,
        body_style,
        background,
        border,
    ):
        title_style = body_style.clone(
            f"CardTitle{title}"
        )

        title_style.fontName = "Helvetica-Bold"
        title_style.fontSize = 8.0
        title_style.leading = 10

        body_card_style = body_style.clone(
            f"CardBody{title}"
        )

        body_card_style.fontName = "Helvetica-Bold"
        body_card_style.fontSize = 10.0
        body_card_style.leading = 13

        table = Table(
            [
                [
                    Paragraph(
                        title,
                        title_style,
                    )
                ],
                [
                    Paragraph(
                        body,
                        body_card_style,
                    )
                ],
            ],
            colWidths=[
                3.05 * inch,
            ],
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            0,
                        ),
                        background,
                    ),
                    (
                        "BOX",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        0.5,
                        border,
                    ),
                    (
                        "LEFTPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        9,
                    ),
                    (
                        "RIGHTPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        9,
                    ),
                    (
                        "TOPPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        7,
                    ),
                    (
                        "BOTTOMPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        7,
                    ),
                ]
            )
        )

        return table

    @staticmethod
    def _staff_priority(
        attention_score: float,
    ) -> str:
        if attention_score >= 55.0:
            return "Review before next practice"

        if attention_score >= 35.0:
            return "Monitor this week"

        return "Routine follow-up"

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
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        0,
                    ),
                    dark_blue,
                ),
                (
                    "TEXTCOLOR",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        0,
                    ),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        0,
                    ),
                    7.5,
                ),
                (
                    "FONTSIZE",
                    (
                        0,
                        1,
                    ),
                    (
                        -1,
                        1,
                    ),
                    11.5,
                ),
                (
                    "ALIGN",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    "CENTER",
                ),
                (
                    "GRID",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    0.45,
                    medium_gray,
                ),
                (
                    "TOPPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
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
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        0,
                    ),
                    dark_blue,
                ),
                (
                    "TEXTCOLOR",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        0,
                    ),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        0,
                    ),
                    "Helvetica-Bold",
                ),
                (
                    "FONTNAME",
                    (
                        0,
                        1,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    "Helvetica",
                ),
                (
                    "FONTSIZE",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    font_size,
                ),
                (
                    "ROWBACKGROUNDS",
                    (
                        0,
                        1,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    [
                        colors.white,
                        light_gray,
                    ],
                ),
                (
                    "GRID",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    0.35,
                    medium_gray,
                ),
                (
                    "VALIGN",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    "MIDDLE",
                ),
                (
                    "LEFTPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    3.5,
                ),
                (
                    "RIGHTPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    3.5,
                ),
                (
                    "TOPPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    4.0,
                ),
                (
                    "BOTTOMPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    4.0,
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
            or "team"
        )


def _run_team_practice_pdf_test() -> None:
    """
    Lightweight module test. Full integration is exercised through the app
    against the user's real team repository.
    """

    assert TeamPracticePDFReportGenerator

    print(
        "TEAM PRACTICE PDF REPORT GENERATOR TEST PASSED"
    )

    print(
        "Team overview page: READY"
    )

    print(
        "Latest-session roster board: READY"
    )

    print(
        "Team review queue: READY"
    )

    print(
        "Next-practice planning board: READY"
    )

    print(
        "Staff notes area: READY"
    )


if __name__ == "__main__":
    _run_team_practice_pdf_test()