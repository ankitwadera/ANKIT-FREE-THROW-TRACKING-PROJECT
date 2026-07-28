from __future__ import annotations

import csv
import io
import json
import os
import shutil
import tempfile
import zipfile
import hashlib
from dataclasses import asdict
from pathlib import Path
from datetime import date, datetime
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio.v2 as imageio
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from tracking_app.analyzer import ShotAnalyzer
from tracking_app.basketball_event_mapper import BasketballEventMapper
from tracking_app.phase_biomechanics import RefinedPhaseBiomechanicsAnalyzer
from tracking_app.research.personal_shot_scoring_engine import (
    PersonalShotScoringEngine,
)
from tracking_app.research.movement_pattern_interpreter import (
    MovementPatternInterpreter,
)
from tracking_app.research.coach_intelligence_layer import (
    CoachIntelligenceLayer,
)
from tracking_app.research.session_history_analyzer import (
    SessionHistoryAnalyzer,
)
from tracking_app.research.player_development_intelligence import (
    PlayerDevelopmentIntelligence,
)
from tracking_app.research.player_history_repository import (
    PlayerHistoryRepository,
)
from tracking_app.research.practice_session_repository import (
    PracticeSessionRepository,
)
from tracking_app.research.organization_team_repository import (
    OrganizationTeamRepository,
)
from tracking_app.research.team_dashboard_engine import (
    TeamDashboardEngine,
)
from tracking_app.research.team_practice_intelligence_engine import (
    TeamPracticeIntelligenceEngine,
)
from tracking_app.research.team_practice_pdf_report_generator import (
    TeamPracticePDFReportGenerator,
)
from tracking_app.research.player_development_timeline import (
    PlayerDevelopmentTimelineEngine,
)
from tracking_app.research.player_development_pdf_report_generator import (
    PlayerDevelopmentPDFReportGenerator,
)
from tracking_app.research.session_analysis_engine import (
    SessionAnalysisEngine,
)
from tracking_app.research.session_comparison_pdf_report_generator import (
    SessionComparisonPDFReportGenerator,
)
from tracking_app.research.session_coach_summary_engine import (
    SessionCoachSummaryEngine,
)
from tracking_app.research.session_pdf_report_generator import (
    SessionPDFReportGenerator,
)
from tracking_app.research.session_grade_action_plan_engine import (
    SessionGradeActionPlanEngine,
)
from tracking_app.research.session_report_insights import (
    build_session_report_insights,
)
from tracking_app.research.feature_metadata import (
    get_feature_metadata,
)
from tracking_app.shot_feature_extractor import ShotFeatureExtractor
from tracking_app.shot_timeline_builder import ShotTimelineBuilder
from tracking_app.timeseries import TimeSeriesAnalyzer
from tracking_app.whole_body_synchronization import (
    WholeBodySynchronizationEngine,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_app_data_root() -> Path:
    """
    Resolve the normal persistent local application data directory.
    """

    configured_value = os.getenv(
        "ANKIT_APP_DATA_ROOT",
        "data",
    ).strip()

    configured_path = Path(
        configured_value
    ).expanduser()

    if not configured_path.is_absolute():
        configured_path = (
            PROJECT_ROOT
            / configured_path
        )

    return configured_path.resolve()


DEFAULT_APP_DATA_ROOT = resolve_app_data_root()

PACKAGED_EXECUTIVE_DEMO_ROOT = (
    PROJECT_ROOT
    / "deployment_data"
)


def is_hosted_portfolio_mode() -> bool:
    """
    Return True when the application is running as the public portfolio site.
    """

    app_mode = os.getenv(
        "APP_MODE",
        "private",
    ).strip().lower()

    return app_mode in {
        "public",
        "portfolio",
        "hosted",
    }


def active_workspace_kind() -> str:
    """
    Return blank, executive_demo, or local.
    """

    if not is_hosted_portfolio_mode():
        return "local"

    return str(
        st.session_state.get(
            "public_workspace_kind",
            "",
        )
    ).strip()


def active_data_root() -> Path:
    """
    Return the data directory for the current browser session.

    Hosted visitors receive an isolated temporary copy. Local users continue
    using the normal persistent data directory.
    """

    if not is_hosted_portfolio_mode():
        return DEFAULT_APP_DATA_ROOT

    workspace_path = st.session_state.get(
        "public_workspace_data_root"
    )

    if not workspace_path:
        raise RuntimeError(
            "Choose Blank Workspace or Executive Demo before opening the platform."
        )

    return Path(
        workspace_path
    )


def initialize_public_workspace(
    workspace_kind: str,
) -> None:
    """
    Create an isolated temporary workspace for one hosted visitor.
    """

    if workspace_kind not in {
        "blank",
        "executive_demo",
    }:
        raise ValueError(
            f"Unsupported workspace kind: {workspace_kind}"
        )

    previous_root = st.session_state.get(
        "public_workspace_data_root"
    )

    if previous_root:
        shutil.rmtree(
            previous_root,
            ignore_errors=True,
        )

    workspace_root = Path(
        tempfile.mkdtemp(
            prefix=(
                "ankit_blank_"
                if workspace_kind == "blank"
                else "ankit_executive_demo_"
            )
        )
    )

    if workspace_kind == "executive_demo":
        if not PACKAGED_EXECUTIVE_DEMO_ROOT.exists():
            raise FileNotFoundError(
                (
                    "The packaged Executive Demo was not found at "
                    f"{PACKAGED_EXECUTIVE_DEMO_ROOT}."
                )
            )

        shutil.copytree(
            PACKAGED_EXECUTIVE_DEMO_ROOT,
            workspace_root,
            dirs_exist_ok=True,
        )

    (
        workspace_root
        / "player_history_files"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    st.session_state[
        "public_workspace_kind"
    ] = workspace_kind

    st.session_state[
        "public_workspace_data_root"
    ] = str(
        workspace_root
    )

    st.session_state[
        "private_workspace"
    ] = (
        "Executive Demo"
        if workspace_kind == "executive_demo"
        else "Home"
    )

    st.session_state[
        "pending_workspace_navigation"
    ] = None


def reset_public_workspace() -> None:
    """
    Delete the visitor's temporary workspace and return to the chooser.
    """

    workspace_root = st.session_state.get(
        "public_workspace_data_root"
    )

    if workspace_root:
        shutil.rmtree(
            workspace_root,
            ignore_errors=True,
        )

    st.session_state.pop(
        "public_workspace_data_root",
        None,
    )

    st.session_state.pop(
        "public_workspace_kind",
        None,
    )

    st.session_state[
        "private_workspace"
    ] = "Home"


def render_public_workspace_chooser() -> None:
    """
    Render the hosted landing page before a workspace is initialized.
    """

    st.markdown(
        """
        <div class="bms-hero">
            <div class="bms-kicker">
                Basketball biomechanics and player-development platform
            </div>
            <h1>ANKIT'S FREE THROW ANALYSIS SOFTWARE</h1>
            <p>
                Analyze structured three-dimensional free-throw tracking data,
                organize practice sessions, compare attempts, and translate
                movement evidence into coach-readable development information.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title(
        "Choose how to begin",
        (
            "Start with an empty workspace for your own compatible tracking "
            "files, or load the populated Executive Demo."
        ),
    )

    blank_column, demo_column = st.columns(
        2,
        gap="large",
    )

    with blank_column:
        st.markdown(
            """
            <div class="bms-score-card" style="min-height:24rem;">
                <div class="bms-kicker" style="color:#c94d18;">
                    Your own data
                </div>
                <h2 style="margin:0 0 0.7rem;color:#152033;">
                    Blank Workspace
                </h2>
                <p style="color:#455066;line-height:1.65;">
                    Begin with no players, teams, sessions, or shot files.
                    Create your own structure and upload compatible free-throw
                    tracking data.
                </p>
                <div class="bms-footer-note" style="margin-top:1rem;">
                    <strong>Accepted uploads:</strong><br>
                    JSON files containing structured 3D basketball free-throw
                    tracking data. Each file should contain one attempt,
                    frame-by-frame tracking, ball XYZ coordinates, player body
                    keypoints, participant ID, trial ID, and made/missed result.
                    <br><br>
                    MP4 video, CSV, Excel, box-score JSON, play-by-play JSON,
                    shot charts, photos, and unrelated JSON are not compatible.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Start Blank Workspace",
            type="primary",
            use_container_width=True,
            key="start_blank_public_workspace",
        ):
            initialize_public_workspace(
                "blank"
            )
            st.rerun()

    with demo_column:
        st.markdown(
            """
            <div class="bms-score-card" style="min-height:24rem;">
                <div class="bms-kicker" style="color:#c94d18;">
                    Portfolio demonstration
                </div>
                <h2 style="margin:0 0 0.7rem;color:#152033;">
                    Executive Demo
                </h2>
                <p style="color:#455066;line-height:1.65;">
                    Load a populated front-office example with organizations,
                    teams, players, sessions, shot files, development timelines,
                    dashboards, comparisons, and professional reports.
                </p>
                <div class="bms-footer-note" style="margin-top:1rem;">
                    The demo is copied into a private temporary workspace for
                    this visitor. You can explore and edit that copy without
                    changing the packaged demonstration or another visitor's
                    workspace.
                    <br><br>
                    Named players are portfolio labels. The underlying tracking
                    evidence originates from SPL Open Data and is not represented
                    as tracking collected from those named athletes.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Load Executive Demo",
            type="primary",
            use_container_width=True,
            key="load_executive_demo_workspace",
        ):
            initialize_public_workspace(
                "executive_demo"
            )
            st.rerun()

    st.markdown(
        """
        <div class="bms-footer">
            Created by Ankit Wadera · ankitwadera2@gmail.com
        </div>
        """,
        unsafe_allow_html=True,
    )


MASTER_FEATURE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "all_shot_features.csv"
)

PERSONAL_DIAGNOSTIC_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "research"
    / "personal_signal_diagnostics"
    / "personal_feature_diagnostics.csv"
)


def is_true_public_mode() -> bool:
    """
    Compatibility alias for hosted portfolio mode.

    Hosted workspaces are isolated and temporary, so visitors may use the
    complete creation, upload, session, roster, and analysis workflows.
    """

    return is_hosted_portfolio_mode()


def render_breadcrumbs(
    workspace: str,
) -> None:
    """Display a compact product-location indicator."""

    group_lookup = {
        "Home": "Start",
        "Global Search": "Start",
        "Executive Demo": "Start",
        "Single Shot Analysis": "Analyze",
        "Comparison Studio": "Analyze",
        "Session Analysis": "Analyze",
        "Session Comparison": "Analyze",
        "Player Profile": "Develop",
        "Player Development": "Develop",
        "Team Dashboard": "Develop",
        "Sessions & Shot Library": "Manage",
        "Roster & Organizations": "Manage",
        "Reports Center": "Reports",
        "Launch Readiness": "Reports",
        "Full Studio": "Advanced",
        "Public Demo": "Start",
    }

    group_name = group_lookup.get(
        workspace,
        "Workspace",
    )

    st.markdown(
        (
            '<div style="font-size:0.82rem;color:#6b7280;'
            'margin:0.15rem 0 0.7rem;">'
            f'ANKIT&apos;S FREE THROW ANALYSIS SOFTWARE'
            f' &nbsp;›&nbsp; {group_name}'
            f' &nbsp;›&nbsp; <strong>{workspace}</strong>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def apply_publisher_chart_style() -> None:
    """Apply one consistent visual language to Matplotlib charts."""

    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#d8dee9",
            "axes.labelcolor": "#364152",
            "axes.titlecolor": "#152033",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.color": "#5e6879",
            "ytick.color": "#5e6879",
            "grid.color": "#e8ecf2",
            "grid.alpha": 0.85,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "lines.linewidth": 2.2,
            "lines.markersize": 5.5,
            "savefig.facecolor": "#ffffff",
            "savefig.bbox": "tight",
        }
    )


def configure_page() -> None:
    apply_publisher_chart_style()

    st.set_page_config(
        page_title="Ankit's Free Throw Analysis Software",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        :root {
            --bms-bg: #f5f7fb;
            --bms-surface: #ffffff;
            --bms-ink: #152033;
            --bms-muted: #667085;
            --bms-border: #e4e8f0;
            --bms-accent: #f26a2e;
            --bms-accent-dark: #c94d18;
            --bms-blue: #2e6eea;
            --bms-green: #159455;
            --bms-amber: #c47a13;
            --bms-red: #c94343;
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(46,110,234,0.06), transparent 28rem),
                linear-gradient(180deg, #f8faff 0%, var(--bms-bg) 100%);
            color: var(--bms-ink);
        }

        .block-container {
            max-width: 1480px;
            padding-top: 1.5rem;
            padding-bottom: 4rem;
        }

        [data-testid="stSidebar"] {
            background: #111b2e;
        }

        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        [data-testid="stSidebar"] .stAlert {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
        }

        [data-testid="stMetric"] {
            background: var(--bms-surface);
            border: 1px solid var(--bms-border);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 22px rgba(28, 39, 60, 0.05);
        }

        [data-testid="stMetricLabel"] {
            color: var(--bms-muted);
        }

        [data-testid="stMetricValue"] {
            color: var(--bms-ink);
        }

        .bms-hero {
            background:
                linear-gradient(135deg, rgba(17,27,46,0.98), rgba(31,53,91,0.95));
            border-radius: 24px;
            padding: 2rem 2.2rem;
            margin-bottom: 1.4rem;
            box-shadow: 0 18px 46px rgba(17,27,46,0.18);
            position: relative;
            overflow: hidden;
        }

        .bms-hero:after {
            content: "";
            position: absolute;
            width: 18rem;
            height: 18rem;
            right: -6rem;
            top: -8rem;
            border-radius: 50%;
            background: rgba(242,106,46,0.18);
        }

        .bms-kicker {
            color: #ffb18e;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }

        .bms-hero h1 {
            color: white;
            margin: 0;
            font-size: clamp(2rem, 4vw, 3.5rem);
            letter-spacing: -0.04em;
        }

        .bms-hero p {
            color: #d5deee;
            max-width: 52rem;
            margin: 0.8rem 0 0;
            font-size: 1.03rem;
            line-height: 1.65;
        }

        .bms-section-title {
            margin: 2rem 0 0.9rem;
        }

        .bms-section-title h2 {
            margin: 0;
            color: var(--bms-ink);
            font-size: 1.45rem;
        }

        .bms-section-title p {
            margin: 0.35rem 0 0;
            color: var(--bms-muted);
        }

        .bms-score-card {
            background: var(--bms-surface);
            border: 1px solid var(--bms-border);
            border-radius: 22px;
            padding: 1.5rem;
            box-shadow: 0 12px 30px rgba(28,39,60,0.06);
            min-height: 18.5rem;
        }

        .bms-score-ring {
            width: 11.5rem;
            height: 11.5rem;
            margin: 0.5rem auto 1rem;
            border-radius: 50%;
            display: grid;
            place-items: center;
            position: relative;
        }

        .bms-score-ring:after {
            content: "";
            width: 8.7rem;
            height: 8.7rem;
            border-radius: 50%;
            background: white;
            position: absolute;
        }

        .bms-score-value {
            z-index: 2;
            text-align: center;
        }

        .bms-score-value strong {
            display: block;
            font-size: 3rem;
            line-height: 1;
            color: var(--bms-ink);
        }

        .bms-score-value span {
            color: var(--bms-muted);
            font-size: 0.82rem;
            font-weight: 700;
        }

        .bms-confidence {
            text-align: center;
            color: var(--bms-muted);
            font-size: 0.9rem;
        }

        .bms-category-card {
            background: var(--bms-surface);
            border: 1px solid var(--bms-border);
            border-radius: 18px;
            padding: 1.1rem;
            min-height: 8.8rem;
            box-shadow: 0 8px 22px rgba(28,39,60,0.04);
        }

        .bms-category-card .label {
            color: var(--bms-muted);
            font-size: 0.83rem;
            font-weight: 700;
            min-height: 2.3rem;
        }

        .bms-category-card .score {
            color: var(--bms-ink);
            font-size: 2rem;
            font-weight: 800;
            margin-top: 0.15rem;
        }

        .bms-category-card .status {
            color: var(--bms-muted);
            font-size: 0.78rem;
            margin-top: 0.25rem;
        }

        .bms-summary-card {
            background: linear-gradient(135deg, #ffffff, #f7f9fd);
            border: 1px solid var(--bms-border);
            border-left: 5px solid var(--bms-accent);
            border-radius: 18px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 8px 24px rgba(28,39,60,0.04);
            margin-bottom: 0.8rem;
        }

        .bms-summary-card h3 {
            margin: 0 0 0.45rem;
            font-size: 1rem;
            color: var(--bms-ink);
        }

        .bms-summary-card p {
            margin: 0;
            color: #455066;
            line-height: 1.58;
        }

        .bms-observation {
            background: white;
            border: 1px solid var(--bms-border);
            border-radius: 18px;
            padding: 1rem 1.15rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 7px 20px rgba(28,39,60,0.04);
        }

        .bms-observation-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.45rem;
        }

        .bms-observation-title {
            font-weight: 800;
            color: var(--bms-ink);
        }

        .bms-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.25rem 0.55rem;
            font-size: 0.72rem;
            font-weight: 800;
            background: #fff0e9;
            color: var(--bms-accent-dark);
            white-space: nowrap;
        }

        .bms-observation p {
            margin: 0;
            color: #455066;
            line-height: 1.55;
        }

        .bms-upload-shell {
            background: white;
            border: 1px solid var(--bms-border);
            border-radius: 20px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 10px 25px rgba(28,39,60,0.04);
            margin-bottom: 1rem;
        }

        div.stButton > button[kind="primary"] {
            background: linear-gradient(90deg, var(--bms-accent), #ff8450);
            border: none;
            border-radius: 12px;
            min-height: 3rem;
            font-weight: 800;
            box-shadow: 0 8px 20px rgba(242,106,46,0.22);
        }

        div.stDownloadButton > button {
            border-radius: 12px;
            min-height: 3rem;
            font-weight: 800;
        }

        [data-testid="stFileUploaderDropzone"] {
            border-radius: 14px;
            border: 1px dashed #b8c1d1;
            background: #f9fbff;
        }

        [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }

        [data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .bms-footer-note {
            color: var(--bms-muted);
            font-size: 0.8rem;
            line-height: 1.55;
            margin-top: 1rem;
        }
        /* Publisher-ready interface polish */
        #MainMenu,
        footer,
        [data-testid="stToolbar"] {
            visibility: hidden;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(255,255,255,0.08);
            min-width: 20rem !important;
            width: 20rem !important;
            transform: none !important;
            visibility: visible !important;
        }

        /* Keep the application sidebar permanently visible. */
        [data-testid="stSidebar"][aria-expanded="false"] {
            min-width: 20rem !important;
            width: 20rem !important;
            margin-left: 0 !important;
            transform: none !important;
            visibility: visible !important;
        }

        [data-testid="stSidebarCollapseButton"],
        [data-testid="collapsedControl"],
        button[data-testid="baseButton-headerNoPadding"] {
            display: none !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            border-radius: 10px;
            padding: 0.34rem 0.45rem;
            margin: 0.05rem 0;
            transition: background 120ms ease, transform 120ms ease;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: rgba(255,255,255,0.08);
            transform: translateX(2px);
        }

        [data-testid="stSidebar"] button {
            border-radius: 10px;
            font-weight: 750;
            background: #24324a !important;
            color: #ffffff !important;
            border: 1px solid rgba(255,255,255,0.16) !important;
            box-shadow: none !important;
        }

        [data-testid="stSidebar"] button:hover {
            background: #31415e !important;
            color: #ffffff !important;
            border-color: rgba(255,177,142,0.65) !important;
        }

        [data-testid="stSidebar"] button:disabled {
            background: #1a2538 !important;
            color: #8f9bb0 !important;
            border-color: rgba(255,255,255,0.08) !important;
            opacity: 1 !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #24324a !important;
            color: #ffffff !important;
            border: 1px solid rgba(255,255,255,0.16) !important;
            min-height: 46px !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] div,
        [data-testid="stSidebar"] [data-baseweb="select"] span,
        [data-testid="stSidebar"] [data-baseweb="select"] input {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] svg {
            fill: #ffffff !important;
            color: #ffffff !important;
        }

        /* Two-level sidebar navigation: selected rows are always obvious. */
        [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] {
            gap: 0.32rem !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            width: 100% !important;
            min-height: 2.65rem !important;
            padding: 0.52rem 0.68rem !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
            background: rgba(255,255,255,0.045) !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            background: linear-gradient(90deg, #f26a2e, #ff8450) !important;
            border-color: #ffb18e !important;
            box-shadow: 0 7px 18px rgba(242,106,46,0.24) !important;
            transform: none !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p,
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) span,
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) div {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            font-weight: 850 !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label input {
            accent-color: #f26a2e !important;
        }

        .bms-nav-heading {
            color: #ffffff;
            font-size: 0.78rem;
            font-weight: 850;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            margin: 0.85rem 0 0.38rem;
        }

        .bms-current-location {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.14);
            border-left: 4px solid #f26a2e;
            border-radius: 12px;
            padding: 0.72rem 0.8rem;
            margin: 0.55rem 0 0.85rem;
        }

        .bms-current-location small {
            display: block;
            color: #aeb9cb;
            font-size: 0.70rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.18rem;
        }

        .bms-current-location strong {
            color: #ffffff;
            font-size: 0.95rem;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label {
            color: #ffffff !important;
        }

        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            color: #ffffff !important;
        }

        .stButton > button,
        .stDownloadButton > button {
            min-height: 2.75rem;
            border-radius: 12px;
            border: 1px solid var(--bms-border);
            font-weight: 750;
            transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(28,39,60,0.10);
            border-color: rgba(242,106,46,0.50);
        }

        [data-testid="stDataFrame"] {
            background: white;
            border: 1px solid var(--bms-border);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 7px 20px rgba(28,39,60,0.04);
        }

        [data-testid="stExpander"] {
            background: white;
            border: 1px solid var(--bms-border);
            border-radius: 14px;
            overflow: hidden;
        }

        [data-testid="stTabs"] button {
            font-weight: 750;
        }

        [data-testid="stException"] a {
            display: none !important;
        }

        [data-testid="stException"] pre {
            max-height: 16rem;
            overflow: auto;
        }

        [data-testid="stFileUploader"] {
            background: white;
            border: 1px dashed #b8c0cf;
            border-radius: 16px;
            padding: 0.35rem;
        }

        .bms-app-status {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
            background: rgba(255,255,255,0.88);
            border: 1px solid var(--bms-border);
            border-radius: 16px;
            padding: 0.75rem 1rem;
            margin: 0 0 1rem;
            box-shadow: 0 7px 20px rgba(28,39,60,0.04);
            backdrop-filter: blur(12px);
        }

        .bms-app-status-left {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            color: var(--bms-ink);
            font-weight: 800;
        }

        .bms-status-dot {
            width: 0.65rem;
            height: 0.65rem;
            border-radius: 50%;
            background: var(--bms-green);
            box-shadow: 0 0 0 5px rgba(21,148,85,0.12);
        }

        .bms-app-status-right {
            color: var(--bms-muted);
            font-size: 0.86rem;
        }

        .bms-sidebar-card {
            background: rgba(255,255,255,0.07);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 14px;
            padding: 0.8rem 0.85rem;
            margin: 0.65rem 0;
        }

        .bms-sidebar-card strong {
            color: #ffffff;
        }

        .bms-sidebar-card p {
            color: #cbd5e1;
            margin: 0.3rem 0 0;
            font-size: 0.82rem;
            line-height: 1.45;
        }

        .bms-footer {
            margin-top: 3rem;
            padding: 1rem 0 0.2rem;
            border-top: 1px solid var(--bms-border);
            color: var(--bms-muted);
            font-size: 0.82rem;
            text-align: center;
        }

        .bms-showcase-banner {
            background:
                radial-gradient(circle at 12% 12%, rgba(255,140,84,0.22), transparent 28%),
                linear-gradient(135deg, #101a2d 0%, #1c2b47 58%, #24385e 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 26px;
            padding: 1.75rem 1.85rem;
            color: white;
            box-shadow: 0 20px 48px rgba(21,32,51,0.22);
            margin: 0.35rem 0 1.2rem;
        }

        .bms-showcase-banner h1,
        .bms-showcase-banner h2 {
            color: white;
            margin: 0;
        }

        .bms-showcase-eyebrow {
            color: #ffad87;
            font-size: 0.74rem;
            font-weight: 850;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }

        .bms-showcase-copy {
            color: #dce4f0;
            line-height: 1.72;
            max-width: 930px;
            margin: 0.55rem 0 0;
        }

        .bms-feature-pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.48rem;
            margin-top: 1rem;
        }

        .bms-feature-pill {
            background: rgba(255,255,255,0.09);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 999px;
            padding: 0.38rem 0.7rem;
            color: #eef3fb;
            font-size: 0.8rem;
            font-weight: 700;
        }

        .bms-empty-state {
            background: linear-gradient(180deg,#ffffff 0%,#f8fafc 100%);
            border: 1px dashed #bdc6d4;
            border-radius: 18px;
            padding: 1.3rem;
            color: var(--bms-muted);
            text-align: center;
        }

        .bms-section-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, #dce2eb, transparent);
            margin: 1.5rem 0;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .bms-hero,
            .bms-showcase-banner {
                padding: 1.35rem;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required project file was not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def safe_result(value: object) -> str:
    normalized = str(value).strip().lower()

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


def validate_uploaded_json(data: object) -> dict:
    if not isinstance(data, dict):
        raise ValueError(
            "This is not a compatible free-throw tracking JSON. The file must " "contain one free-throw trial object."
        )

    tracking = data.get("tracking")

    if not isinstance(tracking, list):
        raise ValueError(
            "This is not a compatible free-throw tracking JSON. A non-empty " "tracking list with ball coordinates and player body keypoints is required."
        )

    if len(tracking) == 0:
        raise ValueError(
            "This is not a compatible free-throw tracking JSON. The tracking " "list is empty."
        )

    return data


def analyze_uploaded_trial(
    trial_data: dict,
    original_filename: str,
    shooting_side: str,
) -> dict:
    """
    Analyze the actual uploaded JSON.

    This intentionally does not ask the scoring engine to find a matching
    trial inside the historical dataset. Instead, it extracts features from
    the upload and compares those exact values with the participant baseline.
    """

    with tempfile.TemporaryDirectory() as temporary_folder:
        temporary_path = (
            Path(temporary_folder)
            / original_filename
        )

        temporary_path.write_text(
            json.dumps(
                trial_data
            ),
            encoding="utf-8",
        )

        shot_analysis = ShotAnalyzer(
            trial_data=trial_data,
            trial_file=temporary_path,
            shooting_wrist=f"{shooting_side}_WRIST",
        ).analyze()

        sampling_rate = float(
            trial_data.get(
                "sampling_rate",
                30,
            )
        )

        time_series = TimeSeriesAnalyzer(
            trial_data=trial_data,
            smoothing_window=5,
        ).analyze()

        phase_analysis = RefinedPhaseBiomechanicsAnalyzer(
            time_series=time_series,
            sampling_rate=sampling_rate,
            motion_start_frame=(
                shot_analysis.motion_start_frame
            ),
            dip_frame=shot_analysis.dip_frame,
            takeoff_frame=shot_analysis.takeoff_frame,
            release_frame=shot_analysis.release_frame,
            shooting_side=shooting_side,
        ).analyze()

        basketball_timeline = BasketballEventMapper(
            time_series=time_series,
            sampling_rate=sampling_rate,
            motion_start_frame=(
                shot_analysis.motion_start_frame
            ),
            dip_frame=shot_analysis.dip_frame,
            takeoff_frame=shot_analysis.takeoff_frame,
            release_frame=shot_analysis.release_frame,
            ball_apex_frame=shot_analysis.ball_apex_frame,
            shooting_side=shooting_side,
        ).build_timeline()

        timeline = ShotTimelineBuilder(
            basketball_timeline=basketball_timeline,
            phase_analysis=phase_analysis,
            landing_frame=shot_analysis.landing_frame,
        ).build()

        synchronization = WholeBodySynchronizationEngine(
            timeline=timeline,
            sampling_rate=sampling_rate,
        ).analyze()

        feature_record = ShotFeatureExtractor(
            trial_file=temporary_path,
            trial_data=trial_data,
            shooting_side=shooting_side,
        ).extract()

    if feature_record.analysis_status != "success":
        raise RuntimeError(
            feature_record.analysis_error
        )

    feature_dictionary = asdict(
        feature_record
    )

    # Historical baseline files are optional in a new Blank Workspace.
    #
    # When they are present, the uploaded shot can be compared with the
    # participant's successful historical pattern. When they are absent,
    # objective biomechanics, shot events, playback, timelines, metrics,
    # and downloads still run normally; only personal-baseline scoring is
    # unavailable.
    master_records = (
        load_csv(
            MASTER_FEATURE_FILE
        )
        if MASTER_FEATURE_FILE.exists()
        else []
    )

    diagnostic_records = (
        load_csv(
            PERSONAL_DIAGNOSTIC_FILE
        )
        if PERSONAL_DIAGNOSTIC_FILE.exists()
        else []
    )

    participant_id = feature_record.participant_id

    participant_records = [
        record
        for record in master_records
        if str(
            record.get(
                "participant_id",
                "",
            )
        ).strip()
        == participant_id
        and str(
            record.get(
                "analysis_status",
                "",
            )
        ).strip().lower()
        == "success"
    ]

    made_records = [
        record
        for record in participant_records
        if safe_result(
            record.get(
                "result",
                "",
            )
        )
        == "made"
    ]

    missed_records = [
        record
        for record in participant_records
        if safe_result(
            record.get(
                "result",
                "",
            )
        )
        == "missed"
    ]

    participant_diagnostics = [
        record
        for record in diagnostic_records
        if str(
            record.get(
                "participant_id",
                "",
            )
        ).strip()
        == participant_id
    ]

    scoring_available = (
        len(made_records) >= 10
        and len(participant_diagnostics) > 0
    )

    score_summary = None
    scored_features = []
    category_scores = []

    if scoring_available:
        scoring_engine = PersonalShotScoringEngine()

        scored_features = scoring_engine.score_features(
            selected_shot=feature_dictionary,
            made_records=made_records,
            diagnostic_records=participant_diagnostics,
        )

        if scored_features:
            category_scores = scoring_engine.build_category_scores(
                scored_features
            )

            score_summary = scoring_engine.build_summary(
                selected_shot=feature_dictionary,
                made_count=len(made_records),
                missed_count=len(missed_records),
                scored_features=scored_features,
                category_scores=category_scores,
            )

    movement_patterns = []
    coach_summary = None
    coach_diagnoses = []

    if scored_features:
        movement_patterns = (
            MovementPatternInterpreter(
                minimum_feature_count=1,
                maximum_patterns=5,
            ).build_patterns(
                scored_features
            )
        )

        (
            coach_summary,
            coach_diagnoses,
        ) = CoachIntelligenceLayer(
            maximum_diagnoses=3,
        ).build(
            movement_patterns=(
                movement_patterns
            ),
            scored_features=(
                scored_features
            ),
            category_scores=(
                category_scores
            ),
            feature_record=(
                feature_record
            ),
        )

    return {
        "feature_record": feature_record,
        "feature_dictionary": feature_dictionary,
        "shot_analysis": shot_analysis,
        "timeline": timeline,
        "synchronization": synchronization,
        "score_summary": score_summary,
        "scored_features": scored_features,
        "category_scores": category_scores,
        "movement_patterns": movement_patterns,
        "coach_summary": coach_summary,
        "coach_diagnoses": coach_diagnoses,
        "baseline_makes": len(made_records),
        "baseline_misses": len(missed_records),
        "scoring_available": scoring_available,
    }


def build_timeline_rows(
    timeline: object,
    sampling_rate: float,
) -> list[dict[str, object]]:
    rows = []

    for event in timeline.events:
        rows.append(
            {
                "Frame": int(event.frame),
                "Time (ms)": round(
                    event.frame
                    / sampling_rate
                    * 1000.0,
                    1,
                ),
                "Event": (
                    event.event_type.value
                    .replace("_", " ")
                    .title()
                ),
                "Source": event.source.value,
                "Signal": event.source_signal,
                "Confidence": round(
                    float(event.confidence),
                    3,
                ),
            }
        )

    return rows


def build_metric_rows(
    feature_record: object,
) -> list[dict[str, object]]:
    return [
        {
            "Category": "Timing",
            "Metric": "Motion start to release",
            "Value": feature_record.motion_to_release_ms,
            "Unit": "ms",
        },
        {
            "Category": "Timing",
            "Metric": "Takeoff to release",
            "Value": feature_record.takeoff_to_release_ms,
            "Unit": "ms",
        },
        {
            "Category": "Release",
            "Metric": "Release angle",
            "Value": feature_record.release_angle_deg,
            "Unit": "degrees",
        },
        {
            "Category": "Release",
            "Metric": "Ball speed at release",
            "Value": feature_record.release_ball_speed_ft_s,
            "Unit": "ft/s",
        },
        {
            "Category": "Release",
            "Metric": "Release height",
            "Value": feature_record.release_height_ft,
            "Unit": "ft",
        },
        {
            "Category": "Lower Body",
            "Metric": "Right knee range of motion",
            "Value": feature_record.right_knee_range_of_motion_deg,
            "Unit": "degrees",
        },
        {
            "Category": "Lower Body",
            "Metric": "Right hip range of motion",
            "Value": feature_record.right_hip_range_of_motion_deg,
            "Unit": "degrees",
        },
        {
            "Category": "Upper Body",
            "Metric": "Right elbow angle at release",
            "Value": feature_record.release_right_elbow_angle_deg,
            "Unit": "degrees",
        },
        {
            "Category": "Upper Body",
            "Metric": "Peak right wrist speed",
            "Value": feature_record.peak_right_wrist_speed_ft_s,
            "Unit": "ft/s",
        },
        {
            "Category": "Coordination",
            "Metric": "Knee-to-elbow timing gap",
            "Value": feature_record.knee_to_elbow_gap_ms,
            "Unit": "ms",
        },
        {
            "Category": "Coordination",
            "Metric": "Elbow-to-release timing gap",
            "Value": feature_record.elbow_to_release_gap_ms,
            "Unit": "ms",
        },
    ]


def build_dashboard_figure(
    result: dict,
):
    summary = result["score_summary"]
    category_scores = result["category_scores"]
    scored_features = result["scored_features"]

    figure = plt.figure(
        figsize=(
            12,
            9,
        )
    )

    grid = figure.add_gridspec(
        3,
        1,
        height_ratios=[
            0.8,
            1.2,
            2.0,
        ],
    )

    score_axis = figure.add_subplot(
        grid[0]
    )

    category_axis = figure.add_subplot(
        grid[1]
    )

    feature_axis = figure.add_subplot(
        grid[2]
    )

    score_axis.barh(
        [0],
        [
            summary.overall_similarity_score
        ],
    )

    score_axis.set_xlim(
        0,
        100,
    )

    score_axis.set_yticks(
        [0],
        labels=[
            "Personal similarity"
        ],
    )

    score_axis.set_xlabel(
        "Score"
    )

    score_axis.set_title(
        "Personal Successful-Baseline Score"
    )

    category_labels = [
        row.category
        for row in category_scores
    ]

    category_values = [
        row.category_similarity_score
        for row in category_scores
    ]

    category_positions = list(
        range(
            len(category_labels)
        )
    )

    category_axis.barh(
        category_positions,
        category_values,
    )

    category_axis.set_yticks(
        category_positions,
        labels=category_labels,
    )

    category_axis.set_xlim(
        0,
        100,
    )

    category_axis.set_xlabel(
        "Similarity score"
    )

    category_axis.set_title(
        "Category Scores"
    )

    top_rows = scored_features[
        :8
    ]

    labels = [
        row.coach_label
        for row in reversed(
            top_rows
        )
    ]

    values = [
        row.feature_similarity_score
        for row in reversed(
            top_rows
        )
    ]

    positions = list(
        range(
            len(labels)
        )
    )

    feature_axis.barh(
        positions,
        values,
    )

    feature_axis.set_yticks(
        positions,
        labels=labels,
        fontsize=8,
    )

    feature_axis.set_xlim(
        0,
        100,
    )

    feature_axis.set_xlabel(
        "Similarity to personal made-shot baseline"
    )

    feature_axis.set_title(
        "Lowest-Similarity Features"
    )

    for axis in (
        score_axis,
        category_axis,
        feature_axis,
    ):
        axis.grid(
            visible=True,
            axis="x",
            alpha=0.25,
        )

    figure.tight_layout()

    return figure


def create_download_zip(
    uploaded_name: str,
    trial_data: dict,
    result: dict,
) -> bytes:
    feature_record = result[
        "feature_record"
    ]

    summary = result[
        "score_summary"
    ]

    timeline_rows = build_timeline_rows(
        result["timeline"],
        feature_record.sampling_rate,
    )

    metric_rows = build_metric_rows(
        feature_record
    )

    recommendation_rows = []

    for rank, row in enumerate(
        result["scored_features"][
            :8
        ],
        start=1,
    ):
        recommendation_rows.append(
            {
                "Rank": rank,
                "Category": row.category,
                "Feature": row.coach_label,
                "Observation": row.coach_observation,
                "Similarity Score": (
                    row.feature_similarity_score
                ),
                "Deviation SD": row.absolute_z_score,
                "Evidence Weight": row.evidence_weight,
            }
        )

    report = {
        "source_file": uploaded_name,
        "feature_record": asdict(
            feature_record
        ),
        "score_summary": (
            None
            if summary is None
            else asdict(summary)
        ),
        "timeline": timeline_rows,
        "metrics": metric_rows,
        "recommendations": recommendation_rows,
        "movement_patterns": [
            asdict(
                pattern
            )
            for pattern in result.get(
                "movement_patterns",
                [],
            )
        ],
        "coach_summary": (
            None
            if result.get(
                "coach_summary"
            )
            is None
            else asdict(
                result[
                    "coach_summary"
                ]
            )
        ),
        "coach_diagnoses": [
            asdict(
                diagnosis
            )
            for diagnosis in result.get(
                "coach_diagnoses",
                [],
            )
        ],
    }

    output = io.BytesIO()

    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr(
            "source_trial.json",
            json.dumps(
                trial_data,
                indent=2,
            ),
        )

        archive.writestr(
            "shot_analysis_report.json",
            json.dumps(
                report,
                indent=2,
            ),
        )

        archive.writestr(
            "shot_features.csv",
            dictionary_to_csv(
                asdict(
                    feature_record
                )
            ),
        )

        archive.writestr(
            "shot_timeline.csv",
            rows_to_csv(
                timeline_rows
            ),
        )

        archive.writestr(
            "shot_metrics.csv",
            rows_to_csv(
                metric_rows
            ),
        )

        archive.writestr(
            "coaching_observations.csv",
            rows_to_csv(
                recommendation_rows
            ),
        )

        archive.writestr(
            "movement_patterns.csv",
            rows_to_csv(
                [
                    {
                        "Rank": rank,
                        "Pattern": pattern.pattern_name,
                        "Category": pattern.category,
                        "Grouped Similarity": (
                            pattern.average_similarity_score
                        ),
                        "Priority Score": (
                            pattern.pattern_priority_score
                        ),
                        "Priority Level": (
                            pattern.pattern_level
                        ),
                        "Headline": pattern.headline,
                        "Summary": pattern.summary,
                        "Strongest Feature": (
                            pattern.strongest_deviation_feature
                        ),
                        "Review Event": (
                            pattern.review_event
                        ),
                        "Evidence": (
                            pattern.evidence_summary
                        ),
                    }
                    for rank, pattern in enumerate(
                        result.get(
                            "movement_patterns",
                            [],
                        ),
                        start=1,
                    )
                ]
            ),
        )

        archive.writestr(
            "coach_diagnoses.csv",
            rows_to_csv(
                [
                    {
                        "Rank": diagnosis.rank,
                        "Diagnosis": (
                            diagnosis.diagnosis_title
                        ),
                        "Pattern": (
                            diagnosis.pattern_name
                        ),
                        "Category": (
                            diagnosis.category
                        ),
                        "Priority": (
                            diagnosis.priority_level
                        ),
                        "Summary": (
                            diagnosis.diagnosis_summary
                        ),
                        "Why It May Matter": (
                            diagnosis.why_it_matters
                        ),
                        "Coaching Focus": (
                            diagnosis.coaching_focus
                        ),
                        "Suggested Drill": (
                            diagnosis.recommended_drill
                        ),
                        "Review Start Frame": (
                            diagnosis.review_start_frame
                        ),
                        "Review End Frame": (
                            diagnosis.review_end_frame
                        ),
                        "Review Instruction": (
                            diagnosis.review_instruction
                        ),
                        "Grouped Similarity": (
                            diagnosis.grouped_similarity_score
                        ),
                        "Evidence Confidence": (
                            diagnosis.evidence_confidence
                        ),
                    }
                    for diagnosis in result.get(
                        "coach_diagnoses",
                        [],
                    )
                ]
            ),
        )

        archive.writestr(
            "report_summary.txt",
            build_text_report(
                result
            ),
        )

    output.seek(0)

    return output.getvalue()


def dictionary_to_csv(
    dictionary: dict[str, object],
) -> str:
    buffer = io.StringIO()

    writer = csv.DictWriter(
        buffer,
        fieldnames=list(
            dictionary
        ),
    )

    writer.writeheader()
    writer.writerow(
        dictionary
    )

    return buffer.getvalue()


def rows_to_csv(
    rows: list[dict[str, object]],
) -> str:
    if not rows:
        return ""

    buffer = io.StringIO()

    writer = csv.DictWriter(
        buffer,
        fieldnames=list(
            rows[0]
        ),
    )

    writer.writeheader()
    writer.writerows(
        rows
    )

    return buffer.getvalue()


def build_text_report(
    result: dict,
) -> str:
    feature_record = result[
        "feature_record"
    ]

    summary = result[
        "score_summary"
    ]

    lines = [
        "PERSONAL FREE-THROW SHOT ANALYSIS",
        "=" * 72,
        "",
        f"Participant: {feature_record.participant_id}",
        f"Trial: {feature_record.trial_id}",
        f"Recorded result: {feature_record.result}",
        "",
        "SHOT EVENTS",
        "-" * 72,
        f"Motion start: {feature_record.motion_start_frame}",
        f"Dip: {feature_record.dip_frame}",
        f"Takeoff: {feature_record.takeoff_frame}",
        f"Release: {feature_record.release_frame}",
        f"Ball apex: {feature_record.ball_apex_frame}",
        f"Landing: {feature_record.landing_frame}",
        "",
    ]

    if summary is None:
        lines.extend(
            [
                "PERSONAL SCORE",
                "-" * 72,
                (
                    "No personal successful-shot baseline is available for "
                    "this participant yet. Objective biomechanics and shot "
                    "event analysis are still included. Import additional "
                    "compatible free-throw tracking JSON files to build a "
                    "personal history."
                ),
            ]
        )

        return "\n".join(
            lines
        )

    lines.extend(
        [
            "PERSONAL SCORE",
            "-" * 72,
            (
                "Overall similarity: "
                f"{summary.overall_similarity_score:.1f} / 100"
            ),
            (
                "Confidence-adjusted score: "
                f"{summary.confidence_adjusted_score:.1f} / 100"
            ),
            (
                "Confidence: "
                f"{summary.score_confidence}"
            ),
            "",
            "ASSESSMENT",
            "-" * 72,
            summary.overall_assessment,
            "",
            "COACHING OBSERVATIONS",
            "-" * 72,
        ]
    )

    for rank, row in enumerate(
        result["scored_features"][
            :8
        ],
        start=1,
    ):
        lines.extend(
            [
                f"{rank}. {row.coach_label}",
                f"   {row.coach_observation}",
                (
                    "   Similarity: "
                    f"{row.feature_similarity_score:.1f} / 100"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "LIMITATION",
            "-" * 72,
            (
                "This report compares the shot with the participant's own "
                "successful history. It identifies deviations, not proven "
                "causes of the make or miss."
            ),
        ]
    )

    return "\n".join(
        lines
    )



def score_status_text(score: float) -> str:
    if score >= 88:
        return "Very close to successful baseline"

    if score >= 75:
        return "Generally similar"

    if score >= 60:
        return "Meaningful differences"

    return "Large differences"


def score_ring_html(
    score: float,
    confidence: str,
) -> str:
    safe_score = max(
        0.0,
        min(
            100.0,
            float(score),
        ),
    )

    return f"""
    <div class="bms-score-card">
        <div class="bms-kicker" style="color:#c94d18;text-align:center;">
            Personal shot score
        </div>
        <div
            class="bms-score-ring"
            style="background:conic-gradient(
                #f26a2e 0deg,
                #f26a2e {safe_score * 3.6:.1f}deg,
                #e9edf5 {safe_score * 3.6:.1f}deg,
                #e9edf5 360deg
            );"
        >
            <div class="bms-score-value">
                <strong>{safe_score:.0f}</strong>
                <span>OUT OF 100</span>
            </div>
        </div>
        <div class="bms-confidence">
            <strong>{score_status_text(safe_score)}</strong><br>
            Confidence: {confidence.title()}
        </div>
    </div>
    """



def render_empty_state(
    title: str,
    message: str,
    guidance: str | None = None,
) -> None:
    """
    Render a consistent empty-state or read-only notice.

    This helper is used by public portfolio mode when database-changing
    management workflows are intentionally unavailable.
    """

    guidance_html = (
        ""
        if not guidance
        else (
            '<div class="bms-footer-note" '
            'style="margin-top:0.65rem;">'
            f"{guidance}"
            "</div>"
        )
    )

    st.markdown(
        f"""
        <div class="bms-empty-state">
            <div style="
                color:#152033;
                font-size:1.05rem;
                font-weight:800;
                margin-bottom:0.35rem;
            ">
                {title}
            </div>
            <div style="
                color:#667085;
                line-height:1.55;
            ">
                {message}
            </div>
            {guidance_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(
    title: str,
    subtitle: str,
) -> None:
    st.markdown(
        f"""
        <div class="bms-section-title">
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_category_cards(
    category_scores: list[object],
) -> None:
    columns = st.columns(
        max(
            1,
            len(
                category_scores
            ),
        )
    )

    for column, category in zip(
        columns,
        category_scores,
    ):
        score = float(
            category.category_similarity_score
        )

        if score >= 85:
            status = "Closely matched"
        elif score >= 70:
            status = "Generally matched"
        elif score >= 50:
            status = "Review differences"
        else:
            status = "Large differences"

        with column:
            st.markdown(
                f"""
                <div class="bms-category-card">
                    <div class="label">{category.category}</div>
                    <div class="score">{score:.1f}</div>
                    <div class="status">{status}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def movement_pattern_frame_range(
    pattern: object,
    feature_record: object,
) -> tuple[
    int | None,
    int | None,
]:
    """
    Translate a configured review event into a useful animation window.
    """

    review_event = pattern.review_event

    motion = feature_record.motion_start_frame
    dip = feature_record.dip_frame
    takeoff = feature_record.takeoff_frame
    release = feature_record.release_frame
    landing = feature_record.landing_frame

    if review_event == "release":
        return (
            None
            if release is None
            else max(
                0,
                release - 6,
            ),
            release,
        )

    if review_event == "takeoff_to_release":
        return (
            takeoff,
            release,
        )

    if review_event == "release_to_landing":
        return (
            release,
            landing,
        )

    if review_event == "motion_to_release":
        return (
            motion,
            release,
        )

    if review_event == "dip_to_release":
        return (
            dip,
            release,
        )

    if review_event in {
        "knee_to_elbow",
        "hip_to_knee",
        "elbow_to_release",
        "wrist_to_release",
        "knee_to_release",
        "hip_to_release",
        "pelvis_to_release",
        "wrist_propulsion",
        "ball_propulsion",
    }:
        return (
            None
            if release is None
            else max(
                0,
                release - 15,
            ),
            release,
        )

    return (
        motion,
        release,
    )


def build_pattern_review_text(
    pattern: object,
    feature_record: object,
) -> str:
    start_frame, end_frame = (
        movement_pattern_frame_range(
            pattern,
            feature_record,
        )
    )

    if (
        start_frame is None
        and end_frame is None
    ):
        return "Review the full tracked shot."

    if start_frame is None:
        return (
            f"Review frame {end_frame}."
        )

    if end_frame is None:
        return (
            f"Review from frame {start_frame} onward."
        )

    if start_frame == end_frame:
        return (
            f"Review frame {start_frame}."
        )

    return (
        f"Review frames {start_frame}–{end_frame}."
    )


def build_coach_summary(
    result: dict,
) -> tuple[
    str,
    str,
    str,
]:
    """
    Build the three headline cards from the Coach Intelligence Layer.
    """

    coach_summary = result.get(
        "coach_summary"
    )

    if coach_summary is None:
        return (
            "No stable movement diagnosis was available.",
            "No category-level consistency result was available.",
            "Use the animation and objective metrics for manual review.",
        )

    return (
        (
            f"{coach_summary.primary_diagnosis}. "
            f"{coach_summary.primary_summary}"
        ),
        coach_summary.what_stayed_consistent,
        coach_summary.session_focus,
    )


def render_coach_summary(
    result: dict,
) -> None:
    (
        diagnosis_text,
        consistent_text,
        focus_text,
    ) = build_coach_summary(
        result
    )

    left, middle, right = st.columns(
        3
    )

    cards = [
        (
            left,
            "Primary diagnosis",
            diagnosis_text,
        ),
        (
            middle,
            "What stayed consistent",
            consistent_text,
        ),
        (
            right,
            "Session focus",
            focus_text,
        ),
    ]

    for column, heading, text in cards:
        with column:
            st.markdown(
                f"""
                <div class="bms-summary-card">
                    <h3>{heading}</h3>
                    <p>{text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def set_review_window(
    start_frame: int | None,
    end_frame: int | None,
    total_frames: int,
) -> None:
    """
    Move playback to the start of a diagnosis review window.
    """

    target_frame = (
        start_frame
        if start_frame is not None
        else end_frame
    )

    if target_frame is None:
        target_frame = 0

    safe_frame = max(
        0,
        min(
            int(
                target_frame
            ),
            total_frames - 1,
        ),
    )

    st.session_state.playback_frame = (
        safe_frame
    )

    st.session_state.playback_playing = (
        False
    )


@st.fragment(
    run_every="100ms",
)
def render_inline_diagnosis_player(
    trial_data: dict,
) -> None:
    """
    Play the selected diagnosis clip directly inside Coach Intelligence.

    This solves the usability problem where the main animation was far above
    the diagnosis buttons and therefore invisible after the user clicked Play.
    """

    initialize_playback_state()

    total_frames = len(
        trial_data.get(
            "tracking",
            [],
        )
    )

    if total_frames <= 0:
        st.warning(
            "No tracking frames were available for diagnosis playback."
        )

        return

    if not st.session_state.diagnosis_segment_active:
        st.info(
            "Choose **Play Diagnosis** below to open and play the relevant "
            "animation segment here."
        )

        return

    segment_start = (
        st.session_state
        .diagnosis_segment_start
    )

    segment_end = (
        st.session_state
        .diagnosis_segment_end
    )

    if segment_start is None:
        segment_start = 0

    if segment_end is None:
        segment_end = (
            total_frames
            - 1
        )

    speed_step_lookup = {
        "Slow": 1,
        "Normal": 3,
        "Fast": 6,
    }

    if st.session_state.playback_playing:
        next_frame = (
            st.session_state.playback_frame
            + speed_step_lookup.get(
                st.session_state.playback_speed,
                3,
            )
        )

        if next_frame >= segment_end:
            next_frame = int(
                segment_end
            )

            st.session_state.playback_playing = (
                False
            )

        set_playback_frame(
            next_frame,
            total_frames,
        )

    overlay_html = diagnosis_overlay_html(
        st.session_state.playback_frame
    )

    if overlay_html:
        st.markdown(
            overlay_html,
            unsafe_allow_html=True,
        )

    player_column, controls_column = st.columns(
        [
            0.73,
            0.27,
        ],
        gap="large",
    )

    with player_column:
        figure = create_playback_figure(
            trial_data=trial_data,
            frame_index=(
                st.session_state
                .playback_frame
            ),
            view_name=(
                st.session_state
                .playback_view
            ),
            show_trajectory=(
                st.session_state
                .playback_show_trajectory
            ),
        )

        st.pyplot(
            figure,
            clear_figure=True,
            use_container_width=True,
        )

    with controls_column:
        st.markdown(
            "### Diagnosis clip"
        )

        st.metric(
            "Current frame",
            st.session_state.playback_frame,
        )

        st.metric(
            "Clip window",
            f"{segment_start}–{segment_end}",
        )

        button_columns = st.columns(
            2
        )

        if button_columns[0].button(
            (
                "⏸ Pause"
                if st.session_state.playback_playing
                else "▶ Resume"
            ),
            key="inline_diagnosis_pause_resume",
            use_container_width=True,
        ):
            st.session_state.playback_playing = (
                not st.session_state.playback_playing
            )

            st.rerun(
                scope="fragment"
            )

        if button_columns[1].button(
            "↺ Replay",
            key="inline_diagnosis_replay",
            use_container_width=True,
        ):
            set_playback_frame(
                segment_start,
                total_frames,
            )

            st.session_state.playback_playing = (
                True
            )

            st.rerun(
                scope="fragment"
            )

        if st.button(
            "Close Diagnosis Clip",
            key="inline_diagnosis_close",
            use_container_width=True,
        ):
            st.session_state.diagnosis_segment_active = (
                False
            )

            st.session_state.playback_playing = (
                False
            )

            st.rerun(
                scope="fragment"
            )

        progress_denominator = max(
            1,
            segment_end
            - segment_start,
        )

        progress_value = (
            st.session_state.playback_frame
            - segment_start
        ) / progress_denominator

        st.progress(
            float(
                max(
                    0.0,
                    min(
                        1.0,
                        progress_value,
                    ),
                )
            )
        )

        st.caption(
            (
                f"Frame "
                f"{st.session_state.playback_frame - segment_start + 1} "
                f"of {segment_end - segment_start + 1}"
            )
        )


def render_coach_diagnoses(
    result: dict,
    trial_data: dict,
) -> None:
    """
    Show coach diagnoses with direct segment playback controls.
    """

    diagnoses = result.get(
        "coach_diagnoses",
        [],
    )

    if not diagnoses:
        st.info(
            "No coach intelligence diagnoses were available for this shot."
        )

        return

    total_frames = len(
        trial_data.get(
            "tracking",
            [],
        )
    )

    render_inline_diagnosis_player(
        trial_data
    )

    for diagnosis in diagnoses:
        with st.container(
            border=True
        ):
            top_left, top_right = st.columns(
                [
                    0.78,
                    0.22,
                ]
            )

            with top_left:
                st.markdown(
                    f"### {diagnosis.rank}. {diagnosis.diagnosis_title}"
                )

                st.write(
                    diagnosis.diagnosis_summary
                )

            with top_right:
                st.metric(
                    "Grouped similarity",
                    (
                        f"{diagnosis.grouped_similarity_score:.1f}/100"
                    ),
                )

                st.caption(
                    (
                        f"Confidence: "
                        f"{diagnosis.evidence_confidence.title()}"
                    )
                )

            why_column, focus_column = st.columns(
                2
            )

            with why_column:
                st.markdown(
                    "**Why it may matter**"
                )

                st.write(
                    diagnosis.why_it_matters
                )

            with focus_column:
                st.markdown(
                    "**Coaching focus**"
                )

                st.write(
                    diagnosis.coaching_focus
                )

            st.markdown(
                "**Suggested drill**"
            )

            st.write(
                diagnosis.recommended_drill
            )

            st.caption(
                (
                    f"{diagnosis.review_instruction} · "
                    f"Supporting measurements: "
                    f"{', '.join(diagnosis.supporting_features[:4])}"
                )
            )

            button_columns = st.columns(
                3
            )

            if button_columns[0].button(
                "▶ Play Diagnosis",
                key=(
                    f"coach_play_"
                    f"{diagnosis.rank}_"
                    f"{diagnosis.pattern_name}"
                ),
                type="primary",
                use_container_width=True,
            ):
                start_diagnosis_segment(
                    diagnosis=diagnosis,
                    total_frames=total_frames,
                )

                st.rerun()

            if button_columns[1].button(
                "↺ Replay",
                key=(
                    f"coach_replay_"
                    f"{diagnosis.rank}_"
                    f"{diagnosis.pattern_name}"
                ),
                use_container_width=True,
            ):
                start_diagnosis_segment(
                    diagnosis=diagnosis,
                    total_frames=total_frames,
                )

                st.rerun()

            if button_columns[2].button(
                "⏸ Pause",
                key=(
                    f"coach_pause_"
                    f"{diagnosis.rank}_"
                    f"{diagnosis.pattern_name}"
                ),
                use_container_width=True,
            ):
                stop_diagnosis_segment()

                st.rerun()

            st.caption(
                (
                    f"Review target: "
                    f"{diagnosis.review_instruction}. "
                    "Play Diagnosis jumps to the start frame, plays only "
                    "the review window, and stops automatically."
                )
            )



def render_movement_patterns(
    result: dict,
) -> None:
    """
    Display grouped movement themes before individual feature details.
    """

    patterns = result.get(
        "movement_patterns",
        [],
    )

    if not patterns:
        st.info(
            "No grouped movement patterns were available for this shot."
        )

        return

    feature_record = result[
        "feature_record"
    ]

    for rank, pattern in enumerate(
        patterns,
        start=1,
    ):
        review_text = build_pattern_review_text(
            pattern,
            feature_record,
        )

        supporting_text = ", ".join(
            pattern.supporting_features[
                :4
            ]
        )

        if len(
            pattern.supporting_features
        ) > 4:
            supporting_text += (
                f", plus "
                f"{len(pattern.supporting_features) - 4} more"
            )

        st.markdown(
            f"""
            <div class="bms-observation">
                <div class="bms-observation-top">
                    <div class="bms-observation-title">
                        {rank}. {pattern.pattern_name}
                    </div>
                    <div class="bms-pill">
                        {pattern.average_similarity_score:.1f}/100 grouped similarity
                    </div>
                </div>
                <p>
                    <strong>{pattern.headline}.</strong>
                    {pattern.summary}
                </p>
                <div class="bms-footer-note">
                    Category: {pattern.category} ·
                    Priority: {pattern.pattern_level.title()} ·
                    {review_text}<br>
                    Supporting measurements: {supporting_text}<br>
                    {pattern.evidence_summary}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )



def render_observations(
    scored_features: list[object],
) -> None:
    for rank, row in enumerate(
        scored_features[
            :6
        ],
        start=1,
    ):
        deviation = (
            "N/A"
            if row.absolute_z_score
            is None
            else (
                f"{row.absolute_z_score:.2f} SD"
            )
        )

        st.markdown(
            f"""
            <div class="bms-observation">
                <div class="bms-observation-top">
                    <div class="bms-observation-title">
                        {rank}. {row.coach_label}
                    </div>
                    <div class="bms-pill">
                        {row.feature_similarity_score:.1f}/100 similarity
                    </div>
                </div>
                <p>{row.coach_observation}</p>
                <div class="bms-footer-note">
                    Category: {row.category} ·
                    Deviation: {deviation} ·
                    Evidence weight: {row.evidence_weight:.3f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )



SKELETON_CONNECTIONS = [
    ("LEFT_EAR", "LEFT_EYE"),
    ("LEFT_EYE", "NOSE"),
    ("NOSE", "RIGHT_EYE"),
    ("RIGHT_EYE", "RIGHT_EAR"),
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ("LEFT_ANKLE", "LEFT_HEEL"),
    ("LEFT_ANKLE", "LEFT_BIG_TOE"),
    ("RIGHT_ANKLE", "RIGHT_HEEL"),
    ("RIGHT_ANKLE", "RIGHT_BIG_TOE"),
    ("LEFT_WRIST", "LEFT_THUMB"),
    ("LEFT_WRIST", "LEFT_PINKY"),
    ("RIGHT_WRIST", "RIGHT_THUMB"),
    ("RIGHT_WRIST", "RIGHT_PINKY"),
]


def finite_point(
    value: object,
) -> np.ndarray | None:
    """
    Convert a tracked XYZ value into a finite NumPy point.
    """

    if not isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return None

    if len(value) < 3:
        return None

    try:
        point = np.asarray(
            value[
                :3
            ],
            dtype=float,
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not np.all(
        np.isfinite(
            point
        )
    ):
        return None

    return point


def frame_data(
    trial_data: dict,
    frame_index: int,
) -> dict:
    tracking = trial_data.get(
        "tracking",
        [],
    )

    if not tracking:
        raise ValueError(
            "The trial does not contain tracking frames."
        )

    safe_index = max(
        0,
        min(
            int(
                frame_index
            ),
            len(
                tracking
            )
            - 1,
        ),
    )

    return tracking[
        safe_index
    ]


def event_frame_lookup(
    timeline: object,
) -> dict[str, int]:
    """
    Build readable event names mapped to their frame numbers.
    """

    output: dict[
        str,
        int,
    ] = {}

    for event in timeline.events:
        label = (
            event.event_type.value
            .replace(
                "_",
                " ",
            )
            .title()
        )

        # Add the frame to the label so duplicate-frame events remain unique.
        output[
            f"{label} — frame {int(event.frame)}"
        ] = int(
            event.frame
        )

    return output


def collect_trial_bounds(
    trial_data: dict,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    """
    Find stable XYZ limits for the entire uploaded trial.
    """

    x_values: list[
        float
    ] = []
    y_values: list[
        float
    ] = []
    z_values: list[
        float
    ] = []

    for frame in trial_data.get(
        "tracking",
        [],
    ):
        data = frame.get(
            "data",
            {},
        )

        ball = finite_point(
            data.get(
                "ball"
            )
        )

        if ball is not None:
            x_values.append(
                float(
                    ball[0]
                )
            )
            y_values.append(
                float(
                    ball[1]
                )
            )
            z_values.append(
                float(
                    ball[2]
                )
            )

        player = data.get(
            "player",
            {},
        )

        if isinstance(
            player,
            dict,
        ):
            for point_value in player.values():
                point = finite_point(
                    point_value
                )

                if point is None:
                    continue

                x_values.append(
                    float(
                        point[0]
                    )
                )
                y_values.append(
                    float(
                        point[1]
                    )
                )
                z_values.append(
                    float(
                        point[2]
                    )
                )

    if not x_values:
        return (
            (
                20.0,
                35.0,
            ),
            (
                -5.0,
                10.0,
            ),
            (
                0.0,
                12.0,
            ),
        )

    def padded_limits(
        values: list[float],
        padding: float,
        minimum_span: float,
    ) -> tuple[
        float,
        float,
    ]:
        lower = float(
            min(
                values
            )
        )

        upper = float(
            max(
                values
            )
        )

        span = max(
            minimum_span,
            upper
            - lower,
        )

        center = (
            lower
            + upper
        ) / 2.0

        half_span = (
            span
            / 2.0
            + padding
        )

        return (
            center
            - half_span,
            center
            + half_span,
        )

    return (
        padded_limits(
            x_values,
            padding=1.5,
            minimum_span=10.0,
        ),
        padded_limits(
            y_values,
            padding=1.5,
            minimum_span=8.0,
        ),
        padded_limits(
            z_values,
            padding=0.8,
            minimum_span=8.0,
        ),
    )


@st.cache_data(
    show_spinner=False,
)
def cached_trial_bounds(
    trial_json_text: str,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    return collect_trial_bounds(
        json.loads(
            trial_json_text
        )
    )


def ball_trajectory_until_frame(
    trial_data: dict,
    frame_index: int,
) -> np.ndarray:
    points: list[
        np.ndarray
    ] = []

    tracking = trial_data.get(
        "tracking",
        [],
    )

    maximum = min(
        len(
            tracking
        ),
        int(
            frame_index
        )
        + 1,
    )

    for frame in tracking[
        :maximum
    ]:
        point = finite_point(
            frame.get(
                "data",
                {},
            ).get(
                "ball"
            )
        )

        if point is not None:
            points.append(
                point
            )

    if not points:
        return np.empty(
            (
                0,
                3,
            ),
            dtype=float,
        )

    return np.vstack(
        points
    )


def project_tracking_point(
    point: np.ndarray,
    view_name: str,
) -> tuple[float, float]:
    x_value = float(point[0])
    y_value = float(point[1])
    z_value = float(point[2])

    if view_name == "Side":
        return x_value, z_value
    if view_name == "Front":
        return y_value, z_value
    if view_name == "Top":
        return x_value, y_value

    return (
        (x_value - y_value) / np.sqrt(2.0),
        z_value,
    )


@st.cache_data(show_spinner=False)
def cached_projected_bounds(
    trial_json_text: str,
    view_name: str,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
]:
    trial_data = json.loads(trial_json_text)
    horizontal_values = []
    vertical_values = []

    for tracking_frame in trial_data.get("tracking", []):
        payload = tracking_frame.get("data", {})
        points = []

        ball = finite_point(payload.get("ball"))
        if ball is not None:
            points.append(ball)

        player = payload.get("player", {})
        if isinstance(player, dict):
            for value in player.values():
                point = finite_point(value)
                if point is not None:
                    points.append(point)

        for point in points:
            projected_x, projected_y = project_tracking_point(
                point,
                view_name,
            )
            horizontal_values.append(projected_x)
            vertical_values.append(projected_y)

    if not horizontal_values:
        return (0.0, 40.0), (0.0, 15.0)

    def padded(values, minimum_span, padding_fraction):
        minimum = float(min(values))
        maximum = float(max(values))
        span = max(minimum_span, maximum - minimum)
        padding = span * padding_fraction
        return minimum - padding, maximum + padding

    return (
        padded(horizontal_values, 8.0, 0.12),
        padded(vertical_values, 8.0, 0.12),
    )


def render_fast_tracking_frame(
    trial_data: dict,
    frame_index: int,
    view_name: str,
    show_trajectory: bool,
    title: str,
    width: int = 640,
    height: int = 520,
) -> np.ndarray:
    canvas = Image.new(
        "RGB",
        (width, height),
        (247, 249, 252),
    )
    draw = ImageDraw.Draw(canvas)

    margin_left = 48
    margin_right = 24
    margin_top = 58
    margin_bottom = 44
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    bounds = cached_projected_bounds(
        json.dumps(trial_data, allow_nan=True),
        view_name,
    )

    horizontal_minimum, horizontal_maximum = bounds[0]
    vertical_minimum, vertical_maximum = bounds[1]
    horizontal_span = max(
        1e-9,
        horizontal_maximum - horizontal_minimum,
    )
    vertical_span = max(
        1e-9,
        vertical_maximum - vertical_minimum,
    )

    def pixel(point):
        projected_x, projected_y = project_tracking_point(
            point,
            view_name,
        )

        x_pixel = (
            margin_left
            + (
                projected_x - horizontal_minimum
            )
            / horizontal_span
            * plot_width
        )
        y_pixel = (
            margin_top
            + plot_height
            - (
                projected_y - vertical_minimum
            )
            / vertical_span
            * plot_height
        )
        return int(round(x_pixel)), int(round(y_pixel))

    draw.rounded_rectangle(
        (
            margin_left,
            margin_top,
            margin_left + plot_width,
            margin_top + plot_height,
        ),
        radius=18,
        fill=(255, 255, 255),
        outline=(221, 227, 236),
        width=2,
    )

    for fraction in (0.25, 0.50, 0.75):
        x_position = int(
            margin_left + plot_width * fraction
        )
        y_position = int(
            margin_top + plot_height * fraction
        )

        draw.line(
            (
                x_position,
                margin_top,
                x_position,
                margin_top + plot_height,
            ),
            fill=(235, 239, 245),
            width=1,
        )
        draw.line(
            (
                margin_left,
                y_position,
                margin_left + plot_width,
                y_position,
            ),
            fill=(235, 239, 245),
            width=1,
        )

    current_frame = frame_data(
        trial_data,
        frame_index,
    )
    payload = current_frame.get("data", {})
    player = payload.get("player", {})

    if show_trajectory and frame_index > 0:
        trajectory = ball_trajectory_until_frame(
            trial_data,
            frame_index,
        )
        trajectory_pixels = [
            pixel(point)
            for point in trajectory
        ]

        if len(trajectory_pixels) >= 2:
            draw.line(
                trajectory_pixels,
                fill=(234, 104, 42),
                width=4,
                joint="curve",
            )

    if isinstance(player, dict):
        for first_name, second_name in SKELETON_CONNECTIONS:
            first_point = finite_point(
                player.get(first_name)
            )
            second_point = finite_point(
                player.get(second_name)
            )

            if (
                first_point is None
                or second_point is None
            ):
                continue

            first_pixel = pixel(first_point)
            second_pixel = pixel(second_point)

            draw.line(
                (
                    first_pixel[0],
                    first_pixel[1],
                    second_pixel[0],
                    second_pixel[1],
                ),
                fill=(31, 55, 91),
                width=5,
            )

        for point_value in player.values():
            point = finite_point(point_value)
            if point is None:
                continue

            center_x, center_y = pixel(point)
            radius = 4

            draw.ellipse(
                (
                    center_x - radius,
                    center_y - radius,
                    center_x + radius,
                    center_y + radius,
                ),
                fill=(31, 55, 91),
            )

    ball = finite_point(payload.get("ball"))

    if ball is not None:
        center_x, center_y = pixel(ball)
        radius = 9

        draw.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ),
            fill=(244, 112, 45),
            outline=(161, 66, 20),
            width=2,
        )

    sampling_rate = float(
        trial_data.get("sampling_rate", 30)
    )
    time_value = current_frame.get(
        "time",
        frame_index / sampling_rate * 1000.0,
    )

    draw.text(
        (margin_left, 18),
        title,
        fill=(20, 32, 51),
    )
    draw.text(
        (margin_left, height - 28),
        (
            f"{view_name} view  |  "
            f"Frame {frame_index}  |  "
            f"{float(time_value):.0f} ms"
        ),
        fill=(91, 103, 123),
    )

    return np.asarray(
        canvas,
        dtype=np.uint8,
    )


def create_playback_figure(
    trial_data: dict,
    frame_index: int,
    view_name: str,
    show_trajectory: bool,
) -> plt.Figure:
    """
    Render one synchronized player-and-ball frame.
    """

    current_frame = frame_data(
        trial_data,
        frame_index,
    )

    data = current_frame.get(
        "data",
        {},
    )

    player = data.get(
        "player",
        {},
    )

    ball = finite_point(
        data.get(
            "ball"
        )
    )

    figure = plt.figure(
        figsize=(
            9.2,
            7.4,
        )
    )

    axis = figure.add_subplot(
        111,
        projection="3d",
    )

    if isinstance(
        player,
        dict,
    ):
        for first_name, second_name in SKELETON_CONNECTIONS:
            first_point = finite_point(
                player.get(
                    first_name
                )
            )

            second_point = finite_point(
                player.get(
                    second_name
                )
            )

            if (
                first_point is None
                or second_point is None
            ):
                continue

            axis.plot(
                [
                    first_point[0],
                    second_point[0],
                ],
                [
                    first_point[1],
                    second_point[1],
                ],
                [
                    first_point[2],
                    second_point[2],
                ],
                linewidth=2.4,
            )

        joint_points = [
            finite_point(
                value
            )
            for value in player.values()
        ]

        joint_points = [
            point
            for point in joint_points
            if point is not None
        ]

        if joint_points:
            joints = np.vstack(
                joint_points
            )

            axis.scatter(
                joints[
                    :,
                    0
                ],
                joints[
                    :,
                    1
                ],
                joints[
                    :,
                    2
                ],
                s=18,
            )

    if (
        show_trajectory
        and frame_index > 0
    ):
        trajectory = ball_trajectory_until_frame(
            trial_data,
            frame_index,
        )

        if len(
            trajectory
        ) >= 2:
            axis.plot(
                trajectory[
                    :,
                    0
                ],
                trajectory[
                    :,
                    1
                ],
                trajectory[
                    :,
                    2
                ],
                linewidth=2.0,
                alpha=0.75,
            )

    if ball is not None:
        axis.scatter(
            [
                ball[0]
            ],
            [
                ball[1]
            ],
            [
                ball[2]
            ],
            s=130,
            marker="o",
        )

    bounds = cached_trial_bounds(
        json.dumps(
            trial_data,
            allow_nan=True,
        )
    )

    axis.set_xlim(
        *bounds[0]
    )

    axis.set_ylim(
        *bounds[1]
    )

    axis.set_zlim(
        *bounds[2]
    )

    view_map = {
        "Side": (
            12,
            -90,
        ),
        "Front": (
            10,
            0,
        ),
        "45°": (
            15,
            -45,
        ),
        "Top": (
            88,
            -90,
        ),
    }

    elevation, azimuth = view_map.get(
        view_name,
        (
            15,
            -45,
        ),
    )

    axis.view_init(
        elev=elevation,
        azim=azimuth,
    )

    axis.set_xlabel(
        "Court X (ft)"
    )

    axis.set_ylabel(
        "Court Y (ft)"
    )

    axis.set_zlabel(
        "Height (ft)"
    )

    time_value = current_frame.get(
        "time",
        (
            frame_index
            / float(
                trial_data.get(
                    "sampling_rate",
                    30,
                )
            )
            * 1000.0
        ),
    )

    axis.set_title(
        (
            f"Frame {frame_index} · "
            f"{float(time_value):.0f} ms"
        ),
        pad=18,
    )

    axis.grid(
        visible=True,
        alpha=0.25,
    )

    figure.tight_layout()

    return figure



def trial_cache_key(
    trial_data: dict,
) -> str:
    """
    Create a stable short identifier for one uploaded trial.
    """

    trial_bytes = json.dumps(
        trial_data,
        sort_keys=True,
        allow_nan=True,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        trial_bytes
    ).hexdigest()[
        :16
    ]


def figure_to_rgb_array(
    figure: plt.Figure,
) -> np.ndarray:
    """
    Convert a Matplotlib figure into an RGB image for video encoding.

    Streamlit may attach a generic FigureCanvasBase to figures. That canvas
    does not provide buffer_rgba(). We therefore render through an explicit
    FigureCanvasAgg, which always provides a pixel buffer suitable for FFmpeg.
    """

    canvas = FigureCanvasAgg(
        figure
    )

    canvas.draw()

    rgba = np.asarray(
        canvas.buffer_rgba()
    )

    return np.ascontiguousarray(
        rgba[
            :,
            :,
            :3,
        ]
    )


@st.cache_data(
    show_spinner=False,
)
def render_tracking_video_bytes(
    trial_json_text: str,
    view_name: str,
    show_trajectory: bool,
    playback_speed: str,
    cache_identifier: str,
) -> bytes:
    """
    Render one complete tracked shot as an MP4.

    The expensive frame rendering happens once and is then cached. Browser
    playback is smooth because Streamlit serves a finished video instead of
    asking Python to redraw every frame during playback.
    """

    del cache_identifier

    trial_data = json.loads(
        trial_json_text
    )

    tracking = trial_data.get(
        "tracking",
        [],
    )

    if not tracking:
        raise ValueError(
            "No tracking frames were available for video rendering."
        )

    source_fps = float(
        trial_data.get(
            "sampling_rate",
            30,
        )
    )

    speed_multiplier = {
        "Real time": 1.0,
        "Half speed": 0.5,
        "Quarter speed": 0.25,
    }.get(
        playback_speed,
        1.0,
    )

    # Keep the encoded video smooth at every speed. Slow motion is created
    # by repeating source frames instead of lowering the video frame rate.
    output_fps = max(
        24.0,
        source_fps,
    )

    frame_repeat = max(
        1,
        int(
            round(
                1.0
                / speed_multiplier
            )
        ),
    )

    with tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False,
    ) as temporary_video:
        temporary_video_path = Path(
            temporary_video.name
        )

    try:
        writer = imageio.get_writer(
            temporary_video_path,
            fps=output_fps,
            codec="libx264",
            quality=7,
            pixelformat="yuv420p",
            macro_block_size=16,
            ffmpeg_log_level="error",
            output_params=[
                "-movflags",
                "+faststart",
                "-preset",
                "veryfast",
            ],
        )

        try:
            for frame_index in range(
                len(
                    tracking
                )
            ):
                frame_image = render_fast_tracking_frame(
                    trial_data=trial_data,
                    frame_index=frame_index,
                    view_name=view_name,
                    show_trajectory=show_trajectory,
                    title=(
                        f"{trial_data.get('participant_id', 'Participant')} · "
                        f"{trial_data.get('trial_id', 'Trial')}"
                    ),
                    width=720,
                    height=560,
                )

                for _ in range(
                    frame_repeat
                ):
                    writer.append_data(
                        frame_image
                    )

        finally:
            writer.close()

        return temporary_video_path.read_bytes()

    finally:
        temporary_video_path.unlink(
            missing_ok=True
        )


def render_smooth_video_player(
    trial_data: dict,
    title: str,
    widget_prefix: str,
) -> None:
    """
    Display a cached browser-native MP4 player for smooth shot playback.
    """

    st.markdown(
        f"### {title}"
    )

    control_columns = st.columns(
        [
            0.34,
            0.33,
            0.33,
        ]
    )

    playback_speed = control_columns[0].selectbox(
        "Playback speed",
        options=[
            "Real time",
            "Half speed",
            "Quarter speed",
        ],
        index=0,
        key=(
            f"{widget_prefix}_video_speed"
        ),
    )

    view_name = control_columns[1].selectbox(
        "Video camera view",
        options=[
            "45°",
            "Side",
            "Front",
            "Top",
        ],
        index=0,
        key=(
            f"{widget_prefix}_video_view"
        ),
    )

    show_trajectory = control_columns[2].toggle(
        "Show trajectory",
        value=True,
        key=(
            f"{widget_prefix}_video_trajectory"
        ),
    )

    cache_identifier = trial_cache_key(
        trial_data
    )

    render_clicked = st.button(
        (
            "Generate Smooth MP4"
            if (
                f"{widget_prefix}_video_bytes"
                not in st.session_state
            )
            else "Regenerate Smooth MP4"
        ),
        key=(
            f"{widget_prefix}_render_video"
        ),
        type="primary",
        use_container_width=True,
    )

    settings_signature = (
        cache_identifier,
        playback_speed,
        view_name,
        show_trajectory,
    )

    stored_signature_key = (
        f"{widget_prefix}_video_signature"
    )

    stored_video_key = (
        f"{widget_prefix}_video_bytes"
    )

    settings_changed = (
        st.session_state.get(
            stored_signature_key
        )
        != settings_signature
    )

    if render_clicked:
        with st.spinner(
            "Rendering the complete shot once. Future playback will be smooth..."
        ):
            video_bytes = render_tracking_video_bytes(
                trial_json_text=json.dumps(
                    trial_data,
                    allow_nan=True,
                ),
                view_name=view_name,
                show_trajectory=show_trajectory,
                playback_speed=playback_speed,
                cache_identifier=cache_identifier,
            )

        st.session_state[
            stored_video_key
        ] = video_bytes

        st.session_state[
            stored_signature_key
        ] = settings_signature

        settings_changed = False

    if settings_changed and (
        stored_video_key
        in st.session_state
    ):
        st.info(
            "Playback settings changed. Select **Regenerate Smooth MP4** "
            "to create the updated version."
        )

    video_bytes = st.session_state.get(
        stored_video_key
    )

    if video_bytes is not None:
        st.video(
            video_bytes,
            format="video/mp4",
            start_time=0,
        )

        st.download_button(
            label="Download Complete Shot MP4",
            data=video_bytes,
            file_name=(
                f"{trial_data.get('participant_id', 'participant')}_"
                f"{trial_data.get('trial_id', 'trial')}_"
                "complete_shot.mp4"
            ),
            mime="video/mp4",
            use_container_width=True,
            key=f"{widget_prefix}_download_video",
        )

        st.caption(
            (
                f"{playback_speed} · {view_name} view · "
                f"{'trajectory shown' if show_trajectory else 'trajectory hidden'}. "
                "Use the browser player's controls for smooth playback, "
                "pausing, seeking, and replay."
            )
        )

    else:
        st.info(
            "Select **Generate Smooth MP4** to create a browser-native video. "
            "The first render may take time, but playback will then be smooth "
            "and the result will be cached."
        )



def combine_rgb_frames(
    left_image: np.ndarray,
    right_image: np.ndarray,
) -> np.ndarray:
    """
    Place two RGB frames side-by-side with matching heights.
    """

    target_height = min(
        left_image.shape[0],
        right_image.shape[0],
    )

    left = left_image[
        :target_height,
        :,
        :
    ]

    right = right_image[
        :target_height,
        :,
        :
    ]

    separator = np.full(
        (
            target_height,
            12,
            3,
        ),
        245,
        dtype=np.uint8,
    )

    return np.concatenate(
        [
            left,
            separator,
            right,
        ],
        axis=1,
    )


def resolve_clip_window(
    preset: str,
    feature_record: object,
    total_frames: int,
    custom_start: int,
    custom_end: int,
) -> tuple[int, int]:
    """
    Resolve the current-shot frame window for smooth playback.
    """

    motion = feature_record.motion_start_frame
    takeoff = feature_record.takeoff_frame
    release = feature_record.release_frame
    landing = feature_record.landing_frame

    if preset == "Full shot":
        start_frame = 0
        end_frame = total_frames - 1

    elif preset == "Release window":
        anchor = (
            release
            if release is not None
            else total_frames // 2
        )

        start_frame = max(
            0,
            int(anchor) - 15,
        )

        end_frame = min(
            total_frames - 1,
            int(anchor) + 10,
        )

    elif preset == "Takeoff to release":
        start_frame = (
            int(takeoff)
            if takeoff is not None
            else max(
                0,
                int(release or 0) - 15,
            )
        )

        end_frame = (
            int(release)
            if release is not None
            else min(
                total_frames - 1,
                start_frame + 20,
            )
        )

    elif preset == "Motion start to release":
        start_frame = (
            int(motion)
            if motion is not None
            else 0
        )

        end_frame = (
            int(release)
            if release is not None
            else total_frames - 1
        )

    elif preset == "Release to landing":
        start_frame = (
            int(release)
            if release is not None
            else 0
        )

        end_frame = (
            int(landing)
            if landing is not None
            else total_frames - 1
        )

    else:
        start_frame = int(
            custom_start
        )

        end_frame = int(
            custom_end
        )

    start_frame = max(
        0,
        min(
            start_frame,
            total_frames - 1,
        ),
    )

    end_frame = max(
        start_frame,
        min(
            end_frame,
            total_frames - 1,
        ),
    )

    return (
        start_frame,
        end_frame,
    )


@st.cache_data(
    show_spinner=False,
)
def render_smooth_comparison_video_bytes(
    current_trial_json_text: str,
    comparison_trial_json_text: str,
    current_filename: str,
    comparison_filename: str,
    shooting_side: str,
    alignment_mode: str,
    view_name: str,
    show_trajectory: bool,
    playback_speed: str,
    start_frame: int,
    end_frame: int,
    cache_identifier: str,
) -> bytes:
    """
    Render one synchronized side-by-side MP4.

    The current-shot frame window controls the clip. Comparison frames are
    aligned to the selected event for every output frame.
    """

    del cache_identifier

    current_trial_data = json.loads(
        current_trial_json_text
    )

    comparison_trial_data = json.loads(
        comparison_trial_json_text
    )

    current_total_frames = len(
        current_trial_data.get(
            "tracking",
            [],
        )
    )

    comparison_total_frames = len(
        comparison_trial_data.get(
            "tracking",
            [],
        )
    )

    if (
        current_total_frames <= 0
        or comparison_total_frames <= 0
    ):
        raise ValueError(
            "Both trials must contain tracking frames."
        )

    current_events = detect_trial_events_for_comparison(
        current_trial_data,
        current_filename,
        shooting_side,
    )

    comparison_events = detect_trial_events_for_comparison(
        comparison_trial_data,
        comparison_filename,
        shooting_side,
    )

    source_fps = float(
        current_trial_data.get(
            "sampling_rate",
            30,
        )
    )

    speed_multiplier = {
        "Real time": 1.0,
        "Half speed": 0.5,
        "Quarter speed": 0.25,
    }.get(
        playback_speed,
        1.0,
    )

    # Keep the encoded video smooth at every speed. Slow motion is created
    # by repeating source frames instead of lowering the video frame rate.
    output_fps = max(
        24.0,
        source_fps,
    )

    frame_repeat = max(
        1,
        int(
            round(
                1.0
                / speed_multiplier
            )
        ),
    )

    with tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False,
    ) as temporary_video:
        temporary_video_path = Path(
            temporary_video.name
        )

    try:
        writer = imageio.get_writer(
            temporary_video_path,
            fps=output_fps,
            codec="libx264",
            quality=7,
            pixelformat="yuv420p",
            macro_block_size=16,
            ffmpeg_log_level="error",
            output_params=[
                "-movflags",
                "+faststart",
                "-preset",
                "veryfast",
            ],
        )

        try:
            for current_frame in range(
                start_frame,
                end_frame + 1,
            ):
                comparison_frame = aligned_comparison_frame(
                    current_frame=current_frame,
                    current_events=current_events,
                    comparison_events=comparison_events,
                    alignment_mode=alignment_mode,
                    comparison_total_frames=(
                        comparison_total_frames
                    ),
                )

                current_image = render_fast_tracking_frame(
                    trial_data=current_trial_data,
                    frame_index=current_frame,
                    view_name=view_name,
                    show_trajectory=show_trajectory,
                    title=(
                        f"Shot 1 · "
                        f"{current_trial_data.get('participant_id', 'Participant')} · "
                        f"{current_trial_data.get('trial_id', 'Trial')}"
                    ),
                    width=640,
                    height=520,
                )

                comparison_image = render_fast_tracking_frame(
                    trial_data=comparison_trial_data,
                    frame_index=comparison_frame,
                    view_name=view_name,
                    show_trajectory=show_trajectory,
                    title=(
                        f"Shot 2 · "
                        f"{comparison_trial_data.get('participant_id', 'Participant')} · "
                        f"{comparison_trial_data.get('trial_id', 'Trial')}"
                    ),
                    width=640,
                    height=520,
                )

                combined_image = combine_rgb_frames(
                    current_image,
                    comparison_image,
                )

                for _ in range(
                    frame_repeat
                ):
                    writer.append_data(
                        combined_image
                    )

        finally:
            writer.close()

        return temporary_video_path.read_bytes()

    finally:
        temporary_video_path.unlink(
            missing_ok=True
        )


def render_smooth_comparison_player(
    result: dict,
    current_trial_data: dict,
    comparison_trial_data: dict,
    current_filename: str,
    comparison_filename: str,
    shooting_side: str,
) -> None:
    """
    Full-featured smooth comparison renderer.
    """

    feature_record = result[
        "feature_record"
    ]

    current_total_frames = len(
        current_trial_data.get(
            "tracking",
            [],
        )
    )

    if current_total_frames <= 0:
        st.warning(
            "The current trial does not contain tracking frames."
        )

        return

    st.markdown(
        "### Synchronized Split-Screen MP4"
    )

    first_row = st.columns(
        4
    )

    playback_speed = first_row[0].selectbox(
        "Comparison speed",
        options=[
            "Real time",
            "Half speed",
            "Quarter speed",
        ],
        index=0,
        key="smooth_comparison_speed",
    )

    alignment_mode = first_row[1].selectbox(
        "Align at",
        options=[
            "Release",
            "Takeoff",
            "Dip",
            "Motion Start",
            "Ball Apex",
            "Landing",
        ],
        index=0,
        key="smooth_comparison_alignment",
    )

    view_name = first_row[2].selectbox(
        "Comparison camera",
        options=[
            "45°",
            "Side",
            "Front",
            "Top",
        ],
        index=0,
        key="smooth_comparison_view",
    )

    show_trajectory = first_row[3].toggle(
        "Show trajectories",
        value=True,
        key="smooth_comparison_trajectory",
    )

    second_row = st.columns(
        [
            0.42,
            0.29,
            0.29,
        ]
    )

    clip_preset = second_row[0].selectbox(
        "Clip preset",
        options=[
            "Full shot",
            "Release window",
            "Takeoff to release",
            "Motion start to release",
            "Release to landing",
            "Custom",
        ],
        index=1,
        key="smooth_comparison_clip_preset",
    )

    default_start = max(
        0,
        int(
            feature_record.release_frame
            or 0
        )
        - 15,
    )

    default_end = min(
        current_total_frames - 1,
        int(
            feature_record.release_frame
            or (
                current_total_frames
                - 1
            )
        )
        + 10,
    )

    custom_start = second_row[1].number_input(
        "Custom start frame",
        min_value=0,
        max_value=current_total_frames - 1,
        value=default_start,
        step=1,
        key="smooth_comparison_custom_start",
        disabled=(
            clip_preset
            != "Custom"
        ),
    )

    custom_end = second_row[2].number_input(
        "Custom end frame",
        min_value=0,
        max_value=current_total_frames - 1,
        value=default_end,
        step=1,
        key="smooth_comparison_custom_end",
        disabled=(
            clip_preset
            != "Custom"
        ),
    )

    start_frame, end_frame = resolve_clip_window(
        preset=clip_preset,
        feature_record=feature_record,
        total_frames=current_total_frames,
        custom_start=int(
            custom_start
        ),
        custom_end=int(
            custom_end
        ),
    )

    clip_length = (
        end_frame
        - start_frame
        + 1
    )

    st.caption(
        (
            f"Current-shot clip: frames {start_frame}–{end_frame} "
            f"({clip_length} frames). The comparison shot is aligned "
            f"at {alignment_mode.lower()}."
        )
    )

    current_key = trial_cache_key(
        current_trial_data
    )

    comparison_key = trial_cache_key(
        comparison_trial_data
    )

    settings_signature = (
        current_key,
        comparison_key,
        playback_speed,
        alignment_mode,
        view_name,
        show_trajectory,
        start_frame,
        end_frame,
    )

    signature_key = (
        "smooth_comparison_video_signature"
    )

    video_key = (
        "smooth_comparison_video_bytes"
    )

    render_clicked = st.button(
        (
            "Generate Split-Screen MP4"
            if video_key
            not in st.session_state
            else "Regenerate Split-Screen MP4"
        ),
        type="primary",
        use_container_width=True,
        key="smooth_comparison_generate",
    )

    if render_clicked:
        with st.spinner(
            "Rendering the optimized synchronized split-screen MP4. The finished video will appear when encoding is complete..."
        ):
            video_bytes = (
                render_smooth_comparison_video_bytes(
                    current_trial_json_text=json.dumps(
                        current_trial_data,
                        allow_nan=True,
                    ),
                    comparison_trial_json_text=json.dumps(
                        comparison_trial_data,
                        allow_nan=True,
                    ),
                    current_filename=current_filename,
                    comparison_filename=comparison_filename,
                    shooting_side=shooting_side,
                    alignment_mode=alignment_mode,
                    view_name=view_name,
                    show_trajectory=show_trajectory,
                    playback_speed=playback_speed,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    cache_identifier=(
                        f"{current_key}_{comparison_key}"
                    ),
                )
            )

        st.session_state[
            video_key
        ] = video_bytes

        st.session_state[
            signature_key
        ] = settings_signature

    saved_video = st.session_state.get(
        video_key
    )

    saved_signature = st.session_state.get(
        signature_key
    )

    if (
        saved_video is not None
        and saved_signature
        != settings_signature
    ):
        st.info(
            "Comparison settings changed. Select **Regenerate Smooth "
            "Comparison** to update the video."
        )

    if saved_video is not None:
        st.video(
            saved_video,
            format="video/mp4",
            start_time=0,
        )

        st.download_button(
            label="Download Split-Screen Comparison MP4",
            data=saved_video,
            file_name=(
                f"{feature_record.participant_id}_"
                f"{feature_record.trial_id}_"
                "comparison.mp4"
            ),
            mime="video/mp4",
            use_container_width=True,
        )

        st.caption(
            (
                f"{playback_speed} · {view_name} view · "
                f"aligned at {alignment_mode} · "
                f"frames {start_frame}–{end_frame}."
            )
        )

    else:
        st.info(
            "Choose the alignment, speed, camera, trajectory, and frame "
            "window, then select **Generate Split-Screen MP4**."
        )


def render_comparison_studio_workspace(
    shooting_side: str,
) -> None:
    """
    Upload or reuse two free throws and render one synchronized split-screen MP4.
    """

    st.markdown(
        """
        <div class="bms-upload-shell">
            <strong>Upload two free-throw tracking files</strong><br>
            Shot 1 can reuse the file already analyzed in Single Shot Analysis.
            Shot 2 is uploaded directly here. The final result is one
            synchronized split-screen MP4 with both shots inside the same video.
        </div>
        """,
        unsafe_allow_html=True,
    )

    existing_data = st.session_state.get(
        "analysis_trial_data"
    )
    existing_filename = st.session_state.get(
        "analysis_filename"
    )

    use_existing = False

    if existing_data is not None:
        use_existing = st.checkbox(
            (
                "Use current Single Shot Analysis file as Shot 1 "
                f"({existing_filename or 'loaded shot'})"
            ),
            value=True,
            key="comparison_use_existing_shot_one",
        )

    left, right = st.columns(
        2,
        gap="large",
    )

    with left:
        shot_one_upload = None

        if use_existing:
            st.success(
                f"Shot 1: {existing_filename or 'current single shot'}"
            )
        else:
            shot_one_upload = st.file_uploader(
                "Shot 1 — Free-Throw Tracking JSON",
                type=["json"],
                accept_multiple_files=False,
                key="comparison_studio_shot_one_upload",
            )

    with right:
        shot_two_upload = st.file_uploader(
            "Shot 2 — Free-Throw Tracking JSON",
            type=["json"],
            accept_multiple_files=False,
            key="comparison_studio_shot_two_upload",
        )

    try:
        if use_existing:
            shot_one_data = existing_data
            shot_one_filename = (
                existing_filename
                or "single_shot.json"
            )
        elif shot_one_upload is not None:
            shot_one_data = validate_uploaded_json(
                json.loads(
                    shot_one_upload.getvalue()
                )
            )
            shot_one_filename = shot_one_upload.name
        else:
            shot_one_data = None
            shot_one_filename = None

        if shot_two_upload is not None:
            shot_two_data = validate_uploaded_json(
                json.loads(
                    shot_two_upload.getvalue()
                )
            )
            shot_two_filename = shot_two_upload.name
        else:
            shot_two_data = None
            shot_two_filename = None

    except Exception as error:
        st.error(
            (
                "Could not read one of the comparison JSON files. "
                f"{type(error).__name__}: {error}"
            )
        )
        return

    if (
        shot_one_data is None
        or shot_two_data is None
    ):
        st.info(
            "Provide both Shot 1 and Shot 2 to build the comparison."
        )
        return

    metrics = st.columns(
        4
    )

    metrics[0].metric(
        "Shot 1 participant",
        shot_one_data.get(
            "participant_id",
            "Unknown",
        ),
    )
    metrics[1].metric(
        "Shot 1 trial",
        shot_one_data.get(
            "trial_id",
            Path(
                shot_one_filename
            ).stem,
        ),
    )
    metrics[2].metric(
        "Shot 2 participant",
        shot_two_data.get(
            "participant_id",
            "Unknown",
        ),
    )
    metrics[3].metric(
        "Shot 2 trial",
        shot_two_data.get(
            "trial_id",
            Path(
                shot_two_filename
            ).stem,
        ),
    )

    prepared_signature = (
        trial_cache_key(
            shot_one_data
        ),
        shot_one_filename,
        shooting_side,
    )

    if (
        st.session_state.get(
            "comparison_primary_result_signature"
        )
        != prepared_signature
    ):
        with st.spinner(
            "Preparing event timing for the synchronized comparison..."
        ):
            try:
                primary_result = analyze_uploaded_trial(
                    trial_data=shot_one_data,
                    original_filename=shot_one_filename,
                    shooting_side=shooting_side,
                )
            except Exception as error:
                st.error(
                    (
                        "Shot 1 could not be prepared. "
                        f"{type(error).__name__}: {error}"
                    )
                )
                return

        st.session_state[
            "comparison_primary_result"
        ] = primary_result
        st.session_state[
            "comparison_primary_result_signature"
        ] = prepared_signature

    primary_result = st.session_state.get(
        "comparison_primary_result"
    )

    if primary_result is None:
        return

    render_smooth_comparison_player(
        result=primary_result,
        current_trial_data=shot_one_data,
        comparison_trial_data=shot_two_data,
        current_filename=shot_one_filename,
        comparison_filename=shot_two_filename,
        shooting_side=shooting_side,
    )


def initialize_playback_state() -> None:
    if "playback_frame" not in st.session_state:
        st.session_state.playback_frame = 0

    if "playback_view" not in st.session_state:
        st.session_state.playback_view = "45°"

    if "playback_show_trajectory" not in st.session_state:
        st.session_state.playback_show_trajectory = True

    if "playback_playing" not in st.session_state:
        st.session_state.playback_playing = False

    if "playback_speed" not in st.session_state:
        st.session_state.playback_speed = "Normal"

    if "playback_loop" not in st.session_state:
        st.session_state.playback_loop = False

    if "diagnosis_segment_active" not in st.session_state:
        st.session_state.diagnosis_segment_active = False

    if "diagnosis_segment_start" not in st.session_state:
        st.session_state.diagnosis_segment_start = None

    if "diagnosis_segment_end" not in st.session_state:
        st.session_state.diagnosis_segment_end = None

    if "diagnosis_segment_title" not in st.session_state:
        st.session_state.diagnosis_segment_title = ""

    if "diagnosis_segment_focus" not in st.session_state:
        st.session_state.diagnosis_segment_focus = ""

    if "comparison_trial_data" not in st.session_state:
        st.session_state.comparison_trial_data = None

    if "comparison_filename" not in st.session_state:
        st.session_state.comparison_filename = None

    if "comparison_file_signature" not in st.session_state:
        st.session_state.comparison_file_signature = None

    if "comparison_enabled" not in st.session_state:
        st.session_state.comparison_enabled = False

    if "comparison_alignment_mode" not in st.session_state:
        st.session_state.comparison_alignment_mode = "Release"

    if "comparison_frame_offset" not in st.session_state:
        st.session_state.comparison_frame_offset = 0

    for video_key in (
        "smooth_comparison_video_bytes",
        "smooth_comparison_video_signature",
    ):
        st.session_state.pop(
            video_key,
            None,
        )


def set_playback_frame(
    frame_value: int,
    total_frames: int,
) -> None:
    """
    Update the canonical playback frame.

    Do not write directly to a Streamlit widget key after that widget has
    already been instantiated during the current app run.
    """

    safe_frame = max(
        0,
        min(
            int(
                frame_value
            ),
            total_frames
            - 1,
        ),
    )

    st.session_state.playback_frame = (
        safe_frame
    )



def joint_angle_degrees(
    first_point: np.ndarray | None,
    center_point: np.ndarray | None,
    third_point: np.ndarray | None,
) -> float:
    """
    Calculate the angle at center_point using three tracked points.
    """

    if (
        first_point is None
        or center_point is None
        or third_point is None
    ):
        return float(
            "nan"
        )

    first_vector = (
        first_point
        - center_point
    )

    second_vector = (
        third_point
        - center_point
    )

    first_length = float(
        np.linalg.norm(
            first_vector
        )
    )

    second_length = float(
        np.linalg.norm(
            second_vector
        )
    )

    if (
        first_length <= 1e-9
        or second_length <= 1e-9
    ):
        return float(
            "nan"
        )

    cosine_value = float(
        np.dot(
            first_vector,
            second_vector,
        )
        / (
            first_length
            * second_length
        )
    )

    cosine_value = float(
        np.clip(
            cosine_value,
            -1.0,
            1.0,
        )
    )

    return float(
        np.degrees(
            np.arccos(
                cosine_value
            )
        )
    )


def finite_difference_speed(
    points: list[np.ndarray | None],
    sampling_rate: float,
) -> np.ndarray:
    """
    Estimate frame-to-frame 3D speed.
    """

    output = np.full(
        len(
            points
        ),
        np.nan,
        dtype=float,
    )

    for index in range(
        1,
        len(
            points
        ),
    ):
        current = points[
            index
        ]

        previous = points[
            index
            - 1
        ]

        if (
            current is None
            or previous is None
        ):
            continue

        output[
            index
        ] = float(
            np.linalg.norm(
                current
                - previous
            )
            * sampling_rate
        )

    return output


def smooth_numeric_series(
    values: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """
    Smooth a numeric series while preserving missing values.
    """

    output = np.full_like(
        values,
        np.nan,
        dtype=float,
    )

    half_window = max(
        1,
        int(
            window
        )
        // 2,
    )

    for index in range(
        len(
            values
        ),
    ):
        start = max(
            0,
            index
            - half_window,
        )

        end = min(
            len(
                values
            ),
            index
            + half_window
            + 1,
        )

        local_values = values[
            start:
            end
        ]

        valid_values = local_values[
            np.isfinite(
                local_values
            )
        ]

        if len(
            valid_values
        ) > 0:
            output[
                index
            ] = float(
                np.mean(
                    valid_values
                )
            )

    return output


@st.cache_data(
    show_spinner=False,
)
def build_synchronized_signals(
    trial_json_text: str,
    shooting_side: str,
) -> dict[str, list[float]]:
    """
    Build frame-aligned signals for the playback dashboard.
    """

    trial_data = json.loads(
        trial_json_text
    )

    tracking = trial_data.get(
        "tracking",
        [],
    )

    sampling_rate = float(
        trial_data.get(
            "sampling_rate",
            30,
        )
    )

    side = shooting_side.upper()

    shoulder_name = (
        f"{side}_SHOULDER"
    )

    elbow_name = (
        f"{side}_ELBOW"
    )

    wrist_name = (
        f"{side}_WRIST"
    )

    hip_name = (
        f"{side}_HIP"
    )

    knee_name = (
        f"{side}_KNEE"
    )

    ankle_name = (
        f"{side}_ANKLE"
    )

    knee_angles: list[
        float
    ] = []

    elbow_angles: list[
        float
    ] = []

    wrist_points: list[
        np.ndarray | None
    ] = []

    ball_points: list[
        np.ndarray | None
    ] = []

    for frame in tracking:
        data = frame.get(
            "data",
            {},
        )

        player = data.get(
            "player",
            {},
        )

        if not isinstance(
            player,
            dict,
        ):
            player = {}

        shoulder = finite_point(
            player.get(
                shoulder_name
            )
        )

        elbow = finite_point(
            player.get(
                elbow_name
            )
        )

        wrist = finite_point(
            player.get(
                wrist_name
            )
        )

        hip = finite_point(
            player.get(
                hip_name
            )
        )

        knee = finite_point(
            player.get(
                knee_name
            )
        )

        ankle = finite_point(
            player.get(
                ankle_name
            )
        )

        knee_angles.append(
            joint_angle_degrees(
                hip,
                knee,
                ankle,
            )
        )

        elbow_angles.append(
            joint_angle_degrees(
                shoulder,
                elbow,
                wrist,
            )
        )

        wrist_points.append(
            wrist
        )

        ball_points.append(
            finite_point(
                data.get(
                    "ball"
                )
            )
        )

    wrist_speed = finite_difference_speed(
        wrist_points,
        sampling_rate,
    )

    ball_speed = finite_difference_speed(
        ball_points,
        sampling_rate,
    )

    ball_height = np.asarray(
        [
            (
                float(
                    point[2]
                )
                if point is not None
                else np.nan
            )
            for point in ball_points
        ],
        dtype=float,
    )

    knee_array = smooth_numeric_series(
        np.asarray(
            knee_angles,
            dtype=float,
        ),
        window=5,
    )

    elbow_array = smooth_numeric_series(
        np.asarray(
            elbow_angles,
            dtype=float,
        ),
        window=5,
    )

    wrist_speed = smooth_numeric_series(
        wrist_speed,
        window=5,
    )

    ball_speed = smooth_numeric_series(
        ball_speed,
        window=5,
    )

    ball_height = smooth_numeric_series(
        ball_height,
        window=5,
    )

    frames = np.arange(
        len(
            tracking
        ),
        dtype=float,
    )

    times_ms = (
        frames
        / sampling_rate
        * 1000.0
    )

    return {
        "frame": frames.tolist(),
        "time_ms": times_ms.tolist(),
        "knee_angle_deg": knee_array.tolist(),
        "elbow_angle_deg": elbow_array.tolist(),
        "wrist_speed_ft_s": wrist_speed.tolist(),
        "ball_speed_ft_s": ball_speed.tolist(),
        "ball_height_ft": ball_height.tolist(),
    }


def safe_series_value(
    values: list[float],
    frame_index: int,
) -> float | None:
    if (
        frame_index < 0
        or frame_index
        >= len(
            values
        )
    ):
        return None

    value = float(
        values[
            frame_index
        ]
    )

    if not np.isfinite(
        value
    ):
        return None

    return value


def create_synchronized_signal_figure(
    signals: dict[str, list[float]],
    current_frame: int,
    event_frames: list[int],
) -> plt.Figure:
    """
    Plot synchronized biomechanical signals with one moving frame cursor.
    """

    frames = np.asarray(
        signals[
            "frame"
        ],
        dtype=float,
    )

    knee_angle = np.asarray(
        signals[
            "knee_angle_deg"
        ],
        dtype=float,
    )

    elbow_angle = np.asarray(
        signals[
            "elbow_angle_deg"
        ],
        dtype=float,
    )

    wrist_speed = np.asarray(
        signals[
            "wrist_speed_ft_s"
        ],
        dtype=float,
    )

    ball_height = np.asarray(
        signals[
            "ball_height_ft"
        ],
        dtype=float,
    )

    figure = plt.figure(
        figsize=(
            12,
            9.5,
        )
    )

    grid = figure.add_gridspec(
        4,
        1,
        hspace=0.32,
    )

    axes = [
        figure.add_subplot(
            grid[index]
        )
        for index in range(
            4
        )
    ]

    plots = [
        (
            axes[0],
            knee_angle,
            "Shooting-side knee angle",
            "Degrees",
        ),
        (
            axes[1],
            elbow_angle,
            "Shooting-side elbow angle",
            "Degrees",
        ),
        (
            axes[2],
            wrist_speed,
            "Shooting-side wrist speed",
            "ft/s",
        ),
        (
            axes[3],
            ball_height,
            "Ball height",
            "ft",
        ),
    ]

    for axis, values, title, y_label in plots:
        axis.plot(
            frames,
            values,
            linewidth=2.0,
        )

        axis.axvline(
            current_frame,
            linewidth=2.0,
            linestyle="--",
        )

        for event_frame in event_frames:
            axis.axvline(
                event_frame,
                linewidth=0.8,
                alpha=0.13,
            )

        if (
            current_frame
            < len(
                values
            )
            and np.isfinite(
                values[
                    current_frame
                ]
            )
        ):
            axis.scatter(
                [
                    current_frame
                ],
                [
                    values[
                        current_frame
                    ]
                ],
                s=55,
                zorder=5,
            )

        axis.set_title(
            title,
            loc="left",
            fontsize=11,
        )

        axis.set_ylabel(
            y_label
        )

        axis.grid(
            visible=True,
            alpha=0.22,
        )

    axes[-1].set_xlabel(
        "Tracking frame"
    )

    figure.suptitle(
        (
            "Synchronized Biomechanics — "
            f"Frame {current_frame}"
        ),
        fontsize=15,
        y=0.995,
    )

    figure.tight_layout()

    return figure


def render_live_signal_panel(
    result: dict,
    trial_data: dict,
    shooting_side: str,
) -> None:
    """
    Render current-frame metrics and synchronized charts.
    """

    signals = build_synchronized_signals(
        json.dumps(
            trial_data,
            allow_nan=True,
        ),
        shooting_side,
    )

    current_frame = int(
        st.session_state
        .playback_frame
    )

    event_frames = sorted(
        {
            int(
                event.frame
            )
            for event in result[
                "timeline"
            ].events
        }
    )

    knee_value = safe_series_value(
        signals[
            "knee_angle_deg"
        ],
        current_frame,
    )

    elbow_value = safe_series_value(
        signals[
            "elbow_angle_deg"
        ],
        current_frame,
    )

    wrist_value = safe_series_value(
        signals[
            "wrist_speed_ft_s"
        ],
        current_frame,
    )

    ball_height_value = safe_series_value(
        signals[
            "ball_height_ft"
        ],
        current_frame,
    )

    metric_columns = st.columns(
        4
    )

    metric_columns[0].metric(
        "Knee angle",
        (
            "N/A"
            if knee_value is None
            else f"{knee_value:.1f}°"
        ),
    )

    metric_columns[1].metric(
        "Elbow angle",
        (
            "N/A"
            if elbow_value is None
            else f"{elbow_value:.1f}°"
        ),
    )

    metric_columns[2].metric(
        "Wrist speed",
        (
            "N/A"
            if wrist_value is None
            else f"{wrist_value:.2f} ft/s"
        ),
    )

    metric_columns[3].metric(
        "Ball height",
        (
            "N/A"
            if ball_height_value is None
            else f"{ball_height_value:.2f} ft"
        ),
    )

    figure = create_synchronized_signal_figure(
        signals=signals,
        current_frame=current_frame,
        event_frames=event_frames,
    )

    st.pyplot(
        figure,
        clear_figure=True,
        use_container_width=True,
    )

    st.caption(
        "The dashed cursor follows the same frame shown in the animation. "
        "Faint vertical lines mark authoritative timeline events."
    )




def clear_comparison_state() -> None:
    """
    Remove the active comparison trial.
    """

    st.session_state.comparison_trial_data = None
    st.session_state.comparison_filename = None
    st.session_state.comparison_file_signature = None
    st.session_state.comparison_enabled = False
    st.session_state.comparison_frame_offset = 0


def load_comparison_upload(
    uploaded_file: object | None,
) -> None:
    """
    Validate and store the optional comparison JSON.
    """

    if uploaded_file is None:
        return

    signature = (
        uploaded_file.name,
        len(
            uploaded_file.getvalue()
        ),
    )

    if (
        st.session_state.comparison_file_signature
        == signature
    ):
        return

    comparison_data = json.loads(
        uploaded_file.getvalue()
    )

    comparison_data = validate_uploaded_json(
        comparison_data
    )

    st.session_state.comparison_trial_data = (
        comparison_data
    )

    st.session_state.comparison_filename = (
        uploaded_file.name
    )

    st.session_state.comparison_file_signature = (
        signature
    )

    st.session_state.comparison_enabled = True


def detect_trial_events_for_comparison(
    trial_data: dict,
    filename: str,
    shooting_side: str,
) -> dict[str, int | None]:
    """
    Detect key events for a comparison shot without running personal scoring.
    """

    with tempfile.TemporaryDirectory() as temporary_folder:
        temporary_path = (
            Path(
                temporary_folder
            )
            / filename
        )

        temporary_path.write_text(
            json.dumps(
                trial_data
            ),
            encoding="utf-8",
        )

        shot_analysis = ShotAnalyzer(
            trial_data=trial_data,
            trial_file=temporary_path,
            shooting_wrist=(
                f"{shooting_side}_WRIST"
            ),
        ).analyze()

    return {
        "motion_start": (
            shot_analysis
            .motion_start_frame
        ),
        "dip": (
            shot_analysis
            .dip_frame
        ),
        "takeoff": (
            shot_analysis
            .takeoff_frame
        ),
        "release": (
            shot_analysis
            .release_frame
        ),
        "ball_apex": (
            shot_analysis
            .ball_apex_frame
        ),
        "landing": (
            shot_analysis
            .landing_frame
        ),
    }


@st.cache_data(
    show_spinner=False,
)
def cached_comparison_events(
    trial_json_text: str,
    filename: str,
    shooting_side: str,
) -> dict[str, int | None]:
    return detect_trial_events_for_comparison(
        trial_data=json.loads(
            trial_json_text
        ),
        filename=filename,
        shooting_side=shooting_side,
    )


def aligned_comparison_frame(
    current_frame: int,
    current_events: dict[str, int | None],
    comparison_events: dict[str, int | None],
    alignment_mode: str,
    comparison_total_frames: int,
) -> int:
    """
    Align two trials around a key event while preserving relative timing.
    """

    alignment_key = (
        alignment_mode
        .strip()
        .lower()
        .replace(
            " ",
            "_",
        )
    )

    current_anchor = current_events.get(
        alignment_key
    )

    comparison_anchor = comparison_events.get(
        alignment_key
    )

    if (
        current_anchor is None
        or comparison_anchor is None
    ):
        aligned_frame = current_frame

    else:
        relative_position = (
            current_frame
            - int(
                current_anchor
            )
        )

        aligned_frame = (
            int(
                comparison_anchor
            )
            + relative_position
        )

    return max(
        0,
        min(
            int(
                aligned_frame
            ),
            comparison_total_frames - 1,
        ),
    )


def current_trial_event_dictionary(
    feature_record: object,
) -> dict[str, int | None]:
    return {
        "motion_start": (
            feature_record
            .motion_start_frame
        ),
        "dip": (
            feature_record
            .dip_frame
        ),
        "takeoff": (
            feature_record
            .takeoff_frame
        ),
        "release": (
            feature_record
            .release_frame
        ),
        "ball_apex": (
            feature_record
            .ball_apex_frame
        ),
        "landing": (
            feature_record
            .landing_frame
        ),
    }


@st.fragment(
    run_every="100ms",
)
def render_side_by_side_comparison(
    result: dict,
    current_trial_data: dict,
    comparison_trial_data: dict,
    comparison_filename: str,
    shooting_side: str,
) -> None:
    """
    Render synchronized current and comparison shots side-by-side.
    """

    initialize_playback_state()

    current_feature_record = result[
        "feature_record"
    ]

    current_total_frames = len(
        current_trial_data.get(
            "tracking",
            [],
        )
    )

    comparison_total_frames = len(
        comparison_trial_data.get(
            "tracking",
            [],
        )
    )

    if (
        current_total_frames <= 0
        or comparison_total_frames <= 0
    ):
        st.warning(
            "Both trials must contain tracking frames for comparison."
        )

        return

    current_events = current_trial_event_dictionary(
        current_feature_record
    )

    comparison_events = cached_comparison_events(
        json.dumps(
            comparison_trial_data,
            allow_nan=True,
        ),
        comparison_filename,
        shooting_side,
    )

    alignment_mode = st.selectbox(
        "Align both shots at",
        options=[
            "Release",
            "Takeoff",
            "Dip",
            "Motion Start",
            "Ball Apex",
            "Landing",
        ],
        key="comparison_alignment_mode",
        help=(
            "Both animations are synchronized around the selected event. "
            "For example, Release places both release frames at the same "
            "relative point in playback."
        ),
    )

    comparison_frame = aligned_comparison_frame(
        current_frame=(
            st.session_state
            .playback_frame
        ),
        current_events=current_events,
        comparison_events=(
            comparison_events
        ),
        alignment_mode=alignment_mode,
        comparison_total_frames=(
            comparison_total_frames
        ),
    )

    current_column, comparison_column = st.columns(
        2,
        gap="large",
    )

    with current_column:
        st.markdown(
            "### Current shot"
        )

        st.caption(
            (
                f"{current_feature_record.participant_id} · "
                f"{current_feature_record.trial_id} · "
                f"Frame {st.session_state.playback_frame}"
            )
        )

        current_figure = create_playback_figure(
            trial_data=current_trial_data,
            frame_index=(
                st.session_state
                .playback_frame
            ),
            view_name=(
                st.session_state
                .playback_view
            ),
            show_trajectory=(
                st.session_state
                .playback_show_trajectory
            ),
        )

        st.pyplot(
            current_figure,
            clear_figure=True,
            use_container_width=True,
        )

    with comparison_column:
        comparison_participant = (
            comparison_trial_data.get(
                "participant_id",
                "Unknown",
            )
        )

        comparison_trial = (
            comparison_trial_data.get(
                "trial_id",
                Path(
                    comparison_filename
                ).stem,
            )
        )

        st.markdown(
            "### Comparison shot"
        )

        st.caption(
            (
                f"{comparison_participant} · "
                f"{comparison_trial} · "
                f"Frame {comparison_frame}"
            )
        )

        comparison_figure = create_playback_figure(
            trial_data=(
                comparison_trial_data
            ),
            frame_index=(
                comparison_frame
            ),
            view_name=(
                st.session_state
                .playback_view
            ),
            show_trajectory=(
                st.session_state
                .playback_show_trajectory
            ),
        )

        st.pyplot(
            comparison_figure,
            clear_figure=True,
            use_container_width=True,
        )

    control_columns = st.columns(
        [
            0.18,
            0.18,
            0.18,
            0.18,
            0.28,
        ]
    )

    if control_columns[0].button(
        "◀ 10",
        key="comparison_back_10",
        use_container_width=True,
    ):
        set_playback_frame(
            st.session_state.playback_frame
            - 10,
            current_total_frames,
        )

        st.rerun(
            scope="fragment"
        )

    if control_columns[1].button(
        "◀ 1",
        key="comparison_back_1",
        use_container_width=True,
    ):
        set_playback_frame(
            st.session_state.playback_frame
            - 1,
            current_total_frames,
        )

        st.rerun(
            scope="fragment"
        )

    if control_columns[2].button(
        "1 ▶",
        key="comparison_forward_1",
        use_container_width=True,
    ):
        set_playback_frame(
            st.session_state.playback_frame
            + 1,
            current_total_frames,
        )

        st.rerun(
            scope="fragment"
        )

    if control_columns[3].button(
        "10 ▶",
        key="comparison_forward_10",
        use_container_width=True,
    ):
        set_playback_frame(
            st.session_state.playback_frame
            + 10,
            current_total_frames,
        )

        st.rerun(
            scope="fragment"
        )

    if control_columns[4].button(
        (
            "⏸ Pause Comparison"
            if st.session_state.playback_playing
            else "▶ Play Comparison"
        ),
        key="comparison_play_pause",
        type=(
            "secondary"
            if st.session_state.playback_playing
            else "primary"
        ),
        use_container_width=True,
    ):
        st.session_state.diagnosis_segment_active = (
            False
        )

        st.session_state.playback_playing = (
            not st.session_state.playback_playing
        )

        st.rerun(
            scope="fragment"
        )

    slider_frame = st.slider(
        "Comparison frame scrubber",
        min_value=0,
        max_value=current_total_frames - 1,
        value=int(
            st.session_state.playback_frame
        ),
        step=1,
        key=(
            f"comparison_slider_"
            f"{st.session_state.playback_frame}"
        ),
    )

    if (
        slider_frame
        != st.session_state.playback_frame
    ):
        st.session_state.playback_playing = (
            False
        )

        set_playback_frame(
            slider_frame,
            current_total_frames,
        )

        st.rerun(
            scope="fragment"
        )

    if st.session_state.playback_playing:
        speed_step_lookup = {
            "Slow": 1,
            "Normal": 3,
            "Fast": 6,
        }

        next_frame = (
            st.session_state.playback_frame
            + speed_step_lookup.get(
                st.session_state.playback_speed,
                3,
            )
        )

        if next_frame >= current_total_frames:
            if st.session_state.playback_loop:
                next_frame = 0
            else:
                next_frame = (
                    current_total_frames
                    - 1
                )

                st.session_state.playback_playing = (
                    False
                )

        set_playback_frame(
            next_frame,
            current_total_frames,
        )

    st.caption(
        (
            f"Alignment: {alignment_mode}. "
            "Both shots use the same camera view and playback controls. "
            "The comparison frame is adjusted so the selected events line up."
        )
    )


def start_diagnosis_segment(
    diagnosis: object,
    total_frames: int,
) -> None:
    """
    Start automatic playback for one diagnosis review segment.
    """

    start_frame = diagnosis.review_start_frame
    end_frame = diagnosis.review_end_frame

    if start_frame is None:
        start_frame = end_frame

    if end_frame is None:
        end_frame = start_frame

    if start_frame is None:
        start_frame = 0

    if end_frame is None:
        end_frame = total_frames - 1

    start_frame = max(
        0,
        min(
            int(start_frame),
            total_frames - 1,
        ),
    )

    end_frame = max(
        start_frame,
        min(
            int(end_frame),
            total_frames - 1,
        ),
    )

    st.session_state.diagnosis_segment_active = True
    st.session_state.diagnosis_segment_start = start_frame
    st.session_state.diagnosis_segment_end = end_frame
    st.session_state.diagnosis_segment_title = (
        diagnosis.diagnosis_title
    )
    st.session_state.diagnosis_segment_focus = (
        diagnosis.coaching_focus
    )

    set_playback_frame(
        start_frame,
        total_frames,
    )

    st.session_state.playback_playing = True


def stop_diagnosis_segment() -> None:
    """
    Stop diagnosis playback while preserving the current frame.
    """

    st.session_state.diagnosis_segment_active = False
    st.session_state.playback_playing = False


def diagnosis_overlay_html(
    current_frame: int,
) -> str:
    """
    Create an on-screen coaching overlay for diagnosis playback.
    """

    if not st.session_state.diagnosis_segment_active:
        return ""

    start_frame = st.session_state.diagnosis_segment_start
    end_frame = st.session_state.diagnosis_segment_end
    title = st.session_state.diagnosis_segment_title
    focus = st.session_state.diagnosis_segment_focus

    return f"""
    <div style="
        background:linear-gradient(135deg,rgba(17,27,46,0.96),rgba(31,53,91,0.94));
        border-radius:16px;
        padding:1rem 1.1rem;
        margin-bottom:0.8rem;
        color:white;
        box-shadow:0 12px 28px rgba(17,27,46,0.18);
    ">
        <div style="
            color:#ffb18e;
            font-size:0.72rem;
            font-weight:800;
            letter-spacing:0.12em;
            text-transform:uppercase;
            margin-bottom:0.35rem;
        ">
            Diagnosis playback
        </div>
        <div style="
            font-size:1.25rem;
            font-weight:800;
            margin-bottom:0.35rem;
        ">
            {title}
        </div>
        <div style="
            color:#d5deee;
            font-size:0.9rem;
            line-height:1.55;
        ">
            Frame {current_frame} · Review window {start_frame}–{end_frame}<br>
            Focus: {focus}
        </div>
    </div>
    """


@st.fragment(
    run_every="100ms",
)
def render_interactive_playback(
    result: dict,
    trial_data: dict,
    shooting_side: str,
) -> None:
    """
    Display synchronized manual and automatic playback.

    The fragment reruns independently every 100 ms. While Play is active,
    it advances the tracked frame without rerunning the entire analysis app.
    """

    initialize_playback_state()

    feature_record = result[
        "feature_record"
    ]

    timeline = result[
        "timeline"
    ]

    total_frames = len(
        trial_data.get(
            "tracking",
            [],
        )
    )

    if total_frames <= 0:
        st.warning(
            "No tracking frames were available for playback."
        )

        return

    speed_step_lookup = {
        "Slow": 1,
        "Normal": 3,
        "Fast": 6,
    }

    if (
        st.session_state.playback_playing
        and not st.session_state.diagnosis_segment_active
    ):
        next_frame = (
            st.session_state.playback_frame
            + speed_step_lookup.get(
                st.session_state.playback_speed,
                3,
            )
        )

        if next_frame >= total_frames:
            if st.session_state.playback_loop:
                next_frame = 0
            else:
                next_frame = (
                    total_frames
                    - 1
                )

                st.session_state.playback_playing = (
                    False
                )

        set_playback_frame(
            next_frame,
            total_frames,
        )

    event_lookup = event_frame_lookup(
        timeline
    )

    playback_left, playback_right = st.columns(
        [
            0.70,
            0.30,
        ],
        gap="large",
    )

    with playback_right:
        st.markdown(
            "### Playback controls"
        )

        play_columns = st.columns(
            2
        )

        if play_columns[0].button(
            (
                "⏸ Pause"
                if st.session_state.playback_playing
                else "▶ Play"
            ),
            type=(
                "primary"
                if not st.session_state.playback_playing
                else "secondary"
            ),
            use_container_width=True,
            key="playback_play_pause",
        ):
            st.session_state.playback_playing = (
                not st.session_state.playback_playing
            )

            st.rerun(
                scope="fragment"
            )

        if play_columns[1].button(
            "⏹ Stop",
            use_container_width=True,
            key="playback_stop",
        ):
            st.session_state.playback_playing = (
                False
            )

            set_playback_frame(
                0,
                total_frames,
            )

            st.rerun(
                scope="fragment"
            )

        st.selectbox(
            "Playback speed",
            options=[
                "Slow",
                "Normal",
                "Fast",
            ],
            key="playback_speed",
            help=(
                "Normal advances three source frames every 100 ms, "
                "which approximates the original 30 Hz tracking speed."
            ),
        )

        st.toggle(
            "Loop playback",
            key="playback_loop",
        )

        view_name = st.selectbox(
            "Camera view",
            options=[
                "45°",
                "Side",
                "Front",
                "Top",
            ],
            key="playback_view",
        )

        show_trajectory = st.toggle(
            "Show ball trajectory",
            key="playback_show_trajectory",
        )

        event_options = [
            "Choose an event"
        ] + list(
            event_lookup
        )

        selected_event = st.selectbox(
            "Jump to event",
            options=event_options,
            key="playback_event_select",
        )

        if (
            selected_event
            != "Choose an event"
            and st.button(
                "Jump to Selected Event",
                use_container_width=True,
                key="playback_jump_event",
            )
        ):
            st.session_state.playback_playing = (
                False
            )

            set_playback_frame(
                event_lookup[
                    selected_event
                ],
                total_frames,
            )

            st.rerun(
                scope="fragment"
            )

        first_row = st.columns(
            3
        )

        if first_row[0].button(
            "◀ 10",
            use_container_width=True,
            key="playback_back_10",
        ):
            st.session_state.playback_playing = (
                False
            )

            set_playback_frame(
                st.session_state.playback_frame
                - 10,
                total_frames,
            )

            st.rerun(
                scope="fragment"
            )

        if first_row[1].button(
            "◀ 1",
            use_container_width=True,
            key="playback_back_1",
        ):
            st.session_state.playback_playing = (
                False
            )

            set_playback_frame(
                st.session_state.playback_frame
                - 1,
                total_frames,
            )

            st.rerun(
                scope="fragment"
            )

        if first_row[2].button(
            "1 ▶",
            use_container_width=True,
            key="playback_forward_1",
        ):
            st.session_state.playback_playing = (
                False
            )

            set_playback_frame(
                st.session_state.playback_frame
                + 1,
                total_frames,
            )

            st.rerun(
                scope="fragment"
            )

        second_row = st.columns(
            3
        )

        if second_row[0].button(
            "10 ▶",
            use_container_width=True,
            key="playback_forward_10",
        ):
            st.session_state.playback_playing = (
                False
            )

            set_playback_frame(
                st.session_state.playback_frame
                + 10,
                total_frames,
            )

            st.rerun(
                scope="fragment"
            )

        if second_row[1].button(
            "Release",
            use_container_width=True,
            key="playback_release",
        ):
            release_frame = (
                feature_record
                .release_frame
            )

            if release_frame is not None:
                st.session_state.playback_playing = (
                    False
                )

                set_playback_frame(
                    release_frame,
                    total_frames,
                )

                st.rerun(
                    scope="fragment"
                )

        if second_row[2].button(
            "Reset",
            use_container_width=True,
            key="playback_reset",
        ):
            st.session_state.playback_playing = (
                False
            )

            set_playback_frame(
                0,
                total_frames,
            )

            st.rerun(
                scope="fragment"
            )

        slider_frame = st.slider(
            "Frame scrubber",
            min_value=0,
            max_value=(
                total_frames
                - 1
            ),
            value=int(
                st.session_state
                .playback_frame
            ),
            step=1,
            key=(
                f"playback_slider_"
                f"{st.session_state.playback_frame}"
            ),
        )

        if (
            slider_frame
            != st.session_state.playback_frame
        ):
            st.session_state.playback_playing = (
                False
            )

            st.session_state.diagnosis_segment_active = (
                False
            )

            set_playback_frame(
                slider_frame,
                total_frames,
            )

            st.rerun(
                scope="fragment"
            )

        sampling_rate = float(
            feature_record
            .sampling_rate
        )

        current_time_ms = (
            st.session_state
            .playback_frame
            / sampling_rate
            * 1000.0
        )

        status_columns = st.columns(
            2
        )

        status_columns[0].metric(
            "Current frame",
            st.session_state.playback_frame,
        )

        status_columns[1].metric(
            "Current time",
            f"{current_time_ms:.1f} ms",
        )

        current_events = [
            (
                event.event_type.value
                .replace(
                    "_",
                    " ",
                )
                .title()
            )
            for event in timeline.events
            if int(
                event.frame
            )
            == int(
                st.session_state
                .playback_frame
            )
        ]

        if current_events:
            st.info(
                "Events at this frame: "
                + ", ".join(
                    current_events
                )
            )

        playback_status = (
            "Playing"
            if st.session_state.playback_playing
            else "Paused"
        )

        st.caption(
            (
                f"Status: {playback_status} · "
                f"Speed: {st.session_state.playback_speed} · "
                f"Frame {st.session_state.playback_frame + 1} "
                f"of {total_frames}"
            )
        )

    with playback_left:
        overlay_html = diagnosis_overlay_html(
            st.session_state.playback_frame
        )

        if overlay_html:
            st.markdown(
                overlay_html,
                unsafe_allow_html=True,
            )

        figure = create_playback_figure(
            trial_data=trial_data,
            frame_index=(
                st.session_state
                .playback_frame
            ),
            view_name=view_name,
            show_trajectory=show_trajectory,
        )

        st.pyplot(
            figure,
            clear_figure=True,
            use_container_width=True,
        )

    st.caption(
        "Press Play for automatic animation. The scrubber, step controls, "
        "camera view, ball trajectory, and event jumps remain synchronized."
    )

    st.markdown(
        "### Live biomechanics"
    )

    render_live_signal_panel(
        result=result,
        trial_data=trial_data,
        shooting_side=shooting_side,
    )



@st.cache_data(
    show_spinner=False,
)
def load_session_history_for_participant(
    participant_id: str,
) -> tuple[
    dict,
    list[dict],
    list[dict],
]:
    """
    Run and cache the historical analysis for one participant.
    """

    analyzer = SessionHistoryAnalyzer()

    (
        summary,
        shot_rows,
        feature_summaries,
    ) = analyzer.run(
        participant_id=participant_id,
    )

    return (
        asdict(
            summary
        ),
        [
            asdict(
                row
            )
            for row in shot_rows
        ],
        [
            asdict(
                row
            )
            for row in feature_summaries
        ],
    )


def rolling_make_percentage(
    results: list[str],
    window: int,
) -> list[float]:
    """
    Calculate a rolling make percentage.
    """

    output: list[
        float
    ] = []

    numeric_results = [
        1.0
        if result == "made"
        else 0.0
        for result in results
    ]

    for index in range(
        len(
            numeric_results
        )
    ):
        start = max(
            0,
            index
            - window
            + 1,
        )

        local_values = numeric_results[
            start:
            index
            + 1
        ]

        output.append(
            float(
                mean(
                    local_values
                )
                * 100.0
            )
        )

    return output


def render_session_history_dashboard(
    participant_id: str,
) -> None:
    """
    Show a coach-facing practice history dashboard.
    """

    try:
        (
            summary,
            shot_rows,
            feature_summaries,
        ) = load_session_history_for_participant(
            participant_id
        )

    except Exception as error:
        st.warning(
            (
                "Practice history could not be loaded for this participant. "
                f"{type(error).__name__}: {error}"
            )
        )

        return

    st.subheader(
        "Practice History"
    )

    st.caption(
        (
            "This section summarizes all successfully analyzed historical "
            "shots for the detected participant."
        )
    )

    top_columns = st.columns(
        5
    )

    top_columns[0].metric(
        "Total shots",
        summary[
            "total_shots"
        ],
    )

    top_columns[1].metric(
        "Made",
        summary[
            "made_shots"
        ],
    )

    top_columns[2].metric(
        "Missed",
        summary[
            "missed_shots"
        ],
    )

    top_columns[3].metric(
        "Make percentage",
        (
            f"{summary['make_percentage']:.1f}%"
        ),
    )

    top_columns[4].metric(
        "History status",
        summary[
            "session_status"
        ],
    )

    highlight_left, highlight_middle, highlight_right = st.columns(
        3
    )

    with highlight_left:
        st.markdown(
            """
            <div class="bms-summary-card">
                <h3>Most consistent</h3>
                <p>{}</p>
            </div>
            """.format(
                summary[
                    "most_consistent_feature"
                ]
            ),
            unsafe_allow_html=True,
        )

    with highlight_middle:
        st.markdown(
            """
            <div class="bms-summary-card">
                <h3>Least consistent</h3>
                <p>{}</p>
            </div>
            """.format(
                summary[
                    "least_consistent_feature"
                ]
            ),
            unsafe_allow_html=True,
        )

    with highlight_right:
        st.markdown(
            """
            <div class="bms-summary-card">
                <h3>Largest trend change</h3>
                <p>{}</p>
            </div>
            """.format(
                summary[
                    "largest_negative_trend_feature"
                ]
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        "### Performance trend"
    )

    rolling_window = st.selectbox(
        "Rolling make-percentage window",
        options=[
            5,
            10,
            20,
        ],
        index=1,
        key=(
            f"history_window_"
            f"{participant_id}"
        ),
    )

    results = [
        row[
            "result"
        ]
        for row in shot_rows
    ]

    rolling_values = rolling_make_percentage(
        results,
        int(
            rolling_window
        ),
    )

    shot_numbers = [
        row[
            "shot_number"
        ]
        for row in shot_rows
    ]

    performance_figure, performance_axis = plt.subplots(
        figsize=(
            12,
            4.5,
        )
    )

    performance_axis.plot(
        shot_numbers,
        rolling_values,
        linewidth=2.0,
    )

    performance_axis.set_ylim(
        0,
        100,
    )

    performance_axis.set_xlabel(
        "Shot number"
    )

    performance_axis.set_ylabel(
        "Rolling make percentage"
    )

    performance_axis.set_title(
        (
            f"{participant_id} rolling make percentage "
            f"({rolling_window}-shot window)"
        )
    )

    performance_axis.grid(
        visible=True,
        alpha=0.25,
    )

    performance_figure.tight_layout()

    st.pyplot(
        performance_figure,
        clear_figure=True,
        use_container_width=True,
    )

    st.markdown(
        "### Feature trend explorer"
    )

    feature_label_lookup = {
        row[
            "feature"
        ]: SessionHistoryAnalyzer.FRIENDLY_NAMES.get(
            row[
                "feature"
            ],
            row[
                "feature"
            ],
        )
        for row in feature_summaries
    }

    feature_options = list(
        feature_label_lookup
    )

    selected_feature = st.selectbox(
        "Biomechanical feature",
        options=feature_options,
        format_func=lambda feature: (
            feature_label_lookup[
                feature
            ]
        ),
        key=(
            f"history_feature_"
            f"{participant_id}"
        ),
    )

    trend_values = [
        row.get(
            selected_feature
        )
        for row in shot_rows
    ]

    made_x: list[
        int
    ] = []

    made_y: list[
        float
    ] = []

    missed_x: list[
        int
    ] = []

    missed_y: list[
        float
    ] = []

    for row, value in zip(
        shot_rows,
        trend_values,
    ):
        if value is None:
            continue

        if row[
            "result"
        ] == "made":
            made_x.append(
                row[
                    "shot_number"
                ]
            )

            made_y.append(
                float(
                    value
                )
            )

        else:
            missed_x.append(
                row[
                    "shot_number"
                ]
            )

            missed_y.append(
                float(
                    value
                )
            )

    trend_figure, trend_axis = plt.subplots(
        figsize=(
            12,
            5.2,
        )
    )

    if made_x:
        trend_axis.scatter(
            made_x,
            made_y,
            label="Made",
            s=28,
        )

    if missed_x:
        trend_axis.scatter(
            missed_x,
            missed_y,
            label="Missed",
            s=28,
        )

    trend_axis.set_xlabel(
        "Shot number"
    )

    trend_axis.set_ylabel(
        feature_label_lookup[
            selected_feature
        ]
    )

    trend_axis.set_title(
        (
            f"{feature_label_lookup[selected_feature]} "
            "across shot history"
        )
    )

    trend_axis.grid(
        visible=True,
        alpha=0.25,
    )

    trend_axis.legend()

    trend_figure.tight_layout()

    st.pyplot(
        trend_figure,
        clear_figure=True,
        use_container_width=True,
    )

    selected_summary = next(
        row
        for row in feature_summaries
        if row[
            "feature"
        ] == selected_feature
    )

    detail_columns = st.columns(
        4
    )

    detail_columns[0].metric(
        "Consistency score",
        (
            f"{selected_summary['consistency_score']:.1f}/100"
        ),
    )

    detail_columns[1].metric(
        "First 10 average",
        (
            "N/A"
            if selected_summary[
                "first_10_mean"
            ]
            is None
            else f"{selected_summary['first_10_mean']:.2f}"
        ),
    )

    detail_columns[2].metric(
        "Last 10 average",
        (
            "N/A"
            if selected_summary[
                "last_10_mean"
            ]
            is None
            else f"{selected_summary['last_10_mean']:.2f}"
        ),
    )

    detail_columns[3].metric(
        "Trend",
        selected_summary[
            "trend_direction"
        ].title(),
    )

    with st.expander(
        "View all feature-history details",
        expanded=False,
    ):
        history_table = []

        for row in feature_summaries:
            history_table.append(
                {
                    "Feature": (
                        SessionHistoryAnalyzer
                        .FRIENDLY_NAMES
                        .get(
                            row[
                                "feature"
                            ],
                            row[
                                "feature"
                            ],
                        )
                    ),
                    "Consistency": round(
                        row[
                            "consistency_score"
                        ],
                        1,
                    ),
                    "First 10": (
                        row[
                            "first_10_mean"
                        ]
                    ),
                    "Last 10": (
                        row[
                            "last_10_mean"
                        ]
                    ),
                    "Change": (
                        row[
                            "first_to_last_change"
                        ]
                    ),
                    "Trend": (
                        row[
                            "trend_direction"
                        ]
                    ),
                    "Made average": (
                        row[
                            "made_mean"
                        ]
                    ),
                    "Missed average": (
                        row[
                            "missed_mean"
                        ]
                    ),
                }
            )

        st.dataframe(
            history_table,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        (
            "A numerical increase or decrease is not automatically an "
            "improvement or decline. Interpret trends using the player's "
            "successful baseline and shot video."
        )
    )



@st.cache_data(
    show_spinner=False,
)
def load_player_development_for_participant(
    participant_id: str,
) -> tuple[
    dict,
    list[dict],
]:
    """
    Run and cache the long-term player-development analysis.
    """

    engine = (
        PlayerDevelopmentIntelligence()
    )

    summary, priorities = engine.run(
        participant_id=participant_id,
    )

    return (
        asdict(
            summary
        ),
        [
            asdict(
                priority
            )
            for priority in priorities
        ],
    )


def development_score_label(
    score: float,
) -> str:
    if score >= 85:
        return "Very consistent"

    if score >= 70:
        return "Generally consistent"

    if score >= 55:
        return "Developing consistency"

    return "High review priority"


def render_player_development_dashboard(
    participant_id: str,
) -> None:
    """
    Show long-term player-development scores and priorities.
    """

    try:
        (
            summary,
            priorities,
        ) = load_player_development_for_participant(
            participant_id
        )

    except Exception as error:
        st.warning(
            (
                "Player Development Intelligence could not be loaded. "
                f"{type(error).__name__}: {error}"
            )
        )

        return

    st.subheader(
        "Player Development Intelligence"
    )

    st.caption(
        (
            "This section analyzes the participant across all successfully "
            "processed historical shots. It identifies long-term consistency "
            "patterns and development priorities rather than diagnosing only "
            "the currently uploaded attempt."
        )
    )

    headline_columns = st.columns(
        4
    )

    headline_columns[0].metric(
        "Tracked shots",
        summary[
            "total_shots"
        ],
    )

    headline_columns[1].metric(
        "Make percentage",
        (
            f"{summary['make_percentage']:.1f}%"
        ),
    )

    headline_columns[2].metric(
        "Overall mechanics",
        (
            f"{summary['overall_mechanics_score']:.1f}/100"
        ),
    )

    headline_columns[3].metric(
        "Evidence confidence",
        summary[
            "evidence_confidence"
        ].title(),
    )

    st.markdown(
        "### Development scorecard"
    )

    score_items = [
        (
            "Overall consistency",
            summary[
                "consistency_score"
            ],
        ),
        (
            "Timing & coordination",
            summary[
                "timing_score"
            ],
        ),
        (
            "Lower body",
            summary[
                "lower_body_score"
            ],
        ),
        (
            "Upper body",
            summary[
                "upper_body_score"
            ],
        ),
        (
            "Ball & release",
            summary[
                "ball_release_score"
            ],
        ),
    ]

    score_columns = st.columns(
        len(
            score_items
        )
    )

    for column, (
        label,
        score,
    ) in zip(
        score_columns,
        score_items,
    ):
        with column:
            st.markdown(
                f"""
                <div class="bms-category-card">
                    <div class="label">{label}</div>
                    <div class="score">{score:.1f}</div>
                    <div class="status">
                        {development_score_label(float(score))}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        "### Development overview"
    )

    overview_left, overview_middle, overview_right = st.columns(
        3
    )

    with overview_left:
        st.markdown(
            f"""
            <div class="bms-summary-card">
                <h3>Strongest area</h3>
                <p>{summary['strongest_area']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with overview_middle:
        st.markdown(
            f"""
            <div class="bms-summary-card">
                <h3>Primary priority</h3>
                <p>{summary['primary_priority']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with overview_right:
        st.markdown(
            f"""
            <div class="bms-summary-card">
                <h3>Development status</h3>
                <p>{summary['development_status']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "### Recommended development priorities"
    )

    if not priorities:
        st.info(
            "No long-term development priorities were available."
        )

        return

    for priority in priorities:
        with st.container(
            border=True
        ):
            title_column, score_column = st.columns(
                [
                    0.78,
                    0.22,
                ]
            )

            with title_column:
                st.markdown(
                    (
                        f"### {priority['rank']}. "
                        f"{priority['display_name']}"
                    )
                )

                st.caption(
                    (
                        f"{priority['category']} · "
                        f"{priority['pattern_group']} · "
                        f"Priority: "
                        f"{priority['priority_level'].title()}"
                    )
                )

            with score_column:
                st.metric(
                    "Development score",
                    (
                        f"{priority['development_score']:.1f}/100"
                    ),
                )

            st.write(
                priority[
                    "diagnosis"
                ]
            )

            why_column, focus_column = st.columns(
                2
            )

            with why_column:
                st.markdown(
                    "**Why it may matter**"
                )

                st.write(
                    priority[
                        "why_it_matters"
                    ]
                )

            with focus_column:
                st.markdown(
                    "**Development focus**"
                )

                st.write(
                    priority[
                        "development_focus"
                    ]
                )

            st.markdown(
                "**Suggested drill**"
            )

            st.write(
                priority[
                    "suggested_drill"
                ]
            )

            evidence_columns = st.columns(
                3
            )

            evidence_columns[0].metric(
                "Consistency",
                (
                    f"{priority['consistency_score']:.1f}/100"
                ),
            )

            evidence_columns[1].metric(
                "First-to-last change",
                (
                    "N/A"
                    if priority[
                        "first_to_last_change"
                    ]
                    is None
                    else (
                        f"{priority['first_to_last_change']:+.2f}"
                    )
                ),
            )

            evidence_columns[2].metric(
                "Made–missed difference",
                (
                    "N/A"
                    if priority[
                        "made_missed_difference"
                    ]
                    is None
                    else (
                        f"{priority['made_missed_difference']:+.2f}"
                    )
                ),
            )

    st.caption(
        (
            "These priorities are long-term review signals. A numerical "
            "increase or decrease is not automatically better or worse. "
            "Use the player's successful-shot baseline and synchronized "
            "animation before making a technical change."
        )
    )



def get_player_history_repository() -> PlayerHistoryRepository:
    """
    Create a repository using the active local or visitor workspace.
    """

    data_root = active_data_root()

    database_file = (
        data_root
        / "player_history.sqlite3"
    )

    storage_folder = (
        data_root
        / "player_history_files"
    )

    data_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    storage_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    return PlayerHistoryRepository(
        database_file=database_file,
        storage_folder=storage_folder,
    )


def player_option_label(
    player: object,
) -> str:
    return (
        f"{player.display_name} "
        f"({player.participant_id})"
    )


def shot_option_label(
    shot: object,
) -> str:
    status = (
        "Active"
        if shot.active
        else "Removed"
    )

    return (
        f"{shot.trial_id} · "
        f"{shot.recorded_result.title()} · "
        f"{shot.original_filename} · "
        f"{status}"
    )


def render_player_history_manager() -> None:
    """
    Manage players and their JSON shot histories.

    This interface supports:

    - creating separate player profiles;
    - bulk uploading many JSON files;
    - enforcing participant-to-player matching;
    - adding more files later;
    - removing files without permanently deleting them;
    - restoring removed files;
    - renaming player profiles;
    - deactivating and reactivating players.
    """

    repository = get_player_history_repository()

    with st.expander(
        "Manage Player History Library",
        expanded=False,
    ):
        st.caption(
            (
                "Use this library to keep every player's files separate. "
                "A JSON file is linked to the player whose participant_id "
                "matches the file. Bulk uploads are supported."
            )
        )

        create_tab, upload_tab, manage_tab = st.tabs(
            [
                "Players",
                "Bulk Upload",
                "Manage Files",
            ]
        )

        # =================================================
        # PLAYERS
        # =================================================

        with create_tab:
            st.markdown(
                "### Create a player profile"
            )

            with st.form(
                "create_player_form",
                clear_on_submit=True,
            ):
                participant_id = st.text_input(
                    "Participant ID",
                    placeholder="Example: P0006",
                )

                display_name = st.text_input(
                    "Display name",
                    placeholder="Example: Jordan Smith",
                )

                create_clicked = st.form_submit_button(
                    "Create Player",
                    type="primary",
                    use_container_width=True,
                )

            if create_clicked:
                try:
                    player = repository.create_or_get_player(
                        participant_id=(
                            participant_id
                        ),
                        display_name=(
                            display_name
                            or participant_id
                        ),
                    )

                    st.success(
                        (
                            f"Player ready: "
                            f"{player.display_name} "
                            f"({player.participant_id})"
                        )
                    )

                    st.cache_data.clear()

                except Exception as error:
                    st.error(
                        (
                            "Could not create the player. "
                            f"{type(error).__name__}: {error}"
                        )
                    )

            st.markdown(
                "### Existing players"
            )

            include_inactive_players = st.toggle(
                "Show inactive players",
                value=False,
                key="history_show_inactive_players",
            )

            players = repository.list_players(
                include_inactive=(
                    include_inactive_players
                )
            )

            if not players:
                st.info(
                    "No player profiles have been created yet."
                )

            else:
                player_rows = [
                    {
                        "Player ID": player.player_id,
                        "Participant ID": (
                            player.participant_id
                        ),
                        "Display Name": (
                            player.display_name
                        ),
                        "Active": player.active,
                        "Created": player.created_at,
                    }
                    for player in players
                ]

                st.dataframe(
                    player_rows,
                    use_container_width=True,
                    hide_index=True,
                )

                selected_player = st.selectbox(
                    "Select a player to edit",
                    options=players,
                    format_func=player_option_label,
                    key="history_player_edit_select",
                )

                edit_columns = st.columns(
                    [
                        0.56,
                        0.22,
                        0.22,
                    ]
                )

                new_display_name = edit_columns[0].text_input(
                    "New display name",
                    value=(
                        selected_player
                        .display_name
                    ),
                    key=(
                        f"history_rename_"
                        f"{selected_player.player_id}"
                    ),
                )

                if edit_columns[1].button(
                    "Rename",
                    use_container_width=True,
                    key=(
                        f"history_rename_button_"
                        f"{selected_player.player_id}"
                    ),
                ):
                    try:
                        repository.rename_player(
                            player_id=(
                                selected_player
                                .player_id
                            ),
                            new_display_name=(
                                new_display_name
                            ),
                        )

                        st.success(
                            "Player name updated."
                        )

                        st.rerun()

                    except Exception as error:
                        st.error(
                            (
                                "Could not rename player. "
                                f"{type(error).__name__}: {error}"
                            )
                        )

                action_label = (
                    "Deactivate"
                    if selected_player.active
                    else "Reactivate"
                )

                if edit_columns[2].button(
                    action_label,
                    use_container_width=True,
                    key=(
                        f"history_active_button_"
                        f"{selected_player.player_id}"
                    ),
                ):
                    try:
                        repository.set_player_active(
                            player_id=(
                                selected_player
                                .player_id
                            ),
                            active=(
                                not selected_player
                                .active
                            ),
                        )

                        st.success(
                            (
                                "Player reactivated."
                                if not selected_player.active
                                else "Player deactivated."
                            )
                        )

                        st.rerun()

                    except Exception as error:
                        st.error(
                            (
                                "Could not update player status. "
                                f"{type(error).__name__}: {error}"
                            )
                        )

        # =================================================
        # BULK UPLOAD
        # =================================================

        with upload_tab:
            active_players = repository.list_players(
                include_inactive=False
            )

            if not active_players:
                st.info(
                    (
                        "Create at least one player profile before "
                        "uploading files into a selected profile."
                    )
                )

            upload_mode = st.radio(
                "How should uploaded files be assigned?",
                options=[
                    "Selected player only",
                    "Automatically by participant_id",
                ],
                index=0,
                key="history_upload_mode",
                help=(
                    "Selected player only is safest. Every JSON must match "
                    "the selected player's participant_id. Automatic mode "
                    "creates or uses the correct player based on each JSON."
                ),
            )

            selected_upload_player = None

            if (
                upload_mode
                == "Selected player only"
                and active_players
            ):
                selected_upload_player = st.selectbox(
                    "Upload files for",
                    options=active_players,
                    format_func=player_option_label,
                    key="history_upload_player_select",
                )

                st.info(
                    (
                        "Protection is enabled: files whose participant_id "
                        f"does not equal "
                        f"{selected_upload_player.participant_id} "
                        "will be rejected."
                    )
                )

            st.info(
                (
                    "Upload one or more JSON files containing structured 3D "
                    "basketball free-throw tracking data. Each file should "
                    "represent one attempt and include a non-empty tracking "
                    "list, ball XYZ coordinates, player body keypoints, "
                    "participant ID, trial ID, and made/missed result. MP4, "
                    "CSV, Excel, box-score JSON, play-by-play JSON, shot-chart "
                    "files, and unrelated JSON are not compatible."
                )
            )

            uploaded_history_files = st.file_uploader(
                "Upload one or many shot JSON files",
                type=[
                    "json",
                ],
                accept_multiple_files=True,
                key="history_bulk_json_upload",
            )

            if uploaded_history_files:
                st.write(
                    (
                        f"Files selected: "
                        f"**{len(uploaded_history_files)}**"
                    )
                )

                preview_rows = []

                for uploaded in uploaded_history_files:
                    try:
                        parsed = json.loads(
                            uploaded.getvalue()
                        )

                        preview_rows.append(
                            {
                                "Filename": uploaded.name,
                                "Participant": parsed.get(
                                    "participant_id",
                                    "Unknown",
                                ),
                                "Trial": parsed.get(
                                    "trial_id",
                                    Path(
                                        uploaded.name
                                    ).stem,
                                ),
                                "Result": parsed.get(
                                    "result",
                                    "Unknown",
                                ),
                                "Frames": len(
                                    parsed.get(
                                        "tracking",
                                        [],
                                    )
                                ),
                                "Validation": "Ready",
                            }
                        )

                    except Exception as error:
                        preview_rows.append(
                            {
                                "Filename": uploaded.name,
                                "Participant": "Unknown",
                                "Trial": "Unknown",
                                "Result": "Unknown",
                                "Frames": 0,
                                "Validation": (
                                    f"Invalid: {error}"
                                ),
                            }
                        )

                st.dataframe(
                    preview_rows,
                    use_container_width=True,
                    hide_index=True,
                )

            import_clicked = st.button(
                "Import Selected Files into History",
                type="primary",
                use_container_width=True,
                disabled=(
                    not uploaded_history_files
                    or (
                        upload_mode
                        == "Selected player only"
                        and selected_upload_player
                        is None
                    )
                ),
                key="history_import_files_button",
            )

            if import_clicked:
                file_payloads = [
                    (
                        uploaded.name,
                        uploaded.getvalue(),
                    )
                    for uploaded in uploaded_history_files
                ]

                try:
                    import_result = (
                        repository.import_json_files(
                            json_files=file_payloads,
                            selected_player_id=(
                                None
                                if (
                                    upload_mode
                                    == "Automatically by participant_id"
                                )
                                else (
                                    selected_upload_player
                                    .player_id
                                )
                            ),
                            enforce_selected_player=(
                                upload_mode
                                == "Selected player only"
                            ),
                        )
                    )

                    result_columns = st.columns(
                        4
                    )

                    result_columns[0].metric(
                        "Files selected",
                        import_result.total_files,
                    )

                    result_columns[1].metric(
                        "Imported",
                        import_result.imported_files,
                    )

                    result_columns[2].metric(
                        "Duplicates skipped",
                        import_result.skipped_duplicates,
                    )

                    result_columns[3].metric(
                        "Failed",
                        import_result.failed_files,
                    )

                    if import_result.imported:
                        st.success(
                            (
                                "Imported: "
                                + ", ".join(
                                    import_result.imported
                                )
                            )
                        )

                    if import_result.duplicates:
                        st.warning(
                            (
                                "Duplicates skipped: "
                                + ", ".join(
                                    import_result.duplicates
                                )
                            )
                        )

                    if import_result.failures:
                        st.error(
                            "Some files were rejected:"
                        )

                        for failure in import_result.failures:
                            st.write(
                                f"- {failure}"
                            )

                    st.cache_data.clear()

                except Exception as error:
                    st.error(
                        (
                            "Bulk import could not be completed. "
                            f"{type(error).__name__}: {error}"
                        )
                    )

        # =================================================
        # MANAGE FILES
        # =================================================

        with manage_tab:
            all_players = repository.list_players(
                include_inactive=True
            )

            if not all_players:
                st.info(
                    "No player profiles are available."
                )

                return

            selected_manage_player = st.selectbox(
                "Player history to manage",
                options=all_players,
                format_func=player_option_label,
                key="history_manage_player_select",
            )

            include_removed_files = st.toggle(
                "Show removed files",
                value=False,
                key="history_show_removed_files",
            )

            shots = repository.list_shot_files(
                player_id=(
                    selected_manage_player
                    .player_id
                ),
                include_inactive=(
                    include_removed_files
                ),
            )

            active_count = sum(
                shot.active
                for shot in shots
            )

            removed_count = sum(
                not shot.active
                for shot in shots
            )

            count_columns = st.columns(
                3
            )

            count_columns[0].metric(
                "Files shown",
                len(
                    shots
                ),
            )

            count_columns[1].metric(
                "Active",
                active_count,
            )

            count_columns[2].metric(
                "Removed",
                removed_count,
            )

            if not shots:
                st.info(
                    (
                        "This player has no files in the selected "
                        "history view."
                    )
                )

                return

            history_rows = [
                {
                    "Shot File ID": (
                        shot.shot_file_id
                    ),
                    "Trial": (
                        shot.trial_id
                    ),
                    "Result": (
                        shot.recorded_result
                    ),
                    "Filename": (
                        shot.original_filename
                    ),
                    "Frames": (
                        shot.total_frames
                    ),
                    "Sampling Rate": (
                        shot.sampling_rate
                    ),
                    "Imported": (
                        shot.imported_at
                    ),
                    "Active": (
                        shot.active
                    ),
                }
                for shot in shots
            ]

            st.dataframe(
                history_rows,
                use_container_width=True,
                hide_index=True,
            )

            active_shots = [
                shot
                for shot in shots
                if shot.active
            ]

            removed_shots = [
                shot
                for shot in shots
                if not shot.active
            ]

            if active_shots:
                selected_remove_shots = st.multiselect(
                    "Select active files to remove from history",
                    options=active_shots,
                    format_func=shot_option_label,
                    key="history_remove_shots_select",
                )

                remove_columns = st.columns(
                    2
                )

                if remove_columns[0].button(
                    "Remove from History",
                    use_container_width=True,
                    disabled=(
                        not selected_remove_shots
                    ),
                    key="history_soft_delete_button",
                ):
                    try:
                        for shot in selected_remove_shots:
                            repository.remove_shot_file(
                                shot_file_id=(
                                    shot.shot_file_id
                                ),
                                delete_physical_file=False,
                            )

                        st.success(
                            (
                                f"Removed "
                                f"{len(selected_remove_shots)} "
                                "file(s) from active history."
                            )
                        )

                        st.cache_data.clear()
                        st.rerun()

                    except Exception as error:
                        st.error(
                            (
                                "Could not remove the selected files. "
                                f"{type(error).__name__}: {error}"
                            )
                        )

                permanently_delete = remove_columns[1].toggle(
                    "Allow permanent deletion",
                    value=False,
                    key="history_permanent_delete_toggle",
                    help=(
                        "Permanent deletion removes both the database record "
                        "and stored JSON file. It cannot be undone."
                    ),
                )

                if permanently_delete:
                    confirmation_text = st.text_input(
                        "Type DELETE to confirm permanent deletion",
                        key="history_delete_confirmation",
                    )

                    if st.button(
                        "Permanently Delete Selected Files",
                        use_container_width=True,
                        disabled=(
                            not selected_remove_shots
                            or confirmation_text
                            != "DELETE"
                        ),
                        key="history_permanent_delete_button",
                    ):
                        try:
                            for shot in selected_remove_shots:
                                repository.remove_shot_file(
                                    shot_file_id=(
                                        shot.shot_file_id
                                    ),
                                    delete_physical_file=True,
                                )

                            st.success(
                                (
                                    f"Permanently deleted "
                                    f"{len(selected_remove_shots)} "
                                    "file(s)."
                                )
                            )

                            st.cache_data.clear()
                            st.rerun()

                        except Exception as error:
                            st.error(
                                (
                                    "Could not permanently delete files. "
                                    f"{type(error).__name__}: {error}"
                                )
                            )

            if removed_shots:
                selected_restore_shots = st.multiselect(
                    "Select removed files to restore",
                    options=removed_shots,
                    format_func=shot_option_label,
                    key="history_restore_shots_select",
                )

                if st.button(
                    "Restore Selected Files",
                    use_container_width=True,
                    disabled=(
                        not selected_restore_shots
                    ),
                    key="history_restore_button",
                ):
                    try:
                        for shot in selected_restore_shots:
                            repository.restore_shot_file(
                                shot_file_id=(
                                    shot.shot_file_id
                                )
                            )

                        st.success(
                            (
                                f"Restored "
                                f"{len(selected_restore_shots)} "
                                "file(s)."
                            )
                        )

                        st.cache_data.clear()
                        st.rerun()

                    except Exception as error:
                        st.error(
                            (
                                "Could not restore files. "
                                f"{type(error).__name__}: {error}"
                            )
                        )



@st.cache_resource
def get_practice_session_repository() -> PracticeSessionRepository:
    """
    Create one persistent practice-session repository for the app.
    """

    history_repository = get_player_history_repository()

    return PracticeSessionRepository(
        database_file=history_repository.database_file
    )


def session_option_label(
    session: object,
) -> str:
    status = (
        "Active"
        if session.active
        else "Inactive"
    )

    return (
        f"{session.session_date} · "
        f"{session.session_name} · "
        f"{session.shot_count} shots · "
        f"{status}"
    )


def session_shot_option_label(
    shot: object,
) -> str:
    return (
        f"{shot.trial_id} · "
        f"{shot.recorded_result.title()} · "
        f"{shot.original_filename}"
    )


def render_practice_session_manager() -> None:
    """
    Create, edit, and organize player practice sessions.
    """

    history_repository = (
        get_player_history_repository()
    )

    session_repository = (
        get_practice_session_repository()
    )

    with st.expander(
        "Manage Practice Sessions",
        expanded=False,
    ):
        st.caption(
            (
                "Group a player's tracked shots into named practices, "
                "workouts, tests, or date-based sessions. Moving a shot "
                "between sessions does not delete the underlying JSON file."
            )
        )

        create_tab, assign_tab, manage_tab = st.tabs(
            [
                "Create Session",
                "Assign Shots",
                "Manage Sessions",
            ]
        )

        all_players = history_repository.list_players(
            include_inactive=False
        )

        if not all_players:
            st.info(
                "Create at least one active player profile first."
            )

            return

        # =================================================
        # CREATE SESSION
        # =================================================

        with create_tab:
            selected_player = st.selectbox(
                "Player",
                options=all_players,
                format_func=player_option_label,
                key="session_create_player",
            )

            with st.form(
                "create_practice_session_form",
                clear_on_submit=True,
            ):
                session_name = st.text_input(
                    "Session name",
                    placeholder="Example: July 27 Free Throws",
                )

                session_date = st.date_input(
                    "Session date",
                    key="session_create_date",
                )

                session_type = st.selectbox(
                    "Session type",
                    options=[
                        "Free Throws",
                        "Shooting Workout",
                        "Fatigue Test",
                        "Baseline Test",
                        "Pre-Practice",
                        "Post-Practice",
                        "Other",
                    ],
                )

                location = st.text_input(
                    "Location",
                    placeholder="Example: Practice Gym",
                )

                notes = st.text_area(
                    "Notes",
                    placeholder=(
                        "Optional context such as drill focus, "
                        "fatigue level, coaching cue, or session goal."
                    ),
                )

                create_session_clicked = (
                    st.form_submit_button(
                        "Create Practice Session",
                        type="primary",
                        use_container_width=True,
                    )
                )

            if create_session_clicked:
                try:
                    session = (
                        session_repository
                        .create_session(
                            player_id=(
                                selected_player.player_id
                            ),
                            session_name=session_name,
                            session_date=session_date,
                            session_type=session_type,
                            location=location,
                            notes=notes,
                        )
                    )

                    st.success(
                        (
                            f"Created session: "
                            f"{session.session_name} "
                            f"for {session.player_display_name}."
                        )
                    )

                    st.rerun()

                except Exception as error:
                    st.error(
                        (
                            "Could not create session. "
                            f"{type(error).__name__}: {error}"
                        )
                    )

            sessions = (
                session_repository
                .list_sessions(
                    player_id=selected_player.player_id,
                    include_inactive=False,
                )
            )

            if sessions:
                st.markdown(
                    "### Existing sessions"
                )

                st.dataframe(
                    [
                        {
                            "Session": session.session_name,
                            "Date": session.session_date,
                            "Type": session.session_type,
                            "Location": session.location,
                            "Shots": session.shot_count,
                            "Notes": session.notes,
                        }
                        for session in sessions
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

        # =================================================
        # ASSIGN SHOTS
        # =================================================

        with assign_tab:
            assignment_player = st.selectbox(
                "Player",
                options=all_players,
                format_func=player_option_label,
                key="session_assign_player",
            )

            active_sessions = (
                session_repository
                .list_sessions(
                    player_id=assignment_player.player_id,
                    include_inactive=False,
                )
            )

            if not active_sessions:
                st.info(
                    "This player has no active practice sessions yet."
                )

            else:
                selected_session = st.selectbox(
                    "Session",
                    options=active_sessions,
                    format_func=session_option_label,
                    key="session_assign_session",
                )

                player_shots = (
                    history_repository
                    .list_shot_files(
                        player_id=assignment_player.player_id,
                        include_inactive=False,
                    )
                )

                current_session_shots = (
                    session_repository
                    .list_session_shots(
                        selected_session.session_id
                    )
                )

                assigned_ids = {
                    shot.shot_file_id
                    for shot in current_session_shots
                }

                available_shots = [
                    shot
                    for shot in player_shots
                    if shot.shot_file_id
                    not in assigned_ids
                ]

                metric_columns = st.columns(
                    3
                )

                metric_columns[0].metric(
                    "Shots in player history",
                    len(player_shots),
                )

                metric_columns[1].metric(
                    "Shots already in session",
                    len(current_session_shots),
                )

                metric_columns[2].metric(
                    "Available to assign",
                    len(available_shots),
                )

                if available_shots:
                    selected_shots = st.multiselect(
                        "Select shots to add or move into this session",
                        options=available_shots,
                        format_func=shot_option_label,
                        key="session_assign_shots",
                        help=(
                            "If a selected shot already belongs to another "
                            "session, it will be moved into this one."
                        ),
                    )

                    if st.button(
                        "Assign Selected Shots",
                        type="primary",
                        use_container_width=True,
                        disabled=not selected_shots,
                        key="session_assign_button",
                    ):
                        try:
                            assigned_count = (
                                session_repository
                                .assign_shots(
                                    session_id=(
                                        selected_session.session_id
                                    ),
                                    shot_file_ids=[
                                        shot.shot_file_id
                                        for shot in selected_shots
                                    ],
                                )
                            )

                            st.success(
                                (
                                    f"Assigned {assigned_count} shot(s) "
                                    f"to {selected_session.session_name}."
                                )
                            )

                            st.cache_data.clear()
                            st.rerun()

                        except Exception as error:
                            st.error(
                                (
                                    "Could not assign shots. "
                                    f"{type(error).__name__}: {error}"
                                )
                            )

                else:
                    st.info(
                        "No additional active shots are available to assign."
                    )

                if current_session_shots:
                    st.markdown(
                        "### Current session shots"
                    )

                    st.dataframe(
                        [
                            {
                                "Order": shot.display_order,
                                "Trial": shot.trial_id,
                                "Result": shot.recorded_result,
                                "Filename": shot.original_filename,
                                "Frames": shot.total_frames,
                            }
                            for shot in current_session_shots
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

                    selected_remove = st.multiselect(
                        "Select shots to remove from this session",
                        options=current_session_shots,
                        format_func=session_shot_option_label,
                        key="session_remove_shots",
                    )

                    if st.button(
                        "Remove Selected from Session",
                        use_container_width=True,
                        disabled=not selected_remove,
                        key="session_remove_button",
                    ):
                        try:
                            removed_count = (
                                session_repository
                                .remove_shots(
                                    session_id=(
                                        selected_session.session_id
                                    ),
                                    shot_file_ids=[
                                        shot.shot_file_id
                                        for shot in selected_remove
                                    ],
                                )
                            )

                            st.success(
                                (
                                    f"Removed {removed_count} shot(s) "
                                    "from the session. The JSON files remain "
                                    "in the player's history."
                                )
                            )

                            st.cache_data.clear()
                            st.rerun()

                        except Exception as error:
                            st.error(
                                (
                                    "Could not remove shots from session. "
                                    f"{type(error).__name__}: {error}"
                                )
                            )

        # =================================================
        # MANAGE SESSIONS
        # =================================================

        with manage_tab:
            manage_player = st.selectbox(
                "Player",
                options=all_players,
                format_func=player_option_label,
                key="session_manage_player",
            )

            show_inactive_sessions = st.toggle(
                "Show inactive sessions",
                value=False,
                key="session_show_inactive",
            )

            sessions = (
                session_repository
                .list_sessions(
                    player_id=manage_player.player_id,
                    include_inactive=show_inactive_sessions,
                )
            )

            if not sessions:
                st.info(
                    "This player has no sessions in the selected view."
                )

                return

            st.dataframe(
                [
                    {
                        "Session ID": session.session_id,
                        "Date": session.session_date,
                        "Session": session.session_name,
                        "Type": session.session_type,
                        "Location": session.location,
                        "Shots": session.shot_count,
                        "Active": session.active,
                    }
                    for session in sessions
                ],
                use_container_width=True,
                hide_index=True,
            )

            selected_session = st.selectbox(
                "Select a session to edit",
                options=sessions,
                format_func=session_option_label,
                key="session_manage_select",
            )

            edit_left, edit_right = st.columns(
                2
            )

            new_name = edit_left.text_input(
                "Session name",
                value=selected_session.session_name,
                key=(
                    f"session_name_"
                    f"{selected_session.session_id}"
                ),
            )

            new_date = edit_right.date_input(
                "Session date",
                value=date.fromisoformat(
                    selected_session.session_date
                ),
                key=(
                    f"session_date_"
                    f"{selected_session.session_id}"
                ),
            )

            new_type = st.selectbox(
                "Session type",
                options=[
                    "Free Throws",
                    "Shooting Workout",
                    "Fatigue Test",
                    "Baseline Test",
                    "Pre-Practice",
                    "Post-Practice",
                    "Other",
                ],
                index=(
                    [
                        "Free Throws",
                        "Shooting Workout",
                        "Fatigue Test",
                        "Baseline Test",
                        "Pre-Practice",
                        "Post-Practice",
                        "Other",
                    ].index(
                        selected_session.session_type
                    )
                    if selected_session.session_type
                    in [
                        "Free Throws",
                        "Shooting Workout",
                        "Fatigue Test",
                        "Baseline Test",
                        "Pre-Practice",
                        "Post-Practice",
                        "Other",
                    ]
                    else 0
                ),
                key=(
                    f"session_type_"
                    f"{selected_session.session_id}"
                ),
            )

            new_location = st.text_input(
                "Location",
                value=selected_session.location,
                key=(
                    f"session_location_"
                    f"{selected_session.session_id}"
                ),
            )

            new_notes = st.text_area(
                "Notes",
                value=selected_session.notes,
                key=(
                    f"session_notes_"
                    f"{selected_session.session_id}"
                ),
            )

            update_columns = st.columns(
                3
            )

            if update_columns[0].button(
                "Save Session Changes",
                type="primary",
                use_container_width=True,
                key=(
                    f"session_save_"
                    f"{selected_session.session_id}"
                ),
            ):
                try:
                    session_repository.update_session(
                        session_id=selected_session.session_id,
                        session_name=new_name,
                        session_date=(
                            new_date
                            if new_date is not None
                            else selected_session.session_date
                        ),
                        session_type=new_type,
                        location=new_location,
                        notes=new_notes,
                    )

                    st.success(
                        "Session updated."
                    )

                    st.rerun()

                except Exception as error:
                    st.error(
                        (
                            "Could not update session. "
                            f"{type(error).__name__}: {error}"
                        )
                    )

            action_label = (
                "Deactivate Session"
                if selected_session.active
                else "Reactivate Session"
            )

            if update_columns[1].button(
                action_label,
                use_container_width=True,
                key=(
                    f"session_active_"
                    f"{selected_session.session_id}"
                ),
            ):
                try:
                    session_repository.set_session_active(
                        session_id=selected_session.session_id,
                        active=(
                            not selected_session.active
                        ),
                    )

                    st.success(
                        (
                            "Session reactivated."
                            if not selected_session.active
                            else "Session deactivated."
                        )
                    )

                    st.rerun()

                except Exception as error:
                    st.error(
                        (
                            "Could not change session status. "
                            f"{type(error).__name__}: {error}"
                        )
                    )

            allow_delete = update_columns[2].toggle(
                "Allow deletion",
                value=False,
                key=(
                    f"session_delete_toggle_"
                    f"{selected_session.session_id}"
                ),
                help=(
                    "Deleting a session removes only the session container. "
                    "Its JSON files remain in player history."
                ),
            )

            if allow_delete:
                delete_confirmation = st.text_input(
                    "Type DELETE SESSION to confirm",
                    key=(
                        f"session_delete_confirmation_"
                        f"{selected_session.session_id}"
                    ),
                )

                if st.button(
                    "Delete Session Container",
                    use_container_width=True,
                    disabled=(
                        delete_confirmation
                        != "DELETE SESSION"
                    ),
                    key=(
                        f"session_delete_"
                        f"{selected_session.session_id}"
                    ),
                ):
                    try:
                        session_repository.delete_session(
                            selected_session.session_id
                        )

                        st.success(
                            (
                                "Session deleted. Its shot files remain "
                                "available in player history."
                            )
                        )

                        st.cache_data.clear()
                        st.rerun()

                    except Exception as error:
                        st.error(
                            (
                                "Could not delete session. "
                                f"{type(error).__name__}: {error}"
                            )
                        )



@st.cache_resource
def get_organization_team_repository() -> OrganizationTeamRepository:
    """
    Create one persistent organization/team repository for the app.
    """

    history_repository = get_player_history_repository()

    return OrganizationTeamRepository(
        database_file=history_repository.database_file
    )


def organization_option_label(
    organization: object,
) -> str:
    status = (
        "Active"
        if organization.active
        else "Inactive"
    )

    return (
        f"{organization.organization_name} · "
        f"{organization.team_count} teams · "
        f"{organization.player_count} players · "
        f"{status}"
    )


def team_option_label(
    team: object,
) -> str:
    season_text = (
        f" · {team.season_name}"
        if team.season_name
        else ""
    )

    status = (
        "Active"
        if team.active
        else "Inactive"
    )

    return (
        f"{team.team_name}{season_text} · "
        f"{team.player_count} players · "
        f"{status}"
    )


def roster_option_label(
    roster_member: object,
) -> str:
    jersey = (
        f"#{roster_member.jersey_number} · "
        if roster_member.jersey_number
        else ""
    )

    position = (
        f"{roster_member.position_group} · "
        if roster_member.position_group
        else ""
    )

    return (
        f"{jersey}"
        f"{roster_member.player_display_name} "
        f"({roster_member.participant_id}) · "
        f"{position}"
        f"{roster_member.roster_role}"
    )


def render_organization_team_manager() -> None:
    """
    Manage organizations, teams, and team rosters.
    """

    history_repository = (
        get_player_history_repository()
    )

    organization_repository = (
        get_organization_team_repository()
    )

    with st.expander(
        "Manage Organizations and Teams",
        expanded=False,
    ):
        st.caption(
            (
                "Create organizations, add teams or seasons, and assign "
                "existing player profiles to one or more rosters. Removing "
                "a player from a roster does not delete their practices or "
                "shot history."
            )
        )

        organization_tab, team_tab, roster_tab = st.tabs(
            [
                "Organizations",
                "Teams",
                "Rosters",
            ]
        )

        # =================================================
        # ORGANIZATIONS
        # =================================================

        with organization_tab:
            st.markdown(
                "### Create an organization"
            )

            with st.form(
                "create_organization_form",
                clear_on_submit=True,
            ):
                organization_name = st.text_input(
                    "Organization name",
                    placeholder="Example: Raptors 905",
                )

                organization_type = st.selectbox(
                    "Organization type",
                    options=[
                        "Professional Team",
                        "National Team",
                        "College Program",
                        "Academy",
                        "Private Clients",
                        "Summer League",
                        "Other",
                    ],
                )

                organization_description = st.text_area(
                    "Description",
                    placeholder=(
                        "Optional notes about the organization, program, "
                        "season, or client group."
                    ),
                )

                create_organization_clicked = (
                    st.form_submit_button(
                        "Create Organization",
                        type="primary",
                        use_container_width=True,
                    )
                )

            if create_organization_clicked:
                try:
                    organization = (
                        organization_repository
                        .create_organization(
                            organization_name=(
                                organization_name
                            ),
                            organization_type=(
                                organization_type
                            ),
                            description=(
                                organization_description
                            ),
                        )
                    )

                    st.success(
                        (
                            f"Created organization: "
                            f"{organization.organization_name}."
                        )
                    )

                    st.rerun()

                except Exception as error:
                    st.error(
                        (
                            "Could not create organization. "
                            f"{type(error).__name__}: {error}"
                        )
                    )

            show_inactive_organizations = st.toggle(
                "Show inactive organizations",
                value=False,
                key="organization_show_inactive",
            )

            organizations = (
                organization_repository
                .list_organizations(
                    include_inactive=(
                        show_inactive_organizations
                    )
                )
            )

            if not organizations:
                st.info(
                    "No organizations are available in this view."
                )

            else:
                st.markdown(
                    "### Existing organizations"
                )

                st.dataframe(
                    [
                        {
                            "Organization": (
                                organization.organization_name
                            ),
                            "Type": (
                                organization.organization_type
                            ),
                            "Teams": (
                                organization.team_count
                            ),
                            "Players": (
                                organization.player_count
                            ),
                            "Active": (
                                organization.active
                            ),
                            "Description": (
                                organization.description
                            ),
                        }
                        for organization in organizations
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

                selected_organization = st.selectbox(
                    "Select organization to edit",
                    options=organizations,
                    format_func=organization_option_label,
                    key="organization_edit_select",
                )

                new_organization_name = st.text_input(
                    "Organization name",
                    value=(
                        selected_organization
                        .organization_name
                    ),
                    key=(
                        f"organization_name_"
                        f"{selected_organization.organization_id}"
                    ),
                )

                new_organization_type = st.selectbox(
                    "Organization type",
                    options=[
                        "Professional Team",
                        "National Team",
                        "College Program",
                        "Academy",
                        "Private Clients",
                        "Summer League",
                        "Other",
                    ],
                    index=(
                        [
                            "Professional Team",
                            "National Team",
                            "College Program",
                            "Academy",
                            "Private Clients",
                            "Summer League",
                            "Other",
                        ].index(
                            selected_organization
                            .organization_type
                        )
                        if (
                            selected_organization
                            .organization_type
                            in [
                                "Professional Team",
                                "National Team",
                                "College Program",
                                "Academy",
                                "Private Clients",
                                "Summer League",
                                "Other",
                            ]
                        )
                        else 0
                    ),
                    key=(
                        f"organization_type_"
                        f"{selected_organization.organization_id}"
                    ),
                )

                new_organization_description = st.text_area(
                    "Description",
                    value=(
                        selected_organization
                        .description
                    ),
                    key=(
                        f"organization_description_"
                        f"{selected_organization.organization_id}"
                    ),
                )

                organization_action_columns = st.columns(
                    2
                )

                if organization_action_columns[0].button(
                    "Save Organization Changes",
                    type="primary",
                    use_container_width=True,
                    key=(
                        f"organization_save_"
                        f"{selected_organization.organization_id}"
                    ),
                ):
                    try:
                        organization_repository.update_organization(
                            organization_id=(
                                selected_organization
                                .organization_id
                            ),
                            organization_name=(
                                new_organization_name
                            ),
                            organization_type=(
                                new_organization_type
                            ),
                            description=(
                                new_organization_description
                            ),
                        )

                        st.success(
                            "Organization updated."
                        )

                        st.rerun()

                    except Exception as error:
                        st.error(
                            (
                                "Could not update organization. "
                                f"{type(error).__name__}: {error}"
                            )
                        )

                organization_action_label = (
                    "Deactivate Organization"
                    if selected_organization.active
                    else "Reactivate Organization"
                )

                if organization_action_columns[1].button(
                    organization_action_label,
                    use_container_width=True,
                    key=(
                        f"organization_active_"
                        f"{selected_organization.organization_id}"
                    ),
                ):
                    try:
                        organization_repository.set_organization_active(
                            organization_id=(
                                selected_organization
                                .organization_id
                            ),
                            active=(
                                not selected_organization
                                .active
                            ),
                        )

                        st.success(
                            (
                                "Organization reactivated."
                                if not selected_organization.active
                                else "Organization deactivated."
                            )
                        )

                        st.rerun()

                    except Exception as error:
                        st.error(
                            (
                                "Could not change organization status. "
                                f"{type(error).__name__}: {error}"
                            )
                        )

        # =================================================
        # TEAMS
        # =================================================

        with team_tab:
            active_organizations = (
                organization_repository
                .list_organizations(
                    include_inactive=False
                )
            )

            if not active_organizations:
                st.info(
                    "Create an active organization before creating teams."
                )

            else:
                selected_team_organization = st.selectbox(
                    "Organization",
                    options=active_organizations,
                    format_func=organization_option_label,
                    key="team_create_organization",
                )

                with st.form(
                    "create_team_form",
                    clear_on_submit=True,
                ):
                    team_name = st.text_input(
                        "Team name",
                        placeholder="Example: Raptors 905",
                    )

                    season_name = st.text_input(
                        "Season or competition",
                        placeholder="Example: 2026 Regular Season",
                    )

                    level = st.selectbox(
                        "Level",
                        options=[
                            "Professional",
                            "G League",
                            "NBA",
                            "National Team",
                            "College",
                            "Academy",
                            "Youth",
                            "Private Clients",
                            "Other",
                        ],
                    )

                    team_description = st.text_area(
                        "Description",
                        placeholder=(
                            "Optional roster, season, or program notes."
                        ),
                    )

                    create_team_clicked = (
                        st.form_submit_button(
                            "Create Team",
                            type="primary",
                            use_container_width=True,
                        )
                    )

                if create_team_clicked:
                    try:
                        team = (
                            organization_repository
                            .create_team(
                                organization_id=(
                                    selected_team_organization
                                    .organization_id
                                ),
                                team_name=team_name,
                                season_name=season_name,
                                level=level,
                                description=team_description,
                            )
                        )

                        st.success(
                            (
                                f"Created team: "
                                f"{team.team_name}."
                            )
                        )

                        st.rerun()

                    except Exception as error:
                        st.error(
                            (
                                "Could not create team. "
                                f"{type(error).__name__}: {error}"
                            )
                        )

            show_inactive_teams = st.toggle(
                "Show inactive teams",
                value=False,
                key="team_show_inactive",
            )

            all_teams = (
                organization_repository
                .list_teams(
                    include_inactive=(
                        show_inactive_teams
                    )
                )
            )

            if all_teams:
                st.markdown(
                    "### Existing teams"
                )

                st.dataframe(
                    [
                        {
                            "Organization": (
                                team.organization_name
                            ),
                            "Team": team.team_name,
                            "Season": team.season_name,
                            "Level": team.level,
                            "Players": team.player_count,
                            "Active": team.active,
                            "Description": team.description,
                        }
                        for team in all_teams
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

                selected_team = st.selectbox(
                    "Select team to edit",
                    options=all_teams,
                    format_func=team_option_label,
                    key="team_edit_select",
                )

                new_team_name = st.text_input(
                    "Team name",
                    value=selected_team.team_name,
                    key=(
                        f"team_name_"
                        f"{selected_team.team_id}"
                    ),
                )

                new_season_name = st.text_input(
                    "Season or competition",
                    value=selected_team.season_name,
                    key=(
                        f"team_season_"
                        f"{selected_team.team_id}"
                    ),
                )

                level_options = [
                    "Professional",
                    "G League",
                    "NBA",
                    "National Team",
                    "College",
                    "Academy",
                    "Youth",
                    "Private Clients",
                    "Other",
                ]

                new_level = st.selectbox(
                    "Level",
                    options=level_options,
                    index=(
                        level_options.index(
                            selected_team.level
                        )
                        if selected_team.level
                        in level_options
                        else 0
                    ),
                    key=(
                        f"team_level_"
                        f"{selected_team.team_id}"
                    ),
                )

                new_team_description = st.text_area(
                    "Description",
                    value=selected_team.description,
                    key=(
                        f"team_description_"
                        f"{selected_team.team_id}"
                    ),
                )

                team_action_columns = st.columns(
                    2
                )

                if team_action_columns[0].button(
                    "Save Team Changes",
                    type="primary",
                    use_container_width=True,
                    key=(
                        f"team_save_"
                        f"{selected_team.team_id}"
                    ),
                ):
                    try:
                        organization_repository.update_team(
                            team_id=selected_team.team_id,
                            team_name=new_team_name,
                            season_name=new_season_name,
                            level=new_level,
                            description=new_team_description,
                        )

                        st.success(
                            "Team updated."
                        )

                        st.rerun()

                    except Exception as error:
                        st.error(
                            (
                                "Could not update team. "
                                f"{type(error).__name__}: {error}"
                            )
                        )

                team_action_label = (
                    "Deactivate Team"
                    if selected_team.active
                    else "Reactivate Team"
                )

                if team_action_columns[1].button(
                    team_action_label,
                    use_container_width=True,
                    key=(
                        f"team_active_"
                        f"{selected_team.team_id}"
                    ),
                ):
                    try:
                        organization_repository.set_team_active(
                            team_id=selected_team.team_id,
                            active=(
                                not selected_team.active
                            ),
                        )

                        st.success(
                            (
                                "Team reactivated."
                                if not selected_team.active
                                else "Team deactivated."
                            )
                        )

                        st.rerun()

                    except Exception as error:
                        st.error(
                            (
                                "Could not change team status. "
                                f"{type(error).__name__}: {error}"
                            )
                        )

        # =================================================
        # ROSTERS
        # =================================================

        with roster_tab:
            st.markdown(
                """<div style="background:#ffffff;border:1px solid #e4e8f0;border-radius:20px;padding:1.15rem 1.3rem;margin-bottom:1rem;box-shadow:0 8px 22px rgba(28,39,60,0.04);">
<div style="color:#c94d18;font-size:0.73rem;font-weight:800;letter-spacing:0.11em;text-transform:uppercase;margin-bottom:0.4rem;">Roster manager</div>
<h3 style="margin:0 0 0.4rem;color:#152033;">Add, edit, remove, or restore players</h3>
<p style="margin:0;color:#455066;line-height:1.65;">A player joins an organization by being assigned to one of that organization's teams. Removing a player from a roster never deletes the player profile, shot history, sessions, or reports.</p>
</div>""",
                unsafe_allow_html=True,
            )

            active_teams = (
                organization_repository
                .list_teams(
                    include_inactive=False
                )
            )

            active_players = (
                history_repository
                .list_players(
                    include_inactive=False
                )
            )

            if not active_teams:
                st.info(
                    "Create an active organization and team before managing a roster."
                )

                return

            if not active_players:
                st.info(
                    "Create at least one active player profile before managing a roster."
                )

                return

            selected_roster_team = st.selectbox(
                "Organization and team",
                options=active_teams,
                format_func=team_option_label,
                key="roster_team_select",
            )

            st.caption(
                (
                    f"Managing: {selected_roster_team.organization_name} · "
                    f"{selected_roster_team.team_name}"
                )
            )

            current_roster = (
                organization_repository
                .list_team_players(
                    team_id=selected_roster_team.team_id,
                    include_inactive=True,
                )
            )

            active_member_ids = {
                member.player_id
                for member in current_roster
                if member.active
            }

            available_players = [
                player
                for player in active_players
                if player.player_id
                not in active_member_ids
            ]

            active_members = [
                member
                for member in current_roster
                if member.active
            ]

            inactive_members = [
                member
                for member in current_roster
                if not member.active
            ]

            roster_metrics = st.columns(
                4
            )

            roster_metrics[0].metric(
                "Active roster",
                len(
                    active_members
                ),
            )

            roster_metrics[1].metric(
                "Former members",
                len(
                    inactive_members
                ),
            )

            roster_metrics[2].metric(
                "Available to add",
                len(
                    available_players
                ),
            )

            roster_metrics[3].metric(
                "Organization",
                selected_roster_team.organization_name,
            )

            (
                active_roster_tab,
                add_player_tab,
                edit_member_tab,
                former_member_tab,
            ) = st.tabs(
                [
                    "Active Roster",
                    "Add Player",
                    "Edit or Remove",
                    "Former Members",
                ]
            )

            # =============================================
            # ACTIVE ROSTER
            # =============================================

            with active_roster_tab:
                st.markdown(
                    "### Current active roster"
                )

                if active_members:
                    st.dataframe(
                        [
                            {
                                "Player": member.player_display_name,
                                "Participant ID": member.participant_id,
                                "Jersey": member.jersey_number,
                                "Position": member.position_group,
                                "Role": member.roster_role,
                                "Organization": member.organization_name,
                                "Team": member.team_name,
                            }
                            for member in active_members
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

                else:
                    st.markdown(
                        """<div class="bms-empty-state">
This team currently has no active roster members. Open <b>Add Player</b> to assign one.
</div>""",
                        unsafe_allow_html=True,
                    )

            # =============================================
            # ADD PLAYER
            # =============================================

            with add_player_tab:
                st.markdown(
                    "### Add player to this roster"
                )

                st.caption(
                    (
                        "Choose an existing player profile or create a brand-new "
                        "player and immediately add them to this organization's "
                        "selected team."
                    )
                )

                add_mode = st.radio(
                    "Player source",
                    options=[
                        "Choose Existing Player",
                        "Create New Player",
                    ],
                    horizontal=True,
                    key=(
                        f"roster_add_mode_"
                        f"{selected_roster_team.team_id}"
                    ),
                )

                roster_columns = st.columns(
                    3
                )

                roster_role = roster_columns[0].selectbox(
                    "Roster role",
                    options=[
                        "Player",
                        "Two-Way Player",
                        "Training Player",
                        "Prospect",
                        "Inactive",
                        "Other",
                    ],
                    key=(
                        f"roster_add_role_"
                        f"{selected_roster_team.team_id}"
                    ),
                )

                jersey_number = roster_columns[1].text_input(
                    "Jersey number",
                    key=(
                        f"roster_add_jersey_"
                        f"{selected_roster_team.team_id}"
                    ),
                )

                position_group = roster_columns[2].selectbox(
                    "Position group",
                    options=[
                        "",
                        "Point Guard",
                        "Guard",
                        "Wing",
                        "Forward",
                        "Center",
                        "Big",
                        "Other",
                    ],
                    key=(
                        f"roster_add_position_"
                        f"{selected_roster_team.team_id}"
                    ),
                )

                if add_mode == "Choose Existing Player":
                    if available_players:
                        player_to_add = st.selectbox(
                            "Existing player profile",
                            options=available_players,
                            format_func=player_option_label,
                            key=(
                                f"roster_add_player_"
                                f"{selected_roster_team.team_id}"
                            ),
                        )

                        if st.button(
                            "Add Existing Player to Roster",
                            type="primary",
                            use_container_width=True,
                            key=(
                                f"roster_add_existing_button_"
                                f"{selected_roster_team.team_id}"
                            ),
                        ):
                            try:
                                organization_repository.assign_player_to_team(
                                    team_id=selected_roster_team.team_id,
                                    player_id=player_to_add.player_id,
                                    roster_role=roster_role,
                                    jersey_number=jersey_number,
                                    position_group=position_group,
                                )

                                st.cache_data.clear()

                                st.success(
                                    (
                                        f"{player_to_add.display_name} was added "
                                        f"to {selected_roster_team.team_name}."
                                    )
                                )

                                st.rerun()

                            except Exception as error:
                                st.error(
                                    (
                                        "Could not add the player to the roster. "
                                        f"{type(error).__name__}: {error}"
                                    )
                                )

                    else:
                        st.info(
                            (
                                "Every active player profile is already on this "
                                "team's active roster. Choose Create New Player "
                                "to add someone new."
                            )
                        )

                else:
                    st.markdown(
                        "#### Create a new player profile"
                    )

                    new_player_columns = st.columns(
                        2
                    )

                    new_participant_id = new_player_columns[0].text_input(
                        "Participant ID",
                        placeholder="Example: P0006",
                        key=(
                            f"roster_new_participant_"
                            f"{selected_roster_team.team_id}"
                        ),
                    )

                    new_player_name = new_player_columns[1].text_input(
                        "Player name",
                        placeholder="Example: Jordan Smith",
                        key=(
                            f"roster_new_player_name_"
                            f"{selected_roster_team.team_id}"
                        ),
                    )

                    st.info(
                        (
                            "This creates the player profile and assigns it to "
                            f"{selected_roster_team.organization_name} · "
                            f"{selected_roster_team.team_name} in one step."
                        )
                    )

                    if st.button(
                        "Create Player and Add to Roster",
                        type="primary",
                        use_container_width=True,
                        key=(
                            f"roster_create_and_add_button_"
                            f"{selected_roster_team.team_id}"
                        ),
                    ):
                        cleaned_participant_id = (
                            new_participant_id.strip()
                        )

                        cleaned_player_name = (
                            new_player_name.strip()
                        )

                        if not cleaned_participant_id:
                            st.error(
                                "Participant ID is required."
                            )

                        else:
                            try:
                                created_player = (
                                    history_repository
                                    .create_or_get_player(
                                        participant_id=(
                                            cleaned_participant_id
                                        ),
                                        display_name=(
                                            cleaned_player_name
                                            or cleaned_participant_id
                                        ),
                                    )
                                )

                                organization_repository.assign_player_to_team(
                                    team_id=selected_roster_team.team_id,
                                    player_id=created_player.player_id,
                                    roster_role=roster_role,
                                    jersey_number=jersey_number,
                                    position_group=position_group,
                                )

                                st.cache_data.clear()

                                st.success(
                                    (
                                        f"{created_player.display_name} "
                                        f"({created_player.participant_id}) was "
                                        "created and added to "
                                        f"{selected_roster_team.team_name}."
                                    )
                                )

                                st.rerun()

                            except Exception as error:
                                st.error(
                                    (
                                        "Could not create and add the player. "
                                        f"{type(error).__name__}: {error}"
                                    )
                                )

            # =============================================
            # EDIT OR REMOVE ACTIVE MEMBER
            # =============================================

            with edit_member_tab:
                st.markdown(
                    "### Edit or remove active roster member"
                )

                if active_members:
                    selected_roster_member = st.selectbox(
                        "Active roster member",
                        options=active_members,
                        format_func=roster_option_label,
                        key=(
                            f"roster_edit_member_"
                            f"{selected_roster_team.team_id}"
                        ),
                    )

                    role_options = [
                        "Player",
                        "Two-Way Player",
                        "Training Player",
                        "Prospect",
                        "Inactive",
                        "Other",
                    ]

                    position_options = [
                        "",
                        "Point Guard",
                        "Guard",
                        "Wing",
                        "Forward",
                        "Center",
                        "Big",
                        "Other",
                    ]

                    edit_roster_columns = st.columns(
                        3
                    )

                    edit_role = edit_roster_columns[0].selectbox(
                        "Roster role",
                        options=role_options,
                        index=(
                            role_options.index(
                                selected_roster_member.roster_role
                            )
                            if selected_roster_member.roster_role
                            in role_options
                            else 0
                        ),
                        key=(
                            f"roster_role_"
                            f"{selected_roster_member.team_id}_"
                            f"{selected_roster_member.player_id}"
                        ),
                    )

                    edit_jersey = edit_roster_columns[1].text_input(
                        "Jersey number",
                        value=selected_roster_member.jersey_number,
                        key=(
                            f"roster_jersey_"
                            f"{selected_roster_member.team_id}_"
                            f"{selected_roster_member.player_id}"
                        ),
                    )

                    edit_position = edit_roster_columns[2].selectbox(
                        "Position group",
                        options=position_options,
                        index=(
                            position_options.index(
                                selected_roster_member.position_group
                            )
                            if selected_roster_member.position_group
                            in position_options
                            else 0
                        ),
                        key=(
                            f"roster_position_"
                            f"{selected_roster_member.team_id}_"
                            f"{selected_roster_member.player_id}"
                        ),
                    )

                    action_columns = st.columns(
                        2
                    )

                    if action_columns[0].button(
                        "Save Roster Changes",
                        type="primary",
                        use_container_width=True,
                        key=(
                            f"roster_save_"
                            f"{selected_roster_member.team_id}_"
                            f"{selected_roster_member.player_id}"
                        ),
                    ):
                        try:
                            organization_repository.update_team_player(
                                team_id=selected_roster_member.team_id,
                                player_id=selected_roster_member.player_id,
                                roster_role=edit_role,
                                jersey_number=edit_jersey,
                                position_group=edit_position,
                            )

                            st.cache_data.clear()

                            st.success(
                                (
                                    f"{selected_roster_member.player_display_name}'s "
                                    "roster details were updated."
                                )
                            )

                            st.rerun()

                        except Exception as error:
                            st.error(
                                (
                                    "Could not update the roster details. "
                                    f"{type(error).__name__}: {error}"
                                )
                            )

                    confirm_remove = st.checkbox(
                        (
                            "I understand this removes the player from the active "
                            "roster but preserves their profile and full history."
                        ),
                        key=(
                            f"roster_confirm_remove_"
                            f"{selected_roster_member.team_id}_"
                            f"{selected_roster_member.player_id}"
                        ),
                    )

                    if action_columns[1].button(
                        "Remove from Active Roster",
                        use_container_width=True,
                        disabled=(
                            not confirm_remove
                        ),
                        key=(
                            f"roster_remove_"
                            f"{selected_roster_member.team_id}_"
                            f"{selected_roster_member.player_id}"
                        ),
                    ):
                        try:
                            organization_repository.set_team_player_active(
                                team_id=selected_roster_member.team_id,
                                player_id=selected_roster_member.player_id,
                                active=False,
                            )

                            st.cache_data.clear()

                            st.success(
                                (
                                    f"{selected_roster_member.player_display_name} "
                                    "was removed from the active roster. Their "
                                    "profile, shots, sessions, and reports remain intact."
                                )
                            )

                            st.rerun()

                        except Exception as error:
                            st.error(
                                (
                                    "Could not remove the player from the roster. "
                                    f"{type(error).__name__}: {error}"
                                )
                            )

                else:
                    st.info(
                        "There are no active roster members to edit or remove."
                    )

            # =============================================
            # FORMER MEMBERS / RESTORE
            # =============================================

            with former_member_tab:
                st.markdown(
                    "### Former roster members"
                )

                st.caption(
                    (
                        "Former memberships are retained so a player can be "
                        "restored without losing any history."
                    )
                )

                if inactive_members:
                    st.dataframe(
                        [
                            {
                                "Player": member.player_display_name,
                                "Participant ID": member.participant_id,
                                "Jersey": member.jersey_number,
                                "Position": member.position_group,
                                "Role": member.roster_role,
                            }
                            for member in inactive_members
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

                    member_to_restore = st.selectbox(
                        "Former roster member",
                        options=inactive_members,
                        format_func=roster_option_label,
                        key=(
                            f"roster_restore_member_"
                            f"{selected_roster_team.team_id}"
                        ),
                    )

                    if st.button(
                        "Restore to Active Roster",
                        type="primary",
                        use_container_width=True,
                        key=(
                            f"roster_restore_button_"
                            f"{member_to_restore.team_id}_"
                            f"{member_to_restore.player_id}"
                        ),
                    ):
                        try:
                            organization_repository.set_team_player_active(
                                team_id=member_to_restore.team_id,
                                player_id=member_to_restore.player_id,
                                active=True,
                            )

                            st.cache_data.clear()

                            st.success(
                                (
                                    f"{member_to_restore.player_display_name} "
                                    "was restored to the active roster."
                                )
                            )

                            st.rerun()

                        except Exception as error:
                            st.error(
                                (
                                    "Could not restore the roster membership. "
                                    f"{type(error).__name__}: {error}"
                                )
                            )

                else:
                    st.info(
                        "This team has no former roster members."
                    )



@st.cache_data(
    show_spinner=False,
)
def load_team_dashboard(
    team_id: int,
) -> tuple[
    dict,
    list[dict],
]:
    """
    Run and cache the selected team's coaching dashboard.
    """

    history_repository = get_player_history_repository()

    engine = TeamDashboardEngine(
        database_file=history_repository.database_file
    )

    summary, player_rows = engine.run(
        team_id=team_id
    )

    return (
        asdict(
            summary
        ),
        [
            asdict(
                row
            )
            for row in player_rows
        ],
    )



@st.cache_data(
    show_spinner=False,
)
def load_team_practice_intelligence(
    team_id: int,
) -> tuple[
    dict,
    list[dict],
]:
    """
    Run and cache the latest-session team practice intelligence.
    """

    history_repository = get_player_history_repository()

    engine = TeamPracticeIntelligenceEngine(
        database_file=history_repository.database_file
    )

    summary, player_rows = engine.run(
        team_id=team_id
    )

    return (
        asdict(
            summary
        ),
        [
            asdict(
                row
            )
            for row in player_rows
        ],
    )



@st.cache_data(
    show_spinner=False,
)
def generate_team_practice_pdf_bytes(
    team_id: int,
) -> tuple[
    bytes,
    str,
]:
    """
    Generate and cache the latest-practice team PDF.
    """

    history_repository = get_player_history_repository()

    intelligence_engine = TeamPracticeIntelligenceEngine(
        database_file=history_repository.database_file
    )

    generator = TeamPracticePDFReportGenerator()

    pdf_path = generator.run(
        team_id=team_id,
        intelligence_engine=intelligence_engine,
    )

    return (
        pdf_path.read_bytes(),
        pdf_path.name,
    )


def render_team_dashboard() -> None:
    team_quick_left, team_quick_middle, team_quick_right = st.columns(
        3
    )

    if team_quick_left.button(
        "Open Player Profiles",
        use_container_width=True,
        key="team_dashboard_open_player_profiles",
    ):
        request_workspace_navigation(
            "Player Profile"
        )

    if team_quick_middle.button(
        "Manage Roster",
        use_container_width=True,
        disabled=False,
        key="team_dashboard_manage_roster",
    ):
        request_workspace_navigation(
            "Roster & Organizations"
        )

    if team_quick_right.button(
        "Open Team Reports",
        use_container_width=True,
        key="team_dashboard_open_reports",
    ):
        request_workspace_navigation(
            "Reports Center"
        )

    """
    Render the coach-facing home dashboard for one team.
    """

    organization_repository = (
        get_organization_team_repository()
    )

    active_teams = (
        organization_repository
        .list_teams(
            include_inactive=False
        )
    )

    if not active_teams:
        st.info(
            (
                "Create an active organization and team before "
                "opening the Team Dashboard."
            )
        )

        return

    team_select_left, team_select_right = st.columns(
        [
            0.82,
            0.18,
        ]
    )

    selected_team = team_select_left.selectbox(
        "Team dashboard",
        options=active_teams,
        format_func=team_option_label,
        key="team_dashboard_team_select",
    )

    if team_select_right.button(
        "Refresh",
        use_container_width=True,
        key=(
            f"team_dashboard_refresh_"
            f"{selected_team.team_id}"
        ),
    ):
        st.cache_data.clear()

        st.success(
            "Team dashboard data refreshed."
        )

        st.rerun()

    try:
        summary, player_rows = load_team_dashboard(
            selected_team.team_id
        )

    except Exception as error:
        st.error(
            (
                "Could not build the team dashboard. "
                f"{type(error).__name__}: {error}"
            )
        )

        return

    st.markdown(
        f"## {summary['team_name']}"
    )

    subtitle_parts = [
        summary[
            "organization_name"
        ],
    ]

    if summary[
        "season_name"
    ]:
        subtitle_parts.append(
            summary[
                "season_name"
            ]
        )

    if summary[
        "level"
    ]:
        subtitle_parts.append(
            summary[
                "level"
            ]
        )

    st.caption(
        " · ".join(
            subtitle_parts
        )
    )

    export_left, export_right = st.columns(
        [
            0.75,
            0.25,
        ]
    )

    export_left.info(
        (
            "Generate a team-wide latest-practice report containing roster "
            "grades, review priorities, recommended drills, and staff notes."
        )
    )

    if export_right.button(
        "Generate Team PDF",
        type="primary",
        use_container_width=True,
        key=(
            f"generate_team_practice_pdf_"
            f"{selected_team.team_id}"
        ),
    ):
        with st.spinner(
            "Generating team practice report..."
        ):
            try:
                (
                    team_pdf_bytes,
                    team_pdf_filename,
                ) = generate_team_practice_pdf_bytes(
                    selected_team.team_id
                )

                st.session_state[
                    "generated_team_pdf_bytes"
                ] = team_pdf_bytes

                st.session_state[
                    "generated_team_pdf_filename"
                ] = team_pdf_filename

                st.session_state[
                    "generated_team_pdf_team_id"
                ] = selected_team.team_id

                st.success(
                    "Team practice PDF generated."
                )

            except Exception as error:
                st.error(
                    (
                        "Could not generate the team PDF. "
                        f"{type(error).__name__}: {error}"
                    )
                )

    if (
        st.session_state.get(
            "generated_team_pdf_team_id"
        )
        == selected_team.team_id
        and st.session_state.get(
            "generated_team_pdf_bytes"
        )
        is not None
    ):
        st.download_button(
            "Download Team Practice PDF",
            data=st.session_state[
                "generated_team_pdf_bytes"
            ],
            file_name=st.session_state[
                "generated_team_pdf_filename"
            ],
            mime="application/pdf",
            use_container_width=True,
            key=(
                f"download_team_practice_pdf_"
                f"{selected_team.team_id}"
            ),
        )

    metric_columns = st.columns(
        6
    )

    metric_columns[0].metric(
        "Active players",
        summary[
            "active_players"
        ],
    )

    metric_columns[1].metric(
        "Tracked shots",
        summary[
            "total_shot_files"
        ],
    )

    metric_columns[2].metric(
        "Team make %",
        (
            f"{summary['team_make_percentage']:.1f}%"
        ),
    )

    metric_columns[3].metric(
        "Practice sessions",
        summary[
            "total_sessions"
        ],
    )

    metric_columns[4].metric(
        "Average mechanics",
        (
            "N/A"
            if summary[
                "average_mechanics_score"
            ]
            is None
            else (
                f"{summary['average_mechanics_score']:.1f}"
            )
        ),
    )

    metric_columns[5].metric(
        "Average consistency",
        (
            "N/A"
            if summary[
                "average_consistency_score"
            ]
            is None
            else (
                f"{summary['average_consistency_score']:.1f}"
            )
        ),
    )

    highlight_left, highlight_middle, highlight_right = st.columns(
        3
    )

    with highlight_left:
        st.markdown(
            f"""<div class="bms-summary-card">
<h3>Highest review attention</h3>
<p>{summary['highest_attention_player']}</p>
</div>""",
            unsafe_allow_html=True,
        )

    with highlight_middle:
        st.markdown(
            f"""<div class="bms-summary-card">
<h3>Strongest consistency</h3>
<p>{summary['strongest_consistency_player']}</p>
</div>""",
            unsafe_allow_html=True,
        )

    with highlight_right:
        st.markdown(
            f"""<div class="bms-summary-card">
<h3>Dashboard status</h3>
<p>{summary['dashboard_status']}</p>
</div>""",
            unsafe_allow_html=True,
        )

    st.markdown(
        "### Latest Team Practice Intelligence"
    )

    try:
        (
            practice_summary,
            practice_players,
        ) = load_team_practice_intelligence(
            selected_team.team_id
        )

    except Exception as error:
        practice_summary = None
        practice_players = []

        st.warning(
            (
                "Latest team practice intelligence could not be built. "
                f"{type(error).__name__}: {error}"
            )
        )

    if practice_summary is not None:
        practice_metric_columns = st.columns(
            6
        )

        practice_metric_columns[0].metric(
            "Players with sessions",
            (
                f"{practice_summary['players_with_sessions']} / "
                f"{practice_summary['active_players']}"
            ),
        )

        practice_metric_columns[1].metric(
            "Players with grades",
            practice_summary[
                "players_with_grades"
            ],
        )

        practice_metric_columns[2].metric(
            "Latest-session shots",
            practice_summary[
                "total_latest_session_shots"
            ],
        )

        practice_metric_columns[3].metric(
            "Average grade",
            (
                "N/A"
                if practice_summary[
                    "average_session_grade"
                ]
                is None
                else (
                    f"{practice_summary['average_session_grade']:.1f}"
                )
            ),
        )

        practice_metric_columns[4].metric(
            "Average make %",
            (
                "N/A"
                if practice_summary[
                    "average_make_percentage"
                ]
                is None
                else (
                    f"{practice_summary['average_make_percentage']:.1f}%"
                )
            ),
        )

        practice_metric_columns[5].metric(
            "Average consistency",
            (
                "N/A"
                if practice_summary[
                    "average_consistency_score"
                ]
                is None
                else (
                    f"{practice_summary['average_consistency_score']:.1f}"
                )
            ),
        )

        practice_highlight_columns = st.columns(
            3
        )

        with practice_highlight_columns[0]:
            st.markdown(
                f"""<div class="bms-summary-card">
<h3>Highest-grade latest session</h3>
<p>{practice_summary['highest_grade_player']}</p>
</div>""",
                unsafe_allow_html=True,
            )

        with practice_highlight_columns[1]:
            st.markdown(
                f"""<div class="bms-summary-card">
<h3>Review first</h3>
<p>{practice_summary['highest_attention_player']}</p>
</div>""",
                unsafe_allow_html=True,
            )

        with practice_highlight_columns[2]:
            st.markdown(
                f"""<div class="bms-summary-card">
<h3>Most common focus</h3>
<p>{practice_summary['most_common_primary_focus']}</p>
</div>""",
                unsafe_allow_html=True,
            )

        if practice_players:
            st.markdown(
                "#### Latest Session Roster Board"
            )

            practice_table = []

            for row in practice_players:
                practice_table.append(
                    {
                        "Player": row[
                            "player_display_name"
                        ],
                        "Jersey": row[
                            "jersey_number"
                        ],
                        "Position": row[
                            "position_group"
                        ],
                        "Latest Session": row[
                            "latest_session_name"
                        ],
                        "Date": row[
                            "latest_session_date"
                        ],
                        "Shots": row[
                            "assigned_shots"
                        ],
                        "Make %": (
                            None
                            if row[
                                "make_percentage"
                            ]
                            is None
                            else round(
                                row[
                                    "make_percentage"
                                ],
                                1,
                            )
                        ),
                        "Grade": (
                            None
                            if row[
                                "session_grade"
                            ]
                            is None
                            else round(
                                row[
                                    "session_grade"
                                ],
                                1,
                            )
                        ),
                        "Consistency": (
                            None
                            if row[
                                "consistency_score"
                            ]
                            is None
                            else round(
                                row[
                                    "consistency_score"
                                ],
                                1,
                            )
                        ),
                        "Primary Focus": row[
                            "primary_focus"
                        ],
                        "Recommended Drill": row[
                            "recommended_drill"
                        ],
                        "Review First": row[
                            "review_first_trial"
                        ],
                        "Attention": round(
                            row[
                                "attention_score"
                            ],
                            1,
                        ),
                        "Status": row[
                            "team_status"
                        ],
                    }
                )

            st.dataframe(
                practice_table,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown(
                "#### Team Review Queue"
            )

            visible_review_count = st.selectbox(
                "Players to show",
                options=[
                    5,
                    10,
                    len(
                        practice_players
                    ),
                ],
                index=0,
                key=(
                    f"team_practice_review_count_"
                    f"{selected_team.team_id}"
                ),
            )

            for rank, row in enumerate(
                practice_players[
                    :int(
                        visible_review_count
                    )
                ],
                start=1,
            ):
                with st.container(
                    border=True
                ):
                    review_left, review_right = st.columns(
                        [
                            0.76,
                            0.24,
                        ]
                    )

                    with review_left:
                        jersey_text = (
                            f"#{row['jersey_number']} · "
                            if row[
                                "jersey_number"
                            ]
                            else ""
                        )

                        st.markdown(
                            (
                                f"### {rank}. "
                                f"{jersey_text}"
                                f"{row['player_display_name']}"
                            )
                        )

                        st.caption(
                            (
                                f"{row['latest_session_name']} · "
                                f"{row['latest_session_date'] or 'No date'} · "
                                f"{row['assigned_shots']} shots"
                            )
                        )

                    with review_right:
                        st.metric(
                            "Team attention",
                            (
                                f"{row['attention_score']:.1f}"
                            ),
                        )

                        st.caption(
                            row[
                                "team_status"
                            ]
                        )

                    detail_columns = st.columns(
                        4
                    )

                    detail_columns[0].metric(
                        "Session grade",
                        (
                            "N/A"
                            if row[
                                "session_grade"
                            ]
                            is None
                            else (
                                f"{row['session_grade']:.1f}"
                            )
                        ),
                    )

                    detail_columns[1].metric(
                        "Make %",
                        (
                            "N/A"
                            if row[
                                "make_percentage"
                            ]
                            is None
                            else (
                                f"{row['make_percentage']:.1f}%"
                            )
                        ),
                    )

                    detail_columns[2].metric(
                        "Consistency",
                        (
                            "N/A"
                            if row[
                                "consistency_score"
                            ]
                            is None
                            else (
                                f"{row['consistency_score']:.1f}"
                            )
                        ),
                    )

                    detail_columns[3].metric(
                        "Review first",
                        row[
                            "review_first_trial"
                        ],
                    )

                    st.markdown(
                        "**Primary coaching focus**"
                    )

                    st.write(
                        row[
                            "primary_focus"
                        ]
                    )

                    st.markdown(
                        "**Recommended drill**"
                    )

                    st.write(
                        row[
                            "recommended_drill"
                        ]
                    )

        st.caption(
            (
                "Team practice intelligence uses each active roster player's "
                "most recent active session. The team attention score orders "
                "the coaching workflow; it is not a talent ranking."
            )
        )

    st.markdown(
        "### Roster development overview"
    )

    if not player_rows:
        st.info(
            "This team has no active roster players."
        )

        return

    roster_table = []

    for row in player_rows:
        roster_table.append(
            {
                "Player": (
                    row[
                        "player_display_name"
                    ]
                ),
                "Participant": (
                    row[
                        "participant_id"
                    ]
                ),
                "Jersey": (
                    row[
                        "jersey_number"
                    ]
                ),
                "Position": (
                    row[
                        "position_group"
                    ]
                ),
                "Shots": (
                    row[
                        "active_shot_files"
                    ]
                ),
                "Make %": round(
                    row[
                        "make_percentage"
                    ],
                    1,
                ),
                "Sessions": (
                    row[
                        "active_sessions"
                    ]
                ),
                "Mechanics": (
                    None
                    if row[
                        "overall_mechanics_score"
                    ]
                    is None
                    else round(
                        row[
                            "overall_mechanics_score"
                        ],
                        1,
                    )
                ),
                "Consistency": (
                    None
                    if row[
                        "consistency_score"
                    ]
                    is None
                    else round(
                        row[
                            "consistency_score"
                        ],
                        1,
                    )
                ),
                "Primary priority": (
                    row[
                        "primary_priority"
                    ]
                ),
                "Attention": round(
                    row[
                        "attention_score"
                    ],
                    1,
                ),
                "Status": (
                    row[
                        "dashboard_status"
                    ]
                ),
            }
        )

    st.dataframe(
        roster_table,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### Coaching attention queue"
    )

    for rank, row in enumerate(
        player_rows,
        start=1,
    ):
        with st.container(
            border=True
        ):
            top_left, top_right = st.columns(
                [
                    0.76,
                    0.24,
                ]
            )

            with top_left:
                jersey_prefix = (
                    f"#{row['jersey_number']} · "
                    if row[
                        "jersey_number"
                    ]
                    else ""
                )

                st.markdown(
                    (
                        f"### {rank}. "
                        f"{jersey_prefix}"
                        f"{row['player_display_name']}"
                    )
                )

                st.caption(
                    (
                        f"{row['position_group'] or 'Position not set'} · "
                        f"{row['roster_role']} · "
                        f"{row['active_shot_files']} tracked shots · "
                        f"{row['active_sessions']} sessions"
                    )
                )

            with top_right:
                st.metric(
                    "Attention score",
                    (
                        f"{row['attention_score']:.1f}/100"
                    ),
                )

                st.caption(
                    row[
                        "dashboard_status"
                    ]
                )

            detail_columns = st.columns(
                4
            )

            detail_columns[0].metric(
                "Make %",
                (
                    f"{row['make_percentage']:.1f}%"
                ),
            )

            detail_columns[1].metric(
                "Mechanics",
                (
                    "N/A"
                    if row[
                        "overall_mechanics_score"
                    ]
                    is None
                    else (
                        f"{row['overall_mechanics_score']:.1f}"
                    )
                ),
            )

            detail_columns[2].metric(
                "Consistency",
                (
                    "N/A"
                    if row[
                        "consistency_score"
                    ]
                    is None
                    else (
                        f"{row['consistency_score']:.1f}"
                    )
                ),
            )

            detail_columns[3].metric(
                "Evidence",
                row[
                    "evidence_confidence"
                ].title(),
            )

            st.markdown(
                "**Primary development priority**"
            )

            st.write(
                row[
                    "primary_priority"
                ]
            )

            if row[
                "most_recent_session_date"
            ]:
                st.caption(
                    (
                        "Most recent session: "
                        f"{row['most_recent_session_date']}"
                    )
                )

    st.caption(
        (
            "The attention score is a coaching workflow priority, not a "
            "talent ranking or player-performance grade."
        )
    )



@st.cache_data(
    show_spinner=False,
)
def load_player_development_timeline(
    player_id: int,
) -> tuple[
    dict,
    list[dict],
    list[dict],
]:
    """
    Run and cache the selected player's session-by-session timeline.
    """

    history_repository = get_player_history_repository()

    engine = PlayerDevelopmentTimelineEngine(
        database_file=history_repository.database_file
    )

    (
        summary,
        session_rows,
        feature_rows,
    ) = engine.run(
        player_id=player_id
    )

    return (
        asdict(
            summary
        ),
        [
            asdict(
                row
            )
            for row in session_rows
        ],
        [
            asdict(
                row
            )
            for row in feature_rows
        ],
    )


def render_player_development_timeline() -> None:
    """
    Display long-term change across assigned practice sessions.
    """

    history_repository = get_player_history_repository()

    active_players = history_repository.list_players(
        include_inactive=False
    )

    if not active_players:
        st.info(
            "Create an active player profile before opening the timeline."
        )

        return

    selected_player = st.selectbox(
        "Player development timeline",
        options=active_players,
        format_func=player_option_label,
        key="development_timeline_player_select",
    )

    try:
        (
            summary,
            session_rows,
            feature_rows,
        ) = load_player_development_timeline(
            selected_player.player_id
        )

    except Exception as error:
        st.error(
            (
                "Could not build the player development timeline. "
                f"{type(error).__name__}: {error}"
            )
        )

        return

    st.markdown(
        f"## {summary['player_display_name']}"
    )

    st.caption(
        (
            f"{summary['participant_id']} · "
            f"{summary['evidence_status']}"
        )
    )

    metric_columns = st.columns(
        6
    )

    metric_columns[0].metric(
        "Sessions analyzed",
        summary[
            "sessions_analyzed"
        ],
    )

    metric_columns[1].metric(
        "Assigned shots",
        summary[
            "total_assigned_shots"
        ],
    )

    metric_columns[2].metric(
        "Matched feature rows",
        summary[
            "total_matched_feature_rows"
        ],
    )

    metric_columns[3].metric(
        "Latest consistency",
        (
            "N/A"
            if summary[
                "latest_consistency_score"
            ]
            is None
            else (
                f"{summary['latest_consistency_score']:.1f}"
            )
        ),
    )

    metric_columns[4].metric(
        "Consistency change",
        (
            "N/A"
            if summary[
                "consistency_change"
            ]
            is None
            else (
                f"{summary['consistency_change']:+.1f}"
            )
        ),
    )

    metric_columns[5].metric(
        "Development direction",
        summary[
            "development_direction"
        ],
    )

    highlight_left, highlight_middle, highlight_right = st.columns(
        3
    )

    with highlight_left:
        st.markdown(
            f"""<div class="bms-summary-card">
<h3>Strongest recent area</h3>
<p>{summary['strongest_recent_area']}</p>
</div>""",
            unsafe_allow_html=True,
        )

    with highlight_middle:
        st.markdown(
            f"""<div class="bms-summary-card">
<h3>Weakest recent area</h3>
<p>{summary['weakest_recent_area']}</p>
</div>""",
            unsafe_allow_html=True,
        )

    with highlight_right:
        st.markdown(
            f"""<div class="bms-summary-card">
<h3>Make-percentage change</h3>
<p>{
    'N/A'
    if summary['make_percentage_change'] is None
    else f"{summary['make_percentage_change']:+.1f} pts"
}</p>
</div>""",
            unsafe_allow_html=True,
        )

    if not session_rows:
        st.info(
            (
                "This player has no active practice sessions. "
                "Create sessions and assign shots before using the timeline."
            )
        )

        return

    st.markdown(
        "### Session timeline"
    )

    session_labels = [
        (
            f"{row['session_date']} · "
            f"{row['session_name']}"
        )
        for row in session_rows
    ]

    session_positions = list(
        range(
            1,
            len(
                session_rows
            )
            + 1,
        )
    )

    consistency_values = [
        row[
            "overall_consistency_score"
        ]
        for row in session_rows
    ]

    make_values = [
        row[
            "make_percentage"
        ]
        for row in session_rows
    ]

    timeline_figure, timeline_axis = plt.subplots(
        figsize=(
            12,
            5.2,
        )
    )

    valid_consistency_x = [
        position
        for position, value in zip(
            session_positions,
            consistency_values,
        )
        if value is not None
    ]

    valid_consistency_y = [
        value
        for value in consistency_values
        if value is not None
    ]

    if valid_consistency_x:
        timeline_axis.plot(
            valid_consistency_x,
            valid_consistency_y,
            marker="o",
            linewidth=2.0,
            label="Consistency score",
        )

    timeline_axis.plot(
        session_positions,
        make_values,
        marker="o",
        linewidth=2.0,
        label="Make percentage",
    )

    timeline_axis.set_ylim(
        0,
        100,
    )

    timeline_axis.set_xticks(
        session_positions,
        labels=session_labels,
        rotation=25,
        ha="right",
    )

    timeline_axis.set_ylabel(
        "Score / percentage"
    )

    timeline_axis.set_title(
        "Practice-to-practice development"
    )

    timeline_axis.grid(
        visible=True,
        alpha=0.25,
    )

    timeline_axis.legend()

    timeline_figure.tight_layout()

    st.pyplot(
        timeline_figure,
        clear_figure=True,
        use_container_width=True,
    )

    st.markdown(
        "### Category development"
    )

    category_options = [
        "Overall Consistency",
        "Timing & Coordination",
        "Lower Body",
        "Upper Body",
        "Ball & Release",
    ]

    selected_category = st.selectbox(
        "Development category",
        options=category_options,
        key="development_timeline_category",
    )

    category_field_lookup = {
        "Overall Consistency": (
            "overall_consistency_score"
        ),
        "Timing & Coordination": (
            "timing_score"
        ),
        "Lower Body": (
            "lower_body_score"
        ),
        "Upper Body": (
            "upper_body_score"
        ),
        "Ball & Release": (
            "ball_release_score"
        ),
    }

    selected_field = category_field_lookup[
        selected_category
    ]

    category_values = [
        row[
            selected_field
        ]
        for row in session_rows
    ]

    category_x = [
        position
        for position, value in zip(
            session_positions,
            category_values,
        )
        if value is not None
    ]

    category_y = [
        value
        for value in category_values
        if value is not None
    ]

    category_figure, category_axis = plt.subplots(
        figsize=(
            12,
            4.8,
        )
    )

    if category_x:
        category_axis.plot(
            category_x,
            category_y,
            marker="o",
            linewidth=2.2,
        )

        category_axis.set_ylim(
            0,
            100,
        )

        category_axis.set_xticks(
            session_positions,
            labels=session_labels,
            rotation=25,
            ha="right",
        )

        category_axis.set_ylabel(
            "Consistency score"
        )

        category_axis.set_title(
            (
                f"{selected_category} "
                "across practice sessions"
            )
        )

        category_axis.grid(
            visible=True,
            alpha=0.25,
        )

    else:
        category_axis.text(
            0.5,
            0.5,
            (
                "No valid category scores are available "
                "for the selected sessions."
            ),
            ha="center",
            va="center",
            transform=category_axis.transAxes,
        )

        category_axis.set_axis_off()

    category_figure.tight_layout()

    st.pyplot(
        category_figure,
        clear_figure=True,
        use_container_width=True,
    )

    st.markdown(
        "### Session-by-session detail"
    )

    st.dataframe(
        [
            {
                "Date": row[
                    "session_date"
                ],
                "Session": row[
                    "session_name"
                ],
                "Type": row[
                    "session_type"
                ],
                "Assigned Shots": row[
                    "assigned_shots"
                ],
                "Matched Rows": row[
                    "matched_feature_rows"
                ],
                "Make %": round(
                    row[
                        "make_percentage"
                    ],
                    1,
                ),
                "Consistency": (
                    None
                    if row[
                        "overall_consistency_score"
                    ]
                    is None
                    else round(
                        row[
                            "overall_consistency_score"
                        ],
                        1,
                    )
                ),
                "Timing": (
                    None
                    if row[
                        "timing_score"
                    ]
                    is None
                    else round(
                        row[
                            "timing_score"
                        ],
                        1,
                    )
                ),
                "Lower Body": (
                    None
                    if row[
                        "lower_body_score"
                    ]
                    is None
                    else round(
                        row[
                            "lower_body_score"
                        ],
                        1,
                    )
                ),
                "Upper Body": (
                    None
                    if row[
                        "upper_body_score"
                    ]
                    is None
                    else round(
                        row[
                            "upper_body_score"
                        ],
                        1,
                    )
                ),
                "Ball & Release": (
                    None
                    if row[
                        "ball_release_score"
                    ]
                    is None
                    else round(
                        row[
                            "ball_release_score"
                        ],
                        1,
                    )
                ),
                "Strongest Area": row[
                    "strongest_area"
                ],
                "Weakest Area": row[
                    "weakest_area"
                ],
                "Data Status": row[
                    "data_status"
                ],
            }
            for row in session_rows
        ],
        use_container_width=True,
        hide_index=True,
    )

    if feature_rows:
        with st.expander(
            "Explore individual feature trends",
            expanded=False,
        ):
            display_name_lookup = {}

            for row in feature_rows:
                display_name_lookup[
                    row[
                        "feature"
                    ]
                ] = row[
                    "display_name"
                ]

            feature_options = sorted(
                display_name_lookup
            )

            selected_feature = st.selectbox(
                "Feature",
                options=feature_options,
                format_func=lambda feature: (
                    display_name_lookup[
                        feature
                    ]
                ),
                key="development_timeline_feature",
            )

            selected_feature_rows = [
                row
                for row in feature_rows
                if row[
                    "feature"
                ]
                == selected_feature
            ]

            selected_feature_rows.sort(
                key=lambda row: (
                    row[
                        "session_date"
                    ],
                    row[
                        "session_id"
                    ],
                )
            )

            feature_labels = [
                (
                    f"{row['session_date']} · "
                    f"{row['session_name']}"
                )
                for row in selected_feature_rows
            ]

            feature_positions = list(
                range(
                    1,
                    len(
                        selected_feature_rows
                    )
                    + 1,
                )
            )

            feature_means = [
                row[
                    "session_mean"
                ]
                for row in selected_feature_rows
            ]

            feature_consistency = [
                row[
                    "consistency_score"
                ]
                for row in selected_feature_rows
            ]

            mean_figure, mean_axis = plt.subplots(
                figsize=(
                    12,
                    4.6,
                )
            )

            mean_axis.plot(
                feature_positions,
                feature_means,
                marker="o",
                linewidth=2.0,
            )

            mean_axis.set_xticks(
                feature_positions,
                labels=feature_labels,
                rotation=25,
                ha="right",
            )

            mean_axis.set_ylabel(
                display_name_lookup[
                    selected_feature
                ]
            )

            mean_axis.set_title(
                (
                    f"{display_name_lookup[selected_feature]} "
                    "session average"
                )
            )

            mean_axis.grid(
                visible=True,
                alpha=0.25,
            )

            mean_figure.tight_layout()

            st.pyplot(
                mean_figure,
                clear_figure=True,
                use_container_width=True,
            )

            consistency_figure, consistency_axis = plt.subplots(
                figsize=(
                    12,
                    4.6,
                )
            )

            consistency_axis.plot(
                feature_positions,
                feature_consistency,
                marker="o",
                linewidth=2.0,
            )

            consistency_axis.set_ylim(
                0,
                100,
            )

            consistency_axis.set_xticks(
                feature_positions,
                labels=feature_labels,
                rotation=25,
                ha="right",
            )

            consistency_axis.set_ylabel(
                "Consistency score"
            )

            consistency_axis.set_title(
                (
                    f"{display_name_lookup[selected_feature]} "
                    "consistency by session"
                )
            )

            consistency_axis.grid(
                visible=True,
                alpha=0.25,
            )

            consistency_figure.tight_layout()

            st.pyplot(
                consistency_figure,
                clear_figure=True,
                use_container_width=True,
            )

    st.caption(
        (
            "Development direction is based on session-to-session "
            "consistency change. A numerical increase or decrease in an "
            "individual biomechanical feature is not automatically better "
            "or worse without video and coaching context."
        )
    )



def render_session_first_upload_workflow() -> None:
    """
    Coach-friendly session workflow.

    Normal order:
        1. Choose or create a player.
        2. Choose or create a practice session.
        3. Upload one or many JSON files.
        4. Import and assign them to the session in one action.

    Existing active duplicate files are not re-imported, but they are still
    attached to the selected session when appropriate.
    """

    import hashlib

    history_repository = get_player_history_repository()
    session_repository = get_practice_session_repository()

    st.markdown(
        """<div style="background:#ffffff;border:1px solid #e4e8f0;border-radius:20px;padding:1.25rem 1.35rem;margin:0.7rem 0 1.2rem;">
<div style="color:#c94d18;font-size:0.75rem;font-weight:800;letter-spacing:0.11em;text-transform:uppercase;margin-bottom:0.4rem;">Session-first workflow</div>
<h3 style="margin:0 0 0.5rem;color:#152033;">Choose player → choose session → upload shots</h3>
<p style="margin:0;color:#455066;line-height:1.65;">Every uploaded shot is automatically attached to the selected practice session. You do not need to import files and assign them in separate screens.</p>
</div>""",
        unsafe_allow_html=True,
    )

    # =====================================================
    # STEP 1 — PLAYER
    # =====================================================

    st.markdown(
        "### 1. Choose the player"
    )

    active_players = history_repository.list_players(
        include_inactive=False
    )

    player_mode_options = [
        "Choose existing player",
        "Create new player",
    ]

    player_mode = st.radio(
        "Player option",
        options=player_mode_options,
        index=(
            0
            if active_players
            else 1
        ),
        horizontal=True,
        key="session_first_player_mode",
        label_visibility="collapsed",
    )

    selected_player = None

    if player_mode == "Choose existing player":
        if not active_players:
            st.info(
                "No active players exist yet. Create a player below."
            )

        else:
            selected_player = st.selectbox(
                "Player",
                options=active_players,
                format_func=player_option_label,
                key="session_first_existing_player",
            )

    else:
        create_player_columns = st.columns(
            2
        )

        new_participant_id = create_player_columns[0].text_input(
            "Participant ID",
            placeholder="Example: P0006",
            key="session_first_new_participant",
        )

        new_player_name = create_player_columns[1].text_input(
            "Player name",
            placeholder="Example: Jordan Smith",
            key="session_first_new_player_name",
        )

        if st.button(
            "Create Player and Continue",
            type="primary",
            use_container_width=True,
            key="session_first_create_player",
        ):
            try:
                created_player = history_repository.create_or_get_player(
                    participant_id=new_participant_id,
                    display_name=(
                        new_player_name
                        or new_participant_id
                    ),
                )

                st.session_state[
                    "session_first_forced_player_id"
                ] = created_player.player_id

                st.success(
                    (
                        f"Player ready: "
                        f"{created_player.display_name} "
                        f"({created_player.participant_id})."
                    )
                )

                st.rerun()

            except Exception as error:
                st.error(
                    (
                        "Could not create player. "
                        f"{type(error).__name__}: {error}"
                    )
                )

    forced_player_id = st.session_state.get(
        "session_first_forced_player_id"
    )

    if forced_player_id is not None:
        try:
            selected_player = history_repository.get_player(
                forced_player_id
            )

            st.info(
                (
                    f"Current player: "
                    f"{selected_player.display_name} "
                    f"({selected_player.participant_id})"
                )
            )

            if st.button(
                "Choose a Different Player",
                use_container_width=True,
                key="session_first_change_player",
            ):
                st.session_state.pop(
                    "session_first_forced_player_id",
                    None,
                )

                st.session_state.pop(
                    "session_first_forced_session_id",
                    None,
                )

                st.rerun()

        except Exception:
            st.session_state.pop(
                "session_first_forced_player_id",
                None,
            )

    if selected_player is None:
        return

    # =====================================================
    # STEP 2 — SESSION
    # =====================================================

    st.markdown(
        "### 2. Choose or create the practice session"
    )

    active_sessions = session_repository.list_sessions(
        player_id=selected_player.player_id,
        include_inactive=False,
    )

    session_mode = st.radio(
        "Session option",
        options=[
            "Use existing session",
            "Create new session",
        ],
        index=(
            0
            if active_sessions
            else 1
        ),
        horizontal=True,
        key=(
            f"session_first_session_mode_"
            f"{selected_player.player_id}"
        ),
        label_visibility="collapsed",
    )

    selected_session = None

    if session_mode == "Use existing session":
        if not active_sessions:
            st.info(
                "This player has no active sessions. Create one below."
            )

        else:
            selected_session = st.selectbox(
                "Practice session",
                options=active_sessions,
                format_func=session_option_label,
                key=(
                    f"session_first_existing_session_"
                    f"{selected_player.player_id}"
                ),
            )

    else:
        session_columns = st.columns(
            2
        )

        new_session_name = session_columns[0].text_input(
            "Session name",
            placeholder="Example: July 27 Free Throws",
            key=(
                f"session_first_name_"
                f"{selected_player.player_id}"
            ),
        )

        new_session_date = session_columns[1].date_input(
            "Session date",
            key=(
                f"session_first_date_"
                f"{selected_player.player_id}"
            ),
        )

        new_session_type = st.selectbox(
            "Session type",
            options=[
                "Free Throws",
                "Shooting Workout",
                "Fatigue Test",
                "Baseline Test",
                "Pre-Practice",
                "Post-Practice",
                "Other",
            ],
            key=(
                f"session_first_type_"
                f"{selected_player.player_id}"
            ),
        )

        detail_columns = st.columns(
            2
        )

        new_location = detail_columns[0].text_input(
            "Location",
            placeholder="Optional",
            key=(
                f"session_first_location_"
                f"{selected_player.player_id}"
            ),
        )

        new_notes = detail_columns[1].text_input(
            "Notes",
            placeholder="Optional focus or context",
            key=(
                f"session_first_notes_"
                f"{selected_player.player_id}"
            ),
        )

        if st.button(
            "Create Session and Continue",
            type="primary",
            use_container_width=True,
            key=(
                f"session_first_create_session_"
                f"{selected_player.player_id}"
            ),
        ):
            try:
                created_session = session_repository.create_session(
                    player_id=selected_player.player_id,
                    session_name=new_session_name,
                    session_date=new_session_date,
                    session_type=new_session_type,
                    location=new_location,
                    notes=new_notes,
                )

                st.session_state[
                    "session_first_forced_session_id"
                ] = created_session.session_id

                st.success(
                    (
                        f"Session ready: "
                        f"{created_session.session_name}."
                    )
                )

                st.rerun()

            except Exception as error:
                st.error(
                    (
                        "Could not create session. "
                        f"{type(error).__name__}: {error}"
                    )
                )

    forced_session_id = st.session_state.get(
        "session_first_forced_session_id"
    )

    if forced_session_id is not None:
        try:
            forced_session = session_repository.get_session(
                forced_session_id
            )

            if (
                forced_session.player_id
                == selected_player.player_id
            ):
                selected_session = forced_session

                st.info(
                    (
                        f"Current session: "
                        f"{selected_session.session_date} · "
                        f"{selected_session.session_name}"
                    )
                )

                if st.button(
                    "Choose a Different Session",
                    use_container_width=True,
                    key="session_first_change_session",
                ):
                    st.session_state.pop(
                        "session_first_forced_session_id",
                        None,
                    )

                    st.rerun()

            else:
                st.session_state.pop(
                    "session_first_forced_session_id",
                    None,
                )

        except Exception:
            st.session_state.pop(
                "session_first_forced_session_id",
                None,
            )

    if selected_session is None:
        return

    # =====================================================
    # STEP 3 — UPLOAD AND AUTO-ASSIGN
    # =====================================================

    st.markdown(
        "### 3. Upload shots into this session"
    )

    st.success(
        (
            f"Destination: "
            f"{selected_player.display_name} → "
            f"{selected_session.session_name}"
        )
    )

    uploaded_files = st.file_uploader(
        "Drop one or many tracked-shot JSON files here",
        type=[
            "json",
        ],
        accept_multiple_files=True,
        key=(
            f"session_first_upload_"
            f"{selected_player.player_id}_"
            f"{selected_session.session_id}"
        ),
    )

    valid_preview = []
    preview_rows = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                parsed = json.loads(
                    uploaded_file.getvalue()
                )

                participant_id = str(
                    parsed.get(
                        "participant_id",
                        "",
                    )
                ).strip()

                matches_player = (
                    participant_id
                    == selected_player.participant_id
                )

                preview_rows.append(
                    {
                        "Filename": uploaded_file.name,
                        "Participant": participant_id or "Missing",
                        "Trial": parsed.get(
                            "trial_id",
                            "Missing",
                        ),
                        "Result": parsed.get(
                            "result",
                            "Unknown",
                        ),
                        "Frames": len(
                            parsed.get(
                                "tracking",
                                [],
                            )
                        ),
                        "Status": (
                            "Ready"
                            if matches_player
                            else (
                                "Rejected — belongs to "
                                f"{participant_id or 'unknown player'}"
                            )
                        ),
                    }
                )

                if matches_player:
                    valid_preview.append(
                        uploaded_file
                    )

            except Exception as error:
                preview_rows.append(
                    {
                        "Filename": uploaded_file.name,
                        "Participant": "Unknown",
                        "Trial": "Unknown",
                        "Result": "Unknown",
                        "Frames": 0,
                        "Status": (
                            f"Invalid JSON: {error}"
                        ),
                    }
                )

        st.dataframe(
            preview_rows,
            use_container_width=True,
            hide_index=True,
        )

    import_button = st.button(
        "Import and Add All Ready Shots to Session",
        type="primary",
        use_container_width=True,
        disabled=(
            not valid_preview
        ),
        key=(
            f"session_first_import_"
            f"{selected_player.player_id}_"
            f"{selected_session.session_id}"
        ),
    )

    if import_button:
        file_payloads = [
            (
                uploaded_file.name,
                uploaded_file.getvalue(),
            )
            for uploaded_file in valid_preview
        ]

        try:
            import_result = history_repository.import_json_files(
                json_files=file_payloads,
                selected_player_id=selected_player.player_id,
                enforce_selected_player=True,
            )

            # Resolve every uploaded file by content hash. This includes:
            # - newly imported records;
            # - restored soft-deleted records;
            # - active duplicates already present in player history.
            resolved_shot_ids = []

            for uploaded_file in valid_preview:
                file_hash = hashlib.sha256(
                    uploaded_file.getvalue()
                ).hexdigest()

                shot_record = (
                    history_repository
                    .get_shot_file_by_hash(
                        file_hash
                    )
                )

                if (
                    shot_record is not None
                    and shot_record.active
                    and shot_record.player_id
                    == selected_player.player_id
                ):
                    resolved_shot_ids.append(
                        shot_record.shot_file_id
                    )

            assigned_count = session_repository.assign_shots(
                session_id=selected_session.session_id,
                shot_file_ids=resolved_shot_ids,
            )

            result_columns = st.columns(
                4
            )

            result_columns[0].metric(
                "Selected",
                len(
                    valid_preview
                ),
            )

            result_columns[1].metric(
                "New or restored",
                import_result.imported_files,
            )

            result_columns[2].metric(
                "Existing duplicates",
                import_result.skipped_duplicates,
            )

            result_columns[3].metric(
                "Added to session",
                assigned_count,
            )

            st.success(
                (
                    f"{assigned_count} shot(s) are now assigned to "
                    f"{selected_session.session_name}."
                )
            )

            if import_result.failures:
                st.error(
                    "Some files could not be imported:"
                )

                for failure in import_result.failures:
                    st.write(
                        f"- {failure}"
                    )

            st.cache_data.clear()

        except Exception as error:
            st.error(
                (
                    "Could not import and assign the selected shots. "
                    f"{type(error).__name__}: {error}"
                )
            )

    # =====================================================
    # CURRENT SESSION
    # =====================================================

    st.markdown(
        "### Current session"
    )

    current_session_shots = (
        session_repository
        .list_session_shots(
            selected_session.session_id
        )
    )

    classified_count = sum(
        shot.recorded_result
        in {
            "made",
            "missed",
        }
        for shot in current_session_shots
    )

    made_count = sum(
        shot.recorded_result
        == "made"
        for shot in current_session_shots
    )

    make_percentage = (
        made_count
        / classified_count
        * 100.0
        if classified_count
        else 0.0
    )

    summary_columns = st.columns(
        4
    )

    summary_columns[0].metric(
        "Shots",
        len(
            current_session_shots
        ),
    )

    summary_columns[1].metric(
        "Made",
        made_count,
    )

    summary_columns[2].metric(
        "Missed",
        sum(
            shot.recorded_result
            == "missed"
            for shot in current_session_shots
        ),
    )

    summary_columns[3].metric(
        "Make %",
        (
            f"{make_percentage:.1f}%"
            if classified_count
            else "N/A"
        ),
    )

    if current_session_shots:
        st.dataframe(
            [
                {
                    "Order": index,
                    "Trial": shot.trial_id,
                    "Result": shot.recorded_result,
                    "Filename": shot.original_filename,
                    "Frames": shot.total_frames,
                }
                for index, shot in enumerate(
                    current_session_shots,
                    start=1,
                )
            ],
            use_container_width=True,
            hide_index=True,
        )

        with st.expander(
            "Move or remove shots",
            expanded=False,
        ):
            selected_manage_shots = st.multiselect(
                "Select shots",
                options=current_session_shots,
                format_func=session_shot_option_label,
                key=(
                    f"session_first_manage_shots_"
                    f"{selected_session.session_id}"
                ),
            )

            other_sessions = [
                session
                for session in session_repository.list_sessions(
                    player_id=selected_player.player_id,
                    include_inactive=False,
                )
                if session.session_id
                != selected_session.session_id
            ]

            manage_columns = st.columns(
                2
            )

            if manage_columns[0].button(
                "Remove from This Session",
                use_container_width=True,
                disabled=(
                    not selected_manage_shots
                ),
                key=(
                    f"session_first_remove_"
                    f"{selected_session.session_id}"
                ),
            ):
                try:
                    session_repository.remove_shots(
                        session_id=selected_session.session_id,
                        shot_file_ids=[
                            shot.shot_file_id
                            for shot in selected_manage_shots
                        ],
                    )

                    st.success(
                        (
                            "Selected shots were removed from this session. "
                            "They remain in the player's shot library."
                        )
                    )

                    st.cache_data.clear()
                    st.rerun()

                except Exception as error:
                    st.error(
                        (
                            "Could not remove shots. "
                            f"{type(error).__name__}: {error}"
                        )
                    )

            if other_sessions:
                destination_session = st.selectbox(
                    "Move selected shots to",
                    options=other_sessions,
                    format_func=session_option_label,
                    key=(
                        f"session_first_move_destination_"
                        f"{selected_session.session_id}"
                    ),
                )

                if manage_columns[1].button(
                    "Move to Selected Session",
                    use_container_width=True,
                    disabled=(
                        not selected_manage_shots
                    ),
                    key=(
                        f"session_first_move_"
                        f"{selected_session.session_id}"
                    ),
                ):
                    try:
                        session_repository.assign_shots(
                            session_id=destination_session.session_id,
                            shot_file_ids=[
                                shot.shot_file_id
                                for shot in selected_manage_shots
                            ],
                        )

                        st.success(
                            (
                                f"Moved {len(selected_manage_shots)} shot(s) "
                                f"to {destination_session.session_name}."
                            )
                        )

                        st.cache_data.clear()
                        st.rerun()

                    except Exception as error:
                        st.error(
                            (
                                "Could not move shots. "
                                f"{type(error).__name__}: {error}"
                            )
                        )

    else:
        st.info(
            "This session has no shots yet. Upload JSON files above."
        )

    with st.expander(
        "Advanced library and session administration",
        expanded=False,
    ):
        st.caption(
            (
                "Use these controls only for restoring removed files, "
                "permanent deletion, renaming profiles, or advanced session "
                "administration."
            )
        )

        render_player_history_manager()
        render_practice_session_manager()



@st.cache_data(
    show_spinner=False,
)
def load_session_analysis(
    session_id: int,
) -> tuple[
    dict,
    list[dict],
    list[dict],
]:
    """
    Run and cache the selected practice-session analysis.
    """

    history_repository = get_player_history_repository()

    engine = SessionAnalysisEngine(
        database_file=history_repository.database_file
    )

    (
        summary,
        shot_rows,
        category_summaries,
    ) = engine.run(
        session_id=session_id
    )

    return (
        asdict(
            summary
        ),
        [
            asdict(
                row
            )
            for row in shot_rows
        ],
        [
            asdict(
                row
            )
            for row in category_summaries
        ],
    )



@st.cache_data(
    show_spinner=False,
)
def load_session_coach_summary(
    session_id: int,
) -> tuple[
    dict,
    list[dict],
]:
    """
    Run and cache the coach-readable interpretation for one session.
    """

    history_repository = get_player_history_repository()

    analysis_engine = SessionAnalysisEngine(
        database_file=history_repository.database_file
    )

    coach_engine = SessionCoachSummaryEngine()

    summary, observations = coach_engine.run(
        session_id=session_id,
        analysis_engine=analysis_engine,
    )

    return (
        asdict(
            summary
        ),
        [
            asdict(
                observation
            )
            for observation in observations
        ],
    )



def analyze_managed_session_shot(
    shot_file_id: int,
    shooting_side: str,
) -> tuple[
    dict,
    dict,
    str,
]:
    """
    Load one managed JSON file from player history and run the complete
    single-shot analysis without requiring another manual upload.
    """

    history_repository = get_player_history_repository()

    shot_record = history_repository.get_shot_file(
        shot_file_id
    )

    stored_path = Path(
        shot_record.stored_path
    )

    if not stored_path.exists():
        raise FileNotFoundError(
            (
                "The stored JSON file could not be found: "
                f"{stored_path}"
            )
        )

    trial_data = json.loads(
        stored_path.read_text(
            encoding="utf-8"
        )
    )

    trial_data = validate_uploaded_json(
        trial_data
    )

    result = analyze_uploaded_trial(
        trial_data=trial_data,
        original_filename=(
            shot_record.original_filename
        ),
        shooting_side=shooting_side,
    )

    return (
        result,
        trial_data,
        shot_record.original_filename,
    )



@st.cache_data(
    show_spinner=False,
)
def generate_session_pdf_bytes(
    session_id: int,
) -> tuple[bytes, str]:
    """
    Generate and cache a PDF report for one practice session.
    """

    history_repository = get_player_history_repository()

    analysis_engine = SessionAnalysisEngine(
        database_file=history_repository.database_file
    )

    generator = SessionPDFReportGenerator()

    pdf_path = generator.run(
        session_id=session_id,
        analysis_engine=analysis_engine,
    )

    return (
        pdf_path.read_bytes(),
        pdf_path.name,
    )



@st.cache_data(
    show_spinner=False,
)
def load_session_grade_action_plan(
    session_id: int,
) -> tuple[
    dict,
    list[dict],
    list[dict],
]:
    """
    Run and cache the session grade and coach action plan.
    """

    history_repository = get_player_history_repository()

    analysis_engine = SessionAnalysisEngine(
        database_file=history_repository.database_file
    )

    grade_engine = SessionGradeActionPlanEngine()

    (
        summary,
        components,
        actions,
    ) = grade_engine.run(
        session_id=session_id,
        analysis_engine=analysis_engine,
    )

    return (
        asdict(
            summary
        ),
        [
            asdict(
                component
            )
            for component in components
        ],
        [
            asdict(
                action
            )
            for action in actions
        ],
    )


def render_session_analysis_workspace() -> None:
    quick_left, quick_middle, quick_right = st.columns(
        3
    )

    if quick_left.button(
        "Open Player Profile",
        use_container_width=True,
        key="session_analysis_open_player_profile",
    ):
        request_workspace_navigation(
            "Player Profile"
        )

    if quick_middle.button(
        "Compare Sessions",
        use_container_width=True,
        key="session_analysis_open_comparison",
    ):
        request_workspace_navigation(
            "Session Comparison"
        )

    if quick_right.button(
        "Open Reports",
        use_container_width=True,
        key="session_analysis_open_reports",
    ):
        request_workspace_navigation(
            "Reports Center"
        )

    """
    Coach-facing session review page.
    """

    history_repository = get_player_history_repository()
    session_repository = get_practice_session_repository()

    active_players = history_repository.list_players(
        include_inactive=False
    )

    if not active_players:
        st.info(
            "Create a player profile before analyzing a session."
        )

        return

    analysis_control_columns = st.columns(
        [
            0.72,
            0.28,
        ]
    )

    selected_player = analysis_control_columns[0].selectbox(
        "Player",
        options=active_players,
        format_func=player_option_label,
        key="session_analysis_player",
    )

    session_analysis_shooting_side = (
        analysis_control_columns[1].selectbox(
            "Shooting side",
            options=[
                "RIGHT",
                "LEFT",
            ],
            index=0,
            key="session_analysis_shooting_side",
        )
    )

    sessions = session_repository.list_sessions(
        player_id=selected_player.player_id,
        include_inactive=False,
    )

    if not sessions:
        st.info(
            "This player has no active practice sessions."
        )

        return

    selected_session = st.selectbox(
        "Practice session",
        options=sessions,
        format_func=session_option_label,
        key="session_analysis_session",
    )

    try:
        (
            summary,
            shot_rows,
            category_summaries,
        ) = load_session_analysis(
            selected_session.session_id
        )

    except Exception as error:
        st.error(
            (
                "Could not analyze the selected session. "
                f"{type(error).__name__}: {error}"
            )
        )

        return

    st.markdown(
        f"## {summary['session_name']}"
    )

    subtitle_parts = [
        summary[
            "player_display_name"
        ],
        summary[
            "session_date"
        ],
        summary[
            "session_type"
        ],
    ]

    if summary[
        "location"
    ]:
        subtitle_parts.append(
            summary[
                "location"
            ]
        )

    st.caption(
        " · ".join(
            subtitle_parts
        )
    )

    report_left, report_right = st.columns(
        [
            0.74,
            0.26,
        ]
    )

    report_left.info(
        (
            "Generate a branded PDF containing the session summary, "
            "coach interpretation, category consistency, and shot review queue."
        )
    )

    with report_right:
        generate_pdf_clicked = st.button(
            "Generate PDF Report",
            type="primary",
            use_container_width=True,
            key=(
                f"generate_session_pdf_"
                f"{selected_session.session_id}"
            ),
        )

    if generate_pdf_clicked:
        with st.spinner(
            "Generating professional session report..."
        ):
            try:
                pdf_bytes, pdf_filename = generate_session_pdf_bytes(
                    selected_session.session_id
                )

                st.session_state[
                    "generated_session_pdf_bytes"
                ] = pdf_bytes

                st.session_state[
                    "generated_session_pdf_filename"
                ] = pdf_filename

                st.session_state[
                    "generated_session_pdf_session_id"
                ] = selected_session.session_id

                st.success(
                    "PDF report generated successfully."
                )

            except Exception as error:
                st.error(
                    (
                        "Could not generate the PDF report. "
                        f"{type(error).__name__}: {error}"
                    )
                )

    if (
        st.session_state.get(
            "generated_session_pdf_session_id"
        )
        == selected_session.session_id
        and st.session_state.get(
            "generated_session_pdf_bytes"
        )
        is not None
    ):
        st.download_button(
            "Download Session PDF",
            data=st.session_state[
                "generated_session_pdf_bytes"
            ],
            file_name=st.session_state[
                "generated_session_pdf_filename"
            ],
            mime="application/pdf",
            use_container_width=True,
            key=(
                f"download_session_pdf_"
                f"{selected_session.session_id}"
            ),
        )

    metric_columns = st.columns(
        6
    )

    metric_columns[0].metric(
        "Assigned shots",
        summary[
            "assigned_shots"
        ],
    )

    metric_columns[1].metric(
        "Matched shots",
        summary[
            "matched_shots"
        ],
    )

    metric_columns[2].metric(
        "Make %",
        (
            f"{summary['make_percentage']:.1f}%"
        ),
    )

    metric_columns[3].metric(
        "Overall consistency",
        (
            "N/A"
            if summary[
                "overall_consistency_score"
            ]
            is None
            else (
                f"{summary['overall_consistency_score']:.1f}"
            )
        ),
    )

    metric_columns[4].metric(
        "Avg similarity",
        (
            "N/A"
            if summary[
                "average_similarity_score"
            ]
            is None
            else (
                f"{summary['average_similarity_score']:.1f}"
            )
        ),
    )

    metric_columns[5].metric(
        "Data status",
        summary[
            "data_status"
        ],
    )

    highlight_left, highlight_middle, highlight_right = st.columns(
        3
    )

    with highlight_left:
        st.markdown(
            f"""<div class="bms-summary-card">
<h3>Strongest category</h3>
<p>{summary['strongest_category']}</p>
</div>""",
            unsafe_allow_html=True,
        )

    with highlight_middle:
        st.markdown(
            f"""<div class="bms-summary-card">
<h3>Weakest category</h3>
<p>{summary['weakest_category']}</p>
</div>""",
            unsafe_allow_html=True,
        )

    with highlight_right:
        st.markdown(
            f"""<div class="bms-summary-card">
<h3>Highest-attention shot</h3>
<p>{summary['highest_attention_trial']}</p>
</div>""",
            unsafe_allow_html=True,
        )

    try:
        (
            grade_summary,
            grade_components,
            grade_actions,
        ) = load_session_grade_action_plan(
            selected_session.session_id
        )

    except Exception as error:
        grade_summary = None
        grade_components = []
        grade_actions = []

        st.warning(
            (
                "Session grade and action plan could not be generated. "
                f"{type(error).__name__}: {error}"
            )
        )

    if grade_summary is not None:
        try:
            insight_object = build_session_report_insights(
                session_summary=type(
                    "SessionSummaryView",
                    (),
                    summary,
                )(),
                shot_rows=[
                    type(
                        "ShotView",
                        (),
                        row,
                    )()
                    for row in shot_rows
                ],
                category_summaries=[
                    type(
                        "CategoryView",
                        (),
                        row,
                    )()
                    for row in category_summaries
                ],
                grade_summary=type(
                    "GradeSummaryView",
                    (),
                    grade_summary,
                )(),
                grade_components=[
                    type(
                        "GradeComponentView",
                        (),
                        row,
                    )()
                    for row in grade_components
                ],
                action_items=[
                    type(
                        "ActionView",
                        (),
                        row,
                    )()
                    for row in grade_actions
                ],
            )

        except Exception:
            insight_object = None

        st.markdown(
            "### Session Grade"
        )

        grade_left, grade_right = st.columns(
            [
                0.32,
                0.68,
            ]
        )

        with grade_left:
            st.markdown(
                f"""<div style="background:#ffffff;border:1px solid #e4e8f0;border-radius:22px;padding:1.4rem;text-align:center;box-shadow:0 8px 22px rgba(28,39,60,0.04);">
<div style="color:#c94d18;font-size:0.75rem;font-weight:800;letter-spacing:0.12em;text-transform:uppercase;">Overall Session Grade</div>
<div style="font-size:3.4rem;font-weight:900;color:#152033;line-height:1.1;margin-top:0.5rem;">{grade_summary['overall_grade']:.1f}</div>
<div style="color:#455066;font-weight:700;">OUT OF 100</div>
<div style="font-size:1.05rem;font-weight:800;color:#152033;margin-top:0.8rem;">{grade_summary['grade_label']}</div>
<div style="color:#6b7486;margin-top:0.35rem;">Confidence: {grade_summary['grade_confidence'].title()}</div>
</div>""",
                unsafe_allow_html=True,
            )

        with grade_right:
            component_columns = st.columns(
                2
            )

            for index, component in enumerate(
                grade_components
            ):
                with component_columns[
                    index
                    % 2
                ]:
                    st.metric(
                        component[
                            "component"
                        ],
                        (
                            f"{component['score']:.1f}"
                        ),
                    )

                    st.caption(
                        (
                            f"{component['status']} · "
                            f"Weight {component['weight']:.0%}"
                        )
                    )

            st.info(
                (
                    f"Primary focus: {grade_summary['primary_focus']} "
                    f"| Recommended drill: "
                    f"{grade_summary['recommended_drill']} "
                    f"| Estimated review time: "
                    f"{grade_summary['estimated_review_minutes']} minutes"
                )
            )

        if insight_object is not None:
            st.markdown(
                "### Why This Session Received Its Grade"
            )

            st.write(
                insight_object.grade_explanation
            )

            story_left, story_right = st.columns(
                2
            )

            with story_left:
                st.markdown(
                    """<div class="bms-summary-card">
<h3>What went well</h3>
</div>""",
                    unsafe_allow_html=True,
                )

                for strength in insight_object.what_went_well:
                    st.write(
                        f"✓ {strength}"
                    )

            with story_right:
                st.markdown(
                    """<div class="bms-summary-card">
<h3>Primary coaching focus</h3>
</div>""",
                    unsafe_allow_html=True,
                )

                st.write(
                    insight_object.primary_focus
                )

                st.caption(
                    insight_object.next_practice_summary
                )

            st.markdown(
                "### Session Story"
            )

            st.info(
                insight_object.session_story
            )

            st.markdown(
                "### Biggest Grade Contributors"
            )

            contributor_columns = st.columns(
                min(
                    5,
                    len(
                        insight_object.contributors
                    ),
                )
                or 1
            )

            for index, contributor in enumerate(
                insight_object.contributors
            ):
                with contributor_columns[
                    index
                    % len(
                        contributor_columns
                    )
                ]:
                    direction_symbol = (
                        "▲"
                        if contributor.direction
                        == "up"
                        else "▼"
                    )

                    st.metric(
                        contributor.label,
                        (
                            f"{direction_symbol} "
                            f"{contributor.impact_points:+.1f}"
                        ),
                    )

                    st.caption(
                        contributor.explanation
                    )

            st.markdown(
                "### Category Consistency Visual"
            )

            for category in category_summaries:
                st.write(
                    (
                        f"**{category['category']}** — "
                        f"{category['average_consistency_score']:.1f}/100"
                    )
                )

                st.progress(
                    float(
                        category[
                            "average_consistency_score"
                        ]
                    )
                    / 100.0
                )

        st.markdown(
            "### Coach Action Plan"
        )

        for action in grade_actions:
            with st.container(
                border=True
            ):
                action_left, action_right = st.columns(
                    [
                        0.78,
                        0.22,
                    ]
                )

                with action_left:
                    st.markdown(
                        (
                            f"### {action['rank']}. "
                            f"{action['title']}"
                        )
                    )

                    st.write(
                        action[
                            "instruction"
                        ]
                    )

                    if action[
                        "related_trials"
                    ]:
                        st.caption(
                            (
                                "Related trials: "
                                f"{action['related_trials']}"
                            )
                        )

                with action_right:
                    st.metric(
                        "Priority",
                        action[
                            "priority"
                        ].title(),
                    )

                    st.caption(
                        (
                            f"{action['estimated_minutes']} min"
                        )
                    )

    try:
        coach_summary, coach_observations = (
            load_session_coach_summary(
                selected_session.session_id
            )
        )

    except Exception as error:
        coach_summary = None
        coach_observations = []

        st.warning(
            (
                "Coach interpretation could not be generated. "
                f"{type(error).__name__}: {error}"
            )
        )

    if coach_summary is not None:
        st.markdown(
            "### Coach Interpretation"
        )

        st.markdown(
            f"""<div style="background:#ffffff;border:1px solid #e4e8f0;border-radius:20px;padding:1.25rem 1.35rem;margin:0.7rem 0 1.2rem;box-shadow:0 8px 22px rgba(28,39,60,0.04);">
<div style="color:#c94d18;font-size:0.74rem;font-weight:800;letter-spacing:0.11em;text-transform:uppercase;margin-bottom:0.45rem;">Executive summary</div>
<p style="margin:0;color:#293548;line-height:1.72;font-size:1rem;">{coach_summary['executive_summary']}</p>
</div>""",
            unsafe_allow_html=True,
        )

        coach_metric_columns = st.columns(
            4
        )

        coach_metric_columns[0].metric(
            "Session status",
            coach_summary[
                "overall_status"
            ],
        )

        coach_metric_columns[1].metric(
            "Evidence",
            coach_summary[
                "evidence_confidence"
            ].title(),
        )

        coach_metric_columns[2].metric(
            "Main review area",
            coach_summary[
                "main_review_area"
            ],
        )

        coach_metric_columns[3].metric(
            "Review first",
            coach_summary[
                "highest_attention_trial"
            ],
        )

        if coach_observations:
            for observation in coach_observations:
                priority = observation[
                    "priority"
                ].title()

                with st.container(
                    border=True
                ):
                    heading_left, heading_right = st.columns(
                        [
                            0.78,
                            0.22,
                        ]
                    )

                    with heading_left:
                        st.markdown(
                            (
                                f"### {observation['rank']}. "
                                f"{observation['title']}"
                            )
                        )

                    with heading_right:
                        st.metric(
                            "Priority",
                            priority,
                        )

                    st.write(
                        observation[
                            "summary"
                        ]
                    )

                    evidence_column, focus_column = st.columns(
                        2
                    )

                    with evidence_column:
                        st.markdown(
                            "**Evidence**"
                        )

                        st.write(
                            observation[
                                "evidence"
                            ]
                        )

                    with focus_column:
                        st.markdown(
                            "**Coaching focus**"
                        )

                        st.write(
                            observation[
                                "coaching_focus"
                            ]
                        )

    if not shot_rows:
        st.info(
            "This session has no active shots."
        )

        return

    st.markdown(
        "### Session progression"
    )

    shot_numbers = [
        row[
            "shot_number"
        ]
        for row in shot_rows
    ]

    similarity_values = [
        row[
            "overall_similarity_score"
        ]
        for row in shot_rows
    ]

    attention_values = [
        row[
            "shot_attention_score"
        ]
        for row in shot_rows
    ]

    progression_figure, progression_axis = plt.subplots(
        figsize=(
            12,
            5.0,
        )
    )

    valid_similarity_x = [
        number
        for number, value in zip(
            shot_numbers,
            similarity_values,
        )
        if value is not None
    ]

    valid_similarity_y = [
        value
        for value in similarity_values
        if value is not None
    ]

    if valid_similarity_x:
        progression_axis.plot(
            valid_similarity_x,
            valid_similarity_y,
            marker="o",
            linewidth=2.0,
            label="Similarity score",
        )

    progression_axis.plot(
        shot_numbers,
        attention_values,
        marker="o",
        linewidth=2.0,
        label="Attention score",
    )

    progression_axis.set_ylim(
        0,
        100,
    )

    progression_axis.set_xlabel(
        "Shot number"
    )

    progression_axis.set_ylabel(
        "Score"
    )

    progression_axis.set_title(
        "Shot-by-shot session progression"
    )

    progression_axis.grid(
        visible=True,
        alpha=0.25,
    )

    progression_axis.legend()

    progression_figure.tight_layout()

    st.pyplot(
        progression_figure,
        clear_figure=True,
        use_container_width=True,
    )

    st.markdown(
        "### Category consistency"
    )

    if category_summaries:
        category_labels = [
            row[
                "category"
            ]
            for row in category_summaries
        ]

        category_values = [
            row[
                "average_consistency_score"
            ]
            for row in category_summaries
        ]

        category_figure, category_axis = plt.subplots(
            figsize=(
                10,
                4.8,
            )
        )

        positions = list(
            range(
                len(
                    category_labels
                )
            )
        )

        category_axis.bar(
            positions,
            category_values,
        )

        category_axis.set_xticks(
            positions,
            labels=category_labels,
            rotation=20,
            ha="right",
        )

        category_axis.set_ylim(
            0,
            100,
        )

        category_axis.set_ylabel(
            "Consistency score"
        )

        category_axis.set_title(
            "Session consistency by category"
        )

        category_axis.grid(
            visible=True,
            axis="y",
            alpha=0.25,
        )

        category_figure.tight_layout()

        st.pyplot(
            category_figure,
            clear_figure=True,
            use_container_width=True,
        )

        st.dataframe(
            [
                {
                    "Category": row[
                        "category"
                    ],
                    "Consistency": round(
                        row[
                            "average_consistency_score"
                        ],
                        1,
                    ),
                    "Features": row[
                        "features_analyzed"
                    ],
                    "Strongest feature": row[
                        "strongest_feature"
                    ],
                    "Weakest feature": row[
                        "weakest_feature"
                    ],
                }
                for row in category_summaries
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        "### Shot browser"
    )

    browser_rows = []

    for row in shot_rows:
        browser_rows.append(
            {
                "Shot": row[
                    "shot_number"
                ],
                "Trial": row[
                    "trial_id"
                ],
                "Result": row[
                    "result"
                ],
                "Similarity": (
                    None
                    if row[
                        "overall_similarity_score"
                    ]
                    is None
                    else round(
                        row[
                            "overall_similarity_score"
                        ],
                        1,
                    )
                ),
                "Adjusted": (
                    None
                    if row[
                        "confidence_adjusted_score"
                    ]
                    is None
                    else round(
                        row[
                            "confidence_adjusted_score"
                        ],
                        1,
                    )
                ),
                "Attention": round(
                    row[
                        "shot_attention_score"
                    ],
                    1,
                ),
                "Release Angle": row[
                    "release_angle_deg"
                ],
                "Ball Speed": row[
                    "release_ball_speed_ft_s"
                ],
                "Release Height": row[
                    "release_height_ft"
                ],
                "Knee→Elbow": row[
                    "knee_to_elbow_gap_ms"
                ],
                "Elbow→Release": row[
                    "elbow_to_release_gap_ms"
                ],
                "Takeoff→Release": row[
                    "takeoff_to_release_ms"
                ],
                "Status": row[
                    "shot_status"
                ],
            }
        )

    st.dataframe(
        browser_rows,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### Review queue"
    )

    review_rows = sorted(
        shot_rows,
        key=lambda row: (
            -row[
                "shot_attention_score"
            ],
            row[
                "shot_number"
            ],
        )
    )

    review_count = st.selectbox(
        "Shots to show",
        options=[
            5,
            10,
            15,
            len(
                review_rows
            ),
        ],
        index=0,
        key="session_analysis_review_count",
    )

    for row in review_rows[
        :int(
            review_count
        )
    ]:
        with st.container(
            border=True
        ):
            left, right = st.columns(
                [
                    0.78,
                    0.22,
                ]
            )

            with left:
                st.markdown(
                    (
                        f"### Shot {row['shot_number']} · "
                        f"{row['trial_id']}"
                    )
                )

                st.caption(
                    (
                        f"{row['result'].title()} · "
                        f"{row['original_filename']} · "
                        f"{row['shot_status']}"
                    )
                )

                if insight_object is not None:
                    matching_reason = next(
                        (
                            reason
                            for reason in insight_object.shot_reasons
                            if reason.trial_id
                            == row[
                                "trial_id"
                            ]
                        ),
                        None,
                    )

                    if matching_reason is not None:
                        st.markdown(
                            (
                                f"**Why review:** "
                                f"{matching_reason.reason}"
                            )
                        )

                        st.caption(
                            matching_reason.detail
                        )

            with right:
                st.metric(
                    "Attention",
                    (
                        f"{row['shot_attention_score']:.1f}"
                    ),
                )

            detail_columns = st.columns(
                4
            )

            detail_columns[0].metric(
                "Similarity",
                (
                    "N/A"
                    if row[
                        "overall_similarity_score"
                    ]
                    is None
                    else (
                        f"{row['overall_similarity_score']:.1f}"
                    )
                ),
            )

            detail_columns[1].metric(
                "Release angle",
                (
                    "N/A"
                    if row[
                        "release_angle_deg"
                    ]
                    is None
                    else (
                        f"{row['release_angle_deg']:.1f}°"
                    )
                ),
            )

            detail_columns[2].metric(
                "Ball speed",
                (
                    "N/A"
                    if row[
                        "release_ball_speed_ft_s"
                    ]
                    is None
                    else (
                        f"{row['release_ball_speed_ft_s']:.2f}"
                    )
                ),
            )

            detail_columns[3].metric(
                "Takeoff→release",
                (
                    "N/A"
                    if row[
                        "takeoff_to_release_ms"
                    ]
                    is None
                    else (
                        f"{row['takeoff_to_release_ms']:.0f} ms"
                    )
                ),
            )

            if st.button(
                "Open Complete Shot Analysis",
                type="primary",
                use_container_width=True,
                key=(
                    f"session_open_shot_"
                    f"{selected_session.session_id}_"
                    f"{row['shot_file_id']}"
                ),
            ):
                with st.spinner(
                    (
                        f"Analyzing {row['trial_id']} from the "
                        "managed session library..."
                    )
                ):
                    try:
                        (
                            opened_result,
                            opened_trial_data,
                            opened_filename,
                        ) = analyze_managed_session_shot(
                            shot_file_id=row[
                                "shot_file_id"
                            ],
                            shooting_side=(
                                session_analysis_shooting_side
                            ),
                        )

                        st.session_state[
                            "session_opened_shot_result"
                        ] = opened_result

                        st.session_state[
                            "session_opened_shot_data"
                        ] = opened_trial_data

                        st.session_state[
                            "session_opened_shot_filename"
                        ] = opened_filename

                        st.session_state[
                            "session_opened_shot_id"
                        ] = row[
                            "shot_file_id"
                        ]

                        st.success(
                            (
                                f"{row['trial_id']} is ready below."
                            )
                        )

                    except Exception as error:
                        st.error(
                            (
                                "Could not open this stored shot. "
                                f"{type(error).__name__}: {error}"
                            )
                        )

    if (
        st.session_state.session_opened_shot_result
        is not None
    ):
        st.markdown(
            "---"
        )

        st.markdown(
            "## Opened Shot Analysis"
        )

        opened_trial_id = (
            st.session_state
            .session_opened_shot_data
            .get(
                "trial_id",
                "Selected shot",
            )
        )

        opened_header_left, opened_header_right = st.columns(
            [
                0.78,
                0.22,
            ]
        )

        opened_header_left.info(
            (
                f"Viewing {opened_trial_id} from "
                f"{selected_session.session_name}."
            )
        )

        if opened_header_right.button(
            "Close Shot",
            use_container_width=True,
            key="session_close_opened_shot",
        ):
            st.session_state.session_opened_shot_result = None
            st.session_state.session_opened_shot_data = None
            st.session_state.session_opened_shot_filename = None
            st.session_state.session_opened_shot_id = None

            st.rerun()

        display_results(
            result=(
                st.session_state
                .session_opened_shot_result
            ),
            uploaded_name=(
                st.session_state
                .session_opened_shot_filename
            ),
            trial_data=(
                st.session_state
                .session_opened_shot_data
            ),
            shooting_side=(
                session_analysis_shooting_side
            ),
        )

    st.caption(
        (
            "The session attention score is a review-ordering tool. "
            "It is not a player talent grade."
        )
    )




@st.cache_data(
    show_spinner=False,
)
def generate_player_development_pdf_bytes(
    player_id: int,
    participant_id: str,
    player_display_name: str,
) -> tuple[
    bytes,
    str,
]:
    """
    Generate and cache one player-development PDF.
    """

    history_repository = get_player_history_repository()

    timeline_engine = PlayerDevelopmentTimelineEngine(
        database_file=history_repository.database_file
    )

    generator = PlayerDevelopmentPDFReportGenerator()

    pdf_path = generator.run(
        player_id=player_id,
        participant_id=participant_id,
        player_display_name=player_display_name,
        timeline_engine=timeline_engine,
    )

    return (
        pdf_path.read_bytes(),
        pdf_path.name,
    )





def source_for_launch_check() -> str:
    """Return the active application source for a lightweight content check."""

    try:
        return Path(
            __file__
        ).read_text(
            encoding="utf-8"
        )

    except Exception:
        return ""


def build_launch_readiness_checks() -> list[dict]:
    """
    Run non-destructive checks for the publishable Version 1 workflow.
    """

    checks = []

    def add_check(
        category: str,
        name: str,
        passed: bool,
        detail: str,
        required: bool = True,
    ) -> None:
        checks.append(
            {
                "category": category,
                "name": name,
                "passed": bool(
                    passed
                ),
                "detail": detail,
                "required": bool(
                    required
                ),
            }
        )

    project_root = Path(
        __file__
    ).resolve().parents[
        2
    ]

    required_modules = [
        (
            "Session analysis engine",
            "session_analysis_engine.py",
        ),
        (
            "Session grade engine",
            "session_grade_action_plan_engine.py",
        ),
        (
            "Coach intelligence",
            "session_coach_summary_engine.py",
        ),
        (
            "Session PDF generator",
            "session_pdf_report_generator.py",
        ),
        (
            "Player development PDF generator",
            "player_development_pdf_report_generator.py",
        ),
        (
            "Team practice PDF generator",
            "team_practice_pdf_report_generator.py",
        ),
        (
            "Session comparison PDF generator",
            "session_comparison_pdf_report_generator.py",
        ),
    ]

    research_folder = (
        project_root
        / "tracking_app"
        / "research"
    )

    for module_name, filename in required_modules:
        module_path = (
            research_folder
            / filename
        )

        add_check(
            category="Core modules",
            name=module_name,
            passed=module_path.exists(),
            detail=str(
                module_path
            ),
        )

    package_checks = [
        (
            "Streamlit",
            "streamlit",
        ),
        (
            "NumPy",
            "numpy",
        ),
        (
            "Matplotlib",
            "matplotlib",
        ),
        (
            "ImageIO",
            "imageio",
        ),
        (
            "ImageIO FFmpeg",
            "imageio_ffmpeg",
        ),
        (
            "ReportLab",
            "reportlab",
        ),
    ]

    for display_name, module_name in package_checks:
        try:
            module = __import__(
                module_name
            )

            version = getattr(
                module,
                "__version__",
                "installed",
            )

            add_check(
                category="Python packages",
                name=display_name,
                passed=True,
                detail=f"Version: {version}",
            )

        except Exception as error:
            add_check(
                category="Python packages",
                name=display_name,
                passed=False,
                detail=(
                    f"{type(error).__name__}: {error}"
                ),
            )

    try:
        history_repository = get_player_history_repository()

        database_file = Path(
            history_repository.database_file
        )

        add_check(
            category="Data repository",
            name="Player-history database",
            passed=database_file.exists(),
            detail=str(
                database_file
            ),
        )

        players = history_repository.list_players(
            include_inactive=False
        )

        add_check(
            category="Demo data",
            name="At least one active player",
            passed=len(
                players
            )
            > 0,
            detail=(
                f"Active players: {len(players)}"
            ),
        )

    except Exception as error:
        players = []

        add_check(
            category="Data repository",
            name="Player-history repository",
            passed=False,
            detail=(
                f"{type(error).__name__}: {error}"
            ),
        )

    sessions_with_shots = []

    try:
        session_repository = get_practice_session_repository()

        active_sessions = []

        for player in players:
            sessions = session_repository.list_sessions(
                player_id=player.player_id,
                include_inactive=False,
            )

            active_sessions.extend(
                sessions
            )

            for session in sessions:
                assigned = session_repository.list_session_shots(
                    session.session_id
                )

                if assigned:
                    sessions_with_shots.append(
                        session
                    )

        add_check(
            category="Demo data",
            name="At least one active session",
            passed=len(
                active_sessions
            )
            > 0,
            detail=(
                f"Active sessions: {len(active_sessions)}"
            ),
        )

        add_check(
            category="Demo data",
            name="At least one session with shots",
            passed=len(
                sessions_with_shots
            )
            > 0,
            detail=(
                f"Sessions containing shots: "
                f"{len(sessions_with_shots)}"
            ),
        )

    except Exception as error:
        add_check(
            category="Demo data",
            name="Practice-session repository",
            passed=False,
            detail=(
                f"{type(error).__name__}: {error}"
            ),
        )

    try:
        organization_repository = get_organization_team_repository()

        active_teams = organization_repository.list_teams(
            include_inactive=False
        )

        add_check(
            category="Demo data",
            name="At least one active team",
            passed=len(
                active_teams
            )
            > 0,
            detail=(
                f"Active teams: {len(active_teams)}"
            ),
            required=False,
        )

    except Exception as error:
        add_check(
            category="Demo data",
            name="Team repository",
            passed=False,
            detail=(
                f"{type(error).__name__}: {error}"
            ),
            required=False,
        )

    if sessions_with_shots:
        test_session = sessions_with_shots[
            0
        ]

        try:
            session_bytes, session_name = (
                generate_session_pdf_bytes(
                    test_session.session_id
                )
            )

            add_check(
                category="Report generation",
                name="Session PDF smoke test",
                passed=len(
                    session_bytes
                )
                > 1000,
                detail=(
                    f"{session_name} · "
                    f"{len(session_bytes):,} bytes"
                ),
            )

        except Exception as error:
            add_check(
                category="Report generation",
                name="Session PDF smoke test",
                passed=False,
                detail=(
                    f"{type(error).__name__}: {error}"
                ),
            )

    else:
        add_check(
            category="Report generation",
            name="Session PDF smoke test",
            passed=False,
            detail=(
                "No session with assigned shots is available."
            ),
        )

    active_source = source_for_launch_check().casefold()

    add_check(
        category="Branding",
        name="Platform title",
        passed=(
            "ankit's free throw analysis software"
            in active_source
        ),
        detail=(
            "ANKIT'S FREE THROW ANALYSIS SOFTWARE"
        ),
    )

    add_check(
        category="Branding",
        name="Creator contact",
        passed=(
            "ankitwadera2@gmail.com"
            in active_source
        ),
        detail=(
            "Ankit Wadera · ankitwadera2@gmail.com"
        ),
    )

    add_check(
        category="Responsible communication",
        name="Exploratory-analysis disclaimer",
        passed=(
            "exploratory"
            in active_source
        ),
        detail=(
            "The interface includes responsible interpretation language."
        ),
    )

    return checks


def render_launch_readiness_workspace() -> None:
    """
    Automated publisher-readiness dashboard.
    """

    st.markdown(
        """<div style="background:linear-gradient(135deg,#152033 0%,#25334d 100%);border-radius:24px;padding:1.55rem 1.7rem;margin:0.5rem 0 1.2rem;color:white;box-shadow:0 16px 34px rgba(21,32,51,0.18);">
<div style="font-size:0.74rem;font-weight:800;letter-spacing:0.13em;text-transform:uppercase;color:#f1a077;margin-bottom:0.45rem;">Version 1.0 release control</div>
<h2 style="margin:0 0 0.55rem;color:white;">Launch Readiness</h2>
<p style="margin:0;color:#dce3ef;line-height:1.7;max-width:900px;">Run automated, non-destructive checks before sharing the platform with coaches, performance staff, or front-office personnel.</p>
</div>""",
        unsafe_allow_html=True,
    )

    control_left, control_right = st.columns(
        [
            0.78,
            0.22,
        ]
    )

    control_left.info(
        (
            "Checks cover core modules, Python packages, demo data, report "
            "generation, branding, and responsible interpretation language."
        )
    )

    if control_right.button(
        "Run Launch Checks",
        type="primary",
        use_container_width=True,
        key="run_launch_readiness_checks",
    ):
        with st.spinner(
            "Running launch-readiness checks..."
        ):
            st.session_state[
                "launch_readiness_checks"
            ] = build_launch_readiness_checks()

            st.session_state[
                "launch_readiness_last_run"
            ] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

    checks = st.session_state.get(
        "launch_readiness_checks"
    )

    if not checks:
        st.info(
            "Run the checks to generate the Version 1 release assessment."
        )

        return

    required_checks = [
        check
        for check in checks
        if check[
            "required"
        ]
    ]

    optional_checks = [
        check
        for check in checks
        if not check[
            "required"
        ]
    ]

    passed_required = sum(
        check[
            "passed"
        ]
        for check in required_checks
    )

    failed_required = len(
        required_checks
    ) - passed_required

    passed_optional = sum(
        check[
            "passed"
        ]
        for check in optional_checks
    )

    readiness_percentage = (
        passed_required
        / len(
            required_checks
        )
        * 100.0
        if required_checks
        else 0.0
    )

    if failed_required == 0:
        release_status = "READY FOR FINAL DEMO REVIEW"

    elif failed_required <= 2:
        release_status = "NEARLY READY"

    else:
        release_status = "NOT READY"

    metric_columns = st.columns(
        5
    )

    metric_columns[0].metric(
        "Readiness",
        f"{readiness_percentage:.0f}%",
    )

    metric_columns[1].metric(
        "Required passed",
        (
            f"{passed_required} / "
            f"{len(required_checks)}"
        ),
    )

    metric_columns[2].metric(
        "Required failed",
        failed_required,
    )

    metric_columns[3].metric(
        "Optional passed",
        (
            f"{passed_optional} / "
            f"{len(optional_checks)}"
        ),
    )

    metric_columns[4].metric(
        "Status",
        release_status,
    )

    st.progress(
        readiness_percentage
        / 100.0
    )

    st.caption(
        (
            "Last run: "
            f"{st.session_state.get('launch_readiness_last_run', 'Unknown')}"
        )
    )

    if failed_required == 0:
        st.success(
            (
                "All required automated checks passed. The platform is ready "
                "for a final visual review and controlled public demonstration."
            )
        )

    else:
        st.warning(
            (
                f"{failed_required} required check(s) still need attention "
                "before publishing."
            )
        )

    categories = []

    for check in checks:
        if check[
            "category"
        ] not in categories:
            categories.append(
                check[
                    "category"
                ]
            )

    for category in categories:
        st.markdown(
            f"### {category}"
        )

        category_checks = [
            check
            for check in checks
            if check[
                "category"
            ]
            == category
        ]

        for check in category_checks:
            status_icon = (
                "✅"
                if check[
                    "passed"
                ]
                else (
                    "⚠️"
                    if not check[
                        "required"
                    ]
                    else "❌"
                )
            )

            requirement_text = (
                "Required"
                if check[
                    "required"
                ]
                else "Optional"
            )

            with st.container(
                border=True
            ):
                check_left, check_right = st.columns(
                    [
                        0.82,
                        0.18,
                    ]
                )

                with check_left:
                    st.markdown(
                        (
                            f"### {status_icon} "
                            f"{check['name']}"
                        )
                    )

                    st.caption(
                        check[
                            "detail"
                        ]
                    )

                with check_right:
                    st.metric(
                        "Type",
                        requirement_text,
                    )

    st.markdown(
        "### Manual Final Review"
    )

    manual_items = [
        "Open Executive Demo and confirm the page is visually clean.",
        "Play at least one smooth tracking video from start to finish.",
        "Open one complete stored-shot analysis from Session Analysis.",
        "Generate and open all four PDF report types.",
        "Confirm no private player information will be shared publicly.",
        "Capture final screenshots or a short demonstration video.",
        "Test the exact launch command from a fresh PowerShell window.",
    ]

    for index, item in enumerate(
        manual_items
    ):
        st.checkbox(
            item,
            key=(
                f"launch_manual_{index}"
            ),
        )

    failed_rows = [
        {
            "Category": check[
                "category"
            ],
            "Check": check[
                "name"
            ],
            "Required": check[
                "required"
            ],
            "Detail": check[
                "detail"
            ],
        }
        for check in checks
        if not check[
            "passed"
        ]
    ]

    if failed_rows:
        st.markdown(
            "### Items Requiring Attention"
        )

        st.dataframe(
            failed_rows,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        """<div style="background:#fff7f2;border:1px solid #f1c7b3;border-radius:18px;padding:1rem 1.15rem;margin-top:1rem;">
<strong style="color:#9d3f17;">Release rule</strong>
<p style="margin:0.35rem 0 0;color:#5d4a42;line-height:1.6;">Automated checks passing does not guarantee that the public presentation is perfect. Publish only after the manual visual review, privacy review, and demonstration walkthrough are complete.</p>
</div>""",
        unsafe_allow_html=True,
    )


def render_global_search_workspace() -> None:
    """
    Search players, sessions, teams, and active tracked-shot records.

    This workspace is read-only. It provides quick summaries and direct
    navigation into the focused workspaces without exposing unrelated tools.
    """

    history_repository = get_player_history_repository()
    session_repository = get_practice_session_repository()
    organization_repository = get_organization_team_repository()

    st.markdown(
        """<div style="background:#ffffff;border:1px solid #e4e8f0;border-radius:22px;padding:1.3rem 1.45rem;margin:0.4rem 0 1.1rem;box-shadow:0 8px 24px rgba(28,39,60,0.04);">
<div style="color:#c94d18;font-size:0.74rem;font-weight:800;letter-spacing:0.11em;text-transform:uppercase;margin-bottom:0.45rem;">Universal navigation</div>
<h2 style="margin:0 0 0.5rem;color:#152033;">Find anything in the platform</h2>
<p style="margin:0;color:#455066;line-height:1.65;">Search by player name, participant ID, session name, session date, team, organization, trial ID, or original JSON filename.</p>
</div>""",
        unsafe_allow_html=True,
    )

    query = st.text_input(
        "Search",
        placeholder=(
            "Examples: P0001, July 15, T0013, Raptors 905, player name..."
        ),
        key="global_platform_search",
    ).strip()

    players = history_repository.list_players(
        include_inactive=False
    )

    teams = organization_repository.list_teams(
        include_inactive=False
    )

    player_results = []
    session_results = []
    shot_results = []
    team_results = []

    normalized_query = query.casefold()

    for player in players:
        player_text = " ".join(
            [
                str(
                    player.display_name
                ),
                str(
                    player.participant_id
                ),
            ]
        ).casefold()

        if (
            not normalized_query
            or normalized_query in player_text
        ):
            player_shots = history_repository.list_shot_files(
                player_id=player.player_id,
                include_inactive=False,
            )

            player_sessions = session_repository.list_sessions(
                player_id=player.player_id,
                include_inactive=False,
            )

            made = sum(
                shot.recorded_result == "made"
                for shot in player_shots
            )

            missed = sum(
                shot.recorded_result == "missed"
                for shot in player_shots
            )

            classified = made + missed

            player_results.append(
                {
                    "record": player,
                    "shots": len(
                        player_shots
                    ),
                    "sessions": len(
                        player_sessions
                    ),
                    "make_percentage": (
                        made
                        / classified
                        * 100.0
                        if classified
                        else None
                    ),
                }
            )

        player_sessions = session_repository.list_sessions(
            player_id=player.player_id,
            include_inactive=False,
        )

        for session in player_sessions:
            session_text = " ".join(
                [
                    str(
                        session.session_name
                    ),
                    str(
                        session.session_date
                    ),
                    str(
                        session.session_type
                    ),
                    str(
                        session.location
                    ),
                    str(
                        player.display_name
                    ),
                    str(
                        player.participant_id
                    ),
                ]
            ).casefold()

            if (
                not normalized_query
                or normalized_query in session_text
            ):
                assigned_shots = session_repository.list_session_shots(
                    session.session_id
                )

                session_results.append(
                    {
                        "player": player,
                        "session": session,
                        "shots": len(
                            assigned_shots
                        ),
                    }
                )

        player_shots = history_repository.list_shot_files(
            player_id=player.player_id,
            include_inactive=False,
        )

        for shot in player_shots:
            shot_text = " ".join(
                [
                    str(
                        shot.trial_id
                    ),
                    str(
                        shot.original_filename
                    ),
                    str(
                        shot.recorded_result
                    ),
                    str(
                        player.display_name
                    ),
                    str(
                        player.participant_id
                    ),
                ]
            ).casefold()

            if (
                normalized_query
                and normalized_query in shot_text
            ):
                shot_results.append(
                    {
                        "player": player,
                        "shot": shot,
                    }
                )

    for team in teams:
        team_text = " ".join(
            [
                str(
                    team.team_name
                ),
                str(
                    team.organization_name
                ),
                str(
                    team.season_name
                ),
                str(
                    team.level
                ),
            ]
        ).casefold()

        if (
            not normalized_query
            or normalized_query in team_text
        ):
            roster = organization_repository.list_team_players(
                team_id=team.team_id,
                include_inactive=False,
            )

            team_results.append(
                {
                    "record": team,
                    "roster_size": len(
                        roster
                    ),
                }
            )

    result_metrics = st.columns(
        4
    )

    result_metrics[0].metric(
        "Players",
        len(
            player_results
        ),
    )

    result_metrics[1].metric(
        "Sessions",
        len(
            session_results
        ),
    )

    result_metrics[2].metric(
        "Tracked shots",
        len(
            shot_results
        ),
    )

    result_metrics[3].metric(
        "Teams",
        len(
            team_results
        ),
    )

    if not query:
        st.info(
            (
                "Enter a search term for precise results. The sections below "
                "currently show a limited platform overview."
            )
        )

    if player_results:
        st.markdown(
            "### Players"
        )

        for result in player_results[
            :12
        ]:
            player = result[
                "record"
            ]

            with st.container(
                border=True
            ):
                player_left, player_right = st.columns(
                    [
                        0.78,
                        0.22,
                    ]
                )

                with player_left:
                    st.markdown(
                        f"### {player.display_name}"
                    )

                    st.caption(
                        player.participant_id
                    )

                with player_right:
                    if st.button(
                        "Open Player Profile",
                        use_container_width=True,
                        key=(
                            f"search_open_player_"
                            f"{player.player_id}"
                        ),
                    ):
                        st.session_state[
                            "player_profile_select"
                        ] = player

                        request_workspace_navigation(
                            "Player Profile"
                        )

                player_metrics = st.columns(
                    3
                )

                player_metrics[0].metric(
                    "Tracked shots",
                    result[
                        "shots"
                    ],
                )

                player_metrics[1].metric(
                    "Sessions",
                    result[
                        "sessions"
                    ],
                )

                player_metrics[2].metric(
                    "Make %",
                    (
                        "N/A"
                        if result[
                            "make_percentage"
                        ]
                        is None
                        else (
                            f"{result['make_percentage']:.1f}%"
                        )
                    ),
                )

    if session_results:
        st.markdown(
            "### Practice Sessions"
        )

        for result in session_results[
            :15
        ]:
            player = result[
                "player"
            ]

            session = result[
                "session"
            ]

            with st.container(
                border=True
            ):
                session_left, session_right = st.columns(
                    [
                        0.75,
                        0.25,
                    ]
                )

                with session_left:
                    st.markdown(
                        f"### {session.session_name}"
                    )

                    st.caption(
                        (
                            f"{player.display_name} · "
                            f"{session.session_date} · "
                            f"{session.session_type} · "
                            f"{result['shots']} shots"
                        )
                    )

                with session_right:
                    if st.button(
                        "Open Session Analysis",
                        use_container_width=True,
                        key=(
                            f"search_open_session_"
                            f"{session.session_id}"
                        ),
                    ):
                        st.session_state[
                            "session_analysis_player"
                        ] = player

                        st.session_state[
                            "session_analysis_session"
                        ] = session

                        request_workspace_navigation(
                            "Session Analysis"
                        )

    if shot_results:
        st.markdown(
            "### Tracked Shots"
        )

        st.dataframe(
            [
                {
                    "Player": result[
                        "player"
                    ].display_name,
                    "Participant": result[
                        "player"
                    ].participant_id,
                    "Trial": result[
                        "shot"
                    ].trial_id,
                    "Result": result[
                        "shot"
                    ].recorded_result,
                    "Filename": result[
                        "shot"
                    ].original_filename,
                }
                for result in shot_results[
                    :50
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            (
                "Open the player's Session Analysis or Shot Library workspace "
                "to launch the complete stored-shot analysis."
            )
        )

    if team_results:
        st.markdown(
            "### Teams"
        )

        for result in team_results[
            :12
        ]:
            team = result[
                "record"
            ]

            with st.container(
                border=True
            ):
                team_left, team_right = st.columns(
                    [
                        0.78,
                        0.22,
                    ]
                )

                with team_left:
                    st.markdown(
                        f"### {team.team_name}"
                    )

                    st.caption(
                        (
                            f"{team.organization_name} · "
                            f"{team.season_name or 'No season'} · "
                            f"{result['roster_size']} active players"
                        )
                    )

                with team_right:
                    if st.button(
                        "Open Team Dashboard",
                        use_container_width=True,
                        key=(
                            f"search_open_team_"
                            f"{team.team_id}"
                        ),
                    ):
                        st.session_state[
                            "team_dashboard_team_select"
                        ] = team

                        request_workspace_navigation(
                            "Team Dashboard"
                        )

    if (
        query
        and not player_results
        and not session_results
        and not shot_results
        and not team_results
    ):
        st.warning(
            (
                "No active player, session, tracked shot, or team matched "
                f"“{query}”."
            )
        )


def render_executive_demo_basic_fallback(
    selected_player: object,
    selected_session: object,
    original_error: Exception,
) -> None:
    session_repository = get_practice_session_repository()
    organization_repository = get_organization_team_repository()

    shots = session_repository.list_session_shots(
        selected_session.session_id
    )

    made_count = sum(
        1
        for shot in shots
        if safe_result(
            shot.recorded_result
        )
        == "made"
    )
    missed_count = sum(
        1
        for shot in shots
        if safe_result(
            shot.recorded_result
        )
        == "missed"
    )
    make_percentage = (
        made_count / len(shots) * 100.0
        if shots
        else 0.0
    )

    st.warning(
        (
            "The hosted Executive Demo is showing its database-backed "
            "overview because the optional historical research baseline is "
            "not packaged in this container. The roster, sessions, tracked "
            "files, shot review, and workspace navigation remain available."
        )
    )

    metrics = st.columns(4)
    metrics[0].metric("Tracked shots", len(shots))
    metrics[1].metric("Made", made_count)
    metrics[2].metric("Missed", missed_count)
    metrics[3].metric(
        "Make percentage",
        f"{make_percentage:.1f}%",
    )

    st.markdown("### Featured session")

    st.markdown(
        f"""
        <div class="bms-score-card">
            <div class="bms-kicker">Selected player and session</div>
            <h3 style="margin:0 0 0.45rem;color:#152033;">
                {selected_player.display_name}
            </h3>
            <p style="margin:0;color:#455066;line-height:1.6;">
                {selected_session.session_name} ·
                {selected_session.session_date} ·
                {selected_session.session_type}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if shots:
        st.dataframe(
            [
                {
                    "Trial": shot.trial_id,
                    "Result": shot.recorded_result,
                    "Original file": shot.original_filename,
                }
                for shot in shots
            ],
            use_container_width=True,
            hide_index=True,
        )

    active_teams = organization_repository.list_teams(
        include_inactive=False
    )

    if active_teams:
        st.markdown(
            "### Organization and team context"
        )
        st.dataframe(
            [
                {
                    "Organization": team.organization_name,
                    "Team": team.team_name,
                    "Season": team.season_name,
                    "Level": team.level,
                }
                for team in active_teams
            ],
            use_container_width=True,
            hide_index=True,
        )

    navigation = st.columns(3)

    if navigation[0].button(
        "Open Session Analysis",
        use_container_width=True,
        key="executive_fallback_open_session",
    ):
        st.session_state[
            "session_analysis_player"
        ] = selected_player
        st.session_state[
            "session_analysis_session"
        ] = selected_session
        request_workspace_navigation(
            "Session Analysis"
        )

    if navigation[1].button(
        "Open Player Development",
        use_container_width=True,
        key="executive_fallback_open_development",
    ):
        request_workspace_navigation(
            "Player Development"
        )

    if navigation[2].button(
        "Open Team Dashboard",
        use_container_width=True,
        key="executive_fallback_open_team",
    ):
        request_workspace_navigation(
            "Team Dashboard"
        )

    with st.expander(
        "Technical deployment note",
        expanded=False,
    ):
        st.code(
            (
                f"{type(original_error).__name__}: "
                f"{original_error}"
            )
        )


def render_executive_demo_workspace() -> None:
    """
    Guided, read-only front-office demonstration using existing project data.
    """

    history_repository = get_player_history_repository()
    session_repository = get_practice_session_repository()
    organization_repository = get_organization_team_repository()

    players = history_repository.list_players(
        include_inactive=False
    )

    active_teams = organization_repository.list_teams(
        include_inactive=False
    )

    st.markdown(
        """<div class="bms-showcase-banner">
<div class="bms-showcase-eyebrow">Front-office demonstration</div>
<h2>ANKIT'S FREE THROW ANALYSIS SOFTWARE</h2>
<p class="bms-showcase-copy">A structured-tracking platform that converts free-throw motion data into smooth playback, biomechanics, session grading, player-development trends, team practice intelligence, coach interpretation, and professional reports.</p>
<div class="bms-feature-pill-row">
<span class="bms-feature-pill">Player-specific baselines</span>
<span class="bms-feature-pill">Coach-ready language</span>
<span class="bms-feature-pill">Session comparison</span>
<span class="bms-feature-pill">Team review queues</span>
<span class="bms-feature-pill">Professional exports</span>
</div>
</div>""",
        unsafe_allow_html=True,
    )

    available = []

    for player in players:
        sessions = session_repository.list_sessions(
            player_id=player.player_id,
            include_inactive=False,
        )

        for session in sessions:
            shots = session_repository.list_session_shots(
                session.session_id
            )

            if shots:
                available.append(
                    (
                        player,
                        session,
                    )
                )

    if not available:
        st.info(
            (
                "The Executive Demo needs at least one active player session "
                "containing tracked shots."
            )
        )

        return

    demo_players = []

    seen = set()

    for player, _ in available:
        if player.player_id not in seen:
            seen.add(
                player.player_id
            )
            demo_players.append(
                player
            )

    selected_player = st.selectbox(
        "Demo player",
        options=demo_players,
        format_func=player_option_label,
        key="executive_demo_player",
    )

    sessions = [
        session
        for player, session in available
        if player.player_id
        == selected_player.player_id
    ]

    selected_session = st.selectbox(
        "Featured session",
        options=sessions,
        format_func=session_option_label,
        key="executive_demo_session",
    )

    try:
        (
            summary,
            shot_rows,
            category_rows,
        ) = load_session_analysis(
            selected_session.session_id
        )

        (
            grade,
            components,
            actions,
        ) = load_session_grade_action_plan(
            selected_session.session_id
        )

        (
            coach_summary,
            coach_observations,
        ) = load_session_coach_summary(
            selected_session.session_id
        )

    except Exception as error:
        render_executive_demo_basic_fallback(
            selected_player=selected_player,
            selected_session=selected_session,
            original_error=error,
        )

        return

    try:
        insights = build_session_report_insights(
            session_summary=type(
                "SessionSummaryView",
                (),
                summary,
            )(),
            shot_rows=[
                type(
                    "ShotView",
                    (),
                    row,
                )()
                for row in shot_rows
            ],
            category_summaries=[
                type(
                    "CategoryView",
                    (),
                    row,
                )()
                for row in category_rows
            ],
            grade_summary=type(
                "GradeSummaryView",
                (),
                grade,
            )(),
            grade_components=[
                type(
                    "GradeComponentView",
                    (),
                    row,
                )()
                for row in components
            ],
            action_items=[],
        )

    except Exception:
        insights = None

    st.markdown(
        "## Practice Session at a Glance"
    )

    metrics = st.columns(
        6
    )

    metrics[0].metric(
        "Session grade",
        f"{grade['overall_grade']:.1f}",
    )

    metrics[1].metric(
        "Grade label",
        grade[
            "grade_label"
        ],
    )

    metrics[2].metric(
        "Shots",
        summary[
            "assigned_shots"
        ],
    )

    metrics[3].metric(
        "Make %",
        f"{summary['make_percentage']:.1f}%",
    )

    metrics[4].metric(
        "Consistency",
        (
            "N/A"
            if summary[
                "overall_consistency_score"
            ]
            is None
            else (
                f"{summary['overall_consistency_score']:.1f}"
            )
        ),
    )

    metrics[5].metric(
        "Evidence",
        grade[
            "grade_confidence"
        ].title(),
    )

    if insights is not None:
        st.info(
            insights.session_story
        )

    highlight_columns = st.columns(
        3
    )

    with highlight_columns[0]:
        st.markdown(
            "### What went well"
        )

        if insights is not None:
            for item in insights.what_went_well:
                st.write(
                    f"✓ {item}"
                )

    with highlight_columns[1]:
        st.markdown(
            "### Primary focus"
        )

        st.write(
            (
                insights.primary_focus
                if insights is not None
                else grade[
                    "primary_focus"
                ]
            )
        )

    with highlight_columns[2]:
        st.markdown(
            "### Next practice"
        )

        st.write(
            grade[
                "recommended_drill"
            ]
        )

        st.caption(
            (
                f"Estimated review time: "
                f"{grade['estimated_review_minutes']} minutes"
            )
        )

    st.markdown(
        "## Transparent Grade Breakdown"
    )

    component_columns = st.columns(
        len(
            components
        )
        or 1
    )

    for index, component in enumerate(
        components
    ):
        with component_columns[
            index
            % len(
                component_columns
            )
        ]:
            st.metric(
                component[
                    "component"
                ],
                f"{component['score']:.1f}",
            )

            st.caption(
                (
                    f"{component['status']} · "
                    f"Weight {component['weight']:.0%}"
                )
            )

    st.markdown(
        "## Category Repeatability"
    )

    for category in category_rows:
        st.write(
            (
                f"**{category['category']}** — "
                f"{category['average_consistency_score']:.1f}/100"
            )
        )

        st.progress(
            float(
                category[
                    "average_consistency_score"
                ]
            )
            / 100.0
        )

    st.markdown(
        "## Visual Summary"
    )

    visual_left, visual_right = st.columns(
        2
    )

    if category_rows:
        radar_labels = [
            row[
                "category"
            ]
            for row in category_rows
        ]

        radar_values = [
            float(
                row[
                    "average_consistency_score"
                ]
            )
            for row in category_rows
        ]

        if len(
            radar_values
        ) >= 3:
            radar_angles = np.linspace(
                0,
                2
                * np.pi,
                len(
                    radar_labels
                ),
                endpoint=False,
            ).tolist()

            radar_values_closed = (
                radar_values
                + radar_values[
                    :1
                ]
            )

            radar_angles_closed = (
                radar_angles
                + radar_angles[
                    :1
                ]
            )

            radar_figure = plt.figure(
                figsize=(
                    6.2,
                    4.6,
                )
            )

            radar_axis = radar_figure.add_subplot(
                111,
                polar=True,
            )

            radar_axis.plot(
                radar_angles_closed,
                radar_values_closed,
                linewidth=2.4,
            )

            radar_axis.fill(
                radar_angles_closed,
                radar_values_closed,
                alpha=0.14,
            )

            radar_axis.set_xticks(
                radar_angles,
                labels=radar_labels,
            )

            radar_axis.set_ylim(
                0,
                100,
            )

            radar_axis.set_yticks(
                [
                    25,
                    50,
                    75,
                    100,
                ]
            )

            radar_axis.set_title(
                "Category repeatability profile",
                pad=18,
            )

            radar_axis.grid(
                alpha=0.35,
            )

            with visual_left:
                st.pyplot(
                    radar_figure,
                    clear_figure=True,
                    use_container_width=True,
                )

    if insights is not None and insights.contributors:
        contributor_labels = [
            item.label
            for item in insights.contributors
        ]

        contributor_values = [
            item.impact_points
            for item in insights.contributors
        ]

        contributor_figure, contributor_axis = plt.subplots(
            figsize=(
                6.2,
                4.6,
            )
        )

        positions = np.arange(
            len(
                contributor_labels
            )
        )

        contributor_axis.barh(
            positions,
            contributor_values,
        )

        contributor_axis.set_yticks(
            positions,
            labels=contributor_labels,
        )

        contributor_axis.axvline(
            0,
            linewidth=1,
        )

        contributor_axis.set_xlabel(
            "Estimated contribution to session grade"
        )

        contributor_axis.set_title(
            "Biggest grade contributors"
        )

        contributor_axis.grid(
            axis="x",
            alpha=0.32,
        )

        contributor_axis.invert_yaxis()

        with visual_right:
            st.pyplot(
                contributor_figure,
                clear_figure=True,
                use_container_width=True,
            )

    st.markdown(
        "## Coach Workflow"
    )

    workflow_columns = st.columns(
        3
    )

    workflow_columns[0].metric(
        "Review first",
        summary[
            "highest_attention_trial"
        ],
    )

    with workflow_columns[1]:
        st.markdown(
            "### Primary message"
        )

        st.write(
            coach_summary[
                "primary_message"
            ]
        )

    with workflow_columns[2]:
        st.markdown(
            "### First actions"
        )

        for action in actions[
            :3
        ]:
            st.write(
                (
                    f"{action['rank']}. "
                    f"{action['title']}"
                )
            )

    st.markdown(
        "## Player Development Context"
    )

    try:
        development_engine = PlayerDevelopmentIntelligence()

        (
            development_summary,
            development_priorities,
        ) = development_engine.run(
            participant_id=selected_player.participant_id
        )

        development_columns = st.columns(
            5
        )

        development_columns[0].metric(
            "Tracked shots",
            development_summary.tracked_shots,
        )

        development_columns[1].metric(
            "Mechanics",
            f"{development_summary.overall_mechanics_score:.1f}",
        )

        development_columns[2].metric(
            "Consistency",
            f"{development_summary.consistency_score:.1f}",
        )

        development_columns[3].metric(
            "Strongest area",
            development_summary.strongest_area,
        )

        development_columns[4].metric(
            "Evidence",
            development_summary.evidence_confidence.title(),
        )

    except Exception as error:
        st.caption(
            (
                "Long-term development is unavailable for this player: "
                f"{type(error).__name__}."
            )
        )

    if active_teams:
        st.markdown(
            "## Team Practice Context"
        )

        selected_team = st.selectbox(
            "Featured team",
            options=active_teams,
            format_func=team_option_label,
            key="executive_demo_team",
        )

        try:
            (
                team_summary,
                team_players,
            ) = load_team_practice_intelligence(
                selected_team.team_id
            )

            team_metrics = st.columns(
                5
            )

            team_metrics[0].metric(
                "Active players",
                team_summary[
                    "active_players"
                ],
            )

            team_metrics[1].metric(
                "Players graded",
                team_summary[
                    "players_with_grades"
                ],
            )

            team_metrics[2].metric(
                "Average grade",
                (
                    "N/A"
                    if team_summary[
                        "average_session_grade"
                    ]
                    is None
                    else (
                        f"{team_summary['average_session_grade']:.1f}"
                    )
                ),
            )

            team_metrics[3].metric(
                "Review first",
                team_summary[
                    "highest_attention_player"
                ],
            )

            team_metrics[4].metric(
                "Highest grade",
                team_summary[
                    "highest_grade_player"
                ],
            )

        except Exception as error:
            st.caption(
                (
                    "Team intelligence is unavailable: "
                    f"{type(error).__name__}."
                )
            )

    st.markdown(
        "## Professional Report Exports"
    )

    report_columns = st.columns(
        3
    )

    with report_columns[0]:
        if st.button(
            "Generate Session PDF",
            type="primary",
            use_container_width=True,
            key="executive_demo_generate_session",
        ):
            try:
                (
                    pdf_bytes,
                    pdf_filename,
                ) = generate_session_pdf_bytes(
                    selected_session.session_id
                )

                st.session_state[
                    "executive_demo_session_pdf_bytes"
                ] = pdf_bytes

                st.session_state[
                    "executive_demo_session_pdf_filename"
                ] = pdf_filename

            except Exception as error:
                st.error(
                    f"{type(error).__name__}: {error}"
                )

        if st.session_state.get(
            "executive_demo_session_pdf_bytes"
        ) is not None:
            st.download_button(
                "Download Session PDF",
                data=st.session_state[
                    "executive_demo_session_pdf_bytes"
                ],
                file_name=st.session_state[
                    "executive_demo_session_pdf_filename"
                ],
                mime="application/pdf",
                use_container_width=True,
                key="executive_demo_download_session",
            )

    with report_columns[1]:
        if st.button(
            "Generate Player PDF",
            type="primary",
            use_container_width=True,
            key="executive_demo_generate_player",
        ):
            try:
                (
                    pdf_bytes,
                    pdf_filename,
                ) = generate_player_development_pdf_bytes(
                    player_id=selected_player.player_id,
                    participant_id=selected_player.participant_id,
                    player_display_name=selected_player.display_name,
                )

                st.session_state[
                    "executive_demo_player_pdf_bytes"
                ] = pdf_bytes

                st.session_state[
                    "executive_demo_player_pdf_filename"
                ] = pdf_filename

            except Exception as error:
                st.error(
                    f"{type(error).__name__}: {error}"
                )

        if st.session_state.get(
            "executive_demo_player_pdf_bytes"
        ) is not None:
            st.download_button(
                "Download Player PDF",
                data=st.session_state[
                    "executive_demo_player_pdf_bytes"
                ],
                file_name=st.session_state[
                    "executive_demo_player_pdf_filename"
                ],
                mime="application/pdf",
                use_container_width=True,
                key="executive_demo_download_player",
            )

    with report_columns[2]:
        if active_teams:
            if st.button(
                "Generate Team PDF",
                type="primary",
                use_container_width=True,
                key="executive_demo_generate_team",
            ):
                try:
                    (
                        pdf_bytes,
                        pdf_filename,
                    ) = generate_team_practice_pdf_bytes(
                        selected_team.team_id
                    )

                    st.session_state[
                        "executive_demo_team_pdf_bytes"
                    ] = pdf_bytes

                    st.session_state[
                        "executive_demo_team_pdf_filename"
                    ] = pdf_filename

                except Exception as error:
                    st.error(
                        f"{type(error).__name__}: {error}"
                    )

            if st.session_state.get(
                "executive_demo_team_pdf_bytes"
            ) is not None:
                st.download_button(
                    "Download Team PDF",
                    data=st.session_state[
                        "executive_demo_team_pdf_bytes"
                    ],
                    file_name=st.session_state[
                        "executive_demo_team_pdf_filename"
                    ],
                    mime="application/pdf",
                    use_container_width=True,
                    key="executive_demo_download_team",
                )

    st.markdown(
        """<div style="background:#fff7f2;border:1px solid #f1c7b3;border-radius:18px;padding:1rem 1.15rem;margin-top:1rem;">
<strong style="color:#9d3f17;">Interpretation note</strong>
<p style="margin:0.35rem 0 0;color:#5d4a42;line-height:1.6;">This demonstration uses structured tracking data and exploratory analytics. Scores and recommendations support review; they do not replace video context, player history, medical judgment, or professional coaching decisions.</p>
</div>""",
        unsafe_allow_html=True,
    )


def render_player_profile_workspace() -> None:
    profile_quick_left, profile_quick_middle, profile_quick_right = st.columns(
        3
    )

    if profile_quick_left.button(
        "Analyze a Session",
        use_container_width=True,
        key="player_profile_open_session_analysis",
    ):
        request_workspace_navigation(
            "Session Analysis"
        )

    if profile_quick_middle.button(
        "Compare Sessions",
        use_container_width=True,
        key="player_profile_open_session_comparison",
    ):
        request_workspace_navigation(
            "Session Comparison"
        )

    if profile_quick_right.button(
        "Generate Reports",
        use_container_width=True,
        key="player_profile_open_reports",
    ):
        request_workspace_navigation(
            "Reports Center"
        )

    """
    NBA-style player profile that combines history, development, sessions,
    team membership, and recent reports in one focused page.
    """

    history_repository = get_player_history_repository()
    session_repository = get_practice_session_repository()
    organization_repository = get_organization_team_repository()

    players = history_repository.list_players(
        include_inactive=False
    )

    if not players:
        st.info(
            "Create a player profile before opening Player Profile."
        )

        return

    selected_player = st.selectbox(
        "Player",
        options=players,
        format_func=player_option_label,
        key="player_profile_select",
    )

    player_shots = history_repository.list_shot_files(
        player_id=selected_player.player_id,
        include_inactive=False,
    )

    player_sessions = session_repository.list_sessions(
        player_id=selected_player.player_id,
        include_inactive=False,
    )

    team_memberships = []

    for team in organization_repository.list_teams(
        include_inactive=False
    ):
        members = organization_repository.list_team_players(
            team_id=team.team_id,
            include_inactive=False,
        )

        for member in members:
            if member.player_id == selected_player.player_id:
                team_memberships.append(
                    {
                        "Organization": member.organization_name,
                        "Team": member.team_name,
                        "Role": member.roster_role,
                        "Jersey": member.jersey_number,
                        "Position": member.position_group,
                    }
                )

    made = sum(
        shot.recorded_result == "made"
        for shot in player_shots
    )

    missed = sum(
        shot.recorded_result == "missed"
        for shot in player_shots
    )

    classified = made + missed

    make_percentage = (
        made
        / classified
        * 100.0
        if classified
        else 0.0
    )

    st.markdown(
        f"## {selected_player.display_name}"
    )

    st.caption(
        selected_player.participant_id
    )

    report_left, report_right = st.columns(
        [
            0.74,
            0.26,
        ]
    )

    report_left.info(
        (
            "Generate a player-development PDF containing the development "
            "snapshot, session timeline, current priorities, and coach plan."
        )
    )

    if report_right.button(
        "Generate Player PDF",
        type="primary",
        use_container_width=True,
        key=(
            f"generate_player_development_pdf_"
            f"{selected_player.player_id}"
        ),
    ):
        with st.spinner(
            "Generating player-development report..."
        ):
            try:
                (
                    player_pdf_bytes,
                    player_pdf_filename,
                ) = generate_player_development_pdf_bytes(
                    player_id=selected_player.player_id,
                    participant_id=(
                        selected_player.participant_id
                    ),
                    player_display_name=(
                        selected_player.display_name
                    ),
                )

                st.session_state[
                    "generated_player_pdf_bytes"
                ] = player_pdf_bytes

                st.session_state[
                    "generated_player_pdf_filename"
                ] = player_pdf_filename

                st.session_state[
                    "generated_player_pdf_player_id"
                ] = selected_player.player_id

                st.success(
                    "Player-development PDF generated."
                )

            except Exception as error:
                st.error(
                    (
                        "Could not generate the player-development PDF. "
                        f"{type(error).__name__}: {error}"
                    )
                )

    if (
        st.session_state.get(
            "generated_player_pdf_player_id"
        )
        == selected_player.player_id
        and st.session_state.get(
            "generated_player_pdf_bytes"
        )
        is not None
    ):
        st.download_button(
            "Download Player Development PDF",
            data=st.session_state[
                "generated_player_pdf_bytes"
            ],
            file_name=st.session_state[
                "generated_player_pdf_filename"
            ],
            mime="application/pdf",
            use_container_width=True,
            key=(
                f"download_player_development_pdf_"
                f"{selected_player.player_id}"
            ),
        )

    profile_metrics = st.columns(
        6
    )

    profile_metrics[0].metric(
        "Tracked shots",
        len(
            player_shots
        ),
    )

    profile_metrics[1].metric(
        "Made",
        made,
    )

    profile_metrics[2].metric(
        "Missed",
        missed,
    )

    profile_metrics[3].metric(
        "Make %",
        (
            f"{make_percentage:.1f}%"
            if classified
            else "N/A"
        ),
    )

    profile_metrics[4].metric(
        "Sessions",
        len(
            player_sessions
        ),
    )

    profile_metrics[5].metric(
        "Teams",
        len(
            team_memberships
        ),
    )

    development_summary = None

    try:
        development_engine = PlayerDevelopmentIntelligence()

        (
            development_summary,
            development_priorities,
        ) = development_engine.run(
            participant_id=selected_player.participant_id
        )

    except Exception:
        development_summary = None
        development_priorities = []

    if development_summary is not None:
        st.markdown(
            "### Development Snapshot"
        )

        development_columns = st.columns(
            5
        )

        development_columns[0].metric(
            "Mechanics",
            (
                f"{development_summary.overall_mechanics_score:.1f}"
            ),
        )

        development_columns[1].metric(
            "Consistency",
            (
                f"{development_summary.consistency_score:.1f}"
            ),
        )

        development_columns[2].metric(
            "Strongest area",
            development_summary.strongest_area,
        )

        development_columns[3].metric(
            "Primary priority",
            development_summary.primary_priority,
        )

        development_columns[4].metric(
            "Evidence",
            development_summary.evidence_confidence.title(),
        )

        if development_priorities:
            st.markdown(
                "#### Current Development Priorities"
            )

            st.dataframe(
                [
                    {
                        "Priority": priority.rank,
                        "Area": priority.display_name,
                        "Score": round(
                            priority.development_score,
                            1,
                        ),
                        "Level": priority.priority_level,
                    }
                    for priority in development_priorities[
                        :5
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    st.markdown(
        "### Practice History"
    )

    if player_sessions:
        session_rows = []

        for session in player_sessions:
            shots = session_repository.list_session_shots(
                session.session_id
            )

            session_made = sum(
                shot.recorded_result == "made"
                for shot in shots
            )

            session_missed = sum(
                shot.recorded_result == "missed"
                for shot in shots
            )

            session_classified = (
                session_made
                + session_missed
            )

            session_make_percentage = (
                session_made
                / session_classified
                * 100.0
                if session_classified
                else 0.0
            )

            session_rows.append(
                {
                    "Date": session.session_date,
                    "Session": session.session_name,
                    "Type": session.session_type,
                    "Shots": len(
                        shots
                    ),
                    "Make %": (
                        round(
                            session_make_percentage,
                            1,
                        )
                        if session_classified
                        else None
                    ),
                    "Location": session.location,
                }
            )

        st.dataframe(
            session_rows,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "No practice sessions have been created for this player."
        )

    st.markdown(
        "### Team Membership"
    )

    if team_memberships:
        st.dataframe(
            team_memberships,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "This player is not assigned to an active team."
        )

    st.markdown(
        "### Quick Actions"
    )

    st.caption(
        (
            "Use the sidebar to open Session Analysis, Player Development, "
            "Session Comparison, or Reports Center for this player's deeper "
            "workflows."
        )
    )



@st.cache_data(
    show_spinner=False,
)
def generate_session_comparison_pdf_bytes(
    first_session_id: int,
    second_session_id: int,
) -> tuple[bytes, str]:
    """Generate and cache a two-session comparison PDF."""

    history_repository = get_player_history_repository()

    analysis_engine = SessionAnalysisEngine(
        database_file=history_repository.database_file
    )

    generator = SessionComparisonPDFReportGenerator()

    pdf_path = generator.run(
        first_session_id=first_session_id,
        second_session_id=second_session_id,
        analysis_engine=analysis_engine,
    )

    return (
        pdf_path.read_bytes(),
        pdf_path.name,
    )


def render_session_comparison_workspace() -> None:
    """
    Compare two complete practice sessions for one player.
    """

    history_repository = get_player_history_repository()
    session_repository = get_practice_session_repository()

    players = history_repository.list_players(
        include_inactive=False
    )

    if not players:
        st.info(
            "Create a player profile before comparing sessions."
        )

        return

    selected_player = st.selectbox(
        "Player",
        options=players,
        format_func=player_option_label,
        key="session_comparison_player",
    )

    sessions = session_repository.list_sessions(
        player_id=selected_player.player_id,
        include_inactive=False,
    )

    if len(
        sessions
    ) < 2:
        st.info(
            "This player needs at least two active sessions for comparison."
        )

        return

    comparison_columns = st.columns(
        2
    )

    first_session = comparison_columns[0].selectbox(
        "Session A",
        options=sessions,
        format_func=session_option_label,
        key="session_comparison_a",
    )

    second_session_options = [
        session
        for session in sessions
        if session.session_id
        != first_session.session_id
    ]

    second_session = comparison_columns[1].selectbox(
        "Session B",
        options=second_session_options,
        format_func=session_option_label,
        key="session_comparison_b",
    )

    comparison_report_left, comparison_report_right = st.columns(
        [
            0.74,
            0.26,
        ]
    )

    comparison_report_left.info(
        (
            "Generate a professional PDF comparing grades, execution, "
            "consistency, category changes, coach interpretation, and next actions."
        )
    )

    if comparison_report_right.button(
        "Generate Comparison PDF",
        type="primary",
        use_container_width=True,
        key=(
            f"generate_session_comparison_pdf_"
            f"{first_session.session_id}_"
            f"{second_session.session_id}"
        ),
    ):
        with st.spinner(
            "Generating session-comparison report..."
        ):
            try:
                (
                    comparison_pdf_bytes,
                    comparison_pdf_filename,
                ) = generate_session_comparison_pdf_bytes(
                    first_session_id=first_session.session_id,
                    second_session_id=second_session.session_id,
                )

                st.session_state[
                    "generated_comparison_pdf_bytes"
                ] = comparison_pdf_bytes

                st.session_state[
                    "generated_comparison_pdf_filename"
                ] = comparison_pdf_filename

                st.session_state[
                    "generated_comparison_pdf_key"
                ] = (
                    first_session.session_id,
                    second_session.session_id,
                )

                st.success(
                    "Session-comparison PDF generated."
                )

            except Exception as error:
                st.error(
                    (
                        "Could not generate the comparison PDF. "
                        f"{type(error).__name__}: {error}"
                    )
                )

    if (
        st.session_state.get(
            "generated_comparison_pdf_key"
        )
        == (
            first_session.session_id,
            second_session.session_id,
        )
        and st.session_state.get(
            "generated_comparison_pdf_bytes"
        )
        is not None
    ):
        st.download_button(
            "Download Session Comparison PDF",
            data=st.session_state[
                "generated_comparison_pdf_bytes"
            ],
            file_name=st.session_state[
                "generated_comparison_pdf_filename"
            ],
            mime="application/pdf",
            use_container_width=True,
            key=(
                f"download_session_comparison_pdf_"
                f"{first_session.session_id}_"
                f"{second_session.session_id}"
            ),
        )

    try:
        (
            first_summary,
            first_shots,
            first_categories,
        ) = load_session_analysis(
            first_session.session_id
        )

        (
            second_summary,
            second_shots,
            second_categories,
        ) = load_session_analysis(
            second_session.session_id
        )

    except Exception as error:
        st.error(
            (
                "Could not compare the selected sessions. "
                f"{type(error).__name__}: {error}"
            )
        )

        return

    st.markdown(
        "### Head-to-Head Summary"
    )

    comparison_table = [
        {
            "Metric": "Assigned shots",
            first_session.session_name: first_summary[
                "assigned_shots"
            ],
            second_session.session_name: second_summary[
                "assigned_shots"
            ],
            "Change": (
                second_summary[
                    "assigned_shots"
                ]
                - first_summary[
                    "assigned_shots"
                ]
            ),
        },
        {
            "Metric": "Make %",
            first_session.session_name: round(
                first_summary[
                    "make_percentage"
                ],
                1,
            ),
            second_session.session_name: round(
                second_summary[
                    "make_percentage"
                ],
                1,
            ),
            "Change": round(
                second_summary[
                    "make_percentage"
                ]
                - first_summary[
                    "make_percentage"
                ],
                1,
            ),
        },
        {
            "Metric": "Overall consistency",
            first_session.session_name: (
                None
                if first_summary[
                    "overall_consistency_score"
                ]
                is None
                else round(
                    first_summary[
                        "overall_consistency_score"
                    ],
                    1,
                )
            ),
            second_session.session_name: (
                None
                if second_summary[
                    "overall_consistency_score"
                ]
                is None
                else round(
                    second_summary[
                        "overall_consistency_score"
                    ],
                    1,
                )
            ),
            "Change": (
                None
                if (
                    first_summary[
                        "overall_consistency_score"
                    ]
                    is None
                    or second_summary[
                        "overall_consistency_score"
                    ]
                    is None
                )
                else round(
                    second_summary[
                        "overall_consistency_score"
                    ]
                    - first_summary[
                        "overall_consistency_score"
                    ],
                    1,
                )
            ),
        },
        {
            "Metric": "Average similarity",
            first_session.session_name: (
                first_summary[
                    "average_similarity_score"
                ]
            ),
            second_session.session_name: (
                second_summary[
                    "average_similarity_score"
                ]
            ),
            "Change": (
                None
                if (
                    first_summary[
                        "average_similarity_score"
                    ]
                    is None
                    or second_summary[
                        "average_similarity_score"
                    ]
                    is None
                )
                else round(
                    second_summary[
                        "average_similarity_score"
                    ]
                    - first_summary[
                        "average_similarity_score"
                    ],
                    1,
                )
            ),
        },
    ]

    st.dataframe(
        comparison_table,
        use_container_width=True,
        hide_index=True,
    )

    category_lookup_a = {
        row[
            "category"
        ]: row[
            "average_consistency_score"
        ]
        for row in first_categories
    }

    category_lookup_b = {
        row[
            "category"
        ]: row[
            "average_consistency_score"
        ]
        for row in second_categories
    }

    all_categories = sorted(
        set(
            category_lookup_a
        )
        | set(
            category_lookup_b
        )
    )

    if all_categories:
        st.markdown(
            "### Category Change"
        )

        category_rows = []

        for category in all_categories:
            first_value = category_lookup_a.get(
                category
            )

            second_value = category_lookup_b.get(
                category
            )

            category_rows.append(
                {
                    "Category": category,
                    first_session.session_name: (
                        None
                        if first_value is None
                        else round(
                            first_value,
                            1,
                        )
                    ),
                    second_session.session_name: (
                        None
                        if second_value is None
                        else round(
                            second_value,
                            1,
                        )
                    ),
                    "Change": (
                        None
                        if (
                            first_value is None
                            or second_value is None
                        )
                        else round(
                            second_value
                            - first_value,
                            1,
                        )
                    ),
                }
            )

        st.dataframe(
            category_rows,
            use_container_width=True,
            hide_index=True,
        )

        positions = list(
            range(
                len(
                    all_categories
                )
            )
        )

        width = 0.36

        comparison_figure, comparison_axis = plt.subplots(
            figsize=(
                11,
                5,
            )
        )

        comparison_axis.bar(
            [
                position
                - width
                / 2
                for position in positions
            ],
            [
                category_lookup_a.get(
                    category,
                    0.0,
                )
                for category in all_categories
            ],
            width=width,
            label=first_session.session_name,
        )

        comparison_axis.bar(
            [
                position
                + width
                / 2
                for position in positions
            ],
            [
                category_lookup_b.get(
                    category,
                    0.0,
                )
                for category in all_categories
            ],
            width=width,
            label=second_session.session_name,
        )

        comparison_axis.set_xticks(
            positions,
            labels=all_categories,
            rotation=20,
            ha="right",
        )

        comparison_axis.set_ylim(
            0,
            100,
        )

        comparison_axis.set_ylabel(
            "Consistency score"
        )

        comparison_axis.set_title(
            "Session category comparison"
        )

        comparison_axis.legend()

        comparison_axis.grid(
            visible=True,
            axis="y",
            alpha=0.25,
        )

        comparison_figure.tight_layout()

        st.pyplot(
            comparison_figure,
            clear_figure=True,
            use_container_width=True,
        )

    make_change = (
        second_summary[
            "make_percentage"
        ]
        - first_summary[
            "make_percentage"
        ]
    )

    consistency_change = (
        None
        if (
            first_summary[
                "overall_consistency_score"
            ]
            is None
            or second_summary[
                "overall_consistency_score"
            ]
            is None
        )
        else (
            second_summary[
                "overall_consistency_score"
            ]
            - first_summary[
                "overall_consistency_score"
            ]
        )
    )

    st.markdown(
        "### Coach Comparison Summary"
    )

    comparison_messages = []

    if make_change >= 5:
        comparison_messages.append(
            (
                f"Make percentage improved by "
                f"{make_change:+.1f} points."
            )
        )

    elif make_change <= -5:
        comparison_messages.append(
            (
                f"Make percentage declined by "
                f"{make_change:+.1f} points."
            )
        )

    else:
        comparison_messages.append(
            "Make percentage remained relatively stable."
        )

    if consistency_change is not None:
        if consistency_change >= 5:
            comparison_messages.append(
                (
                    f"Overall consistency improved by "
                    f"{consistency_change:+.1f} points."
                )
            )

        elif consistency_change <= -5:
            comparison_messages.append(
                (
                    f"Overall consistency declined by "
                    f"{consistency_change:+.1f} points."
                )
            )

        else:
            comparison_messages.append(
                "Overall consistency remained relatively stable."
            )

    st.info(
        " ".join(
            comparison_messages
        )
    )


def render_reports_center_workspace() -> None:
    """
    Central report-generation workspace.
    """

    history_repository = get_player_history_repository()
    session_repository = get_practice_session_repository()

    players = history_repository.list_players(
        include_inactive=False
    )

    if not players:
        st.info(
            "Create a player profile before opening Reports Center."
        )

        return

    selected_player = st.selectbox(
        "Player",
        options=players,
        format_func=player_option_label,
        key="reports_center_player",
    )

    sessions = session_repository.list_sessions(
        player_id=selected_player.player_id,
        include_inactive=False,
    )

    st.markdown(
        "### Available Reports"
    )

    st.markdown(
        """<div style="background:#ffffff;border:1px solid #e4e8f0;border-radius:20px;padding:1.15rem 1.25rem;margin-bottom:1rem;">
<h3 style="margin:0 0 0.45rem;color:#152033;">Session Report</h3>
<p style="margin:0;color:#455066;line-height:1.6;">Includes session overview, coach interpretation, category consistency, and shot review queue.</p>
</div>""",
        unsafe_allow_html=True,
    )

    if not sessions:
        st.info(
            "This player has no active sessions available for reporting."
        )

        return

    selected_session = st.selectbox(
        "Practice session",
        options=sessions,
        format_func=session_option_label,
        key="reports_center_session",
    )

    if st.button(
        "Generate Session PDF",
        type="primary",
        use_container_width=True,
        key=(
            f"reports_center_generate_"
            f"{selected_session.session_id}"
        ),
    ):
        with st.spinner(
            "Generating session PDF..."
        ):
            try:
                pdf_bytes, pdf_filename = generate_session_pdf_bytes(
                    selected_session.session_id
                )

                st.session_state[
                    "reports_center_pdf_bytes"
                ] = pdf_bytes

                st.session_state[
                    "reports_center_pdf_filename"
                ] = pdf_filename

                st.session_state[
                    "reports_center_pdf_session_id"
                ] = selected_session.session_id

                st.success(
                    "Session report generated."
                )

            except Exception as error:
                st.error(
                    (
                        "Could not generate session report. "
                        f"{type(error).__name__}: {error}"
                    )
                )

    if (
        st.session_state.get(
            "reports_center_pdf_session_id"
        )
        == selected_session.session_id
        and st.session_state.get(
            "reports_center_pdf_bytes"
        )
        is not None
    ):
        st.download_button(
            "Download Session Report",
            data=st.session_state[
                "reports_center_pdf_bytes"
            ],
            file_name=st.session_state[
                "reports_center_pdf_filename"
            ],
            mime="application/pdf",
            use_container_width=True,
            key=(
                f"reports_center_download_"
                f"{selected_session.session_id}"
            ),
        )

    st.markdown(
        "### Player Development Report"
    )

    if st.button(
        "Generate Player Development PDF",
        type="primary",
        use_container_width=True,
        key=(
            f"reports_center_generate_player_"
            f"{selected_player.player_id}"
        ),
    ):
        with st.spinner(
            "Generating player-development PDF..."
        ):
            try:
                (
                    player_pdf_bytes,
                    player_pdf_filename,
                ) = generate_player_development_pdf_bytes(
                    player_id=selected_player.player_id,
                    participant_id=(
                        selected_player.participant_id
                    ),
                    player_display_name=(
                        selected_player.display_name
                    ),
                )

                st.session_state[
                    "reports_center_player_pdf_bytes"
                ] = player_pdf_bytes

                st.session_state[
                    "reports_center_player_pdf_filename"
                ] = player_pdf_filename

                st.session_state[
                    "reports_center_player_pdf_player_id"
                ] = selected_player.player_id

                st.success(
                    "Player-development report generated."
                )

            except Exception as error:
                st.error(
                    (
                        "Could not generate player-development report. "
                        f"{type(error).__name__}: {error}"
                    )
                )

    if (
        st.session_state.get(
            "reports_center_player_pdf_player_id"
        )
        == selected_player.player_id
        and st.session_state.get(
            "reports_center_player_pdf_bytes"
        )
        is not None
    ):
        st.download_button(
            "Download Player Development Report",
            data=st.session_state[
                "reports_center_player_pdf_bytes"
            ],
            file_name=st.session_state[
                "reports_center_player_pdf_filename"
            ],
            mime="application/pdf",
            use_container_width=True,
            key=(
                f"reports_center_download_player_"
                f"{selected_player.player_id}"
            ),
        )

    st.markdown(
        "### Team Practice Report"
    )

    organization_repository = (
        get_organization_team_repository()
    )

    active_teams = (
        organization_repository
        .list_teams(
            include_inactive=False
        )
    )

    if active_teams:
        reports_team = st.selectbox(
            "Team",
            options=active_teams,
            format_func=team_option_label,
            key="reports_center_team",
        )

        if st.button(
            "Generate Team Practice PDF",
            type="primary",
            use_container_width=True,
            key=(
                f"reports_center_generate_team_"
                f"{reports_team.team_id}"
            ),
        ):
            with st.spinner(
                "Generating team practice PDF..."
            ):
                try:
                    (
                        team_pdf_bytes,
                        team_pdf_filename,
                    ) = generate_team_practice_pdf_bytes(
                        reports_team.team_id
                    )

                    st.session_state[
                        "reports_center_team_pdf_bytes"
                    ] = team_pdf_bytes

                    st.session_state[
                        "reports_center_team_pdf_filename"
                    ] = team_pdf_filename

                    st.session_state[
                        "reports_center_team_pdf_team_id"
                    ] = reports_team.team_id

                    st.success(
                        "Team practice report generated."
                    )

                except Exception as error:
                    st.error(
                        (
                            "Could not generate team practice report. "
                            f"{type(error).__name__}: {error}"
                        )
                    )

        if (
            st.session_state.get(
                "reports_center_team_pdf_team_id"
            )
            == reports_team.team_id
            and st.session_state.get(
                "reports_center_team_pdf_bytes"
            )
            is not None
        ):
            st.download_button(
                "Download Team Practice Report",
                data=st.session_state[
                    "reports_center_team_pdf_bytes"
                ],
                file_name=st.session_state[
                    "reports_center_team_pdf_filename"
                ],
                mime="application/pdf",
                use_container_width=True,
                key=(
                    f"reports_center_download_team_"
                    f"{reports_team.team_id}"
                ),
            )

    else:
        st.info(
            "No active teams are available for reporting."
        )

    st.markdown(
        "### Session Comparison Report"
    )

    comparison_sessions = session_repository.list_sessions(
        player_id=selected_player.player_id,
        include_inactive=False,
    )

    if len(
        comparison_sessions
    ) >= 2:
        comparison_report_columns = st.columns(
            2
        )

        reports_first_session = comparison_report_columns[0].selectbox(
            "Comparison Session A",
            options=comparison_sessions,
            format_func=session_option_label,
            key="reports_center_comparison_a",
        )

        reports_second_options = [
            session
            for session in comparison_sessions
            if session.session_id
            != reports_first_session.session_id
        ]

        reports_second_session = comparison_report_columns[1].selectbox(
            "Comparison Session B",
            options=reports_second_options,
            format_func=session_option_label,
            key="reports_center_comparison_b",
        )

        if st.button(
            "Generate Session Comparison PDF",
            type="primary",
            use_container_width=True,
            key=(
                f"reports_center_generate_comparison_"
                f"{reports_first_session.session_id}_"
                f"{reports_second_session.session_id}"
            ),
        ):
            with st.spinner(
                "Generating session-comparison PDF..."
            ):
                try:
                    (
                        comparison_pdf_bytes,
                        comparison_pdf_filename,
                    ) = generate_session_comparison_pdf_bytes(
                        first_session_id=reports_first_session.session_id,
                        second_session_id=reports_second_session.session_id,
                    )

                    st.session_state[
                        "reports_center_comparison_pdf_bytes"
                    ] = comparison_pdf_bytes

                    st.session_state[
                        "reports_center_comparison_pdf_filename"
                    ] = comparison_pdf_filename

                    st.session_state[
                        "reports_center_comparison_pdf_key"
                    ] = (
                        reports_first_session.session_id,
                        reports_second_session.session_id,
                    )

                    st.success(
                        "Session-comparison report generated."
                    )

                except Exception as error:
                    st.error(
                        (
                            "Could not generate session-comparison report. "
                            f"{type(error).__name__}: {error}"
                        )
                    )

        if (
            st.session_state.get(
                "reports_center_comparison_pdf_key"
            )
            == (
                reports_first_session.session_id,
                reports_second_session.session_id,
            )
            and st.session_state.get(
                "reports_center_comparison_pdf_bytes"
            )
            is not None
        ):
            st.download_button(
                "Download Session Comparison Report",
                data=st.session_state[
                    "reports_center_comparison_pdf_bytes"
                ],
                file_name=st.session_state[
                    "reports_center_comparison_pdf_filename"
                ],
                mime="application/pdf",
                use_container_width=True,
                key=(
                    f"reports_center_download_comparison_"
                    f"{reports_first_session.session_id}_"
                    f"{reports_second_session.session_id}"
                ),
            )

    else:
        st.info(
            (
                "The selected player needs at least two active sessions "
                "for a comparison report."
            )
        )

    st.markdown(
        "### Reports Center Status"
    )

    st.caption(
        (
            "Session, player-development, team-practice, and session-"
            "comparison PDF reporting are now available."
        )
    )


def display_results(
    result: dict,
    uploaded_name: str,
    trial_data: dict,
    shooting_side: str,
) -> None:
    feature_record = result[
        "feature_record"
    ]

    summary = result[
        "score_summary"
    ]

    st.success(
        "Analysis completed successfully."
    )

    section_title(
        "Shot overview",
        "The uploaded trial was processed through the event, biomechanics, "
        "timeline, feature, and personal-baseline pipelines.",
    )

    first, second, third, fourth = st.columns(
        4
    )

    first.metric(
        "Participant",
        feature_record.participant_id,
    )

    second.metric(
        "Trial",
        feature_record.trial_id,
    )

    third.metric(
        "Recorded result",
        str(
            feature_record.result
        ).title(),
    )

    fourth.metric(
        "Timeline events",
        feature_record.authoritative_event_count,
    )

    section_title(
        "Smooth MP4 shot playback",
        (
            "Generate one browser-native MP4 and use the normal video controls "
            "to play, pause, scrub, seek, replay, enter fullscreen, or download "
            "the shot."
        ),
    )

    with st.expander(
        "Generate and control the smooth shot MP4",
        expanded=True,
    ):
        render_smooth_video_player(
            trial_data=trial_data,
            title="Complete Shot MP4",
            widget_prefix="current_shot",
        )

    if summary is None:
        st.warning(
            "The biomechanics analysis worked, but no personal baseline "
            "was available for this participant. Personal scoring requires "
            "historical made shots and personal diagnostic results."
        )

    else:
        section_title(
            "Personal successful-baseline score",
            "Similarity to this player's own made-shot history—not a universal "
            "free-throw grade.",
        )

        score_column, assessment_column = st.columns(
            [
                0.34,
                0.66,
            ],
            gap="large",
        )

        with score_column:
            st.markdown(
                score_ring_html(
                    score=(
                        summary
                        .overall_similarity_score
                    ),
                    confidence=(
                        summary
                        .score_confidence
                    ),
                ),
                unsafe_allow_html=True,
            )

        with assessment_column:
            assessment_heading = score_status_text(
                summary.overall_similarity_score
            )

            assessment_text = (
                summary.overall_assessment
            )

            st.markdown(
                f"""
                <div class="bms-score-card">
                    <div class="bms-kicker" style="color:#c94d18;">
                        Coach interpretation
                    </div>
                    <h2 style="
                        margin:0.7rem 0 0.75rem;
                        color:#152033;
                        font-size:1.7rem;
                    ">
                        {assessment_heading}
                    </h2>
                    <p style="
                        margin:0;
                        color:#455066;
                        line-height:1.7;
                        font-size:1rem;
                    ">
                        {assessment_text}
                    </p>
                    <div style="
                        margin-top:1.1rem;
                        padding-top:1rem;
                        border-top:1px solid #e4e8f0;
                        color:#667085;
                        font-size:0.84rem;
                        line-height:1.55;
                    ">
                        Confidence-adjusted score:
                        <strong>{summary.confidence_adjusted_score:.1f}/100</strong>
                        &nbsp;·&nbsp;
                        Baseline:
                        <strong>{result['baseline_makes']} made</strong>
                        and
                        <strong>{result['baseline_misses']} missed</strong>
                        shots.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        section_title(
            "Mechanics breakdown",
            "Each area measures similarity to the player's successful pattern.",
        )

        render_category_cards(
            result[
                "category_scores"
            ]
        )

        section_title(
            "Coach summary",
            "The three clearest takeaways from this tracked shot.",
        )

        render_coach_summary(
            result
        )

        section_title(
            "Coach intelligence",
            "The primary diagnosis, why it may matter, a coaching focus, and the exact animation window to review.",
        )

        render_coach_diagnoses(
            result=result,
            trial_data=trial_data,
        )

        with st.expander(
            "View grouped movement-pattern evidence",
            expanded=False,
        ):
            render_movement_patterns(
                result
            )

        section_title(
            "Supporting feature details",
            "The individual measurements supporting the grouped movement patterns.",
        )

        render_observations(
            result[
                "scored_features"
            ]
        )

        section_title(
            "Visual score dashboard",
            "A compact view of the overall score, categories, and largest deviations.",
        )

        figure = build_dashboard_figure(
            result
        )

        st.pyplot(
            figure,
            clear_figure=True,
            use_container_width=True,
        )

    section_title(
        "Detailed analysis",
        "Review the authoritative timeline, objective measurements, and full feature record.",
    )

    (
        timeline_tab,
        metrics_tab,
        history_tab,
        development_tab,
        data_tab,
    ) = st.tabs(
        [
            "Shot Timeline",
            "Objective Metrics",
            "Practice History",
            "Player Development",
            "Feature Record",
        ]
    )

    with timeline_tab:
        st.dataframe(
            build_timeline_rows(
                result["timeline"],
                feature_record.sampling_rate,
            ),
            use_container_width=True,
            hide_index=True,
        )

    with metrics_tab:
        st.dataframe(
            build_metric_rows(
                feature_record
            ),
            use_container_width=True,
            hide_index=True,
        )

    with history_tab:
        render_session_history_dashboard(
            feature_record.participant_id
        )

    with development_tab:
        render_player_development_dashboard(
            feature_record.participant_id
        )

    with data_tab:
        st.json(
            result[
                "feature_dictionary"
            ],
            expanded=False,
        )

    package = create_download_zip(
        uploaded_name=uploaded_name,
        trial_data=trial_data,
        result=result,
    )

    section_title(
        "Export",
        "Download the source trial, report, timeline, metrics, and observations as one ZIP package.",
    )

    st.download_button(
        label="Download Complete Report Package",
        data=package,
        file_name=(
            f"{feature_record.participant_id}_"
            f"{feature_record.trial_id}_"
            "shot_analysis.zip"
        ),
        mime="application/zip",
        use_container_width=True,
        type="primary",
    )

    st.markdown(
        """
        <div class="bms-footer-note">
            This prototype describes tracked movement and compares it with the
            participant's historical successful pattern. It does not establish
            a universal ideal or prove why a shot was made or missed.
        </div>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state() -> None:
    if "pending_workspace_navigation" not in st.session_state:
        st.session_state.pending_workspace_navigation = None

    if (
        is_hosted_portfolio_mode()
        and "public_workspace_kind"
        not in st.session_state
    ):
        st.session_state.public_workspace_kind = ""

    if (
        is_hosted_portfolio_mode()
        and "public_workspace_data_root"
        not in st.session_state
    ):
        st.session_state.public_workspace_data_root = None

    if "launch_readiness_checks" not in st.session_state:
        st.session_state.launch_readiness_checks = None

    if "launch_readiness_last_run" not in st.session_state:
        st.session_state.launch_readiness_last_run = None

    if "executive_demo_session_pdf_bytes" not in st.session_state:
        st.session_state.executive_demo_session_pdf_bytes = None

    if "executive_demo_session_pdf_filename" not in st.session_state:
        st.session_state.executive_demo_session_pdf_filename = None

    if "executive_demo_player_pdf_bytes" not in st.session_state:
        st.session_state.executive_demo_player_pdf_bytes = None

    if "executive_demo_player_pdf_filename" not in st.session_state:
        st.session_state.executive_demo_player_pdf_filename = None

    if "executive_demo_team_pdf_bytes" not in st.session_state:
        st.session_state.executive_demo_team_pdf_bytes = None

    if "executive_demo_team_pdf_filename" not in st.session_state:
        st.session_state.executive_demo_team_pdf_filename = None

    if "reports_center_comparison_pdf_bytes" not in st.session_state:
        st.session_state.reports_center_comparison_pdf_bytes = None

    if "reports_center_comparison_pdf_filename" not in st.session_state:
        st.session_state.reports_center_comparison_pdf_filename = None

    if "reports_center_comparison_pdf_key" not in st.session_state:
        st.session_state.reports_center_comparison_pdf_key = None

    if "generated_comparison_pdf_bytes" not in st.session_state:
        st.session_state.generated_comparison_pdf_bytes = None

    if "generated_comparison_pdf_filename" not in st.session_state:
        st.session_state.generated_comparison_pdf_filename = None

    if "generated_comparison_pdf_key" not in st.session_state:
        st.session_state.generated_comparison_pdf_key = None

    if "reports_center_player_pdf_bytes" not in st.session_state:
        st.session_state.reports_center_player_pdf_bytes = None

    if "reports_center_player_pdf_filename" not in st.session_state:
        st.session_state.reports_center_player_pdf_filename = None

    if "reports_center_player_pdf_player_id" not in st.session_state:
        st.session_state.reports_center_player_pdf_player_id = None

    if "generated_player_pdf_bytes" not in st.session_state:
        st.session_state.generated_player_pdf_bytes = None

    if "generated_player_pdf_filename" not in st.session_state:
        st.session_state.generated_player_pdf_filename = None

    if "generated_player_pdf_player_id" not in st.session_state:
        st.session_state.generated_player_pdf_player_id = None

    if "reports_center_team_pdf_bytes" not in st.session_state:
        st.session_state.reports_center_team_pdf_bytes = None

    if "reports_center_team_pdf_filename" not in st.session_state:
        st.session_state.reports_center_team_pdf_filename = None

    if "reports_center_team_pdf_team_id" not in st.session_state:
        st.session_state.reports_center_team_pdf_team_id = None

    if "generated_team_pdf_bytes" not in st.session_state:
        st.session_state.generated_team_pdf_bytes = None

    if "generated_team_pdf_filename" not in st.session_state:
        st.session_state.generated_team_pdf_filename = None

    if "generated_team_pdf_team_id" not in st.session_state:
        st.session_state.generated_team_pdf_team_id = None

    if "reports_center_pdf_bytes" not in st.session_state:
        st.session_state.reports_center_pdf_bytes = None

    if "reports_center_pdf_filename" not in st.session_state:
        st.session_state.reports_center_pdf_filename = None

    if "reports_center_pdf_session_id" not in st.session_state:
        st.session_state.reports_center_pdf_session_id = None

    if "generated_session_pdf_bytes" not in st.session_state:
        st.session_state.generated_session_pdf_bytes = None

    if "generated_session_pdf_filename" not in st.session_state:
        st.session_state.generated_session_pdf_filename = None

    if "generated_session_pdf_session_id" not in st.session_state:
        st.session_state.generated_session_pdf_session_id = None

    if "session_opened_shot_result" not in st.session_state:
        st.session_state.session_opened_shot_result = None

    if "session_opened_shot_data" not in st.session_state:
        st.session_state.session_opened_shot_data = None

    if "session_opened_shot_filename" not in st.session_state:
        st.session_state.session_opened_shot_filename = None

    if "session_opened_shot_id" not in st.session_state:
        st.session_state.session_opened_shot_id = None

    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None

    if "analysis_filename" not in st.session_state:
        st.session_state.analysis_filename = None

    if "analysis_trial_data" not in st.session_state:
        st.session_state.analysis_trial_data = None

    if "analysis_file_signature" not in st.session_state:
        st.session_state.analysis_file_signature = None


def clear_stale_result(
    uploaded_file: object,
) -> None:
    if uploaded_file is None:
        current_signature = None
    else:
        current_signature = (
            uploaded_file.name,
            len(
                uploaded_file.getvalue()
            ),
        )

    if (
        st.session_state.analysis_file_signature
        != current_signature
    ):
        st.session_state.analysis_result = None
        st.session_state.analysis_filename = None
        st.session_state.analysis_trial_data = None
        st.session_state.analysis_file_signature = (
            current_signature
        )
        st.session_state.playback_frame = 0
        st.session_state.playback_playing = False
        st.session_state.diagnosis_segment_active = False
        st.session_state.diagnosis_segment_start = None
        st.session_state.diagnosis_segment_end = None
        st.session_state.diagnosis_segment_title = ""
        st.session_state.diagnosis_segment_focus = ""

        for video_key in (
            "current_shot_video_bytes",
            "current_shot_video_signature",
        ):
            st.session_state.pop(
                video_key,
                None,
            )



def application_mode() -> str:
    """
    Return the configured application mode for compatibility.
    """

    return (
        "public"
        if is_true_public_mode()
        else "private"
    )


def is_public_mode() -> bool:
    """
    Legacy presentation-mode helper.

    This intentionally remains False so the unified full navigation and all
    analysis workspaces stay visible. Database-changing actions use
    is_true_public_mode() for read-only protection.
    """

    return False


def render_public_demo_guidance() -> None:
    """
    Explain the public demonstration workflow without exposing admin tools.
    """

    st.markdown(
        """<div style="background:#ffffff;border:1px solid #e4e8f0;border-radius:20px;padding:1.25rem 1.35rem;margin:0.8rem 0 1.3rem;box-shadow:0 8px 22px rgba(28,39,60,0.04);">
<div style="color:#c94d18;font-size:0.75rem;font-weight:800;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.45rem;">Public demonstration</div>
<h3 style="margin:0 0 0.55rem;color:#152033;">Explore the complete tracked-shot workflow</h3>
<p style="margin:0;color:#455066;line-height:1.65;">This public version is designed to demonstrate shot playback, biomechanics, personal-baseline scoring, coaching interpretation, comparison tools, practice history, and player-development intelligence. Player-library administration and permanent data management are intentionally hidden from public visitors.</p>
</div>""",
        unsafe_allow_html=True,
    )


def request_workspace_navigation(
    workspace_name: str,
) -> None:
    """
    Queue workspace navigation for the next Streamlit rerun.

    Streamlit does not allow a widget-backed session-state key to be changed
    after that widget has already been created during the current run.
    """

    st.session_state[
        "pending_workspace_navigation"
    ] = workspace_name

    st.rerun()


def main() -> None:
    configure_page()
    initialize_session_state()
    initialize_playback_state()

    if (
        is_hosted_portfolio_mode()
        and not active_workspace_kind()
    ):
        render_public_workspace_chooser()
        return

    pending_workspace = st.session_state.pop(
        "pending_workspace_navigation",
        None,
    )

    if pending_workspace is not None:
        st.session_state[
            "private_workspace"
        ] = pending_workspace

    public_mode = (
        is_public_mode()
        or is_true_public_mode()
    )

    workspace = st.session_state.get(
        "private_workspace",
        "Home",
    )

    st.markdown(
        f"""<div class="bms-app-status">
<div class="bms-app-status-left">
<span class="bms-status-dot"></span>
<span>Platform operational</span>
</div>
<div class="bms-app-status-right">Page: {workspace} · {'Blank Workspace' if active_workspace_kind() == 'blank' else ('Executive Demo Workspace' if active_workspace_kind() == 'executive_demo' else 'Local Coaching Workspace')} · Version 1.0 full platform</div>
</div>""",
        unsafe_allow_html=True,
    )



    render_breadcrumbs(
        workspace
    )

    st.markdown(
        """
        <div class="bms-hero">
            <div class="bms-kicker">
                Basketball biomechanics and player-development platform
            </div>
            <h1>ANKIT'S FREE THROW ANALYSIS SOFTWARE</h1>
            <p>
                A data-driven platform for analyzing free-throw mechanics,
                comparing shots, tracking player development, and translating
                three-dimensional motion data into coach-readable insights.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if (
        is_public_mode()
        or workspace == "Home"
    ):
        st.markdown(
            """<div style="background:#ffffff;border:1px solid #e4e8f0;border-radius:22px;padding:1.5rem 1.6rem;margin-bottom:1.2rem;box-shadow:0 10px 28px rgba(28,39,60,0.05);">
    <div style="color:#c94d18;font-size:0.78rem;font-weight:800;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.55rem;">Why I created this platform</div>
    <h2 style="margin:0 0 0.75rem;color:#152033;font-size:1.55rem;">Turning complex motion data into practical basketball decisions</h2>
    <p style="margin:0;color:#455066;line-height:1.75;font-size:1rem;">I created this platform to explore how player-tracking data can be translated into useful information for coaches, athletes, performance staff, and basketball front offices. The goal is not to define one universal “perfect” free throw. The goal is to understand each player's own movement pattern, identify meaningful differences between successful and unsuccessful attempts, and make those findings easier to review through synchronized animation, biomechanics, comparison tools, and long-term development tracking.</p>
    </div>""",
            unsafe_allow_html=True,
        )

        public_left, public_middle, public_right = st.columns(
            3
        )

        with public_left:
            st.markdown(
                """
                <div class="bms-summary-card">
                    <h3>Shot Analysis</h3>
                    <p>
                        Detect key events, calculate objective biomechanics, and
                        review each tracked free throw frame by frame.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with public_middle:
            st.markdown(
                """
                <div class="bms-summary-card">
                    <h3>Player Development</h3>
                    <p>
                        Compare attempts with personal made-shot baselines and
                        track consistency, trends, and development priorities.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with public_right:
            st.markdown(
                """
                <div class="bms-summary-card">
                    <h3>Coach Communication</h3>
                    <p>
                        Convert technical movement data into clear diagnoses,
                        review windows, coaching focuses, and suggested drills.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """<div style="background:linear-gradient(135deg,#111b2e,#1f355b);border-radius:20px;padding:1.25rem 1.4rem;margin:1.1rem 0 1.5rem;color:#ffffff;box-shadow:0 12px 30px rgba(17,27,46,0.16);">
    <div style="color:#ffb18e;font-size:0.75rem;font-weight:800;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.45rem;">Created by</div>
    <div style="font-size:1.25rem;font-weight:800;margin-bottom:0.25rem;">Ankit Wadera</div>
    <div style="color:#d5deee;line-height:1.6;">Basketball Coach and Analytics Developer<br>Contact: <a href="mailto:ankitwadera2@gmail.com" style="color:#ffb18e;text-decoration:none;font-weight:700;">ankitwadera2@gmail.com</a></div>
    </div>""",
            unsafe_allow_html=True,
        )

    shooting_side = "RIGHT"

    with st.sidebar:
        st.markdown(
            "## ANKIT'S FREE THROW ANALYSIS SOFTWARE"
        )

        st.caption(
            "Created by Ankit Wadera"
        )

        st.markdown(
            "[ankitwadera2@gmail.com](mailto:ankitwadera2@gmail.com)"
        )

        st.caption(
            (
                "Unified Full Platform"
                if is_public_mode()
                else "Unified Full Platform"
            )
        )

        st.caption(
            (
                "Public demo data"
                if is_true_public_mode()
                else "Local coaching data"
            )
        )

        if True:
            st.markdown(
                "---"
            )

            st.markdown(
                "## Workspace"
            )

            workspace_groups = {
                "Start": [
                    "Home",
                    "Global Search",
                    "Executive Demo",
                ],
                "Analyze": [
                    "Single Shot Analysis",
                    "Comparison Studio",
                    "Session Analysis",
                    "Session Comparison",
                ],
                "Develop": [
                    "Player Profile",
                    "Player Development",
                    "Team Dashboard",
                ],
                "Manage": [
                    "Sessions & Shot Library",
                    "Roster & Organizations",
                ],
                "Reports": [
                    "Reports Center",
                    "Launch Readiness",
                ],
                "Advanced": [
                    "Full Studio",
                ],
            }

            workspace_icons = {
                "Home": "🏠  Home",
                "Global Search": "🔎  Global Search",
                "Single Shot Analysis": "🎯  Single Shot Analysis",
                "Comparison Studio": "↔️  Comparison Studio",
                "Executive Demo": "🏢  Executive Demo",
                "Player Profile": "👤  Player Profile",
                "Player Development": "📈  Player Development",
                "Team Dashboard": "👥  Team Dashboard",
                "Roster & Organizations": "🏀  Roster & Organizations",
                "Sessions & Shot Library": "📁  Sessions & Shot Library",
                "Session Analysis": "📊  Session Analysis",
                "Session Comparison": "🧭  Session Comparison",
                "Reports Center": "📄  Reports Center",
                "Launch Readiness": "🚀  Launch Readiness",
                "Full Studio": "🛠️  Full Studio",
            }

            current_workspace = st.session_state.get(
                "private_workspace",
                "Home",
            )

            workspace_to_group = {
                item: group_name
                for group_name, items in workspace_groups.items()
                for item in items
            }

            group_names = list(
                workspace_groups
            )

            default_group = workspace_to_group.get(
                current_workspace,
                "Start",
            )

            st.markdown(
                '<div class="bms-nav-heading">Navigation section</div>',
                unsafe_allow_html=True,
            )

            selected_group = st.radio(
                "Navigation section",
                options=group_names,
                index=group_names.index(
                    default_group
                ),
                key="private_workspace_group",
                horizontal=True,
                label_visibility="collapsed",
                help="Choose the section of the platform you want to work in.",
            )

            group_options = workspace_groups[
                selected_group
            ]

            if current_workspace not in group_options:
                st.session_state[
                    "private_workspace"
                ] = group_options[
                    0
                ]

            st.markdown(
                '<div class="bms-nav-heading">Pages</div>',
                unsafe_allow_html=True,
            )

            workspace = st.radio(
                "Workspace",
                options=group_options,
                key="private_workspace",
                format_func=lambda value: workspace_icons[
                    value
                ],
                label_visibility="collapsed",
                help=(
                    "Choose the exact page you want to open."
                ),
            )

            st.markdown(
                (
                    '<div class="bms-current-location">'
                    '<small>Currently viewing</small>'
                    f'<strong>{workspace}</strong>'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )

            st.markdown(
                """<div class="bms-sidebar-card">
<strong>Recommended demonstration</strong>
<p>Use Executive Demo for a front-office overview. Use Global Search for fast navigation.</p>
</div>""",
                unsafe_allow_html=True,
            )



            if is_hosted_portfolio_mode():
                workspace_label = (
                    "Blank Workspace"
                    if active_workspace_kind() == "blank"
                    else "Executive Demo"
                )

                st.markdown(
                    f"""<div class="bms-sidebar-card">
<strong>Active data workspace</strong>
<p>{workspace_label}<br>Data is isolated to this visitor and temporary.</p>
</div>""",
                    unsafe_allow_html=True,
                )

                if st.button(
                    "Choose Different Workspace",
                    use_container_width=True,
                    key="sidebar_reset_public_workspace",
                ):
                    reset_public_workspace()
                    st.rerun()

            quick_left, quick_right = st.columns(2)

            if quick_left.button(
                "Executive Demo",
                use_container_width=True,
                key="sidebar_quick_executive_demo",
            ):
                request_workspace_navigation(
                    "Executive Demo"
                )

            if quick_right.button(
                "Reports",
                use_container_width=True,
                key="sidebar_quick_reports",
            ):
                request_workspace_navigation(
                    "Reports Center"
                )

        if (
            is_public_mode()
            or workspace
            in {
                "Single Shot Analysis",
                "Comparison Studio",
                "Full Studio",
            }
        ):
            st.markdown(
                "---"
            )

            st.markdown(
                "## Analysis controls"
            )

            shooting_side = st.selectbox(
                "Shooting side",
                options=[
                    "RIGHT",
                    "LEFT",
                ],
                index=0,
            )

        if (
            is_public_mode()
            or workspace
            in {
                "Single Shot Analysis",
                "Comparison Studio",
                "Full Studio",
            }
        ):
            st.markdown(
                "---"
            )

            st.markdown(
                "### Personal scoring requirements"
            )

            st.caption(
                "The participant must already exist in the historical feature "
                "dataset and have personal diagnostics built from previous shots."
            )

            master_status = (
                "Ready"
                if MASTER_FEATURE_FILE.exists()
                else "Missing"
            )

            diagnostic_status = (
                "Ready"
                if PERSONAL_DIAGNOSTIC_FILE.exists()
                else "Missing"
            )

            st.write(
                f"Historical features: **{master_status}**"
            )

            st.write(
                f"Personal diagnostics: **{diagnostic_status}**"
            )

        st.markdown(
            "---"
        )

        st.caption(
            "Current version: professional interface prototype with "
            "diagnosis clips and synchronized shot comparison."
        )

    st.info(
        (
            "Version 1 platform: this application demonstrates an end-to-end "
            "basketball motion-analysis workflow using structured 3D tracking "
            "data. Scores and coaching interpretations are exploratory and "
            "should be reviewed alongside video and professional judgment."
        )
    )

    if is_public_mode():
        render_public_demo_guidance()

    else:
        if workspace == "Home":
            st.markdown(
                """<div style="background:#ffffff;border:1px solid #e4e8f0;border-radius:22px;padding:1.4rem 1.5rem;margin:0.8rem 0 1.2rem;">
<div style="color:#c94d18;font-size:0.76rem;font-weight:800;letter-spacing:0.11em;text-transform:uppercase;margin-bottom:0.45rem;">Choose a workspace</div>
<h2 style="margin:0 0 0.6rem;color:#152033;">Start with the job you need to complete</h2>
<p style="margin:0;color:#455066;line-height:1.65;">The platform is now separated into focused workspaces. Select one from the sidebar so you only see the tools related to that task.</p>
</div>""",
                unsafe_allow_html=True,
            )

            demo_left, demo_right = st.columns(
                [
                    0.72,
                    0.28,
                ]
            )

            demo_left.markdown(
                """<div class="bms-showcase-banner">
<div class="bms-showcase-eyebrow">Executive front-office demonstration</div>
<h2>See the complete player-development workflow in minutes</h2>
<p class="bms-showcase-copy">Move from a tracked free throw to smooth playback, biomechanics, coach interpretation, practice-session grading, long-term development, team intelligence, and professional PDF reporting.</p>
<div class="bms-feature-pill-row">
<span class="bms-feature-pill">Smooth playback</span>
<span class="bms-feature-pill">Biomechanics</span>
<span class="bms-feature-pill">Session grading</span>
<span class="bms-feature-pill">Player development</span>
<span class="bms-feature-pill">Team intelligence</span>
<span class="bms-feature-pill">PDF reporting</span>
</div>
</div>""",
                unsafe_allow_html=True,
            )

            with demo_right:
                st.markdown(
                    "### Showcase controls"
                )

                st.caption(
                    (
                        "Launch a front-office overview using "
                        "the data already stored in the platform."
                    )
                )

                if st.button(
                    "Open Executive Demo",
                    type="primary",
                    use_container_width=True,
                    key="home_start_guided_demo",
                ):
                    request_workspace_navigation(
                        "Executive Demo"
                    )

                if st.button(
                    "Open Reports Center",
                    use_container_width=True,
                    key="home_open_reports_center",
                ):
                    request_workspace_navigation(
                        "Reports Center"
                    )

            home_columns = st.columns(
                3
            )

            home_items = [
                (
                    "Global Search",
                    "Find players, sessions, teams, and tracked shots from one place.",
                ),
                (
                    "Single Shot Analysis",
                    "Upload and analyze one tracked free throw.",
                ),
                (
                    "Comparison Studio",
                    "Compare two shots with synchronized playback.",
                ),
                (
                    "Executive Demo",
                    "Open a front-office overview using existing data.",
                ),
                (
                    "Player Profile",
                    "Open one athlete's complete development overview.",
                ),
                (
                    "Player Development",
                    "Review long-term progress without unrelated tools.",
                ),
                (
                    "Team Dashboard",
                    "Review roster activity and coaching priorities.",
                ),
                (
                    "Roster & Organizations",
                    "Manage organizations, teams, and roster membership.",
                ),
                (
                    "Sessions & Shot Library",
                    "Upload files and organize them into practice sessions.",
                ),
                (
                    "Session Analysis",
                    "Review one practice session without unrelated tools.",
                ),
                (
                    "Session Comparison",
                    "Compare two complete practice sessions.",
                ),
                (
                    "Reports Center",
                    "Generate and download professional reports.",
                ),
                (
                    "Launch Readiness",
                    "Run automated checks before sharing the platform publicly.",
                ),
            ]

            for index, (
                heading,
                description,
            ) in enumerate(
                home_items
            ):
                with home_columns[
                    index
                    % 3
                ]:
                    st.markdown(
                        f"""<div class="bms-summary-card">
<h3>{heading}</h3>
<p>{description}</p>
</div>""",
                        unsafe_allow_html=True,
                    )

            return

        if workspace == "Global Search":
            section_title(
                "Global Search",
                "Find players, sessions, teams, and tracked shots without navigating through unrelated tools.",
            )

            render_global_search_workspace()

            return

        if workspace == "Team Dashboard":
            section_title(
                "Coach Team Dashboard",
                "Review roster-level activity, development scores, and coaching priorities.",
            )

            render_team_dashboard()

            return

        if workspace == "Executive Demo":
            section_title(
                "Executive Demo",
                "A front-office overview of session analysis, player development, team intelligence, and reporting.",
            )

            render_executive_demo_workspace()

            return

        if workspace == "Player Profile":
            section_title(
                "Player Profile",
                "Review one athlete's complete history, development snapshot, sessions, and team membership.",
            )

            render_player_profile_workspace()

            return

        if workspace == "Player Development":
            section_title(
                "Long-Term Player Development",
                "Track practice-to-practice consistency, category development, and individual biomechanical trends.",
            )

            render_player_development_timeline()

            return

        if workspace == "Roster & Organizations":
            section_title(
                "Roster and Organization Management",
                "Manage organizations, teams, seasons, and roster membership.",
            )

            render_organization_team_manager()

            return

        if workspace == "Sessions & Shot Library":
            section_title(
                "Sessions and Shot Library",
                "Choose a player and session, then upload shots directly into that session.",
            )

            render_session_first_upload_workflow()

            return

        if workspace == "Session Analysis":
            section_title(
                "Session Analysis",
                "Review one practice session, its shot progression, category consistency, and highest-priority attempts.",
            )

            render_session_analysis_workspace()

            return

        if workspace == "Session Comparison":
            section_title(
                "Session Comparison",
                "Compare two complete practice sessions for the same player.",
            )

            render_session_comparison_workspace()

            return

        if workspace == "Reports Center":
            section_title(
                "Reports Center",
                "Generate and download professional player and session reports.",
            )

            render_reports_center_workspace()

            return

        if workspace == "Launch Readiness":
            section_title(
                "Launch Readiness",
                "Run automated Version 1 checks and complete the final publishing checklist.",
            )

            render_launch_readiness_workspace()

            return

        if workspace == "Full Studio":
            section_title(
                "Coach Team Dashboard",
                "Review roster-level activity, development scores, and coaching priorities before opening individual shots.",
            )

            render_team_dashboard()

            section_title(
                "Long-Term Player Development Timeline",
                "Track practice-to-practice consistency, category development, and individual biomechanical trends.",
            )

            render_player_development_timeline()

            section_title(
                "Session Analysis",
                "Review one practice session, its shot progression, and highest-priority attempts.",
            )

            render_session_analysis_workspace()

            render_organization_team_manager()
            render_player_history_manager()
            render_practice_session_manager()

    if workspace == "Comparison Studio":
        section_title(
            "Comparison Studio",
            (
                "Upload two compatible free-throw tracking JSON files and "
                "generate one smooth synchronized split-screen MP4."
            ),
        )

        render_comparison_studio_workspace(
            shooting_side=shooting_side,
        )

        return

    section_title(
        "Analyze a tracked free throw",
        "Upload one JSON trial, verify the detected participant, then run the analysis.",
    )

    st.markdown(
        '<div class="bms-upload-shell">',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a free-throw tracking JSON",
        type=[
            "json",
        ],
        accept_multiple_files=False,
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    clear_stale_result(
        uploaded_file
    )

    if uploaded_file is None:
        st.info(
            "Upload a JSON trial to begin."
        )

        return

    try:
        trial_data = json.loads(
            uploaded_file.getvalue()
        )

        trial_data = validate_uploaded_json(
            trial_data
        )

    except Exception as error:
        st.error(
            f"Could not read the uploaded JSON: {error}"
        )

        return

    participant_id = trial_data.get(
        "participant_id",
        "Unknown",
    )

    trial_id = trial_data.get(
        "trial_id",
        Path(
            uploaded_file.name
        ).stem,
    )

    detected_one, detected_two, detected_three = st.columns(
        3
    )

    detected_one.metric(
        "Detected participant",
        participant_id,
    )

    detected_two.metric(
        "Detected trial",
        trial_id,
    )

    detected_three.metric(
        "File size",
        (
            f"{len(uploaded_file.getvalue()) / 1_000_000:.2f} MB"
        ),
    )

    analyze_clicked = st.button(
        "Run Complete Shot Analysis",
        type="primary",
        use_container_width=True,
    )

    if analyze_clicked:
        with st.spinner(
            "Analyzing tracking, biomechanics, timing, and personal baseline..."
        ):
            try:
                result = analyze_uploaded_trial(
                    trial_data=trial_data,
                    original_filename=(
                        uploaded_file.name
                    ),
                    shooting_side=(
                        shooting_side
                    ),
                )

            except Exception as error:
                st.error(
                    (
                        "The analysis could not be completed.\n\n"
                        f"{type(error).__name__}: {error}"
                    )
                )

                return

        st.session_state.analysis_result = (
            result
        )

        st.session_state.analysis_filename = (
            uploaded_file.name
        )

        st.session_state.analysis_trial_data = (
            trial_data
        )

    if (
        st.session_state.analysis_result
        is not None
    ):
        display_results(
            result=(
                st.session_state
                .analysis_result
            ),
            uploaded_name=(
                st.session_state
                .analysis_filename
            ),
            trial_data=(
                st.session_state
                .analysis_trial_data
            ),
            shooting_side=(
                shooting_side
            ),
        )

    st.markdown(
        """<div style="margin-top:3rem;padding-top:1.25rem;border-top:1px solid #e4e8f0;text-align:center;color:#667085;font-size:0.84rem;line-height:1.7;">
<strong style="color:#152033;">ANKIT'S FREE THROW ANALYSIS SOFTWARE</strong><br>
Created by Ankit Wadera · <a href="mailto:ankitwadera2@gmail.com" style="color:#c94d18;text-decoration:none;font-weight:700;">ankitwadera2@gmail.com</a><br>
Public basketball analytics and player-development prototype
</div>""",
        unsafe_allow_html=True,
    )


    st.markdown(
        """<div class="bms-footer">
ANKIT'S FREE THROW ANALYSIS SOFTWARE · Created by Ankit Wadera ·
<a href="mailto:ankitwadera2@gmail.com">ankitwadera2@gmail.com</a><br>
Exploratory structured-tracking analytics for coaching review and player development.
</div>""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
