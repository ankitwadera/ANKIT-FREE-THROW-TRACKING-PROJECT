# 🏀 ANKIT FREE THROW TRACKING PROJECT

## Basketball Biomechanics & Player Development Platform

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Render](https://img.shields.io/badge/Cloud-Deployable-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

> **A Python-based basketball analytics platform demonstrating how structured
> Hawk-Eye-style optical tracking data can be transformed into player development,
> biomechanics analysis, coaching workflows, and basketball operations intelligence.**

---

# Project Overview

ANKIT FREE THROW TRACKING PROJECT demonstrates how modern optical player-tracking
technology can extend beyond data collection and become a complete basketball
player-development platform.

Using structured three-dimensional free-throw tracking data, the software:

- reconstructs player movement
- identifies key shooting events
- evaluates biomechanics
- measures consistency
- compares shooting sessions
- tracks player development
- generates professional coaching reports

The goal is not to define a single "perfect" shooting motion. Instead, the platform
builds an individual movement profile for each athlete and measures development over time.

---

# Key Features

## Motion Analysis

✅ 3D Skeleton Reconstruction

✅ Ball Trajectory Visualization

✅ Automatic Shot Event Detection

✅ Release Analysis

✅ Smooth MP4 Playback

✅ Split-Screen Comparison Studio

---

## Biomechanics

- Release Biomechanics
- Phase Analysis
- Kinetic Chain Analysis
- Joint Sequencing
- Timing Analysis
- Consistency Metrics

---

## Player Development

- Practice Session Management
- Development Timelines
- Historical Progress Tracking
- Session Grading
- Coaching Recommendations
- Improvement Trends

---

## Basketball Operations

- Organization Management
- Team Management
- Player Profiles
- Roster Management
- Team Dashboards
- Executive Reporting

---

## Professional Reporting

- Session Reports
- Player Development Reports
- Team Reports
- Comparison Reports
- Executive PDF Reports

---

# Public Portfolio Modes

| Workspace | Purpose |
|-----------|---------|
| **Blank Workspace** | Create your own organization and upload compatible basketball free-throw tracking JSON files. |
| **Executive Demo** | Explore a fully populated demonstration showing the complete coaching workflow. |

---

# Supported Tracking Data

Compatible JSON files contain:

- Ball coordinates
- Player body keypoints
- Participant ID
- Trial ID
- Shot result
- Frame timing
- Tracking sequence

---

# Unsupported Formats

- Video
- CSV
- Excel
- Shot Charts
- Images
- Generic JSON

---

# Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Framework | Streamlit |
| Motion Processing | NumPy • SciPy |
| Visualization | Matplotlib • PyVista |
| Video Rendering | ImageIO • FFmpeg |
| Reporting | ReportLab |
| Database | SQLite |
| Deployment | Docker • Render |
| Version Control | GitHub |

---

# Software Workflow

```text
Tracking JSON
      │
      ▼
Motion Reconstruction
      │
      ▼
Shot Event Detection
      │
      ▼
Biomechanics Analysis
      │
      ▼
Player Development Intelligence
      │
      ▼
Professional Reports
```

---

# Screenshots

Add screenshots or GIFs here.

Suggested order:

1. Dashboard
2. Smooth MP4 Playback
3. Comparison Studio
4. Player Development Timeline
5. Team Dashboard
6. PDF Report

---

# Local Installation

## Clone Repository

```bash
git clone https://github.com/ankitwadera/ANKIT-FREE-THROW-TRACKING-PROJECT.git
```

## Create Virtual Environment

```bash
python -m venv .venv
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

PowerShell

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:APP_MODE = "private"
$env:ANKIT_APP_DATA_ROOT = "data"

.\.venv\Scripts\python.exe -m streamlit run tracking_app\research\shot_analysis_app.py
```

---

# Docker

Build

```bash
docker build -t ankit-free-throw-analysis .
```

Run

```bash
docker run --rm -p 10000:10000 ankit-free-throw-analysis
```

Open

```
http://localhost:10000
```

---

# Cloud Deployment

The project includes Docker and Render configuration.

Deployment requires:

- GitHub
- Docker
- Render

The public deployment provides:

- Blank Workspace
- Executive Demo
- Browser-based analysis
- Professional reporting

---

# Roadmap

Future improvements include:

- Multi-camera synchronization
- Additional basketball movements
- AI-assisted coaching recommendations
- Expanded biomechanics dashboards
- Organization-level analytics
- Multi-user authentication

---

# Data Disclaimer

The demonstration dataset originates from the SPL Open Data basketball
free-throw dataset.

Player names shown within the Executive Demo are portfolio labels used only
to demonstrate software workflows and should not be interpreted as actual
tracking data collected from those athletes or organizations.

---

# Author

## Ankit Wadera

Basketball Analytics • Player Development • Software Engineering

📧 ankitwadera2@gmail.com

---

MIT License
