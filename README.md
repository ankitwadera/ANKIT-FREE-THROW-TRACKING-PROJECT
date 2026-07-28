ANKIT'S FREE THROW TRACKING PROJECT
Basketball Biomechanics & Player Development Platform

ANKIT'S FREE THROW TRACKING PROJECT is a Python-based basketball analytics platform that demonstrates how structured 3D optical tracking data (Hawk-Eye-style) can be transformed into practical coaching and player-development intelligence.

The software reconstructs basketball free throws from tracking data, automatically detects key shooting events, analyzes biomechanics and kinetic sequencing, evaluates consistency over time, and converts the results into coach-readable reports, player development insights, and front-office decision-support tools.

The project was developed to demonstrate how modern tracking technology can extend beyond data collection and become a practical workflow for coaches, player-development staff, performance analysts, and basketball operations personnel.

Project Objectives

This project explores how structured 3D basketball tracking data can be used to:

Reconstruct complete free-throw movements.
Automatically detect key shot events.
Evaluate shooting biomechanics.
Measure movement consistency.
Track player development over time.
Compare successful and unsuccessful shooting patterns.
Generate coach-ready feedback.
Support basketball operations through organized reporting and analytics.

Rather than attempting to define a single "perfect" shooting motion, the software focuses on identifying each player's individual movement profile and monitoring meaningful changes across training sessions.

Key Features
Motion Reconstruction
3D skeleton visualization
Ball trajectory reconstruction
Shot event detection
Release analysis
Smooth MP4 playback
Side-by-side comparison
Biomechanics
Joint sequencing
Kinetic-chain analysis
Phase biomechanics
Release mechanics
Timing analysis
Consistency measurements
Player Development
Session grading
Development timelines
Historical progress
Coaching priorities
Drill recommendations
Practice summaries
Team Management
Organizations
Teams
Rosters
Practice sessions
Player profiles
Team dashboards
Reporting
Session Reports
Player Development Reports
Team Practice Reports
Session Comparison Reports
Executive summaries
Professional PDF exports
Data

The application is designed for structured basketball free-throw tracking JSON files.

Each file contains a complete tracked free-throw attempt, including:

Ball coordinates
Player body keypoints
Frame timing
Participant ID
Trial ID
Shot outcome (made/missed)

The public portfolio includes a curated demonstration dataset based on the SPL Open Data free-throw dataset.

Technology Stack
Python
Streamlit
NumPy
Matplotlib
PyVista
VTK
ImageIO / FFmpeg
ReportLab
SQLite
Docker
GitHub
Render
Workflow

The platform supports two workflows.

Blank Workspace

Designed for users who want to analyze their own compatible free-throw tracking JSON files.

Users can:

Create organizations
Create teams
Create players
Upload tracking files
Build practice sessions
Analyze mechanics
Generate reports
Executive Demo

A fully populated demonstration environment showcasing the complete workflow using a curated portfolio dataset.

The demonstration includes:

Organizations
Teams
Players
Practice sessions
Player development history
Team dashboards
Executive reporting
PDF exports
Technical Demonstration

This repository demonstrates experience in:

Sports analytics
Basketball biomechanics
Motion analysis
Software engineering
Python application development
Data visualization
Database design
User interface design
Product development
Cloud deployment
Disclaimer

This software demonstrates analytical workflows using structured basketball tracking data.

The demonstration dataset is derived from the SPL Open Data free-throw dataset. Player names used within the Executive Demo are portfolio labels intended to demonstrate software functionality and do not represent tracking data collected from those athletes.

Contact

Ankit Wadera
📧 ankitwadera2@gmail.com
