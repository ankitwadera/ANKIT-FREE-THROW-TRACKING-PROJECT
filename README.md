# ANKIT'S FREE THROW ANALYSIS SOFTWARE

A Python and Streamlit platform for structured three-dimensional basketball
free-throw tracking analysis, practice-session intelligence, player
development, roster workflows, and professional PDF reporting.

Created by **Ankit Wadera**  
Contact: **ankitwadera2@gmail.com**

## Public portfolio experience

The hosted application begins with two choices:

### Blank Workspace

Start with no organizations, teams, players, sessions, or shot files. Users can
create their own workspace and upload compatible free-throw tracking JSON data.

Compatible uploads should contain:

- One basketball free-throw attempt per JSON file
- A non-empty frame-by-frame `tracking` list
- Ball X/Y/Z coordinates
- Player body keypoints
- Participant ID
- Trial ID
- Made or missed result
- Sampling-rate or frame-timing information

The application does not currently accept ordinary video, CSV, Excel,
box-score JSON, play-by-play JSON, shot charts, photographs, or unrelated
JSON formats.

### Executive Demo

Load an isolated temporary copy of the curated portfolio dataset, including:

- Toronto Raptors Basketball Operations
- Toronto Raptors and Raptors 905
- Five portfolio player labels
- Fifteen practice sessions
- Seventy-five curated SPL Open Data tracking files
- Session analysis and comparison
- Player-development timelines
- Team dashboards
- PDF reports

## Data disclaimer

The packaged tracking files originate from the SPL Open Data basketball
free-throw dataset. Named professional players are portfolio labels used to
demonstrate product workflows. The software does not claim that the underlying
tracking data was collected from those named athletes or organizations.

## Core capabilities

- Three-dimensional skeleton and ball reconstruction
- Smooth playback and frame-by-frame review
- Automated shot-event detection
- Release and phase biomechanics
- Kinetic-chain and synchronization analysis
- Single-shot and side-by-side comparison
- Practice-session management
- Session grading and coach-readable action plans
- Player-development intelligence
- Team practice intelligence
- Organization, team, and roster management
- Session, player-development, team, and comparison PDF reports

## Local development

Requirements:

- Python 3.12
- Windows, macOS, or Linux
- Packages listed in `requirements.txt`

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Run the complete local coaching workspace:

```bash
export PYTHONPATH="$PWD"
export APP_MODE=private
export ANKIT_APP_DATA_ROOT=data
python -m streamlit run tracking_app/research/shot_analysis_app.py
```

PowerShell:

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:APP_MODE = "private"
$env:ANKIT_APP_DATA_ROOT = "data"
.\.venv\Scripts\python.exe -m streamlit run tracking_app\research\shot_analysis_app.py
```

## Docker

Build:

```bash
docker build -t ankit-free-throw-analysis .
```

Run:

```bash
docker run --rm -p 10000:10000 ankit-free-throw-analysis
```

Open:

```text
http://localhost:10000
```

## Render

This repository includes `render.yaml`.

1. Push the repository to GitHub.
2. Sign in to Render.
3. Create a new Blueprint.
4. Select this repository.
5. Render builds the Docker image and creates the public web service.
6. Open the generated `onrender.com` URL.

The Render service runs the public Blank Workspace / Executive Demo experience.

## Repository structure

```text
tracking_app/             Application and analysis modules
assets/brand/             Logo and application icons
deployment_data/          Curated public portfolio database and JSON files
.streamlit/config.toml    Streamlit visual configuration
Dockerfile                Production container
start.sh                  Production startup command
render.yaml               Render Blueprint
requirements.txt          Python dependencies
```

## Full SPL dataset

The complete SPL Open Data dataset is intentionally not committed to this
repository because of its size. Download the original dataset separately and
use the included builder scripts to create a full local environment.

## License

MIT License. See `LICENSE`.
