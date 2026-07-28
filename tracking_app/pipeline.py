from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tracking_app.analyzer import ShotAnalysis, ShotAnalyzer
from tracking_app.events import ShotEvents, detect_shot_events


@dataclass
class PipelineResult:
    """
    Stores everything produced by the analysis pipeline.

    For now, the pipeline returns:

    1. Shot-level metrics
    2. Detected shot events

    Future stages can add filtered tracking data,
    time-series curves, quality checks and model predictions.
    """

    analysis: ShotAnalysis
    events: ShotEvents


class AnalysisPipeline:
    """
    Coordinate the full analysis of one basketball shot.

    The pipeline is responsible for deciding the order in which
    each analytics stage runs.

    Current pipeline
    ----------------
    Raw tracking data
            ↓
    Event detection
            ↓
    Shot analysis
            ↓
    Pipeline result
    """

    def __init__(
        self,
        trial_data: dict,
        trial_file: Path | None = None,
        shooting_wrist: str = "RIGHT_WRIST",
    ) -> None:
        self.trial_data = trial_data
        self.trial_file = trial_file
        self.shooting_wrist = shooting_wrist

        self.tracking_frames = trial_data.get(
            "tracking",
            [],
        )

        self.sampling_rate = trial_data.get(
            "sampling_rate",
            30,
        )

        self._validate_trial()

    def _validate_trial(self) -> None:
        """
        Confirm that the trial contains the minimum data
        required by the analysis pipeline.
        """

        if not isinstance(self.trial_data, dict):
            raise TypeError(
                "trial_data must be a dictionary."
            )

        if not isinstance(self.tracking_frames, list):
            raise TypeError(
                "The tracking section must be a list of frames."
            )

        if not self.tracking_frames:
            raise ValueError(
                "The selected trial contains no tracking frames."
            )

        if not isinstance(
            self.sampling_rate,
            (int, float),
        ):
            raise TypeError(
                "The sampling rate must be numeric."
            )

        if self.sampling_rate <= 0:
            raise ValueError(
                "The sampling rate must be greater than zero."
            )

    def _detect_events(self) -> ShotEvents:
        """
        Run the event-detection stage.

        Events currently include:

        - motion start;
        - dip;
        - takeoff;
        - release;
        - ball apex;
        - landing.
        """

        return detect_shot_events(
            tracking_frames=self.tracking_frames,
            sampling_rate=float(self.sampling_rate),
            shooting_wrist=self.shooting_wrist,
        )

    def _analyze_shot(self) -> ShotAnalysis:
        """
        Run the permanent ShotAnalyzer.
        """

        analyzer = ShotAnalyzer(
            trial_data=self.trial_data,
            trial_file=self.trial_file,
            shooting_wrist=self.shooting_wrist,
        )

        return analyzer.analyze()

    def run(self) -> PipelineResult:
        """
        Execute the complete analysis pipeline.

        The order matters:

        1. Detect important basketball events.
        2. Calculate shot and biomechanics metrics.
        3. Package everything into one reusable result.
        """

        events = self._detect_events()
        analysis = self._analyze_shot()

        # Both systems should identify the same release.
        # If they disagree, something in the analysis logic
        # needs to be reviewed instead of silently continuing.
        if (
            events.release_frame is not None
            and events.release_frame
            != analysis.release_frame
        ):
            raise ValueError(
                "Release-frame disagreement detected.\n"
                f"Event detector: {events.release_frame}\n"
                f"ShotAnalyzer: {analysis.release_frame}"
            )

        return PipelineResult(
            analysis=analysis,
            events=events,
        )