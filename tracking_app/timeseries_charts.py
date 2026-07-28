from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from tracking_app.analyzer import ShotAnalyzer
from tracking_app.data_loader import load_first_trial
from tracking_app.timeseries import ShotTimeSeries, TimeSeriesAnalyzer


@dataclass(frozen=True)
class EventFrames:
    """
    Detected shot-event frame numbers used by the chart panel.
    """

    motion_start: int | None
    dip: int | None
    takeoff: int | None
    release: int | None
    ball_apex: int | None
    landing: int | None

    def as_items(self) -> list[tuple[str, int]]:
        values = [
            ("Motion start", self.motion_start),
            ("Dip", self.dip),
            ("Takeoff", self.takeoff),
            ("Release", self.release),
            ("Ball apex", self.ball_apex),
            ("Landing", self.landing),
        ]

        return [
            (name, int(frame))
            for name, frame in values
            if frame is not None
        ]


class TimeSeriesChartPanel:
    """
    Matplotlib chart manager for basketball shot time series.
    """

    def __init__(
        self,
        time_series: ShotTimeSeries,
        event_frames: EventFrames,
    ) -> None:
        self.time_series = time_series
        self.event_frames = event_frames

        self.figure: Figure | None = None
        self.axes: list[Axes] = []
        self.cursor_lines: list[Line2D] = []

        self.current_frame = 0

    def build(self) -> Figure:
        """
        Create the complete four-chart biomechanics dashboard.
        """

        self.figure, axes_array = plt.subplots(
            nrows=4,
            ncols=1,
            figsize=(13, 10),
            sharex=True,
            constrained_layout=True,
        )

        self.axes = list(axes_array)
        frame_index = self.time_series.frame_index

        self._plot_series(
            axis=self.axes[0],
            x=frame_index,
            series=[
                (self.time_series.ball_height, "Ball height"),
                (self.time_series.pelvis_height, "Pelvis height"),
                (
                    self.time_series.average_ankle_height,
                    "Average ankle height",
                ),
            ],
            title="VERTICAL POSITION",
            ylabel="Height (ft)",
        )

        self._plot_series(
            axis=self.axes[1],
            x=frame_index,
            series=[
                (self.time_series.ball_speed, "Ball speed"),
                (
                    self.time_series.right_wrist_speed,
                    "Right wrist speed",
                ),
                (
                    self.time_series.left_wrist_speed,
                    "Left wrist speed",
                ),
            ],
            title="SPEED",
            ylabel="Speed (ft/s)",
        )

        self._plot_series(
            axis=self.axes[2],
            x=frame_index,
            series=[
                (
                    self.time_series.right_elbow_angle,
                    "Right elbow",
                ),
                (
                    self.time_series.left_elbow_angle,
                    "Left elbow",
                ),
                (
                    self.time_series.right_shoulder_angle,
                    "Right shoulder",
                ),
                (
                    self.time_series.left_shoulder_angle,
                    "Left shoulder",
                ),
            ],
            title="UPPER-BODY JOINT ANGLES",
            ylabel="Angle (degrees)",
        )

        self._plot_series(
            axis=self.axes[3],
            x=frame_index,
            series=[
                (
                    self.time_series.right_knee_angle,
                    "Right knee",
                ),
                (
                    self.time_series.left_knee_angle,
                    "Left knee",
                ),
                (
                    self.time_series.right_hip_angle,
                    "Right hip",
                ),
                (
                    self.time_series.left_hip_angle,
                    "Left hip",
                ),
            ],
            title="LOWER-BODY JOINT ANGLES",
            ylabel="Angle (degrees)",
        )

        self.axes[-1].set_xlabel("Frame")

        for axis in self.axes:
            self._add_event_markers(axis)

            cursor = axis.axvline(
                x=self.current_frame,
                linewidth=2,
                linestyle="-",
                label="_current_frame",
            )

            self.cursor_lines.append(cursor)

            axis.grid(
                visible=True,
                alpha=0.25,
            )

            axis.set_xlim(
                0,
                max(
                    1,
                    self.time_series.total_frames - 1,
                ),
            )

        self.figure.suptitle(
            "BASKETBALL SHOT TIME-SERIES ANALYSIS",
            fontsize=15,
            fontweight="bold",
        )

        return self.figure

    @staticmethod
    def _plot_series(
        axis: Axes,
        x: np.ndarray,
        series: list[tuple[np.ndarray, str]],
        title: str,
        ylabel: str,
    ) -> None:
        for values, label in series:
            axis.plot(
                x,
                values,
                linewidth=1.6,
                label=label,
            )

        axis.set_title(
            title,
            fontsize=11,
            fontweight="bold",
        )

        axis.set_ylabel(ylabel)

        axis.legend(
            loc="upper left",
            fontsize=8,
            ncols=2,
        )

    def _add_event_markers(
        self,
        axis: Axes,
    ) -> None:
        for _, frame in self.event_frames.as_items():
            axis.axvline(
                x=frame,
                linewidth=1,
                linestyle="--",
                alpha=0.45,
                label="_event_marker",
            )

        if axis is not self.axes[0]:
            return

        y_min, y_max = axis.get_ylim()
        label_height = (
            y_min
            + (y_max - y_min) * 0.96
        )

        for event_name, frame in self.event_frames.as_items():
            axis.text(
                frame,
                label_height,
                event_name,
                rotation=90,
                va="top",
                ha="right",
                fontsize=7,
                alpha=0.75,
            )

    def update_cursor(
        self,
        frame_index: int,
        redraw: bool = True,
    ) -> None:
        """
        Move the vertical current-frame cursor on every chart.
        """

        self.current_frame = max(
            0,
            min(
                int(frame_index),
                self.time_series.total_frames - 1,
            ),
        )

        for cursor in self.cursor_lines:
            cursor.set_xdata(
                [
                    self.current_frame,
                    self.current_frame,
                ]
            )

        if (
            redraw
            and self.figure is not None
            and self.figure.canvas is not None
        ):
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()

    def show(
        self,
        block: bool = False,
    ) -> None:
        if self.figure is None:
            self.build()

        plt.show(block=block)

    def save(
        self,
        output_path: str | Path,
        dpi: int = 160,
    ) -> Path:
        if self.figure is None:
            self.build()

        destination = Path(output_path)
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.figure.savefig(
            destination,
            dpi=dpi,
            bbox_inches="tight",
        )

        return destination

    def close(self) -> None:
        if self.figure is not None:
            plt.close(self.figure)


def build_event_frames(
    analysis: object,
) -> EventFrames:
    """
    Convert ShotAnalysis fields into the chart event structure.
    """

    return EventFrames(
        motion_start=getattr(
            analysis,
            "motion_start_frame",
            None,
        ),
        dip=getattr(
            analysis,
            "dip_frame",
            None,
        ),
        takeoff=getattr(
            analysis,
            "takeoff_frame",
            None,
        ),
        release=getattr(
            analysis,
            "release_frame",
            None,
        ),
        ball_apex=getattr(
            analysis,
            "ball_apex_frame",
            None,
        ),
        landing=getattr(
            analysis,
            "landing_frame",
            None,
        ),
    )


def _run_chart_test() -> None:
    """
    Build and export a chart dashboard for the first trial.
    """

    trial_file, trial_data = load_first_trial()

    shot_analysis = ShotAnalyzer(
        trial_data=trial_data,
        trial_file=trial_file,
        shooting_wrist="RIGHT_WRIST",
    ).analyze()

    time_series = TimeSeriesAnalyzer(
        trial_data=trial_data,
        smoothing_window=5,
    ).analyze()

    panel = TimeSeriesChartPanel(
        time_series=time_series,
        event_frames=build_event_frames(
            shot_analysis
        ),
    )

    panel.build()
    panel.update_cursor(
        shot_analysis.release_frame,
        redraw=False,
    )

    output_path = panel.save(
        Path("outputs")
        / "first_trial_timeseries.png"
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert len(panel.axes) == 4
    assert len(panel.cursor_lines) == 4

    print("TIME-SERIES CHART TEST PASSED")
    print(f"Trial file: {trial_file.name}")
    print(f"Charts created: {len(panel.axes)}")
    print(
        "Cursor frame: "
        f"{panel.current_frame}"
    )
    print(
        "Output file: "
        f"{output_path.resolve()}"
    )

    panel.close()


if __name__ == "__main__":
    _run_chart_test()
