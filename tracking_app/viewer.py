from __future__ import annotations

from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from tracking_app.analyzer import ShotAnalysis, ShotAnalyzer
from tracking_app.ball import get_longest_ball_segment
from tracking_app.court import add_basketball_court
from tracking_app.data_loader import load_first_trial
from tracking_app.playback import PlaybackController
from tracking_app.skeleton import (
    get_usable_connections,
    is_valid_point,
)
from tracking_app.timeseries import ShotTimeSeries, TimeSeriesAnalyzer
from tracking_app.timeseries_charts import (
    EventFrames,
    TimeSeriesChartPanel,
    build_event_frames,
)


class TrackingViewer:
    """
    Interactive basketball tracking and shot-analysis workstation.

    Frame controls
    --------------
    N             Next frame
    P             Previous frame
    Right Arrow   Jump forward 10 frames
    Left Arrow    Jump backward 10 frames

    Event controls
    --------------
    1             Motion start
    2             Dip
    3             Takeoff
    4             Release
    5             Ball apex
    6             Landing
    G             Jump to release

    Playback controls
    -----------------
    Space         Play or pause
    7             0.25x speed
    8             0.50x speed
    9             1.00x speed
    0             2.00x speed
    L             Toggle looping

    Viewer controls
    ---------------
    T             Show or hide trajectory
    C             Show or hide synchronized charts
    R             Reset camera
    """

    def __init__(self) -> None:
        # =================================================
        # LOAD THE TRIAL
        # =================================================

        self.trial_file, self.trial_data = load_first_trial()

        self.tracking_frames = self.trial_data.get(
            "tracking",
            [],
        )

        if not self.tracking_frames:
            raise ValueError(
                "The selected trial contains no tracking frames."
            )

        self.participant = self.trial_data.get(
            "participant_id",
            "Unknown",
        )

        self.trial = self.trial_data.get(
            "trial_id",
            "Unknown",
        )

        self.result = self.trial_data.get(
            "result",
            "Unknown",
        )

        self.entry_angle = self.trial_data.get(
            "entry_angle",
            None,
        )

        self.sampling_rate = self._get_sampling_rate()

        # =================================================
        # RUN THE SHOT ANALYZER
        # =================================================

        self.analysis: ShotAnalysis | None = None
        self.analysis_error: str | None = None

        try:
            analyzer = ShotAnalyzer(
                trial_data=self.trial_data,
                trial_file=self.trial_file,
                shooting_wrist="RIGHT_WRIST",
            )

            self.analysis = analyzer.analyze()

        except (
            ValueError,
            TypeError,
            IndexError,
            KeyError,
        ) as error:
            self.analysis_error = str(error)

        # =================================================
        # BUILD FRAME-BY-FRAME TIME SERIES
        # =================================================

        self.time_series: ShotTimeSeries | None = None
        self.time_series_error: str | None = None

        try:
            self.time_series = TimeSeriesAnalyzer(
                trial_data=self.trial_data,
                smoothing_window=5,
            ).analyze()

        except (
            ValueError,
            TypeError,
            IndexError,
            KeyError,
        ) as error:
            self.time_series_error = str(error)

        # =================================================
        # IDENTIFY SKELETON CONNECTIONS
        # =================================================

        first_player_data = (
            self.tracking_frames[0]
            .get("data", {})
            .get("player", {})
        )

        self.usable_connections = get_usable_connections(
            first_player_data.keys()
        )

        if not self.usable_connections:
            raise ValueError(
                "No usable skeleton connections were found."
            )

        # =================================================
        # VIEWER STATE
        # =================================================

        self.current_frame = 0
        self.camera_initialized = False
        self.slider_widget = None

        self.trajectory_visible = True
        self.trajectory_actor = None
        self.release_marker_actor = None

        self.chart_panel: TimeSeriesChartPanel | None = None
        self.chart_event_frames: EventFrames | None = None
        self.chart_visible = False

        if (
            self.analysis is not None
            and self.time_series is not None
        ):
            self.chart_event_frames = build_event_frames(
                self.analysis
            )

        # =================================================
        # CREATE THE WINDOW
        # =================================================

        self.plotter = pv.Plotter(
            window_size=(1600, 950),
        )

        self.plotter.set_background("black")
        self.plotter.add_axes()

        self.joint_actors: dict[str, object] = {}

        self.connection_meshes: dict[
            tuple[str, str],
            pv.PolyData,
        ] = {}

        self.connection_actors: dict[
            tuple[str, str],
            object,
        ] = {}

        self.ball_actor = None

        # =================================================
        # PLAYBACK ENGINE
        # =================================================

        self.playback = PlaybackController(
            total_frames=len(self.tracking_frames),
            source_fps=self.sampling_rate,
            frame_callback=self._display_playback_frame,
            loop_enabled=True,
        )

        # =================================================
        # BUILD VIEWER
        # =================================================

        self._create_scene()
        self._register_controls()
        self._create_frame_slider()

    # =====================================================
    # DATA SETTINGS
    # =====================================================

    def _get_sampling_rate(self) -> float:
        """
        Read and validate the source tracking frame rate.
        """

        sampling_rate = self.trial_data.get(
            "sampling_rate",
            30,
        )

        try:
            numeric_rate = float(sampling_rate)

        except (TypeError, ValueError):
            numeric_rate = 30.0

        if numeric_rate <= 0:
            numeric_rate = 30.0

        return numeric_rate

    # =====================================================
    # CREATE THE SCENE
    # =====================================================

    def _create_scene(self) -> None:
        """
        Create the court, skeleton, basketball,
        trajectory and analytics panels.
        """

        add_basketball_court(self.plotter)

        joint_names = sorted(
            {
                joint_name
                for connection in self.usable_connections
                for joint_name in connection
            }
        )

        # -------------------------------------------------
        # BODY JOINTS
        # -------------------------------------------------

        for joint_name in joint_names:
            joint_mesh = pv.Sphere(
                radius=0.065,
                center=(0, 0, 0),
            )

            joint_actor = self.plotter.add_mesh(
                joint_mesh,
                color="lightgray",
                smooth_shading=True,
                show_edges=False,
            )

            joint_actor.SetVisibility(False)

            self.joint_actors[joint_name] = joint_actor

        # -------------------------------------------------
        # BODY SEGMENTS
        # -------------------------------------------------

        for connection in self.usable_connections:
            line_mesh = pv.Line(
                pointa=(0, 0, 0),
                pointb=(0, 0, 0),
            )

            line_actor = self.plotter.add_mesh(
                line_mesh,
                color="navy",
                line_width=7,
            )

            line_actor.SetVisibility(False)

            self.connection_meshes[connection] = line_mesh
            self.connection_actors[connection] = line_actor

        # -------------------------------------------------
        # CURRENT BASKETBALL
        # -------------------------------------------------

        ball_mesh = pv.Sphere(
            radius=0.16,
            center=(0, 0, 0),
        )

        self.ball_actor = self.plotter.add_mesh(
            ball_mesh,
            color="orange",
            smooth_shading=True,
            show_edges=False,
        )

        self.ball_actor.SetVisibility(False)

        # -------------------------------------------------
        # TRAJECTORY AND RELEASE MARKER
        # -------------------------------------------------

        self._create_clean_ball_trajectory()
        self._create_release_marker()

        # -------------------------------------------------
        # ANALYSIS PANEL
        # -------------------------------------------------

        self._create_combined_analysis_panel()

        # -------------------------------------------------
        # CONTROL INSTRUCTIONS
        # -------------------------------------------------

        self.plotter.add_text(
            (
                "FRAME CONTROLS\n"
                "N / P: +/- 1 frame\n"
                "Left / Right: +/- 10\n"
                "\n"
                "PLAYBACK\n"
                "Space: play / pause\n"
                "7: 0.25x\n"
                "8: 0.50x\n"
                "9: 1.00x\n"
                "0: 2.00x\n"
                "L: toggle loop\n"
                "\n"
                "EVENTS\n"
                "1: motion start\n"
                "2: dip\n"
                "3: takeoff\n"
                "4: release\n"
                "5: ball apex\n"
                "6: landing\n"
                "\n"
                "G: release\n"
                "T: trajectory\n"
                "C: charts\n"
                "R: reset camera"
            ),
            position=(1280, 610),
            font_size=7,
            color="white",
            name="control_instructions",
        )

    # =====================================================
    # COMBINED LEFT PANEL
    # =====================================================

    def _create_combined_analysis_panel(self) -> None:
        """
        Display shot analysis, biomechanics and events
        as one continuous left-side panel.
        """

        def format_value(
            value: float | None,
            suffix: str = "",
        ) -> str:
            if value is None:
                return "N/A"

            return f"{value:.2f}{suffix}"

        def format_frame(
            value: int | None,
        ) -> str:
            if value is None:
                return "Not detected"

            return str(value)

        if self.analysis is None:
            panel_text = (
                "SHOT ANALYSIS\n"
                "------------------------------\n"
                "Unavailable\n"
                f"{self.analysis_error or 'Unknown error'}"
            )

        else:
            panel_text = (
                "SHOT ANALYSIS\n"
                "------------------------------\n"
                f"Result: {str(self.analysis.result).title()}\n"
                f"Release frame: {self.analysis.release_frame}\n"
                f"Release height: "
                f"{self.analysis.release_height:.2f} ft\n"
                f"Ball speed: "
                f"{self.analysis.ball_speed:.2f} ft/s\n"
                f"Horizontal speed: "
                f"{self.analysis.ball_horizontal_speed:.2f} ft/s\n"
                f"Vertical speed: "
                f"{self.analysis.ball_vertical_speed:.2f} ft/s\n"
                f"Release angle: "
                f"{self.analysis.release_angle:.2f} degrees\n"
                f"Wrist speed: "
                f"{format_value(self.analysis.wrist_speed, ' ft/s')}\n"
                f"Ball/wrist separation: "
                f"{self.analysis.ball_wrist_separation:.2f} ft\n"
                "\n"
                "RELEASE BIOMECHANICS\n"
                "------------------------------\n"
                f"Right elbow: "
                f"{format_value(self.analysis.right_elbow_angle, ' degrees')}\n"
                f"Left elbow: "
                f"{format_value(self.analysis.left_elbow_angle, ' degrees')}\n"
                f"Right knee: "
                f"{format_value(self.analysis.right_knee_angle, ' degrees')}\n"
                f"Left knee: "
                f"{format_value(self.analysis.left_knee_angle, ' degrees')}\n"
                f"Right hip: "
                f"{format_value(self.analysis.right_hip_angle, ' degrees')}\n"
                f"Left hip: "
                f"{format_value(self.analysis.left_hip_angle, ' degrees')}\n"
                f"Right shoulder: "
                f"{format_value(self.analysis.right_shoulder_angle, ' degrees')}\n"
                f"Left shoulder: "
                f"{format_value(self.analysis.left_shoulder_angle, ' degrees')}\n"
                "\n"
                "SHOT EVENTS\n"
                "------------------------------\n"
                f"1  Motion start: "
                f"{format_frame(self.analysis.motion_start_frame)}\n"
                f"2  Dip: "
                f"{format_frame(self.analysis.dip_frame)}\n"
                f"3  Takeoff: "
                f"{format_frame(self.analysis.takeoff_frame)}\n"
                f"4  Release: "
                f"{format_frame(self.analysis.release_frame)}\n"
                f"5  Ball apex: "
                f"{format_frame(self.analysis.ball_apex_frame)}\n"
                f"6  Landing: "
                f"{format_frame(self.analysis.landing_frame)}"
            )

        self.plotter.add_text(
            panel_text,
            position=(15, 125),
            font_size=8,
            color="white",
            name="combined_analysis_panel",
        )

    # =====================================================
    # RELEASE MARKER
    # =====================================================

    def _create_release_marker(self) -> None:
        """
        Place a yellow marker at the detected release position.
        """

        if self.analysis is None:
            return

        release_frame = self.analysis.release_frame

        if (
            release_frame < 0
            or release_frame >= len(self.tracking_frames)
        ):
            return

        release_ball = (
            self.tracking_frames[release_frame]
            .get("data", {})
            .get("ball")
        )

        if not is_valid_point(release_ball):
            return

        marker_mesh = pv.Sphere(
            radius=0.24,
            center=release_ball,
        )

        self.release_marker_actor = self.plotter.add_mesh(
            marker_mesh,
            color="yellow",
            opacity=0.75,
            smooth_shading=True,
            show_edges=True,
        )

    # =====================================================
    # BALL TRAJECTORY
    # =====================================================

    def _create_clean_ball_trajectory(self) -> None:
        """
        Draw the longest clean ball-tracking segment.
        """

        clean_segment = get_longest_ball_segment(
            tracking_frames=self.tracking_frames,
            maximum_frame_jump=2.5,
            minimum_segment_length=4,
        )

        if clean_segment is None:
            print("No clean ball trajectory was found.")
            return

        trajectory_line = pv.lines_from_points(
            clean_segment,
            close=False,
        )

        trajectory_tube = trajectory_line.tube(
            radius=0.045,
            n_sides=12,
            capping=True,
        )

        self.trajectory_actor = self.plotter.add_mesh(
            trajectory_tube,
            color="orange",
            opacity=0.70,
            smooth_shading=True,
        )

        print(
            f"Clean trajectory points: {len(clean_segment)}"
        )

    def toggle_trajectory(self) -> None:
        """
        Show or hide the trajectory and release marker.
        """

        self.trajectory_visible = (
            not self.trajectory_visible
        )

        if self.trajectory_actor is not None:
            self.trajectory_actor.SetVisibility(
                self.trajectory_visible
            )

        if self.release_marker_actor is not None:
            self.release_marker_actor.SetVisibility(
                self.trajectory_visible
            )

        self.plotter.render()

    # =====================================================
    # SYNCHRONIZED TIME-SERIES CHARTS
    # =====================================================

    def _create_chart_panel(self) -> bool:
        """
        Build a fresh synchronized chart panel.

        A fresh panel is created whenever the user reopens charts
        after closing them, because Matplotlib figures cannot always
        be safely reused after their native window is destroyed.
        """

        if (
            self.time_series is None
            or self.chart_event_frames is None
        ):
            return False

        self.chart_panel = TimeSeriesChartPanel(
            time_series=self.time_series,
            event_frames=self.chart_event_frames,
        )

        self.chart_panel.build()
        self.chart_panel.update_cursor(
            self.current_frame,
            redraw=False,
        )

        return True

    def _chart_window_exists(self) -> bool:
        """
        Return True while the Matplotlib chart window still exists.
        """

        if (
            self.chart_panel is None
            or self.chart_panel.figure is None
        ):
            return False

        return bool(
            plt.fignum_exists(
                self.chart_panel.figure.number
            )
        )

    def show_charts(self) -> None:
        """
        Open or restore the synchronized chart window.
        """

        if self._chart_window_exists():
            self.chart_visible = True
            self.chart_panel.update_cursor(
                self.current_frame,
                redraw=True,
            )
            return

        if not self._create_chart_panel():
            print(
                "Time-series charts are unavailable: "
                f"{self.time_series_error or self.analysis_error or 'Unknown error'}"
            )
            return

        self.chart_panel.show(
            block=False,
        )

        self.chart_visible = True

    def hide_charts(self) -> None:
        """
        Close the synchronized chart window.
        """

        if self.chart_panel is not None:
            self.chart_panel.close()

        self.chart_visible = False
        self.chart_panel = None

    def toggle_charts(self) -> None:
        """
        Show or hide the synchronized time-series dashboard.
        """

        if self._chart_window_exists():
            self.hide_charts()
        else:
            self.show_charts()

    def _update_chart_cursor(self) -> None:
        """
        Synchronize the chart cursor with the current 3D frame.
        """

        if not self._chart_window_exists():
            self.chart_visible = False
            return

        self.chart_visible = True

        self.chart_panel.update_cursor(
            self.current_frame,
            redraw=True,
        )

    def _process_chart_events(self) -> None:
        """
        Keep the Matplotlib chart window responsive.

        PyVista owns the main application loop, so Matplotlib's GUI
        event queue must also be processed during each loop iteration.
        """

        if not self._chart_window_exists():
            self.chart_visible = False
            return

        figure = self.chart_panel.figure

        if (
            figure is not None
            and figure.canvas is not None
        ):
            figure.canvas.flush_events()

    def _display_playback_frame(
        self,
        frame_index: int,
    ) -> None:
        """
        Display a frame requested by the PlaybackController.
        """

        self.display_frame(
            frame_index=frame_index,
            update_slider=True,
            synchronize_playback=False,
        )

    # =====================================================
    # PLAYBACK COMMANDS
    # =====================================================

    def toggle_playback(self) -> None:
        """
        Start or pause automatic playback.
        """

        self.playback.toggle_playback()
        self._update_playback_status()
        self.plotter.render()

    def set_speed_quarter(self) -> None:
        self._set_playback_speed(0.25)

    def set_speed_half(self) -> None:
        self._set_playback_speed(0.50)

    def set_speed_normal(self) -> None:
        self._set_playback_speed(1.00)

    def set_speed_double(self) -> None:
        self._set_playback_speed(2.00)

    def _set_playback_speed(
        self,
        speed_multiplier: float,
    ) -> None:
        """
        Change playback speed and refresh the status display.
        """

        self.playback.set_speed(speed_multiplier)
        self._update_playback_status()
        self.plotter.render()

    def toggle_loop(self) -> None:
        """
        Enable or disable automatic looping.
        """

        self.playback.toggle_loop()
        self._update_playback_status()
        self.plotter.render()

    # =====================================================
    # KEYBOARD CONTROLS
    # =====================================================

    def _register_controls(self) -> None:
        """
        Register keyboard controls.
        """

        self.plotter.add_key_event(
            "n",
            self.next_frame,
        )

        self.plotter.add_key_event(
            "p",
            self.previous_frame,
        )

        self.plotter.add_key_event(
            "Right",
            self.jump_forward,
        )

        self.plotter.add_key_event(
            "Left",
            self.jump_backward,
        )

        self.plotter.add_key_event(
            "space",
            self.toggle_playback,
        )

        self.plotter.add_key_event(
            "7",
            self.set_speed_quarter,
        )

        self.plotter.add_key_event(
            "8",
            self.set_speed_half,
        )

        self.plotter.add_key_event(
            "9",
            self.set_speed_normal,
        )

        self.plotter.add_key_event(
            "0",
            self.set_speed_double,
        )

        self.plotter.add_key_event(
            "l",
            self.toggle_loop,
        )

        self.plotter.add_key_event(
            "t",
            self.toggle_trajectory,
        )

        self.plotter.add_key_event(
            "c",
            self.toggle_charts,
        )

        self.plotter.add_key_event(
            "g",
            self.jump_to_release,
        )

        self.plotter.add_key_event(
            "r",
            self.reset_camera,
        )

        self.plotter.add_key_event(
            "1",
            self.jump_to_motion_start,
        )

        self.plotter.add_key_event(
            "2",
            self.jump_to_dip,
        )

        self.plotter.add_key_event(
            "3",
            self.jump_to_takeoff,
        )

        self.plotter.add_key_event(
            "4",
            self.jump_to_release,
        )

        self.plotter.add_key_event(
            "5",
            self.jump_to_ball_apex,
        )

        self.plotter.add_key_event(
            "6",
            self.jump_to_landing,
        )

    # =====================================================
    # FRAME SLIDER
    # =====================================================

    def _create_frame_slider(self) -> None:
        """
        Add the frame-selection slider.
        """

        self.slider_widget = self.plotter.add_slider_widget(
            callback=self._slider_changed,
            rng=(
                0,
                len(self.tracking_frames) - 1,
            ),
            value=0,
            title="Frame",
            pointa=(0.23, 0.055),
            pointb=(0.80, 0.055),
            color="navy",
            title_color="white",
            fmt="%.0f",
            interaction_event="end",
        )

    def _slider_changed(
        self,
        value: float,
    ) -> None:
        """
        Pause playback and move to the selected slider frame.
        """

        selected_frame = int(round(value))

        self.playback.pause()

        self.playback.set_frame(
            selected_frame,
            notify_viewer=False,
        )

        self.display_frame(
            selected_frame,
            update_slider=False,
            synchronize_playback=False,
        )

    def _update_slider_position(self) -> None:
        """
        Synchronize the slider with frame navigation.
        """

        if self.slider_widget is None:
            return

        slider_representation = (
            self.slider_widget.GetRepresentation()
        )

        slider_representation.SetValue(
            float(self.current_frame)
        )

    # =====================================================
    # DISPLAY FRAME
    # =====================================================

    def display_frame(
        self,
        frame_index: int,
        update_slider: bool = True,
        synchronize_playback: bool = True,
    ) -> None:
        """
        Display one selected tracking frame.
        """

        self.current_frame = max(
            0,
            min(
                int(frame_index),
                len(self.tracking_frames) - 1,
            ),
        )

        if synchronize_playback:
            self.playback.set_frame(
                self.current_frame,
                notify_viewer=False,
            )

        frame = self.tracking_frames[
            self.current_frame
        ]

        frame_data = frame.get("data", {})
        player_data = frame_data.get("player", {})
        ball_data = frame_data.get("ball")

        self._update_joints(player_data)
        self._update_connections(player_data)
        self._update_ball(ball_data)
        self._initialize_camera(player_data)
        self._update_information(frame)
        self._update_event_status()
        self._update_playback_status()
        self._update_chart_cursor()

        if update_slider:
            self._update_slider_position()

        self.plotter.render()

    # =====================================================
    # UPDATE SKELETON
    # =====================================================

    def _update_joints(
        self,
        player_data: dict[str, object],
    ) -> None:
        """
        Move all valid joint markers.
        """

        for joint_name, joint_actor in self.joint_actors.items():
            point = player_data.get(joint_name)

            if not is_valid_point(point):
                joint_actor.SetVisibility(False)
                continue

            joint_actor.SetVisibility(True)

            joint_actor.SetPosition(
                float(point[0]),
                float(point[1]),
                float(point[2]),
            )

    def _update_connections(
        self,
        player_data: dict[str, object],
    ) -> None:
        """
        Update all valid body segments.
        """

        for connection, line_mesh in self.connection_meshes.items():
            joint_a, joint_b = connection

            point_a = player_data.get(joint_a)
            point_b = player_data.get(joint_b)

            line_actor = self.connection_actors[
                connection
            ]

            if (
                not is_valid_point(point_a)
                or not is_valid_point(point_b)
            ):
                line_actor.SetVisibility(False)
                continue

            line_actor.SetVisibility(True)

            line_mesh.points[0] = np.array(
                point_a,
                dtype=float,
            )

            line_mesh.points[-1] = np.array(
                point_b,
                dtype=float,
            )

            line_mesh.Modified()

    # =====================================================
    # UPDATE BALL
    # =====================================================

    def _update_ball(
        self,
        ball_data: object,
    ) -> None:
        """
        Move or hide the current basketball marker.
        """

        if not is_valid_point(ball_data):
            self.ball_actor.SetVisibility(False)
            return

        self.ball_actor.SetVisibility(True)

        self.ball_actor.SetPosition(
            float(ball_data[0]),
            float(ball_data[1]),
            float(ball_data[2]),
        )

    # =====================================================
    # CAMERA
    # =====================================================

    def _initialize_camera(
        self,
        player_data: dict[str, object],
    ) -> None:
        """
        Set the initial elevated analysis camera.
        """

        if self.camera_initialized:
            return

        left_hip = player_data.get("LEFT_HIP")
        right_hip = player_data.get("RIGHT_HIP")

        if (
            not is_valid_point(left_hip)
            or not is_valid_point(right_hip)
        ):
            return

        hip_center = (
            np.array(left_hip, dtype=float)
            + np.array(right_hip, dtype=float)
        ) / 2

        self.plotter.camera.focal_point = tuple(
            hip_center
        )

        self.plotter.camera.position = (
            hip_center[0] + 12,
            hip_center[1] - 16,
            hip_center[2] + 10,
        )

        self.plotter.camera.up = (0, 0, 1)

        self.camera_initialized = True

    def reset_camera(self) -> None:
        """
        Return to the default elevated player-analysis camera.
        """

        self.camera_initialized = False

        frame = self.tracking_frames[
            self.current_frame
        ]

        player_data = (
            frame
            .get("data", {})
            .get("player", {})
        )

        self._initialize_camera(player_data)
        self.plotter.render()

    # =====================================================
    # FRAME INFORMATION
    # =====================================================

    def _update_information(
        self,
        frame: dict,
    ) -> None:
        """
        Update trial and frame information.
        """

        frame_number = frame.get(
            "frame",
            self.current_frame,
        )

        frame_time = frame.get(
            "time",
            0,
        )

        entry_angle_text = "N/A"

        if isinstance(
            self.entry_angle,
            (int, float),
        ):
            entry_angle_text = (
                f"{self.entry_angle:.1f} degrees"
            )

        self.plotter.add_text(
            (
                f"Participant {self.participant} | "
                f"Trial {self.trial}\n"
                f"Frame {frame_number} / "
                f"{len(self.tracking_frames) - 1}\n"
                f"Time: {frame_time}\n"
                f"Sampling rate: "
                f"{self.sampling_rate:g} FPS\n"
                f"Recorded entry angle: "
                f"{entry_angle_text}"
            ),
            position="upper_left",
            font_size=10,
            color="white",
            name="frame_information",
        )

    # =====================================================
    # EVENT STATUS
    # =====================================================

    def _update_event_status(self) -> None:
        """
        Identify whether the selected frame is a detected event.
        """

        if self.analysis is None:
            status = "Shot event analysis unavailable"

        else:
            event_lookup = {
                self.analysis.motion_start_frame: "MOTION START",
                self.analysis.dip_frame: "DIP",
                self.analysis.takeoff_frame: "TAKEOFF",
                self.analysis.release_frame: "RELEASE",
                self.analysis.ball_apex_frame: "BALL APEX",
                self.analysis.landing_frame: "LANDING",
            }

            status = event_lookup.get(
                self.current_frame,
                "Between detected events",
            )

        self.plotter.add_text(
            status,
            position=(1250, 100),
            font_size=10,
            color="yellow",
            name="event_status",
        )

    def _update_playback_status(self) -> None:
        """
        Display playback mode, speed and loop status.
        """

        self.plotter.add_text(
            self.playback.status_text(),
            position=(1250, 65),
            font_size=10,
            color="white",
            name="playback_status",
        )

    # =====================================================
    # FRAME NAVIGATION
    # =====================================================

    def next_frame(self) -> None:
        self.playback.step(1)

    def previous_frame(self) -> None:
        self.playback.step(-1)

    def jump_forward(self) -> None:
        self.playback.step(10)

    def jump_backward(self) -> None:
        self.playback.step(-10)

    # =====================================================
    # EVENT NAVIGATION
    # =====================================================

    def _jump_to_event(
        self,
        frame_number: int | None,
    ) -> None:
        """
        Pause playback and jump to a detected event.
        """

        if frame_number is None:
            return

        self.playback.pause()
        self.playback.set_frame(frame_number)

    def jump_to_motion_start(self) -> None:
        if self.analysis is None:
            return

        self._jump_to_event(
            self.analysis.motion_start_frame
        )

    def jump_to_dip(self) -> None:
        if self.analysis is None:
            return

        self._jump_to_event(
            self.analysis.dip_frame
        )

    def jump_to_takeoff(self) -> None:
        if self.analysis is None:
            return

        self._jump_to_event(
            self.analysis.takeoff_frame
        )

    def jump_to_release(self) -> None:
        if self.analysis is None:
            return

        self._jump_to_event(
            self.analysis.release_frame
        )

    def jump_to_ball_apex(self) -> None:
        if self.analysis is None:
            return

        self._jump_to_event(
            self.analysis.ball_apex_frame
        )

    def jump_to_landing(self) -> None:
        if self.analysis is None:
            return

        self._jump_to_event(
            self.analysis.landing_frame
        )

    # =====================================================
    # OPEN VIEWER
    # =====================================================

    def show(self) -> None:
        """
        Open the complete tracking workstation.
        """

        self.display_frame(0)

        print("BASKETBALL MOTION STUDIO READY")
        print(f"Trial file: {self.trial_file.name}")
        print(
            f"Total frames: "
            f"{len(self.tracking_frames)}"
        )
        print(
            f"Sampling rate: "
            f"{self.sampling_rate:g} FPS"
        )
        print(
            "Playback controls: "
            "Space, 7, 8, 9, 0 and L"
        )
        print(
            "Chart control: C"
        )

        if self.analysis is not None:
            print("Detected shot events:")
            print(
                f"Motion start: "
                f"{self.analysis.motion_start_frame}"
            )
            print(
                f"Dip: {self.analysis.dip_frame}"
            )
            print(
                f"Takeoff: "
                f"{self.analysis.takeoff_frame}"
            )
            print(
                f"Release: "
                f"{self.analysis.release_frame}"
            )
            print(
                f"Ball apex: "
                f"{self.analysis.ball_apex_frame}"
            )
            print(
                f"Landing: "
                f"{self.analysis.landing_frame}"
            )

        else:
            print(
                f"Analysis unavailable: "
                f"{self.analysis_error}"
            )

        # Open the synchronized time-series dashboard beside the
        # 3D viewer when complete analysis data is available.
        self.show_charts()

        # Open the window in non-blocking interactive mode. PyVista
        # then expects Plotter.update() to be called repeatedly.
        self.plotter.show(
            title="Basketball Motion Studio",
            interactive=True,
            auto_close=False,
            interactive_update=True,
        )

        try:
            while not bool(getattr(self.plotter, "_closed", False)):
                # The standalone controller determines whether enough
                # real time has elapsed to advance the tracking frame.
                self.playback.tick()

                # Process Windows/VTK input events and redraw the scene.
                # This keeps keyboard, mouse, slider and camera controls
                # responsive while playback is running.
                self.plotter.update(
                    stime=1,
                    force_redraw=False,
                )

                self._process_chart_events()

                # Avoid consuming an entire CPU core while paused.
                sleep(0.001)

        except (KeyboardInterrupt, RuntimeError):
            # KeyboardInterrupt supports Ctrl+C from the VS Code terminal.
            # RuntimeError safely handles a window closed during an update.
            pass

        finally:
            self.hide_charts()

            if not bool(getattr(self.plotter, "_closed", False)):
                self.plotter.close()


def show_tracking_viewer() -> None:
    """
    Main function called by app.py.
    """

    viewer = TrackingViewer()
    viewer.show()


if __name__ == "__main__":
    show_tracking_viewer()