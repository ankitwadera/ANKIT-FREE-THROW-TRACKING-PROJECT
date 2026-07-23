from __future__ import annotations

import numpy as np
import pyvista as pv

from tracking_app.data_loader import load_first_trial
from tracking_app.skeleton import (
    get_usable_connections,
    is_valid_point,
)


class TrackingViewer:
    """
    Interactive frame-by-frame basketball tracking viewer.

    Controls
    --------
    N            Next frame
    P            Previous frame
    Right Arrow  Jump forward 10 frames
    Left Arrow   Jump backward 10 frames
    Slider       Jump directly to any frame
    R            Reset camera
    """

    def __init__(self) -> None:
        # -------------------------------------------------
        # LOAD THE FIRST FREE-THROW TRIAL
        # -------------------------------------------------

        self.trial_file, self.trial_data = load_first_trial()

        self.tracking_frames = self.trial_data.get("tracking", [])

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

        # -------------------------------------------------
        # VIEWER STATE
        # -------------------------------------------------

        self.current_frame = 0
        self.camera_initialized = False
        self.slider_widget = None

        # -------------------------------------------------
        # CREATE THE 3D WINDOW
        # -------------------------------------------------

        self.plotter = pv.Plotter(
            window_size=(1200, 850),
        )

        self.plotter.set_background("white")
        self.plotter.add_axes()

        # These store the objects we update each frame.
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

        self._create_scene()
        self._register_controls()
        self._create_frame_slider()

    def _create_scene(self) -> None:
        """
        Create all permanent 3D objects.
        """

        joint_names = sorted(
            {
                joint_name
                for connection in self.usable_connections
                for joint_name in connection
            }
        )

        # -------------------------------------------------
        # CREATE BODY-JOINT SPHERES
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
        # CREATE BODY-SEGMENT LINES
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
        # CREATE THE BASKETBALL
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
        # ADD CONTROL INSTRUCTIONS
        # -------------------------------------------------

        self.plotter.add_text(
            (
                "N: next frame\n"
                "P: previous frame\n"
                "Right Arrow: +10 frames\n"
                "Left Arrow: -10 frames\n"
                "R: reset camera"
            ),
            position="upper_right",
            font_size=10,
            color="black",
        )

    def _register_controls(self) -> None:
        """
        Connect keyboard keys to viewer actions.
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

    def _create_frame_slider(self) -> None:
        """
        Add a slider that can jump directly to any frame.
        """

        self.slider_widget = self.plotter.add_slider_widget(
            callback=self._slider_changed,
            rng=(
                0,
                len(self.tracking_frames) - 1,
            ),
            value=0,
            title="Frame",
            pointa=(0.20, 0.08),
            pointb=(0.80, 0.08),
            color="navy",
            title_color="black",
            fmt="%0.0f",
            interaction_event="end",
        )

    def _slider_changed(self, value: float) -> None:
        """
        Display the frame selected by the slider.
        """

        selected_frame = int(round(value))

        self.display_frame(
            selected_frame,
            update_slider=False,
        )

    def _update_slider_position(self) -> None:
        """
        Keep the slider synchronized with keyboard controls.
        """

        if self.slider_widget is None:
            return

        slider_representation = (
            self.slider_widget.GetRepresentation()
        )

        slider_representation.SetValue(
            float(self.current_frame)
        )

    def display_frame(
        self,
        frame_index: int,
        update_slider: bool = True,
    ) -> None:
        """
        Display one selected tracking frame.
        """

        self.current_frame = max(
            0,
            min(
                frame_index,
                len(self.tracking_frames) - 1,
            ),
        )

        frame = self.tracking_frames[self.current_frame]

        frame_data = frame.get("data", {})
        player_data = frame_data.get("player", {})
        ball_data = frame_data.get("ball")

        self._update_joints(player_data)
        self._update_connections(player_data)
        self._update_ball(ball_data)
        self._initialize_camera(player_data)
        self._update_information(frame)

        if update_slider:
            self._update_slider_position()

        self.plotter.render()

    def _update_joints(
        self,
        player_data: dict[str, object],
    ) -> None:
        """
        Move all visible body-joint markers.
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
        Move every visible body segment.
        """

        for connection, line_mesh in self.connection_meshes.items():
            joint_a, joint_b = connection

            point_a = player_data.get(joint_a)
            point_b = player_data.get(joint_b)

            line_actor = self.connection_actors[connection]

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

    def _update_ball(
        self,
        ball_data: object,
    ) -> None:
        """
        Move or hide the basketball.
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

    def _initialize_camera(
        self,
        player_data: dict[str, object],
    ) -> None:
        """
        Position the camera around the athlete once.
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
            hip_center[0] + 6,
            hip_center[1] - 8,
            hip_center[2] + 4,
        )

        self.plotter.camera.up = (0, 0, 1)

        self.camera_initialized = True

    def _update_information(
        self,
        frame: dict,
    ) -> None:
        """
        Update the information shown in the upper-left.
        """

        frame_number = frame.get(
            "frame",
            self.current_frame,
        )

        frame_time = frame.get(
            "time",
            0,
        )

        self.plotter.add_text(
            (
                f"Participant {self.participant} | "
                f"Trial {self.trial}\n"
                f"Frame {frame_number} / "
                f"{len(self.tracking_frames) - 1}\n"
                f"Time: {frame_time}"
            ),
            position="upper_left",
            font_size=12,
            color="black",
            name="frame_information",
        )

    def next_frame(self) -> None:
        """
        Move forward by one frame.
        """

        self.display_frame(
            self.current_frame + 1
        )

    def previous_frame(self) -> None:
        """
        Move backward by one frame.
        """

        self.display_frame(
            self.current_frame - 1
        )

    def jump_forward(self) -> None:
        """
        Move forward by ten frames.
        """

        self.display_frame(
            self.current_frame + 10
        )

    def jump_backward(self) -> None:
        """
        Move backward by ten frames.
        """

        self.display_frame(
            self.current_frame - 10
        )

    def show(self) -> None:
        """
        Open the tracking viewer.
        """

        self.display_frame(0)

        print("TRACKING ANALYSIS WORKSTATION READY")
        print(f"Trial file: {self.trial_file.name}")
        print(f"Total frames: {len(self.tracking_frames)}")
        print("Use the frame slider to jump through the shot.")

        self.plotter.show()


def show_tracking_viewer() -> None:
    """
    Main function used by app.py.
    """

    viewer = TrackingViewer()
    viewer.show()


if __name__ == "__main__":
    show_tracking_viewer()