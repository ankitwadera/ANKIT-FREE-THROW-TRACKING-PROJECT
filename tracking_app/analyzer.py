from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tracking_app.biomechanics import calculate_angles_at_frame
from tracking_app.events import ShotEvents, detect_shot_events
from tracking_app.kinematics import calculate_frame_velocity
from tracking_app.release import get_release_details
from tracking_app.shot_metrics import calculate_release_angle


@dataclass
class ShotAnalysis:
    """
    Stores the complete analysis for one free throw.
    """

    participant_id: str
    trial_id: str
    result: Any
    trial_file: str | None

    sampling_rate: float
    total_frames: int

    motion_start_frame: int | None
    dip_frame: int | None
    takeoff_frame: int | None
    release_frame: int
    ball_apex_frame: int | None
    landing_frame: int | None

    release_time: Any
    release_height: float

    ball_speed: float
    ball_horizontal_speed: float
    ball_vertical_speed: float
    release_angle: float

    wrist_speed: float | None
    ball_wrist_separation: float

    right_elbow_angle: float | None
    left_elbow_angle: float | None

    right_knee_angle: float | None
    left_knee_angle: float | None

    right_hip_angle: float | None
    left_hip_angle: float | None

    right_shoulder_angle: float | None
    left_shoulder_angle: float | None


class ShotAnalyzer:
    """
    Convert one raw tracking trial into basketball analytics.
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
        Confirm that the trial contains usable tracking data.
        """

        if not isinstance(self.trial_data, dict):
            raise TypeError(
                "trial_data must be a dictionary."
            )

        if not isinstance(self.tracking_frames, list):
            raise TypeError(
                "The tracking section must be a list."
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

    def detect_events(self) -> ShotEvents:
        """
        Detect all major free-throw events once.
        """

        return detect_shot_events(
            tracking_frames=self.tracking_frames,
            sampling_rate=float(self.sampling_rate),
            shooting_wrist=self.shooting_wrist,
        )

    def analyze(self) -> ShotAnalysis:
        """
        Run the complete shot analysis.
        """

        events = self.detect_events()

        if events.release_frame is None:
            raise ValueError(
                "No release frame could be detected."
            )

        release_frame = events.release_frame

        release_details = get_release_details(
            tracking_frames=self.tracking_frames,
            release_frame_index=release_frame,
            sampling_rate=float(self.sampling_rate),
            shooting_wrist=self.shooting_wrist,
        )

        ball_velocity = calculate_frame_velocity(
            tracking_frames=self.tracking_frames,
            frame_index=release_frame,
            object_name="ball",
            sampling_rate=float(self.sampling_rate),
        )

        if ball_velocity is None:
            raise ValueError(
                "Ball velocity could not be calculated "
                "at the release frame."
            )

        wrist_velocity = calculate_frame_velocity(
            tracking_frames=self.tracking_frames,
            frame_index=release_frame,
            object_name="player",
            joint_name=self.shooting_wrist,
            sampling_rate=float(self.sampling_rate),
        )

        release_angle = calculate_release_angle(
            horizontal_speed=ball_velocity[
                "horizontal_speed"
            ],
            vertical_speed=ball_velocity[
                "vertical_speed"
            ],
        )

        release_angles = calculate_angles_at_frame(
            tracking_frames=self.tracking_frames,
            frame_index=release_frame,
        )

        wrist_speed = None

        if wrist_velocity is not None:
            wrist_speed = wrist_velocity["speed"]

        trial_file_name = None

        if self.trial_file is not None:
            trial_file_name = self.trial_file.name

        return ShotAnalysis(
            participant_id=self.trial_data.get(
                "participant_id",
                "Unknown",
            ),
            trial_id=self.trial_data.get(
                "trial_id",
                "Unknown",
            ),
            result=self.trial_data.get(
                "result",
                "Unknown",
            ),
            trial_file=trial_file_name,
            sampling_rate=float(self.sampling_rate),
            total_frames=len(self.tracking_frames),

            motion_start_frame=events.motion_start_frame,
            dip_frame=events.dip_frame,
            takeoff_frame=events.takeoff_frame,
            release_frame=release_frame,
            ball_apex_frame=events.ball_apex_frame,
            landing_frame=events.landing_frame,

            release_time=release_details["time"],
            release_height=release_details[
                "ball_height"
            ],

            ball_speed=ball_velocity["speed"],
            ball_horizontal_speed=ball_velocity[
                "horizontal_speed"
            ],
            ball_vertical_speed=ball_velocity[
                "vertical_speed"
            ],
            release_angle=release_angle,

            wrist_speed=wrist_speed,
            ball_wrist_separation=release_details[
                "ball_wrist_separation"
            ],

            right_elbow_angle=release_angles[
                "right_elbow_angle"
            ],
            left_elbow_angle=release_angles[
                "left_elbow_angle"
            ],

            right_knee_angle=release_angles[
                "right_knee_angle"
            ],
            left_knee_angle=release_angles[
                "left_knee_angle"
            ],

            right_hip_angle=release_angles[
                "right_hip_angle"
            ],
            left_hip_angle=release_angles[
                "left_hip_angle"
            ],

            right_shoulder_angle=release_angles[
                "right_shoulder_angle"
            ],
            left_shoulder_angle=release_angles[
                "left_shoulder_angle"
            ],
        )