from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tracking_app.data_loader import load_first_trial
from tracking_app.skeleton import is_valid_point


@dataclass(frozen=True)
class ShotTimeSeries:
    """
    Complete frame-by-frame biomechanical and ball-tracking series.

    Every array has the same length as the trial's tracking-frame list.
    Missing or unusable measurements are stored as numpy.nan.
    """

    frame_index: np.ndarray
    frame_number: np.ndarray
    time: np.ndarray

    ball_height: np.ndarray
    ball_speed: np.ndarray
    ball_horizontal_speed: np.ndarray
    ball_vertical_speed: np.ndarray

    right_wrist_speed: np.ndarray
    left_wrist_speed: np.ndarray

    right_elbow_angle: np.ndarray
    left_elbow_angle: np.ndarray
    right_knee_angle: np.ndarray
    left_knee_angle: np.ndarray
    right_hip_angle: np.ndarray
    left_hip_angle: np.ndarray
    right_shoulder_angle: np.ndarray
    left_shoulder_angle: np.ndarray

    pelvis_height: np.ndarray
    left_ankle_height: np.ndarray
    right_ankle_height: np.ndarray
    average_ankle_height: np.ndarray

    @property
    def total_frames(self) -> int:
        return int(self.frame_index.size)

    def available_series(self) -> dict[str, np.ndarray]:
        """
        Return all analysis curves by stable display name.
        """

        return {
            "Ball Height": self.ball_height,
            "Ball Speed": self.ball_speed,
            "Ball Horizontal Speed": self.ball_horizontal_speed,
            "Ball Vertical Speed": self.ball_vertical_speed,
            "Right Wrist Speed": self.right_wrist_speed,
            "Left Wrist Speed": self.left_wrist_speed,
            "Right Elbow Angle": self.right_elbow_angle,
            "Left Elbow Angle": self.left_elbow_angle,
            "Right Knee Angle": self.right_knee_angle,
            "Left Knee Angle": self.left_knee_angle,
            "Right Hip Angle": self.right_hip_angle,
            "Left Hip Angle": self.left_hip_angle,
            "Right Shoulder Angle": self.right_shoulder_angle,
            "Left Shoulder Angle": self.left_shoulder_angle,
            "Pelvis Height": self.pelvis_height,
            "Left Ankle Height": self.left_ankle_height,
            "Right Ankle Height": self.right_ankle_height,
            "Average Ankle Height": self.average_ankle_height,
        }


class TimeSeriesAnalyzer:
    """
    Build complete frame-by-frame motion-analysis curves.

    This module is intentionally independent of the PyVista viewer.
    It can later be reused by:

    - live viewer graphs;
    - batch shot analysis;
    - CSV/database export;
    - Streamlit dashboards;
    - machine-learning feature generation.
    """

    def __init__(
        self,
        trial_data: dict,
        smoothing_window: int = 5,
    ) -> None:
        self.trial_data = trial_data

        self.tracking_frames = trial_data.get(
            "tracking",
            [],
        )

        if not self.tracking_frames:
            raise ValueError(
                "Time-series analysis requires tracking frames."
            )

        sampling_rate = trial_data.get(
            "sampling_rate",
            30,
        )

        try:
            self.sampling_rate = float(sampling_rate)
        except (TypeError, ValueError):
            self.sampling_rate = 30.0

        if self.sampling_rate <= 0:
            self.sampling_rate = 30.0

        self.smoothing_window = max(
            1,
            int(smoothing_window),
        )

        if self.smoothing_window % 2 == 0:
            self.smoothing_window += 1

    # =====================================================
    # PUBLIC ANALYSIS
    # =====================================================

    def analyze(self) -> ShotTimeSeries:
        """
        Calculate all currently supported time-series curves.
        """

        frame_index = np.arange(
            len(self.tracking_frames),
            dtype=int,
        )

        frame_number = np.array(
            [
                self._safe_float(
                    frame.get("frame", index),
                )
                for index, frame in enumerate(
                    self.tracking_frames
                )
            ],
            dtype=float,
        )

        time = np.array(
            [
                self._safe_float(
                    frame.get("time", np.nan),
                )
                for frame in self.tracking_frames
            ],
            dtype=float,
        )

        ball_positions = self._extract_points(
            "ball",
            player_joint=False,
        )

        right_wrist_positions = self._extract_points(
            "RIGHT_WRIST",
            player_joint=True,
        )

        left_wrist_positions = self._extract_points(
            "LEFT_WRIST",
            player_joint=True,
        )

        ball_velocity = self._calculate_velocity(
            ball_positions
        )

        right_wrist_velocity = self._calculate_velocity(
            right_wrist_positions
        )

        left_wrist_velocity = self._calculate_velocity(
            left_wrist_positions
        )

        ball_height = ball_positions[:, 2]

        ball_speed = self._vector_magnitude(
            ball_velocity
        )

        ball_horizontal_speed = np.sqrt(
            np.square(ball_velocity[:, 0])
            + np.square(ball_velocity[:, 1])
        )

        ball_vertical_speed = ball_velocity[:, 2]

        right_wrist_speed = self._vector_magnitude(
            right_wrist_velocity
        )

        left_wrist_speed = self._vector_magnitude(
            left_wrist_velocity
        )

        right_elbow_angle = self._joint_angle_series(
            first_joint="RIGHT_SHOULDER",
            vertex_joint="RIGHT_ELBOW",
            third_joint="RIGHT_WRIST",
        )

        left_elbow_angle = self._joint_angle_series(
            first_joint="LEFT_SHOULDER",
            vertex_joint="LEFT_ELBOW",
            third_joint="LEFT_WRIST",
        )

        right_knee_angle = self._joint_angle_series(
            first_joint="RIGHT_HIP",
            vertex_joint="RIGHT_KNEE",
            third_joint="RIGHT_ANKLE",
        )

        left_knee_angle = self._joint_angle_series(
            first_joint="LEFT_HIP",
            vertex_joint="LEFT_KNEE",
            third_joint="LEFT_ANKLE",
        )

        right_hip_angle = self._joint_angle_series(
            first_joint="RIGHT_SHOULDER",
            vertex_joint="RIGHT_HIP",
            third_joint="RIGHT_KNEE",
        )

        left_hip_angle = self._joint_angle_series(
            first_joint="LEFT_SHOULDER",
            vertex_joint="LEFT_HIP",
            third_joint="LEFT_KNEE",
        )

        right_shoulder_angle = self._joint_angle_series(
            first_joint="RIGHT_HIP",
            vertex_joint="RIGHT_SHOULDER",
            third_joint="RIGHT_ELBOW",
        )

        left_shoulder_angle = self._joint_angle_series(
            first_joint="LEFT_HIP",
            vertex_joint="LEFT_SHOULDER",
            third_joint="LEFT_ELBOW",
        )

        pelvis_height = self._pelvis_height_series()

        left_ankle_height = self._joint_height_series(
            "LEFT_ANKLE"
        )

        right_ankle_height = self._joint_height_series(
            "RIGHT_ANKLE"
        )

        average_ankle_height = self._nanmean_pair(
            left_ankle_height,
            right_ankle_height,
        )

        return ShotTimeSeries(
            frame_index=frame_index,
            frame_number=frame_number,
            time=time,
            ball_height=self._smooth(ball_height),
            ball_speed=self._smooth(ball_speed),
            ball_horizontal_speed=self._smooth(
                ball_horizontal_speed
            ),
            ball_vertical_speed=self._smooth(
                ball_vertical_speed
            ),
            right_wrist_speed=self._smooth(
                right_wrist_speed
            ),
            left_wrist_speed=self._smooth(
                left_wrist_speed
            ),
            right_elbow_angle=self._smooth(
                right_elbow_angle
            ),
            left_elbow_angle=self._smooth(
                left_elbow_angle
            ),
            right_knee_angle=self._smooth(
                right_knee_angle
            ),
            left_knee_angle=self._smooth(
                left_knee_angle
            ),
            right_hip_angle=self._smooth(
                right_hip_angle
            ),
            left_hip_angle=self._smooth(
                left_hip_angle
            ),
            right_shoulder_angle=self._smooth(
                right_shoulder_angle
            ),
            left_shoulder_angle=self._smooth(
                left_shoulder_angle
            ),
            pelvis_height=self._smooth(
                pelvis_height
            ),
            left_ankle_height=self._smooth(
                left_ankle_height
            ),
            right_ankle_height=self._smooth(
                right_ankle_height
            ),
            average_ankle_height=self._smooth(
                average_ankle_height
            ),
        )

    # =====================================================
    # POINT EXTRACTION
    # =====================================================

    def _extract_points(
        self,
        point_name: str,
        player_joint: bool,
    ) -> np.ndarray:
        """
        Extract one 3D point for every frame.
        """

        points = np.full(
            (
                len(self.tracking_frames),
                3,
            ),
            np.nan,
            dtype=float,
        )

        for index, frame in enumerate(
            self.tracking_frames
        ):
            frame_data = frame.get(
                "data",
                {},
            )

            if player_joint:
                point = (
                    frame_data
                    .get("player", {})
                    .get(point_name)
                )
            else:
                point = frame_data.get(point_name)

            if not is_valid_point(point):
                continue

            points[index] = np.asarray(
                point,
                dtype=float,
            )

        return points

    def _joint_height_series(
        self,
        joint_name: str,
    ) -> np.ndarray:
        """
        Extract the Z coordinate of one body joint.
        """

        return self._extract_points(
            joint_name,
            player_joint=True,
        )[:, 2]

    def _pelvis_height_series(self) -> np.ndarray:
        """
        Use MID_HIP when valid, otherwise average both hips.
        """

        result = np.full(
            len(self.tracking_frames),
            np.nan,
            dtype=float,
        )

        for index, frame in enumerate(
            self.tracking_frames
        ):
            player = (
                frame
                .get("data", {})
                .get("player", {})
            )

            mid_hip = player.get("MID_HIP")

            if is_valid_point(mid_hip):
                mid_hip_array = np.asarray(
                    mid_hip,
                    dtype=float,
                )

                # Some trials contain a present but unusable MID_HIP.
                # Reject coordinates that collapse near the origin.
                if np.linalg.norm(mid_hip_array) > 0.25:
                    result[index] = mid_hip_array[2]
                    continue

            left_hip = player.get("LEFT_HIP")
            right_hip = player.get("RIGHT_HIP")

            if (
                is_valid_point(left_hip)
                and is_valid_point(right_hip)
            ):
                result[index] = (
                    float(left_hip[2])
                    + float(right_hip[2])
                ) / 2.0

            elif is_valid_point(left_hip):
                result[index] = float(left_hip[2])

            elif is_valid_point(right_hip):
                result[index] = float(right_hip[2])

        return result

    # =====================================================
    # VELOCITY
    # =====================================================

    def _calculate_velocity(
        self,
        positions: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate 3D velocity using frame-to-frame differences.

        Velocity is only calculated when both adjacent positions are
        valid. This prevents missing points from creating false spikes.
        """

        velocity = np.full_like(
            positions,
            np.nan,
            dtype=float,
        )

        seconds_per_frame = (
            1.0 / self.sampling_rate
        )

        for index in range(
            1,
            len(positions),
        ):
            previous_point = positions[index - 1]
            current_point = positions[index]

            if (
                np.any(np.isnan(previous_point))
                or np.any(np.isnan(current_point))
            ):
                continue

            velocity[index] = (
                current_point
                - previous_point
            ) / seconds_per_frame

        return velocity

    @staticmethod
    def _vector_magnitude(
        vectors: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate vector magnitude while preserving missing rows.
        """

        result = np.full(
            len(vectors),
            np.nan,
            dtype=float,
        )

        valid_rows = ~np.any(
            np.isnan(vectors),
            axis=1,
        )

        result[valid_rows] = np.linalg.norm(
            vectors[valid_rows],
            axis=1,
        )

        return result

    # =====================================================
    # JOINT ANGLES
    # =====================================================

    def _joint_angle_series(
        self,
        first_joint: str,
        vertex_joint: str,
        third_joint: str,
    ) -> np.ndarray:
        """
        Calculate one three-dimensional joint angle per frame.
        """

        result = np.full(
            len(self.tracking_frames),
            np.nan,
            dtype=float,
        )

        for index, frame in enumerate(
            self.tracking_frames
        ):
            player = (
                frame
                .get("data", {})
                .get("player", {})
            )

            point_a = player.get(first_joint)
            vertex = player.get(vertex_joint)
            point_c = player.get(third_joint)

            angle = self._calculate_angle(
                point_a=point_a,
                vertex=vertex,
                point_c=point_c,
            )

            if angle is not None:
                result[index] = angle

        return result

    @staticmethod
    def _calculate_angle(
        point_a: object,
        vertex: object,
        point_c: object,
    ) -> float | None:
        """
        Calculate the 3D angle A-vertex-C in degrees.
        """

        if (
            not is_valid_point(point_a)
            or not is_valid_point(vertex)
            or not is_valid_point(point_c)
        ):
            return None

        vector_a = (
            np.asarray(point_a, dtype=float)
            - np.asarray(vertex, dtype=float)
        )

        vector_c = (
            np.asarray(point_c, dtype=float)
            - np.asarray(vertex, dtype=float)
        )

        magnitude_a = float(
            np.linalg.norm(vector_a)
        )

        magnitude_c = float(
            np.linalg.norm(vector_c)
        )

        if (
            magnitude_a <= 1e-12
            or magnitude_c <= 1e-12
        ):
            return None

        cosine_value = float(
            np.dot(vector_a, vector_c)
            / (magnitude_a * magnitude_c)
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
                np.arccos(cosine_value)
            )
        )

    # =====================================================
    # SMOOTHING
    # =====================================================

    def _smooth(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """
        Apply a centered NaN-aware moving average.

        Missing values remain missing when a window contains no valid
        measurements. No interpolation is performed at this stage.
        """

        numeric_values = np.asarray(
            values,
            dtype=float,
        )

        if (
            self.smoothing_window <= 1
            or numeric_values.size == 0
        ):
            return numeric_values.copy()

        half_window = (
            self.smoothing_window // 2
        )

        smoothed = np.full_like(
            numeric_values,
            np.nan,
            dtype=float,
        )

        for index in range(
            numeric_values.size
        ):
            start = max(
                0,
                index - half_window,
            )

            end = min(
                numeric_values.size,
                index + half_window + 1,
            )

            window = numeric_values[start:end]
            valid_values = window[
                ~np.isnan(window)
            ]

            if valid_values.size > 0:
                smoothed[index] = float(
                    np.mean(valid_values)
                )

        return smoothed

    # =====================================================
    # HELPERS
    # =====================================================

    @staticmethod
    def _nanmean_pair(
        first: np.ndarray,
        second: np.ndarray,
    ) -> np.ndarray:
        """
        Average two arrays without warnings for fully missing pairs.
        """

        result = np.full_like(
            first,
            np.nan,
            dtype=float,
        )

        first_valid = ~np.isnan(first)
        second_valid = ~np.isnan(second)

        both_valid = (
            first_valid
            & second_valid
        )

        result[both_valid] = (
            first[both_valid]
            + second[both_valid]
        ) / 2.0

        only_first = (
            first_valid
            & ~second_valid
        )

        result[only_first] = first[only_first]

        only_second = (
            second_valid
            & ~first_valid
        )

        result[only_second] = second[only_second]

        return result

    @staticmethod
    def _safe_float(
        value: object,
    ) -> float:
        """
        Convert a value to float or return NaN.
        """

        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")


def _count_valid(
    values: np.ndarray,
) -> int:
    return int(
        np.count_nonzero(
            ~np.isnan(values)
        )
    )


def _run_time_series_test() -> None:
    """
    Run a standalone test using the first free-throw trial.
    """

    trial_file, trial_data = load_first_trial()

    analyzer = TimeSeriesAnalyzer(
        trial_data=trial_data,
        smoothing_window=5,
    )

    result = analyzer.analyze()

    expected_frames = len(
        trial_data.get("tracking", [])
    )

    assert result.total_frames == expected_frames

    for series_name, values in (
        result.available_series().items()
    ):
        assert len(values) == expected_frames, (
            f"{series_name} has the wrong length."
        )

    assert _count_valid(
        result.ball_height
    ) > 0

    assert _count_valid(
        result.right_elbow_angle
    ) > 0

    assert _count_valid(
        result.pelvis_height
    ) > 0

    print("TIME-SERIES ENGINE TEST PASSED")
    print(f"Trial file: {trial_file.name}")
    print(f"Total frames: {result.total_frames}")
    print(
        "Available curves: "
        f"{len(result.available_series())}"
    )
    print(
        "Valid ball-height frames: "
        f"{_count_valid(result.ball_height)}"
    )
    print(
        "Valid ball-speed frames: "
        f"{_count_valid(result.ball_speed)}"
    )
    print(
        "Valid right-elbow frames: "
        f"{_count_valid(result.right_elbow_angle)}"
    )
    print(
        "Valid pelvis-height frames: "
        f"{_count_valid(result.pelvis_height)}"
    )


if __name__ == "__main__":
    _run_time_series_test()
