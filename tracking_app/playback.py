from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic


FrameCallback = Callable[[int], None]


@dataclass(frozen=True)
class PlaybackState:
    """
    Read-only snapshot of the current playback state.
    """

    current_frame: int
    total_frames: int
    source_fps: float
    speed_multiplier: float
    is_playing: bool
    loop_enabled: bool

    @property
    def effective_fps(self) -> float:
        """
        Return the playback FPS after applying the speed multiplier.
        """

        return self.source_fps * self.speed_multiplier

    @property
    def final_frame(self) -> int:
        """
        Return the final valid frame index.
        """

        return self.total_frames - 1


class PlaybackController:
    """
    Time-based playback controller.

    This class manages:

    - Play and pause
    - Playback speed
    - Frame stepping
    - Looping
    - Real-time frame advancement

    It does not depend on PyVista. The viewer provides a callback
    that is called whenever the displayed frame should change.
    """

    ALLOWED_SPEEDS = (
        0.25,
        0.50,
        1.00,
        2.00,
    )

    def __init__(
        self,
        total_frames: int,
        source_fps: float,
        frame_callback: FrameCallback,
        loop_enabled: bool = True,
    ) -> None:
        if total_frames <= 0:
            raise ValueError(
                "Playback requires at least one frame."
            )

        if source_fps <= 0:
            raise ValueError(
                "Source FPS must be greater than zero."
            )

        if not callable(frame_callback):
            raise TypeError(
                "frame_callback must be callable."
            )

        self.total_frames = int(total_frames)
        self.source_fps = float(source_fps)
        self.frame_callback = frame_callback

        self.current_frame = 0
        self.speed_multiplier = 1.0

        self.is_playing = False
        self.loop_enabled = bool(loop_enabled)

        self._last_tick_time: float | None = None
        self._elapsed_accumulator = 0.0

    # =====================================================
    # STATE
    # =====================================================

    @property
    def final_frame(self) -> int:
        """
        Return the final valid frame index.
        """

        return self.total_frames - 1

    @property
    def effective_fps(self) -> float:
        """
        Return playback FPS after applying the speed multiplier.
        """

        return self.source_fps * self.speed_multiplier

    @property
    def frame_interval_seconds(self) -> float:
        """
        Return the time required between frames.
        """

        return 1.0 / self.effective_fps

    def get_state(self) -> PlaybackState:
        """
        Return a read-only snapshot of the playback state.
        """

        return PlaybackState(
            current_frame=self.current_frame,
            total_frames=self.total_frames,
            source_fps=self.source_fps,
            speed_multiplier=self.speed_multiplier,
            is_playing=self.is_playing,
            loop_enabled=self.loop_enabled,
        )

    # =====================================================
    # PLAY AND PAUSE
    # =====================================================

    def play(self) -> None:
        """
        Start or resume playback.

        When playback starts from the final frame and looping is
        enabled, playback restarts from Frame 0.
        """

        if self.is_playing:
            return

        if self.current_frame >= self.final_frame:
            if self.loop_enabled:
                self.set_frame(0)
            else:
                return

        self.is_playing = True
        self._reset_clock()

    def pause(self) -> None:
        """
        Pause playback at the current frame.
        """

        self.is_playing = False
        self._reset_clock()

    def toggle_playback(self) -> None:
        """
        Switch between playing and paused.
        """

        if self.is_playing:
            self.pause()
        else:
            self.play()

    # =====================================================
    # SPEED
    # =====================================================

    def set_speed(
        self,
        speed_multiplier: float,
    ) -> None:
        """
        Set the playback speed.

        Supported values:

        - 0.25
        - 0.50
        - 1.00
        - 2.00
        """

        requested_speed = float(speed_multiplier)

        if requested_speed not in self.ALLOWED_SPEEDS:
            allowed_text = ", ".join(
                f"{speed:g}x"
                for speed in self.ALLOWED_SPEEDS
            )

            raise ValueError(
                "Unsupported playback speed. "
                f"Allowed speeds: {allowed_text}."
            )

        self.speed_multiplier = requested_speed
        self._reset_clock()

    # =====================================================
    # LOOPING
    # =====================================================

    def set_loop(
        self,
        enabled: bool,
    ) -> None:
        """
        Enable or disable playback looping.
        """

        self.loop_enabled = bool(enabled)

    def toggle_loop(self) -> None:
        """
        Switch looping on or off.
        """

        self.loop_enabled = not self.loop_enabled

    # =====================================================
    # FRAME CONTROL
    # =====================================================

    def set_frame(
        self,
        frame_index: int,
        notify_viewer: bool = True,
    ) -> int:
        """
        Move to a specific frame.

        Values outside the available range are clamped.
        """

        clamped_frame = max(
            0,
            min(
                int(frame_index),
                self.final_frame,
            ),
        )

        self.current_frame = clamped_frame
        self._reset_clock()

        if notify_viewer:
            self.frame_callback(self.current_frame)

        return self.current_frame

    def step(
        self,
        frame_change: int,
    ) -> int:
        """
        Pause playback and move by a specific frame count.
        """

        self.pause()

        return self.set_frame(
            self.current_frame + int(frame_change)
        )

    def reset(self) -> int:
        """
        Pause playback and return to Frame 0.
        """

        self.pause()

        return self.set_frame(0)

    # =====================================================
    # TIMER UPDATE
    # =====================================================

    def tick(
        self,
        current_time: float | None = None,
    ) -> bool:
        """
        Advance playback based on elapsed real time.

        The PyVista viewer will call this method repeatedly using
        a timer event.

        Returns
        -------
        bool
            True when the frame changed.
            False when no frame change was required.
        """

        if not self.is_playing:
            return False

        now = (
            monotonic()
            if current_time is None
            else float(current_time)
        )

        if self._last_tick_time is None:
            self._last_tick_time = now
            return False

        elapsed = max(
            0.0,
            now - self._last_tick_time,
        )

        self._last_tick_time = now
        self._elapsed_accumulator += elapsed

        frame_interval = self.frame_interval_seconds

        # This tolerance avoids normal floating-point rounding
        # problems at exact timing boundaries.
        timing_tolerance = frame_interval * 1e-9

        frames_to_advance = int(
            (
                self._elapsed_accumulator
                + timing_tolerance
            )
            / frame_interval
        )

        if frames_to_advance <= 0:
            return False

        self._elapsed_accumulator -= (
            frames_to_advance
            * frame_interval
        )

        if self._elapsed_accumulator < 0:
            self._elapsed_accumulator = 0.0

        next_frame = (
            self.current_frame
            + frames_to_advance
        )

        if next_frame <= self.final_frame:
            self.current_frame = next_frame
            self.frame_callback(self.current_frame)
            return True

        if self.loop_enabled:
            self.current_frame = (
                next_frame
                % self.total_frames
            )

            self.frame_callback(self.current_frame)
            return True

        self.current_frame = self.final_frame
        self.frame_callback(self.current_frame)
        self.pause()

        return True

    # =====================================================
    # STATUS DISPLAY
    # =====================================================

    def status_text(self) -> str:
        """
        Return viewer-ready playback status text.
        """

        playback_status = (
            "PLAYING"
            if self.is_playing
            else "PAUSED"
        )

        loop_status = (
            "ON"
            if self.loop_enabled
            else "OFF"
        )

        return (
            f"{playback_status} | "
            f"{self.speed_multiplier:g}x | "
            f"Loop {loop_status}"
        )

    # =====================================================
    # INTERNAL CLOCK
    # =====================================================

    def _reset_clock(self) -> None:
        """
        Clear real-time playback timing.
        """

        self._last_tick_time = None
        self._elapsed_accumulator = 0.0


def _run_playback_test() -> None:
    """
    Test the playback controller without opening PyVista.
    """

    displayed_frames: list[int] = []

    controller = PlaybackController(
        total_frames=10,
        source_fps=10,
        frame_callback=displayed_frames.append,
        loop_enabled=True,
    )

    # -----------------------------------------------------
    # INITIAL STATE
    # -----------------------------------------------------

    assert controller.current_frame == 0
    assert controller.final_frame == 9
    assert controller.effective_fps == 10
    assert controller.is_playing is False

    # -----------------------------------------------------
    # SPEED AND BASIC PLAYBACK
    # -----------------------------------------------------

    controller.set_speed(2.0)

    assert controller.effective_fps == 20
    assert controller.frame_interval_seconds == 0.05

    controller.play()

    controller.tick(current_time=0.00)
    controller.tick(current_time=0.05)
    controller.tick(current_time=0.10)

    assert displayed_frames == [1, 2]
    assert controller.current_frame == 2

    # -----------------------------------------------------
    # PAUSE AND STEP
    # -----------------------------------------------------

    controller.pause()

    assert controller.is_playing is False

    controller.step(3)

    assert controller.current_frame == 5
    assert displayed_frames[-1] == 5

    # -----------------------------------------------------
    # LOOPING
    # -----------------------------------------------------

    controller.set_frame(8)

    assert controller.current_frame == 8

    controller.play()

    controller.tick(current_time=1.00)

    # At 20 FPS, 0.10 seconds advances two frames:
    # Frame 8 -> Frame 9 -> Frame 0.
    controller.tick(current_time=1.10)

    assert controller.current_frame == 0
    assert displayed_frames[-1] == 0

    # -----------------------------------------------------
    # FINAL OUTPUT
    # -----------------------------------------------------

    print("PLAYBACK ENGINE TEST PASSED")
    print(f"Current frame: {controller.current_frame}")
    print(f"Source FPS: {controller.source_fps:g}")
    print(
        f"Playback speed: "
        f"{controller.speed_multiplier:g}x"
    )
    print(
        f"Effective FPS: "
        f"{controller.effective_fps:g}"
    )
    print(
        f"Loop enabled: "
        f"{controller.loop_enabled}"
    )
    print(
        f"Status: "
        f"{controller.status_text()}"
    )


if __name__ == "__main__":
    _run_playback_test()