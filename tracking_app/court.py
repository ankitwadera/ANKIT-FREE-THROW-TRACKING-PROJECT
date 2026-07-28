from __future__ import annotations

import numpy as np
import pyvista as pv


# =========================================================
# COURT DIMENSIONS — FEET
# =========================================================

# SPL tracking convention:
# X = length of the court
# Y = width of the court
# Z = height

COURT_LENGTH = 94.0
COURT_WIDTH = 50.0

HALF_LENGTH = COURT_LENGTH / 2
HALF_WIDTH = COURT_WIDTH / 2

HOOP_HEIGHT = 10.0
HOOP_RADIUS = 0.75

HOOP_DISTANCE_FROM_BASELINE = 5.25
HOOP_X = HALF_LENGTH - HOOP_DISTANCE_FROM_BASELINE

BACKBOARD_DISTANCE_FROM_BASELINE = 4.0
BACKBOARD_X = HALF_LENGTH - BACKBOARD_DISTANCE_FROM_BASELINE

KEY_WIDTH = 16.0
FREE_THROW_LINE_DISTANCE = 19.0

LINE_HEIGHT = 0.025
LINE_THICKNESS = 0.06


# =========================================================
# GEOMETRY HELPERS
# =========================================================

def create_line(
    point_a: tuple[float, float, float],
    point_b: tuple[float, float, float],
    thickness: float = LINE_THICKNESS,
) -> pv.PolyData:
    """
    Create one court marking as a narrow 3D tube.
    """

    return pv.Tube(
        pointa=point_a,
        pointb=point_b,
        radius=thickness,
        n_sides=10,
        capping=True,
    )


def create_circle(
    center_x: float,
    center_y: float,
    radius: float,
    height: float,
    number_of_points: int = 100,
) -> pv.PolyData:
    """
    Create a flat circular marking parallel to the floor.
    """

    angles = np.linspace(
        0,
        2 * np.pi,
        number_of_points,
        endpoint=True,
    )

    points = np.column_stack(
        (
            center_x + radius * np.cos(angles),
            center_y + radius * np.sin(angles),
            np.full(number_of_points, height),
        )
    )

    circle = pv.lines_from_points(
        points,
        close=True,
    )

    return circle.tube(
        radius=LINE_THICKNESS,
        n_sides=10,
        capping=True,
    )


# =========================================================
# COURT DRAWING
# =========================================================

def add_basketball_court(plotter: pv.Plotter) -> None:
    """
    Add a full basketball court to an existing PyVista viewer.

    Coordinate system:
    X = court length
    Y = court width
    Z = height
    """

    # -----------------------------------------------------
    # FLOOR
    # -----------------------------------------------------

    floor = pv.Box(
        bounds=(
            -HALF_LENGTH,
            HALF_LENGTH,
            -HALF_WIDTH,
            HALF_WIDTH,
            -0.08,
            0.0,
        )
    )

    plotter.add_mesh(
        floor,
        color="burlywood",
        show_edges=False,
    )

    court_lines: list[pv.PolyData] = []

    # -----------------------------------------------------
    # OUTER BOUNDARY
    # -----------------------------------------------------

    court_lines.extend(
        [
            # Baselines
            create_line(
                (-HALF_LENGTH, -HALF_WIDTH, LINE_HEIGHT),
                (-HALF_LENGTH, HALF_WIDTH, LINE_HEIGHT),
            ),
            create_line(
                (HALF_LENGTH, -HALF_WIDTH, LINE_HEIGHT),
                (HALF_LENGTH, HALF_WIDTH, LINE_HEIGHT),
            ),

            # Sidelines
            create_line(
                (-HALF_LENGTH, -HALF_WIDTH, LINE_HEIGHT),
                (HALF_LENGTH, -HALF_WIDTH, LINE_HEIGHT),
            ),
            create_line(
                (-HALF_LENGTH, HALF_WIDTH, LINE_HEIGHT),
                (HALF_LENGTH, HALF_WIDTH, LINE_HEIGHT),
            ),

            # Half-court line
            create_line(
                (0, -HALF_WIDTH, LINE_HEIGHT),
                (0, HALF_WIDTH, LINE_HEIGHT),
            ),
        ]
    )

    # -----------------------------------------------------
    # CENTER CIRCLE
    # -----------------------------------------------------

    center_circle = create_circle(
        center_x=0,
        center_y=0,
        radius=6.0,
        height=LINE_HEIGHT,
    )

    # -----------------------------------------------------
    # KEYS AND FREE-THROW CIRCLES
    # -----------------------------------------------------

    for court_side in (-1, 1):
        baseline_x = court_side * HALF_LENGTH

        free_throw_x = court_side * (
            HALF_LENGTH - FREE_THROW_LINE_DISTANCE
        )

        key_bottom_y = -KEY_WIDTH / 2
        key_top_y = KEY_WIDTH / 2

        court_lines.extend(
            [
                create_line(
                    (baseline_x, key_bottom_y, LINE_HEIGHT),
                    (free_throw_x, key_bottom_y, LINE_HEIGHT),
                ),
                create_line(
                    (baseline_x, key_top_y, LINE_HEIGHT),
                    (free_throw_x, key_top_y, LINE_HEIGHT),
                ),
                create_line(
                    (free_throw_x, key_bottom_y, LINE_HEIGHT),
                    (free_throw_x, key_top_y, LINE_HEIGHT),
                ),
            ]
        )

        free_throw_circle = create_circle(
            center_x=free_throw_x,
            center_y=0,
            radius=6.0,
            height=LINE_HEIGHT,
        )

        plotter.add_mesh(
            free_throw_circle,
            color="white",
        )

    # -----------------------------------------------------
    # ADD THE FLOOR MARKINGS
    # -----------------------------------------------------

    for court_line in court_lines:
        plotter.add_mesh(
            court_line,
            color="white",
        )

    plotter.add_mesh(
        center_circle,
        color="white",
    )

    # -----------------------------------------------------
    # HOOPS AND BACKBOARDS
    # -----------------------------------------------------

    for court_side in (-1, 1):
        hoop_x = court_side * HOOP_X
        backboard_x = court_side * BACKBOARD_X

        rim = create_circle(
            center_x=hoop_x,
            center_y=0,
            radius=HOOP_RADIUS,
            height=HOOP_HEIGHT,
        )

        plotter.add_mesh(
            rim,
            color="orange",
        )

        # Backboard is thin along X and extends across Y.
        backboard = pv.Box(
            bounds=(
                backboard_x - 0.05,
                backboard_x + 0.05,
                -3.0,
                3.0,
                9.0,
                12.5,
            )
        )

        plotter.add_mesh(
            backboard,
            color="white",
            opacity=0.65,
            show_edges=True,
        )

        support = create_line(
            (hoop_x, 0, HOOP_HEIGHT),
            (backboard_x, 0, HOOP_HEIGHT),
            thickness=0.05,
        )

        plotter.add_mesh(
            support,
            color="gray",
        )


# =========================================================
# STANDALONE COURT TEST
# =========================================================

def test_court() -> None:
    """
    Display the corrected court by itself.
    """

    plotter = pv.Plotter(
        window_size=(1200, 850),
    )

    plotter.set_background("black")

    add_basketball_court(plotter)

    plotter.add_axes()
    plotter.camera_position = "iso"
    plotter.reset_camera()

    plotter.add_text(
        "Basketball Motion Studio — Coordinate-Aligned Court",
        position="upper_left",
        font_size=13,
        color="white",
    )

    print("CORRECTED COURT TEST READY")
    print("X = court length")
    print("Y = court width")
    print("Z = height")

    plotter.show()


if __name__ == "__main__":
    test_court()