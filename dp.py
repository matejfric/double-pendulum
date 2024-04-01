import numpy as np
from dataclasses import dataclass
import pygame
import pygame.gfxdraw
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

# Inspired by:
# https://thecodingtrain.com/challenges/93-double-pendulum
# https://www.myphysicslab.com/pendulum/double-pendulum-en.html


# region Constants
G = 9.81  # m/s^2
PI = np.pi

WIDTH = 800  # Width of the screen.
HEIGHT = 600  # Height of the screen.
CX = WIDTH // 2  # X-axis mount point of the pendulum.
CY = HEIGHT // 3  # Y-axis mount point of the pendulum.

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

BLUE = (31, 119, 180)
ORANGE = (255, 127, 14)
PURPLE = (148, 103, 189)
RED = (214, 39, 40)
GREEN = (44, 160, 44)

COLOR1 = (33, 177, 255)
COLOR2 = (255, 33, 140)
COLOR3 = (33, 177, 255)
# endregion


@dataclass
class Pendulum:
    r: float  # length
    m: float  # mass
    a: float  # angle
    v: float  # angular velocity


def double_pendulum_ode(
    t: float, y: ArrayLike, p1: Pendulum, p2: Pendulum
) -> list[float]:
    """
    ODE function for the double pendulum.

    Parameters
    ----------
    t: float
        Time variable (not used, because the system is time-independent).
    y: ArrayLike
        Array containing the angles and angular velocities of the pendulums.
    p1: Pendulum
        First pendulum object.
    p2: Pendulum
        Second pendulum object.

    Returns
    -------
    list[float]
        Array containing the derivatives of the angles and angular velocities.
    """
    A1, A2, V1, V2 = y

    R1 = p1.r
    R2 = p2.r
    M1 = p1.m
    M2 = p2.m

    num1 = -G * (2 * M1 + M2) * np.sin(A1)
    num2 = -M2 * G * np.sin(A1 - 2 * A2)
    num3 = -2 * np.sin(A1 - A2) * M2
    num4 = (V2**2) * R2 + (V1**2) * R1 * np.cos(A1 - A2)
    den = R1 * (2 * M1 + M2 - M2 * np.cos(2 * A1 - 2 * A2))
    a1 = (num1 + num2 + num3 * num4) / den

    num1 = 2 * np.sin(A1 - A2)
    num2 = (V1**2) * R1 * (M1 + M2)
    num3 = G * (M1 + M2) * np.cos(A1)
    num4 = (V2**2) * R2 * M2 * np.cos(A1 - A2)
    den = R2 * (2 * M1 + M2 - M2 * np.cos(2 * A1 - 2 * A2))
    a2 = (num1 * (num2 + num3 + num4)) / den

    return [V1, V2, a1, a2]


def draw_thick_aaline(display, color, point0, point1, w):
    # https://gist.github.com/gerryjenkinslb/8d433632ab541ad282f0c4fd49371b54
    import math

    x0, y0 = point0
    x1, y1 = point1
    proportion = w / math.hypot(x1 - x0, y1 - y0) / 2
    adjx = (x1 - x0) * proportion  # x side
    adjy = (y1 - y0) * proportion  # y side

    pts = (
        (x0 - adjy, y0 + adjx),  # A
        (x0 + adjy, y0 - adjx),  # B
        (x1 + adjy, y1 - adjx),  # C
        (x1 - adjy, y1 + adjx),  # D
    )

    pygame.draw.aalines(display, color, True, pts, True)  # outline
    pygame.draw.polygon(display, color, pts, 0)  # fill it


def draw_aa_filled_circle(surf, color, center, radius):
    pygame.gfxdraw.aacircle(surf, *center, radius + 1, color)
    pygame.gfxdraw.aacircle(surf, *center, radius, color)
    pygame.gfxdraw.aacircle(surf, *center, radius - 1, color)
    pygame.draw.circle(surf, color, center, radius)


def draw_double_pendulum(
    screen: pygame.Surface,
    p1: Pendulum,
    p2: Pendulum,
    trajectory_surface: pygame.Surface,
    prev_x2: float,
    prev_y2: float,
) -> tuple[float, float]:
    x1 = CX + p1.r * np.sin(p1.a)
    y1 = CY + p1.r * np.cos(p1.a)
    x2 = x1 + p2.r * np.sin(p2.a)
    y2 = y1 + p2.r * np.cos(p2.a)

    # Draw trajectory of the second pendulum
    if prev_x2 and prev_y2:
        draw_thick_aaline(trajectory_surface, COLOR3, (prev_x2, prev_y2), (x2, y2), 0.8)

    screen.fill(BLACK)
    screen.blit(trajectory_surface, (0, 0))

    draw_thick_aaline(screen, COLOR1, (CX, CY), (int(x1), int(y1)), 3)
    draw_thick_aaline(screen, COLOR2, (int(x1), int(y1)), (int(x2), int(y2)), 3)

    draw_aa_filled_circle(screen, COLOR1, (CX, CY), 4)
    draw_aa_filled_circle(screen, COLOR1, (int(x1), int(y1)), 8)
    draw_aa_filled_circle(screen, COLOR2, (int(x2), int(y2)), 8)

    pygame.display.flip()

    return x2, y2


def simulate_double_pendulum():
    """Simulate the double pendulum with ODE45 numerical integration."""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Double Pendulum Simulation")
    clock = pygame.time.Clock()

    # Pendulum parameters
    p1 = Pendulum(r=125, m=20, a=PI / 3, v=0.00)
    p2 = Pendulum(r=130, m=20, a=PI / 2, v=0.00)

    # Create a surface for drawing the trajectory of the second pendulum.
    trajectory_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    # Initial conditions
    y0 = [p1.a, p2.a, p1.v, p2.v]

    t0 = 0  # Start time
    t1 = 1000  # End time
    dt = 0.1  # Time step

    # Solve ODEs
    sol = solve_ivp(
        double_pendulum_ode,
        [t0, t1],  # Time interval
        y0,  # Initial conditions
        args=(p1, p2),  # Additional arguments for the ODE function
        t_eval=np.arange(t0, t0 + t1, dt),
    )

    idx = 0
    prev_x2, prev_y2 = None, None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update pendulum angles
        p1.a, p2.a = sol.y[0, idx], sol.y[1, idx]

        # Draw pendulum
        prev_x2, prev_y2 = draw_double_pendulum(
            screen, p1, p2, trajectory_surface, prev_x2, prev_y2
        )

        idx = (idx + 1) % len(sol.t)

        clock.tick(180)

    pygame.quit()


if __name__ == "__main__":
    simulate_double_pendulum()
