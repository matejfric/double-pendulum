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
RED = (255, 0, 0)
BLUE = (0, 0, 255)
PURPLE = (128, 0, 128)
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


def update_double_pendulum(p1: Pendulum, p2: Pendulum) -> tuple[Pendulum, Pendulum]:
    # Constants
    R1 = p1.r
    R2 = p2.r
    M1 = p1.m
    M2 = p2.m
    A1 = p1.a
    A2 = p2.a
    V1 = p1.v
    V2 = p2.v
    EPS = 1e-3

    # Acceleration
    num1 = -G * (2 * M1 + M2) * np.sin(A1)
    num2 = -M2 * G * np.sin(A1 - 2 * A2)
    num3 = -2 * np.sin(A1 - A2) * M2
    num4 = (V2**2) * R2 + (V1**2) * R1 * np.cos(A1 - A2)
    den = R1 * (2 * M1 + M2 - M2 * np.cos(2 * A1 - 2 * A2))
    a1 = (num1 + num2 + num3 * num4) / (den + EPS)

    num1 = 2 * np.sin(A1 - A2)
    num2 = ((V1**2) * R1 * (M1 + M2))
    num3 = G * (M1 + M2) * np.cos(A1)
    num4 = (V2**2) * R2 * M2 * np.cos(A1 - A2)
    den = R2 * (2 * M1 + M2 - M2 * np.cos(2 * A1 - 2 * A2))
    a2 = (num1 * (num2 + num3 + num4)) / (den + EPS)

    # Update velocity
    p1.v += a1
    p2.v += a2

    # Update angle
    p1.a += p1.v
    p2.a += p2.v

    # Damping of velocity
    p1.v *= 0.995
    p2.v *= 0.995

    return p1, p2


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
        pygame.draw.aaline(
            trajectory_surface, BLUE, (int(prev_x2), int(prev_y2)), (int(x2), int(y2))
        )
        # pygame.gfxdraw.line(trajectory_surface, int(prev_x2), int(prev_y2), int(x2), int(y2), PURPLE)

    screen.fill(WHITE)
    screen.blit(trajectory_surface, (0, 0))

    # pygame.gfxdraw.line(screen, CX, CY, int(x1), int(y1), RED)
    # pygame.gfxdraw.line(screen, int(x1), int(y1), int(x2), int(y2), BLUE)
    pygame.draw.aaline(screen, RED, (CX, CY), (int(x1), int(y1)))
    pygame.draw.aaline(screen, BLUE, (int(x1), int(y1)), (int(x2), int(y2)))

    draw_aa_filled_circle(screen, BLACK, (CX, CY), 4)
    draw_aa_filled_circle(screen, RED, (int(x1), int(y1)), 8)
    draw_aa_filled_circle(screen, BLUE, (int(x2), int(y2)), 8)
    # pygame.gfxdraw.filled_circle(screen, CX, CY, 5, BLACK)
    # pygame.gfxdraw.filled_circle(screen, int(x1), int(y1), 10, RED)
    # pygame.gfxdraw.filled_circle(screen, int(x2), int(y2), 10, BLUE)
    # pygame.draw.circle(screen, RED, (int(x1), int(y1)), 10)
    # pygame.draw.circle(screen, BLUE, (int(x2), int(y2)), 10)

    pygame.display.flip()

    return x2, y2


def main_euler():
    """The numerical errors accumulate and the pendulum
    quickly diverges from the expected behavior.
    """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Double Pendulum Simulation")
    clock = pygame.time.Clock()

    # Pendulum parameters
    p1 = Pendulum(r=100, m=20, a=PI / 3, v=0.00)
    p2 = Pendulum(r=100, m=20, a=PI / 2, v=0.00)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update and draw pendulum
        p1, p2 = update_double_pendulum(p1, p2)
        draw_double_pendulum(screen, p1, p2)

        clock.tick(60)

    pygame.quit()


def main_ode45():
    """The ODE45 solver is more stable and the pendulum behaves as expected."""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Double Pendulum Simulation")
    clock = pygame.time.Clock()

    # Pendulum parameters
    p1 = Pendulum(r=100, m=20, a=PI / 3, v=0.00)
    p2 = Pendulum(r=100, m=20, a=PI / 2, v=0.00)

    # Create a surface for drawing the trajectory
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
    main_ode45()
