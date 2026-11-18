"""Global camera C(t) of the time-lapse (v1.1): a function of the video time, independent of the version being rendered.

It starts looking from above at the equatorial plane (theta ~ 0: the view of the 2D simulations of v0.1/v0.2) and tilts up to theta = 80 degrees
while the azimuth phi rotates at a constant rate around the hole; distance and FOV fixed, so the hole has the same size in all versions.
"""
import math

THETA0 = 0.5      # degrees: not exactly 0, so the camera basis does not degenerate in the old renderers
THETA1 = 80.0     # degrees, reached at TILT_END * total and held afterwards
TILT_END = 0.60   # fraction of the video
PHI_RATE = 4.0    # degrees per second of video (constant)
DIST = 40.0       # M
REF_TOTAL = 143.0  # duration of the original video (v1.1): the camera is a function of the absolute time with this reference, so adding segments does not change the frames already rendered
FOV = 40.0        # degrees (vertical), as in v0.4-v0.6


def smoothstep(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


class Camera:
    def __init__(self, total):
        self.total = float(total)

    def theta(self, t):
        return THETA0 + (THETA1 - THETA0) * smoothstep(t / (TILT_END * self.total))

    def phi(self, t):
        return PHI_RATE * t

    def state(self, t):
        return (DIST, self.theta(t), self.phi(t))

    def dtheta(self, t, h=1e-4):
        return (self.theta(t + h) - self.theta(t - h)) / (2 * h)
