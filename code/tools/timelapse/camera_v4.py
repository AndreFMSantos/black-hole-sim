"""Global camera of the v4 video: constant inclination (after a start seen from above for the 2D scenes) and ONE full orbit around the hole over the whole video (phi = 360 t / total).

Distance and FOV are fixed, so the hole has the same size in every version. The inclination is 75 degrees (the 2D scenes of v0.1/v0.2 are
checked at that angle). Same interface as camera.py (Camera(total).state(t)); REF_TOTAL = None means "use the real total duration".
"""
THETA = 75.0
THETA_TOP = 0.5     # the 2D scenes of v0.1/v0.2 are drawn in the orbital plane: seen from above (not exactly 0, so the camera basis does not degenerate)
TILT_T0, TILT_T1 = 9.0, 17.0   # s02 (9-17 s) tilts the view from above to THETA, ready for the 3D renders that start at 17 s
DIST = 40.0
FOV = 40.0
REF_TOTAL = None


class Camera:
    def __init__(self, total):
        self.total = float(total)

    def theta(self, t):
        x = min(1.0, max(0.0, (t - TILT_T0) / (TILT_T1 - TILT_T0)))
        return THETA_TOP + (THETA - THETA_TOP) * x * x * (3.0 - 2.0 * x)

    def phi(self, t):
        return 360.0 * t / self.total

    def state(self, t):
        return (DIST, self.theta(t), self.phi(t))

    def dtheta(self, t, h=1e-4):
        return 0.0
