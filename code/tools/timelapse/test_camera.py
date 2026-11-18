import math
import unittest

import camera as C


class CameraTests(unittest.TestCase):
    cam = C.Camera(132.0)

    def test_starts_top_down_ends_at_80(self):
        self.assertAlmostEqual(self.cam.theta(0.0), C.THETA0, places=9)
        self.assertAlmostEqual(self.cam.theta(0.6 * 132.0), 80.0, places=9)
        self.assertAlmostEqual(self.cam.theta(132.0), 80.0, places=9)

    def test_theta_monotonic_and_smooth(self):
        prev = -1
        worst = 0.0
        for i in range(0, 1321):
            t = i * 0.1
            th = self.cam.theta(t)
            self.assertGreaterEqual(th, prev - 1e-12)
            prev = th
            worst = max(worst, abs(self.cam.dtheta(t)))
        # maximum tilt rate: 1.5 * 79.5 / 79.2 s = 1.5 degrees/s (maximum derivative of smoothstep = 1.5)
        self.assertLess(worst, 1.6)

    def test_phi_continuous_constant_rate(self):
        for t in (0.0, 10.0, 77.7, 131.9):
            self.assertAlmostEqual(self.cam.phi(t + 1.0) - self.cam.phi(t), C.PHI_RATE, places=9)

    def test_independent_of_segments(self):
        # the camera only depends on the global time: the same t always gives the same state
        self.assertEqual(self.cam.state(33.3), C.Camera(132.0).state(33.3))


if __name__ == "__main__":
    unittest.main()
