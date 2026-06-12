import unittest

from main import SwipeDetector, calculate_swipe_threshold


class SwipeDetectorTests(unittest.TestCase):
    def make_detector(self):
        return SwipeDetector(
            threshold_pixels=50,
            cooldown_seconds=1.0,
            history_seconds=1.0,
            hand_loss_grace_seconds=0.25,
            min_samples=4,
        )

    def test_left_wave_requests_next_page(self):
        detector = self.make_detector()

        results = [
            detector.update(x, timestamp)
            for timestamp, x in (
                (0.00, 320),
                (0.08, 300),
                (0.16, 270),
                (0.24, 245),
            )
        ]

        self.assertEqual(results[-1], "right")

    def test_right_wave_requests_previous_page(self):
        detector = self.make_detector()

        results = [
            detector.update(x, timestamp)
            for timestamp, x in (
                (0.00, 245),
                (0.08, 265),
                (0.16, 300),
                (0.24, 325),
            )
        ]

        self.assertEqual(results[-1], "left")

    def test_small_movement_does_not_trigger(self):
        detector = self.make_detector()

        results = [
            detector.update(x, timestamp)
            for timestamp, x in (
                (0.00, 300),
                (0.08, 295),
                (0.16, 290),
                (0.24, 285),
            )
        ]

        self.assertTrue(all(result is None for result in results))

    def test_short_tracking_loss_preserves_gesture(self):
        detector = self.make_detector()
        detector.update(320, 0.00)
        detector.update(305, 0.08)
        detector.mark_hand_missing(0.20)

        detector.update(275, 0.24)
        result = detector.update(245, 0.32)

        self.assertEqual(result, "right")

    def test_cooldown_blocks_return_movement(self):
        detector = self.make_detector()
        for timestamp, x in (
            (0.00, 320),
            (0.08, 300),
            (0.16, 270),
            (0.24, 245),
        ):
            first_result = detector.update(x, timestamp)

        return_results = [
            detector.update(x, timestamp)
            for timestamp, x in (
                (0.32, 265),
                (0.40, 290),
                (0.48, 315),
                (0.56, 335),
            )
        ]

        self.assertEqual(first_result, "right")
        self.assertTrue(all(result is None for result in return_results))

    def test_threshold_scales_for_camera_width(self):
        self.assertEqual(calculate_swipe_threshold(320), 45)
        self.assertEqual(calculate_swipe_threshold(640), 51)
        self.assertEqual(calculate_swipe_threshold(1920), 80)


if __name__ == "__main__":
    unittest.main()
