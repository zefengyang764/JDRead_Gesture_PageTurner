"""Use hand-wave gestures to turn pages in a browser reader."""

from collections import deque
from statistics import median
import time

import cv2
import mediapipe as mp
import pyautogui


CAMERA_INDEX = 0
SWIPE_THRESHOLD_RATIO = 0.055
MIN_SWIPE_THRESHOLD_PIXELS = 32
MAX_SWIPE_THRESHOLD_PIXELS = 60
COOLDOWN_SECONDS = 0.8
GESTURE_WINDOW_SECONDS = 1.2
TARGET_SWIPE_SECONDS = 0.8
HAND_LOSS_GRACE_SECONDS = 0.35
MIN_TRACKED_SAMPLES = 3
MIN_DIRECTION_CONSISTENCY = 0.55
MIN_SAMPLE_MOVEMENT_PIXELS = 2
PALM_BASE_LANDMARKS = (5, 9, 13, 17)
WINDOW_NAME = "JDRead Gesture Page Turner"


class SwipeDetector:
    """Detect quick horizontal wrist movement from a short sample history."""

    def __init__(
        self,
        threshold_pixels,
        cooldown_seconds=COOLDOWN_SECONDS,
        history_seconds=GESTURE_WINDOW_SECONDS,
        hand_loss_grace_seconds=HAND_LOSS_GRACE_SECONDS,
        min_samples=MIN_TRACKED_SAMPLES,
        min_direction_consistency=MIN_DIRECTION_CONSISTENCY,
        min_sample_movement_pixels=MIN_SAMPLE_MOVEMENT_PIXELS,
    ):
        self.threshold_pixels = threshold_pixels
        self.cooldown_seconds = cooldown_seconds
        self.history_seconds = history_seconds
        self.hand_loss_grace_seconds = hand_loss_grace_seconds
        self.min_samples = min_samples
        self.min_direction_consistency = min_direction_consistency
        self.min_sample_movement_pixels = min_sample_movement_pixels
        self.samples = deque()
        self.last_seen_time = None
        self.last_trigger_time = float("-inf")

    def update(self, hand_x, current_time):
        """Return the arrow key for a detected swipe, otherwise return None."""
        self.last_seen_time = current_time

        while (
            self.samples
            and current_time - self.samples[0][0] > self.history_seconds
        ):
            self.samples.popleft()

        if current_time - self.last_trigger_time < self.cooldown_seconds:
            # Track a fresh origin during cooldown so the return movement
            # cannot trigger another page turn.
            self.samples.clear()
            self.samples.append((current_time, hand_x))
            return None

        self.samples.append((current_time, hand_x))
        if len(self.samples) < self.min_samples:
            return None

        samples = list(self.samples)
        smoothing_count = min(2, len(samples))
        current_x = median(
            sample_x for _, sample_x in samples[-smoothing_count:]
        )

        # Search the whole recent history for a local starting edge. The hand
        # can already be moving when tracking becomes stable, so requiring the
        # first sample to be the exact gesture origin causes missed swipes.
        anchor_samples = samples[:-1]
        left_anchor_index = max(
            range(len(anchor_samples)),
            key=lambda index: anchor_samples[index][1],
        )
        right_anchor_index = min(
            range(len(anchor_samples)),
            key=lambda index: anchor_samples[index][1],
        )

        left_displacement = (
            anchor_samples[left_anchor_index][1] - current_x
        )
        right_displacement = (
            current_x - anchor_samples[right_anchor_index][1]
        )

        left_valid = self._is_directional_swipe(
            samples[left_anchor_index:],
            left_displacement,
            direction=-1,
        )
        right_valid = self._is_directional_swipe(
            samples[right_anchor_index:],
            right_displacement,
            direction=1,
        )

        if left_valid and (
            not right_valid or left_displacement >= right_displacement
        ):
            key = "right"
        elif right_valid:
            key = "left"
        else:
            return None

        self.last_trigger_time = current_time
        self.samples.clear()
        self.samples.append((current_time, hand_x))
        return key

    def _is_directional_swipe(self, samples, displacement, direction):
        """Check distance, speed, and directional consistency."""
        if displacement < self.threshold_pixels or len(samples) < 2:
            return False

        duration = samples[-1][0] - samples[0][0]
        if duration <= 0 or duration > self.history_seconds:
            return False

        # A slow posture drift should not turn a page. Requiring the average
        # speed to cover the threshold within the history window keeps the
        # detector responsive without accepting gradual movement.
        minimum_speed = self.threshold_pixels / TARGET_SWIPE_SECONDS
        if displacement / duration < minimum_speed:
            return False

        deltas = [
            current[1] - previous[1]
            for previous, current in zip(samples, samples[1:])
        ]
        significant_deltas = [
            delta
            for delta in deltas
            if abs(delta) >= self.min_sample_movement_pixels
        ]
        if not significant_deltas:
            return False

        matching_deltas = sum(
            1
            for delta in significant_deltas
            if (delta < 0 if direction < 0 else delta > 0)
        )
        consistency = matching_deltas / len(significant_deltas)
        return consistency >= self.min_direction_consistency

    def mark_hand_missing(self, current_time):
        """Reset only after a meaningful tracking gap, not a single bad frame."""
        if (
            self.last_seen_time is not None
            and current_time - self.last_seen_time
            > self.hand_loss_grace_seconds
        ):
            self.samples.clear()
            self.last_seen_time = None

    def cooldown_remaining(self, current_time):
        """Return the number of seconds left in the trigger cooldown."""
        return max(
            0.0,
            self.cooldown_seconds
            - (current_time - self.last_trigger_time),
        )


def calculate_swipe_threshold(frame_width):
    """Scale sensitivity for camera resolution while keeping useful limits."""
    return min(
        MAX_SWIPE_THRESHOLD_PIXELS,
        max(
            MIN_SWIPE_THRESHOLD_PIXELS,
            int(frame_width * SWIPE_THRESHOLD_RATIO),
        ),
    )


def calculate_hand_position_x(hand_landmarks, frame_width):
    """Blend wrist and palm-base landmarks for a steadier hand position."""
    wrist_x = hand_landmarks.landmark[0].x
    palm_base_x = median(
        hand_landmarks.landmark[index].x
        for index in PALM_BASE_LANDMARKS
    )
    normalized_x = (wrist_x + palm_base_x) / 2
    return int(normalized_x * frame_width)


def draw_instructions(frame, status_text, swipe_threshold) -> None:
    """Draw usage instructions on the camera preview."""
    lines = (
        "Wave Left = Next Page",
        "Wave Right = Previous Page",
        "Press Q to Quit",
        f"Status: {status_text}",
        f"Sensitivity: {swipe_threshold}px",
    )

    for index, text in enumerate(lines):
        y = 35 + index * 35
        color = (0, 255, 255) if index == 3 else (0, 255, 0)
        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            color,
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    """Open the camera, detect horizontal wrist movement, and press page keys."""
    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    swipe_detector = None
    status_text = "Show one hand"
    status_until = 0.0

    try:
        if not camera.isOpened():
            raise RuntimeError(
                "Could not open the default camera. "
                "Check camera access and try again."
            )

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils

        # Keep pyautogui's emergency fail-safe enabled. Moving the mouse to the
        # upper-left corner stops key automation cleanly.
        pyautogui.FAILSAFE = True

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        ) as hands:
            while True:
                success, frame = camera.read()
                if not success:
                    print("Failed to read a frame from the camera.")
                    break

                # Flip before detection so the preview behaves like a mirror.
                # Moving left in the mirrored preview decreases wrist x, while
                # moving right increases it.
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                current_time = time.monotonic()
                hand_detected = bool(results.multi_hand_landmarks)

                if swipe_detector is None:
                    swipe_threshold = calculate_swipe_threshold(frame.shape[1])
                    swipe_detector = SwipeDetector(swipe_threshold)

                if hand_detected:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    frame_width = frame.shape[1]
                    hand_x = calculate_hand_position_x(
                        hand_landmarks,
                        frame_width,
                    )

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                    )

                    key = swipe_detector.update(hand_x, current_time)
                    if key == "right":
                        # Physical left wave in the mirrored preview:
                        # press Right Arrow to go to the next page.
                        pyautogui.press("right")
                        status_text = "Next Page"
                        status_until = current_time + 0.7
                        print("Wave Left -> Next Page")
                    elif key == "left":
                        # Physical right wave in the mirrored preview:
                        # press Left Arrow to go to the previous page.
                        pyautogui.press("left")
                        status_text = "Previous Page"
                        status_until = current_time + 0.7
                        print("Wave Right -> Previous Page")
                    elif current_time >= status_until:
                        cooldown = swipe_detector.cooldown_remaining(
                            current_time
                        )
                        status_text = (
                            f"Cooldown {cooldown:.1f}s"
                            if cooldown > 0
                            else "Hand detected - Ready"
                        )
                else:
                    swipe_detector.mark_hand_missing(current_time)
                    if current_time >= status_until:
                        status_text = "Show one hand"

                draw_instructions(
                    frame,
                    status_text,
                    swipe_detector.threshold_pixels,
                )
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                try:
                    if cv2.getWindowProperty(
                        WINDOW_NAME, cv2.WND_PROP_VISIBLE
                    ) < 1:
                        break
                except cv2.error:
                    # The user may close the preview between waitKey and this
                    # check, in which case OpenCV no longer knows the window.
                    break
    except pyautogui.FailSafeException:
        print("PyAutoGUI fail-safe triggered. Exiting.")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
