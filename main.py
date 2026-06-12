"""Use hand-wave gestures to turn pages in a browser reader."""

from collections import deque
from statistics import median
import time

import cv2
import mediapipe as mp
import pyautogui


CAMERA_INDEX = 0
SWIPE_THRESHOLD_RATIO = 0.08
MIN_SWIPE_THRESHOLD_PIXELS = 45
MAX_SWIPE_THRESHOLD_PIXELS = 80
COOLDOWN_SECONDS = 1.0
GESTURE_WINDOW_SECONDS = 1.0
HAND_LOSS_GRACE_SECONDS = 0.25
MIN_TRACKED_SAMPLES = 4
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
    ):
        self.threshold_pixels = threshold_pixels
        self.cooldown_seconds = cooldown_seconds
        self.history_seconds = history_seconds
        self.hand_loss_grace_seconds = hand_loss_grace_seconds
        self.min_samples = min_samples
        self.samples = deque()
        self.last_seen_time = None
        self.last_trigger_time = float("-inf")

    def update(self, wrist_x, current_time):
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
            self.samples.append((current_time, wrist_x))
            return None

        self.samples.append((current_time, wrist_x))
        if len(self.samples) < self.min_samples:
            return None

        smoothing_count = min(2, len(self.samples))
        origin_x = median(
            sample_x for _, sample_x in list(self.samples)[:smoothing_count]
        )
        current_x = median(
            sample_x for _, sample_x in list(self.samples)[-smoothing_count:]
        )
        movement_x = current_x - origin_x

        if movement_x <= -self.threshold_pixels:
            key = "right"
        elif movement_x >= self.threshold_pixels:
            key = "left"
        else:
            return None

        self.last_trigger_time = current_time
        self.samples.clear()
        self.samples.append((current_time, wrist_x))
        return key

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
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
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
                    wrist_x = int(
                        hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                        * frame_width
                    )

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                    )

                    key = swipe_detector.update(wrist_x, current_time)
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
