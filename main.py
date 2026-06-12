"""Use hand-wave gestures to turn pages in a browser reader."""

import time

import cv2
import mediapipe as mp
import pyautogui


CAMERA_INDEX = 0
SWIPE_THRESHOLD_PIXELS = 80
COOLDOWN_SECONDS = 1.0
GESTURE_WINDOW_SECONDS = 0.75
WINDOW_NAME = "JDRead Gesture Page Turner"


def draw_instructions(frame) -> None:
    """Draw usage instructions on the camera preview."""
    lines = (
        "Wave Left = Next Page",
        "Wave Right = Previous Page",
        "Press Q to Quit",
    )

    for index, text in enumerate(lines):
        y = 35 + index * 35
        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    """Open the camera, detect horizontal wrist movement, and press page keys."""
    camera = cv2.VideoCapture(CAMERA_INDEX)
    previous_wrist_x = None
    gesture_start_time = None
    last_trigger_time = 0.0

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
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
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

                    if previous_wrist_x is not None:
                        movement_x = wrist_x - previous_wrist_x
                        cooldown_ready = (
                            current_time - last_trigger_time >= COOLDOWN_SECONDS
                        )

                        if not cooldown_ready:
                            # Reset the gesture origin throughout the cooldown
                            # so returning the hand does not cause a late trigger.
                            previous_wrist_x = wrist_x
                            gesture_start_time = current_time
                        elif (
                            gesture_start_time is not None
                            and current_time - gesture_start_time
                            > GESTURE_WINDOW_SECONDS
                        ):
                            # Slow hand drift is not a wave. Start a new short
                            # measurement window from the current position.
                            previous_wrist_x = wrist_x
                            gesture_start_time = current_time
                        elif movement_x <= -SWIPE_THRESHOLD_PIXELS:
                            # Physical left wave in the mirrored preview:
                            # press Right Arrow to go to the next page.
                            pyautogui.press("right")
                            last_trigger_time = current_time
                            previous_wrist_x = wrist_x
                            gesture_start_time = current_time
                            print("Wave Left -> Next Page")
                        elif movement_x >= SWIPE_THRESHOLD_PIXELS:
                            # Physical right wave in the mirrored preview:
                            # press Left Arrow to go to the previous page.
                            pyautogui.press("left")
                            last_trigger_time = current_time
                            previous_wrist_x = wrist_x
                            gesture_start_time = current_time
                            print("Wave Right -> Previous Page")

                    # Measure displacement over a short window so deliberate
                    # waves trigger while slow posture changes do not.
                    if previous_wrist_x is None:
                        previous_wrist_x = wrist_x
                        gesture_start_time = current_time
                else:
                    # Require a newly detected hand to establish a fresh start.
                    previous_wrist_x = None
                    gesture_start_time = None

                draw_instructions(frame)
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
