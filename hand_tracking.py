import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)


WINDOW_NAME = "Reconnaissance des mains"
COLOR_WINDOW_NAME = "Couleur"
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detection de mains via webcam avec points rouges et lignes vertes."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Index de la camera a utiliser. Par defaut: 0.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Largeur de capture souhaitee.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Hauteur de capture souhaitee.",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=20,
        help="Nombre maximum de mains a detecter. Par defaut: 20.",
    )
    return parser.parse_args()


def noop(_value: int) -> None:
    pass


def create_color_controls() -> None:
    cv2.namedWindow(COLOR_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(COLOR_WINDOW_NAME, 320, 160)
    cv2.createTrackbar("R", COLOR_WINDOW_NAME, 255, 255, noop)
    cv2.createTrackbar("G", COLOR_WINDOW_NAME, 0, 255, noop)
    cv2.createTrackbar("B", COLOR_WINDOW_NAME, 0, 255, noop)
    cv2.createTrackbar("Clear", COLOR_WINDOW_NAME, 0, 1, noop)


def selected_color() -> tuple[int, int, int]:
    red = cv2.getTrackbarPos("R", COLOR_WINDOW_NAME)
    green = cv2.getTrackbarPos("G", COLOR_WINDOW_NAME)
    blue = cv2.getTrackbarPos("B", COLOR_WINDOW_NAME)
    return blue, green, red


def clear_requested() -> bool:
    return cv2.getTrackbarPos("Clear", COLOR_WINDOW_NAME) == 1


def reset_clear_request() -> None:
    cv2.setTrackbarPos("Clear", COLOR_WINDOW_NAME, 0)


def point(landmark, width: int, height: int) -> tuple[int, int]:
    return int(landmark.x * width), int(landmark.y * height)


def distance(first, second) -> float:
    return ((first.x - second.x) ** 2 + (first.y - second.y) ** 2) ** 0.5


def axis_projection(origin, target, axis_x: float, axis_y: float) -> float:
    return (target.x - origin.x) * axis_x + (target.y - origin.y) * axis_y


def finger_is_extended(hand_landmarks, tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
    wrist = hand_landmarks[0]
    middle_mcp = hand_landmarks[9]
    axis_x = middle_mcp.x - wrist.x
    axis_y = middle_mcp.y - wrist.y
    axis_length = (axis_x**2 + axis_y**2) ** 0.5
    if axis_length == 0:
        return False

    axis_x /= axis_length
    axis_y /= axis_length

    tip = axis_projection(wrist, hand_landmarks[tip_idx], axis_x, axis_y)
    pip = axis_projection(wrist, hand_landmarks[pip_idx], axis_x, axis_y)
    mcp = axis_projection(wrist, hand_landmarks[mcp_idx], axis_x, axis_y)
    return tip > pip + 0.025 and pip > mcp


def thumb_is_closed(hand_landmarks) -> bool:
    thumb_tip = hand_landmarks[4]
    index_mcp = hand_landmarks[5]
    middle_mcp = hand_landmarks[9]
    pinky_mcp = hand_landmarks[17]
    palm_width = distance(index_mcp, pinky_mcp)
    palm_center_x = (index_mcp.x + middle_mcp.x + pinky_mcp.x) / 3
    palm_center_y = (index_mcp.y + middle_mcp.y + pinky_mcp.y) / 3
    thumb_distance = ((thumb_tip.x - palm_center_x) ** 2 + (thumb_tip.y - palm_center_y) ** 2) ** 0.5
    return thumb_distance < palm_width * 0.75


def only_index_is_up(hand_landmarks) -> bool:
    index_up = finger_is_extended(hand_landmarks, 8, 6, 5)
    middle_up = finger_is_extended(hand_landmarks, 12, 10, 9)
    ring_up = finger_is_extended(hand_landmarks, 16, 14, 13)
    pinky_up = finger_is_extended(hand_landmarks, 20, 18, 17)
    return index_up and not middle_up and not ring_up and not pinky_up and thumb_is_closed(hand_landmarks)


def draw_hand(frame, hand_landmarks, connections) -> None:
    height, width, _ = frame.shape

    for connection in connections:
        start = hand_landmarks[connection.start]
        end = hand_landmarks[connection.end]
        cv2.line(frame, point(start, width, height), point(end, width, height), (0, 255, 0), 3)

    for landmark in hand_landmarks:
        center = point(landmark, width, height)
        cv2.circle(frame, center, 6, (0, 0, 255), -1)
        cv2.circle(frame, center, 8, (255, 255, 255), 1)


def draw_status(frame, hand_count: int, fps: float) -> None:
    cv2.rectangle(frame, (20, 20), (330, 115), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"FPS = {fps:.1f}",
        (35, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"nb_de_main = {hand_count}",
        (35, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_clear_feedback(frame) -> None:
    _, width, _ = frame.shape
    text = "canvas cleared"
    cv2.rectangle(frame, (width - 280, 20), (width - 20, 72), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (width - 262, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def clear_canvas(canvas) -> float:
    if canvas is not None:
        canvas[:] = 0
    return time.perf_counter() + 0.8


def main() -> int:
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(
            f"Impossible d'ouvrir la camera {args.camera}. "
            "Essaie avec --camera 1 si tu as plusieurs cameras.",
            file=sys.stderr,
        )
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    create_color_controls()

    connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=max(1, args.max_hands),
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.65,
        min_tracking_confidence=0.65,
    )

    frame_timestamp_ms = 0
    canvas = None
    last_draw_point = None
    last_time = time.perf_counter()
    clear_feedback_until = 0.0
    fps = 0.0

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Impossible de lire l'image de la camera.", file=sys.stderr)
                break

            frame = cv2.flip(frame, 1)
            if canvas is None or canvas.shape != frame.shape:
                canvas = np.zeros_like(frame)
                last_draw_point = None

            now = time.perf_counter()
            if clear_requested():
                clear_feedback_until = clear_canvas(canvas)
                reset_clear_request()
                last_draw_point = None

            delta = now - last_time
            last_time = now
            if delta > 0:
                current_fps = 1 / delta
                fps = current_fps if fps == 0 else (fps * 0.9) + (current_fps * 0.1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            frame_timestamp_ms += 1

            hand_count = 0
            draw_point = None
            if results.hand_landmarks:
                hand_count = len(results.hand_landmarks)
                for hand_landmarks in results.hand_landmarks:
                    draw_hand(frame, hand_landmarks, connections)
                    if draw_point is None and only_index_is_up(hand_landmarks):
                        draw_point = point(hand_landmarks[8], frame.shape[1], frame.shape[0])

            if draw_point is not None:
                if last_draw_point is not None:
                    cv2.line(canvas, last_draw_point, draw_point, selected_color(), 8)
                last_draw_point = draw_point
            else:
                last_draw_point = None

            frame = cv2.add(frame, canvas)
            draw_status(frame, hand_count, fps)
            if time.perf_counter() < clear_feedback_until:
                draw_clear_feedback(frame)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("c") and canvas is not None:
                clear_feedback_until = clear_canvas(canvas)
                last_draw_point = None

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
