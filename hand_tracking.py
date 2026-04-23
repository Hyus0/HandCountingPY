import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)


WINDOW_NAME = "Reconnaissance des mains"
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


def draw_hand(frame, hand_landmarks, connections) -> None:
    height, width, _ = frame.shape

    for connection in connections:
        start = hand_landmarks[connection.start]
        end = hand_landmarks[connection.end]
        start_point = (int(start.x * width), int(start.y * height))
        end_point = (int(end.x * width), int(end.y * height))
        cv2.line(frame, start_point, end_point, (0, 255, 0), 3)

    for landmark in hand_landmarks:
        center = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(frame, center, 6, (0, 0, 255), -1)
        cv2.circle(frame, center, 8, (255, 255, 255), 1)


def draw_counter(frame, hand_count: int) -> None:
    cv2.rectangle(frame, (20, 20), (310, 80), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"nb_de_main = {hand_count}",
        (35, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


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
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Impossible de lire l'image de la camera.", file=sys.stderr)
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            frame_timestamp_ms += 1

            hand_count = 0
            if results.hand_landmarks:
                hand_count = len(results.hand_landmarks)
                for hand_landmarks in results.hand_landmarks:
                    draw_hand(frame, hand_landmarks, connections)

            draw_counter(frame, hand_count)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
