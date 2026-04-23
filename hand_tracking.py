import argparse
import math
import random
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
FX_STYLES = [
    {
        "name": "Inferno",
        "core": (255, 255, 180),
        "aura": (0, 150, 255),
        "ring": (0, 210, 255),
        "spark": (120, 255, 255),
        "impact": (0, 135, 255),
        "alt_impact": (80, 240, 255),
        "shape": "circle",
        "particles": "fire",
    },
    {
        "name": "Plasma",
        "core": (255, 180, 255),
        "aura": (255, 60, 190),
        "ring": (255, 90, 235),
        "spark": (255, 220, 255),
        "impact": (255, 40, 180),
        "alt_impact": (255, 210, 255),
        "shape": "diamond",
        "particles": "spark",
    },
    {
        "name": "Frost",
        "core": (255, 255, 255),
        "aura": (255, 210, 70),
        "ring": (255, 240, 120),
        "spark": (255, 255, 220),
        "impact": (255, 220, 70),
        "alt_impact": (255, 255, 255),
        "shape": "snow",
        "particles": "ice",
    },
    {
        "name": "Toxic",
        "core": (160, 255, 120),
        "aura": (70, 255, 80),
        "ring": (80, 255, 40),
        "spark": (210, 255, 80),
        "impact": (80, 255, 40),
        "alt_impact": (0, 190, 120),
        "shape": "triangle",
        "particles": "orb",
    },
]


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
    cv2.resizeWindow(COLOR_WINDOW_NAME, 360, 210)
    cv2.createTrackbar("R", COLOR_WINDOW_NAME, 255, 255, noop)
    cv2.createTrackbar("G", COLOR_WINDOW_NAME, 0, 255, noop)
    cv2.createTrackbar("B", COLOR_WINDOW_NAME, 0, 255, noop)
    cv2.createTrackbar("Clear", COLOR_WINDOW_NAME, 0, 1, noop)
    cv2.createTrackbar("Style", COLOR_WINDOW_NAME, 0, len(FX_STYLES) - 1, noop)


def selected_color() -> tuple[int, int, int]:
    red = cv2.getTrackbarPos("R", COLOR_WINDOW_NAME)
    green = cv2.getTrackbarPos("G", COLOR_WINDOW_NAME)
    blue = cv2.getTrackbarPos("B", COLOR_WINDOW_NAME)
    return blue, green, red


def selected_style() -> dict:
    index = cv2.getTrackbarPos("Style", COLOR_WINDOW_NAME)
    return FX_STYLES[max(0, min(index, len(FX_STYLES) - 1))]


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


def is_fist(hand_landmarks) -> bool:
    index_up = finger_is_extended(hand_landmarks, 8, 6, 5)
    middle_up = finger_is_extended(hand_landmarks, 12, 10, 9)
    ring_up = finger_is_extended(hand_landmarks, 16, 14, 13)
    pinky_up = finger_is_extended(hand_landmarks, 20, 18, 17)
    return not index_up and not middle_up and not ring_up and not pinky_up and thumb_is_closed(hand_landmarks)


def palm_center(hand_landmarks, width: int, height: int) -> tuple[int, int]:
    palm_points = [hand_landmarks[index] for index in (0, 5, 9, 13, 17)]
    x = sum(landmark.x for landmark in palm_points) / len(palm_points)
    y = sum(landmark.y for landmark in palm_points) / len(palm_points)
    return int(x * width), int(y * height)


def random_range(low: float, high: float) -> float:
    return low + random.random() * (high - low)


def spawn_charge_particle(center: tuple[int, int], charge: float, particles: list[dict], style: dict) -> None:
    angle = random.random() * math.tau
    distance_from_center = random_range(60, 160)
    speed = random_range(1.5, 4.0) * (charge / 100 + 0.3)
    particle_type = style["particles"]
    if particle_type == "fire":
        particle_type = "fire" if random.random() > 0.5 else "spark"
    particles.append(
        {
            "x": center[0] + math.cos(angle) * distance_from_center,
            "y": center[1] + math.sin(angle) * distance_from_center,
            "vx": -math.cos(angle) * speed,
            "vy": -math.sin(angle) * speed,
            "life": 1.0,
            "size": random_range(2, 6),
            "type": particle_type,
            "color": style["spark"],
        }
    )


def spawn_charge_ring(center: tuple[int, int], rings: list[dict], style: dict) -> None:
    rings.append(
        {
            "x": center[0],
            "y": center[1],
            "radius": random_range(35, 75),
            "max_radius": random_range(110, 230),
            "life": 1.0,
            "width": random_range(1, 3),
            "color": style["ring"],
        }
    )


def spawn_release_ring(center: tuple[int, int], charge: float, release_rings: list[dict], style: dict) -> None:
    release_rings.append(
        {
            "x": center[0],
            "y": center[1],
            "radius": 18,
            "speed": random_range(10, 18) * (charge / 100 + 0.35),
            "life": 1.0,
            "width": random_range(5, 10),
            "color": style["ring"],
        }
    )


def spawn_impact_line(center: tuple[int, int], charge: float, impact_lines: list[dict], style: dict) -> None:
    angle = random.random() * math.tau
    length = random_range(90, 260) * (charge / 100 + 0.25)
    impact_lines.append(
        {
            "x": center[0],
            "y": center[1],
            "angle": angle,
            "length": length,
            "life": 1.0,
            "width": random_range(2, 5),
            "color": style["impact"] if random.random() > 0.35 else style["alt_impact"],
        }
    )


def spawn_lightning(center: tuple[int, int], charge: float, lightning: list[dict], style: dict) -> None:
    angle = random.random() * math.tau
    length = random_range(40, 110) * (charge / 100 + 0.3)
    start_x = center[0] + math.cos(angle) * 45
    start_y = center[1] + math.sin(angle) * 45
    points = [(start_x, start_y)]
    steps = int(random_range(4, 9))
    dx = math.cos(angle) * length / steps
    dy = math.sin(angle) * length / steps
    x = start_x
    y = start_y

    for _ in range(steps):
        x += dx + random_range(-18, 18)
        y += dy + random_range(-18, 18)
        points.append((x, y))

    lightning.append(
        {
            "points": points,
            "life": 1.0,
            "color": style["spark"] if random.random() > 0.3 else style["core"],
        }
    )


def update_charge_effects(center: tuple[int, int], charge: float, particles: list[dict], rings: list[dict], lightning: list[dict], style: dict) -> None:
    if charge > 10 and random.random() < charge / 170:
        spawn_charge_particle(center, charge, particles, style)
    if charge > 20 and random.random() < charge / 330:
        spawn_charge_ring(center, rings, style)
    if charge > 30 and random.random() < charge / 260:
        spawn_lightning(center, charge, lightning, style)

    for particle in particles:
        particle["x"] += particle["vx"]
        particle["y"] += particle["vy"]
        particle["vx"] *= 0.95
        particle["vy"] *= 0.95
        particle["life"] -= 0.03
    particles[:] = [particle for particle in particles if particle["life"] > 0]

    for ring in rings:
        ring["radius"] += 3 * ring["life"]
        ring["life"] -= 0.025
    rings[:] = [ring for ring in rings if ring["life"] > 0 and ring["radius"] < ring["max_radius"]]

    for bolt in lightning:
        bolt["life"] -= 0.12
    lightning[:] = [bolt for bolt in lightning if bolt["life"] > 0]


def trigger_release_explosion(
    center: tuple[int, int],
    charge: float,
    release_particles: list[dict],
    release_rings: list[dict],
    impact_lines: list[dict],
    lightning: list[dict],
    style: dict,
) -> None:
    if charge < 8:
        return

    particle_count = int(18 + charge * 0.42)
    line_count = int(10 + charge * 0.16)
    ring_count = 2 + int(charge > 45) + int(charge > 75)

    for _ in range(particle_count):
        spawn_charge_particle(center, charge, release_particles, style)
        particle = release_particles[-1]
        particle["vx"] *= -1.7
        particle["vy"] *= -1.7
        particle["life"] = 1.15
        particle["size"] *= 1.35

    for _ in range(ring_count):
        spawn_release_ring(center, charge, release_rings, style)

    for _ in range(line_count):
        spawn_impact_line(center, charge, impact_lines, style)

    for _ in range(max(3, int(charge / 14))):
        spawn_lightning(center, charge, lightning, style)


def update_release_effects(
    release_particles: list[dict],
    release_rings: list[dict],
    impact_lines: list[dict],
    lightning: list[dict],
) -> None:
    for particle in release_particles:
        particle["x"] += particle["vx"]
        particle["y"] += particle["vy"]
        particle["vx"] *= 0.92
        particle["vy"] *= 0.92
        particle["life"] -= 0.025
    release_particles[:] = [particle for particle in release_particles if particle["life"] > 0]

    for ring in release_rings:
        ring["radius"] += ring["speed"]
        ring["speed"] *= 0.96
        ring["life"] -= 0.035
    release_rings[:] = [ring for ring in release_rings if ring["life"] > 0]

    for line in impact_lines:
        line["life"] -= 0.045
    impact_lines[:] = [line for line in impact_lines if line["life"] > 0]

    for bolt in lightning:
        bolt["life"] -= 0.1
    lightning[:] = [bolt for bolt in lightning if bolt["life"] > 0]


def scaled_color(color: tuple[int, int, int], scale: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(channel * scale))) for channel in color)


def draw_charge_aura(frame, center: tuple[int, int], charge: float, style: dict) -> None:
    glow = charge / 100
    if glow <= 0.02:
        return

    overlay = np.zeros_like(frame)
    max_radius = int(95 + glow * 110)
    for radius in range(max_radius, 8, -8):
        alpha = (1 - radius / max_radius) * 0.18 * glow
        color = scaled_color(style["aura"], 0.55 + 0.45 * glow)
        cv2.circle(overlay, center, radius, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1.0, 0, frame)
        overlay[:] = 0


def draw_charge_particles(frame, particles: list[dict]) -> None:
    for particle in particles:
        life = max(0.0, min(particle["life"], 1.0))
        x = int(particle["x"])
        y = int(particle["y"])
        color = scaled_color(particle["color"], 0.45 + 0.55 * life)
        if particle["type"] == "fire":
            size = max(1, int(particle["size"] * life))
            cv2.circle(frame, (x, y), size, color, -1, cv2.LINE_AA)
        elif particle["type"] == "ice":
            size = max(3, int(particle["size"] * life * 2))
            cv2.line(frame, (x - size, y), (x + size, y), color, 1, cv2.LINE_AA)
            cv2.line(frame, (x, y - size), (x, y + size), color, 1, cv2.LINE_AA)
            cv2.line(frame, (x - size, y - size), (x + size, y + size), color, 1, cv2.LINE_AA)
            cv2.line(frame, (x - size, y + size), (x + size, y - size), color, 1, cv2.LINE_AA)
        elif particle["type"] == "orb":
            size = max(1, int(particle["size"] * life * 1.3))
            cv2.circle(frame, (x, y), size, color, 1, cv2.LINE_AA)
        else:
            end = (int(x + particle["vx"] * 4), int(y + particle["vy"] * 4))
            cv2.line(frame, (x, y), end, color, max(1, int(particle["size"] * 0.4)), cv2.LINE_AA)


def draw_charge_rings(frame, rings: list[dict]) -> None:
    for ring in rings:
        life = max(0.0, min(ring["life"], 1.0))
        color = scaled_color(ring["color"], 0.45 + 0.55 * life)
        cv2.circle(
            frame,
            (int(ring["x"]), int(ring["y"])),
            int(ring["radius"]),
            color,
            max(1, int(ring["width"] * life * 2)),
            cv2.LINE_AA,
        )


def draw_release_rings(frame, release_rings: list[dict]) -> None:
    for ring in release_rings:
        life = max(0.0, min(ring["life"], 1.0))
        color = scaled_color(ring["color"], 0.5 + 0.5 * life)
        cv2.circle(
            frame,
            (int(ring["x"]), int(ring["y"])),
            int(ring["radius"]),
            color,
            max(1, int(ring["width"] * life)),
            cv2.LINE_AA,
        )


def draw_impact_lines(frame, impact_lines: list[dict]) -> None:
    for line in impact_lines:
        life = max(0.0, min(line["life"], 1.0))
        start_dist = line["length"] * (1 - life) * 0.4
        end_dist = start_dist + line["length"] * life
        start = (
            int(line["x"] + math.cos(line["angle"]) * start_dist),
            int(line["y"] + math.sin(line["angle"]) * start_dist),
        )
        end = (
            int(line["x"] + math.cos(line["angle"]) * end_dist),
            int(line["y"] + math.sin(line["angle"]) * end_dist),
        )
        color = tuple(int(channel * life) for channel in line["color"])
        cv2.line(frame, start, end, color, max(1, int(line["width"] * life)), cv2.LINE_AA)


def draw_lightning(frame, lightning: list[dict]) -> None:
    for bolt in lightning:
        points = np.array(bolt["points"], dtype=np.int32)
        color = tuple(int(channel * bolt["life"]) for channel in bolt["color"])
        cv2.polylines(frame, [points], False, color, 2, cv2.LINE_AA)


def draw_charge_bar(frame, center: tuple[int, int], charge: float) -> None:
    width = 190
    height = 12
    x = max(20, min(center[0] - width // 2, frame.shape[1] - width - 20))
    y = max(135, min(center[1] + 100, frame.shape[0] - 35))
    fill = int(width * charge / 100)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (25, 25, 25), -1)
    cv2.rectangle(frame, (x, y), (x + fill, y + height), (0, 170, 255), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 1)
    cv2.putText(
        frame,
        f"charge = {int(charge)}%",
        (x, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 240, 180),
        2,
        cv2.LINE_AA,
    )


def regular_polygon(center: tuple[int, int], radius: int, sides: int, rotation: float = 0.0) -> np.ndarray:
    return np.array(
        [
            (
                int(center[0] + math.cos(rotation + index * math.tau / sides) * radius),
                int(center[1] + math.sin(rotation + index * math.tau / sides) * radius),
            )
            for index in range(sides)
        ],
        dtype=np.int32,
    )


def draw_style_core(frame, center: tuple[int, int], charge: float, style: dict) -> None:
    pulse = 1 + math.sin(time.perf_counter() * 14) * 0.06 * (charge / 100)
    outer_radius = int((28 + charge * 0.45) * pulse)
    inner_radius = int((15 + charge * 0.25) * pulse)
    ring_color = style["ring"]
    core_color = style["core"]

    if style["shape"] == "diamond":
        outer = regular_polygon(center, outer_radius, 4, math.pi / 4)
        inner = regular_polygon(center, inner_radius, 4, math.pi / 4)
        cv2.polylines(frame, [outer], True, ring_color, 3, cv2.LINE_AA)
        cv2.fillPoly(frame, [inner], core_color, cv2.LINE_AA)
    elif style["shape"] == "snow":
        cv2.circle(frame, center, outer_radius, ring_color, 2, cv2.LINE_AA)
        for index in range(6):
            angle = index * math.tau / 6
            end = (
                int(center[0] + math.cos(angle) * outer_radius),
                int(center[1] + math.sin(angle) * outer_radius),
            )
            cv2.line(frame, center, end, ring_color, 2, cv2.LINE_AA)
        cv2.circle(frame, center, inner_radius, core_color, -1, cv2.LINE_AA)
    elif style["shape"] == "triangle":
        outer = regular_polygon(center, outer_radius, 3, -math.pi / 2)
        inner = regular_polygon(center, inner_radius, 3, -math.pi / 2)
        cv2.polylines(frame, [outer], True, ring_color, 3, cv2.LINE_AA)
        cv2.fillPoly(frame, [inner], core_color, cv2.LINE_AA)
    else:
        cv2.circle(frame, center, outer_radius, ring_color, 3, cv2.LINE_AA)
        cv2.circle(frame, center, inner_radius, core_color, -1, cv2.LINE_AA)


def draw_style_label(frame, style: dict) -> None:
    text = f"style = {style['name']}"
    cv2.rectangle(frame, (20, 122), (330, 162), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (35, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        style["ring"],
        2,
        cv2.LINE_AA,
    )


def draw_charge_effect(frame, center: tuple[int, int], charge: float, particles: list[dict], rings: list[dict], lightning: list[dict], style: dict) -> None:
    draw_charge_aura(frame, center, charge, style)
    draw_charge_rings(frame, rings)
    draw_lightning(frame, lightning)
    draw_charge_particles(frame, particles)
    draw_style_core(frame, center, charge, style)
    draw_charge_bar(frame, center, charge)


def draw_release_effect(
    frame,
    release_particles: list[dict],
    release_rings: list[dict],
    impact_lines: list[dict],
    lightning: list[dict],
) -> None:
    draw_release_rings(frame, release_rings)
    draw_impact_lines(frame, impact_lines)
    draw_lightning(frame, lightning)
    draw_charge_particles(frame, release_particles)


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
    charge_level = 0.0
    charge_particles = []
    charge_rings = []
    charge_lightning = []
    release_particles = []
    release_rings = []
    impact_lines = []
    release_lightning = []
    was_charging = False
    last_charge_center = None
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
            style = selected_style()

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
            fist_hand = None
            if results.hand_landmarks:
                hand_count = len(results.hand_landmarks)
                for hand_landmarks in results.hand_landmarks:
                    draw_hand(frame, hand_landmarks, connections)
                    if draw_point is None and only_index_is_up(hand_landmarks):
                        draw_point = point(hand_landmarks[8], frame.shape[1], frame.shape[0])
                    if fist_hand is None and is_fist(hand_landmarks):
                        fist_hand = hand_landmarks

            if draw_point is not None:
                if last_draw_point is not None:
                    cv2.line(canvas, last_draw_point, draw_point, selected_color(), 8)
                last_draw_point = draw_point
            else:
                last_draw_point = None

            frame = cv2.add(frame, canvas)
            if fist_hand is not None:
                charge_level = min(100.0, charge_level + delta * 38)
                charge_center = palm_center(fist_hand, frame.shape[1], frame.shape[0])
                last_charge_center = charge_center
                was_charging = True
                update_charge_effects(charge_center, charge_level, charge_particles, charge_rings, charge_lightning, style)
                draw_charge_effect(frame, charge_center, charge_level, charge_particles, charge_rings, charge_lightning, style)
            else:
                if was_charging and last_charge_center is not None:
                    trigger_release_explosion(
                        last_charge_center,
                        charge_level,
                        release_particles,
                        release_rings,
                        impact_lines,
                        release_lightning,
                        style,
                    )
                was_charging = False
                charge_level = max(0.0, charge_level - delta * 80)
                charge_particles.clear()
                charge_rings.clear()
                charge_lightning.clear()

            update_release_effects(release_particles, release_rings, impact_lines, release_lightning)
            draw_release_effect(frame, release_particles, release_rings, impact_lines, release_lightning)
            draw_status(frame, hand_count, fps)
            draw_style_label(frame, style)
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
