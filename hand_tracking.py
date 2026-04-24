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
MODE_DRAW = 1
MODE_CUBE = 2
MODE_POWER = 3
MODE_NERD = 4
MODE_SWORD = 5
GAME_MODES = {
    MODE_DRAW: "Dessin",
    MODE_CUBE: "Cube",
    MODE_POWER: "Pouvoir",
    MODE_NERD: "Nerd",
    MODE_SWORD: "Epee",
}
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
        "theme": "inferno",
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
        "theme": "plasma",
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
        "theme": "frost",
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
        "particles": "toxic_cloud",
        "theme": "toxic",
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
    cv2.resizeWindow(COLOR_WINDOW_NAME, 390, 250)
    cv2.createTrackbar("Mode", COLOR_WINDOW_NAME, MODE_DRAW, MODE_SWORD, noop)
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


def selected_game_mode() -> int:
    mode = cv2.getTrackbarPos("Mode", COLOR_WINDOW_NAME)
    if mode not in GAME_MODES:
        cv2.setTrackbarPos("Mode", COLOR_WINDOW_NAME, MODE_DRAW)
        return MODE_DRAW
    return mode


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


def pinch_ratio(hand_landmarks) -> float:
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    index_mcp = hand_landmarks[5]
    pinky_mcp = hand_landmarks[17]
    palm_width = max(distance(index_mcp, pinky_mcp), 0.001)
    return distance(thumb_tip, index_tip) / palm_width


def cube_size_from_pinch(hand_landmarks) -> int:
    ratio = pinch_ratio(hand_landmarks)
    size = 45 + ratio * 170
    return int(max(35, min(size, 260)))


def draw_cube(frame, center: tuple[int, int], size: int) -> None:
    x, y = center
    half = size // 2
    depth = max(18, int(size * 0.35))

    front = np.array(
        [
            [x - half, y - half],
            [x + half, y - half],
            [x + half, y + half],
            [x - half, y + half],
        ],
        dtype=np.int32,
    )
    back = front + np.array([depth, -depth], dtype=np.int32)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [front], (60, 180, 255))
    cv2.fillPoly(overlay, [back], (40, 120, 220))
    side = np.array([front[1], back[1], back[2], front[2]], dtype=np.int32)
    top = np.array([front[0], back[0], back[1], front[1]], dtype=np.int32)
    cv2.fillPoly(overlay, [side], (35, 95, 200))
    cv2.fillPoly(overlay, [top], (90, 210, 255))
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    for square in (front, back):
        cv2.polylines(frame, [square], True, (0, 255, 255), 3, cv2.LINE_AA)
    for index in range(4):
        cv2.line(frame, tuple(front[index]), tuple(back[index]), (0, 255, 255), 3, cv2.LINE_AA)


def hand_axis(hand_landmarks, width: int, height: int) -> tuple[tuple[float, float], tuple[int, int], float]:
    wrist = point(hand_landmarks[0], width, height)
    middle_mcp = point(hand_landmarks[9], width, height)
    dx = middle_mcp[0] - wrist[0]
    dy = middle_mcp[1] - wrist[1]
    length = math.hypot(dx, dy)
    if length < 1:
        return (0.0, -1.0), wrist, 1.0
    return (dx / length, dy / length), wrist, length


def draw_sword(frame, hand_landmarks) -> None:
    height, width, _ = frame.shape
    center = palm_center(hand_landmarks, width, height)
    (dir_x, dir_y), wrist, axis_length = hand_axis(hand_landmarks, width, height)
    perp_x = -dir_y
    perp_y = dir_x
    palm_width = max(
        math.hypot(
            point(hand_landmarks[5], width, height)[0] - point(hand_landmarks[17], width, height)[0],
            point(hand_landmarks[5], width, height)[1] - point(hand_landmarks[17], width, height)[1],
        ),
        axis_length * 0.9,
    )

    scale = max(34, palm_width * 0.55)
    guard_center = (
        int(center[0] + dir_x * scale * 0.3),
        int(center[1] + dir_y * scale * 0.3),
    )
    pommel_center = (
        int(center[0] - dir_x * scale * 1.2),
        int(center[1] - dir_y * scale * 1.2),
    )
    blade_base = (
        int(guard_center[0] + dir_x * scale * 0.36),
        int(guard_center[1] + dir_y * scale * 0.36),
    )
    tip = (
        int(blade_base[0] + dir_x * scale * 4.3),
        int(blade_base[1] + dir_y * scale * 4.3),
    )

    blade_half = scale * 0.22
    blade_depth = scale * 0.14
    grip_half = scale * 0.16
    grip_length = scale * 1.15
    guard_half = scale * 0.7
    guard_depth = scale * 0.2

    def p(origin, along: float = 0.0, side: float = 0.0, depth: float = 0.0) -> tuple[int, int]:
        return (
            int(origin[0] + dir_x * along + perp_x * side + (perp_x + dir_x * 0.35) * depth),
            int(origin[1] + dir_y * along + perp_y * side + (perp_y + dir_y * 0.35) * depth),
        )

    blade_front = np.array(
        [
            p(blade_base, side=-blade_half),
            p(blade_base, side=blade_half),
            p(tip),
        ],
        dtype=np.int32,
    )
    blade_side = np.array(
        [
            p(blade_base, side=blade_half),
            p(blade_base, side=blade_half, depth=blade_depth),
            p(tip, depth=blade_depth * 0.5),
            p(tip),
        ],
        dtype=np.int32,
    )
    guard_front = np.array(
        [
            p(guard_center, side=-guard_half, depth=-guard_depth * 0.4),
            p(guard_center, side=guard_half, depth=-guard_depth * 0.4),
            p(guard_center, along=scale * 0.18, side=guard_half, depth=guard_depth * 0.35),
            p(guard_center, along=scale * 0.18, side=-guard_half, depth=guard_depth * 0.35),
        ],
        dtype=np.int32,
    )
    grip_front = np.array(
        [
            p(center, along=-grip_length, side=-grip_half),
            p(center, along=-grip_length, side=grip_half),
            p(center, along=scale * 0.15, side=grip_half * 0.92),
            p(center, along=scale * 0.15, side=-grip_half * 0.92),
        ],
        dtype=np.int32,
    )

    overlay = frame.copy()
    cv2.fillPoly(overlay, [blade_front], (215, 225, 235), cv2.LINE_AA)
    cv2.fillPoly(overlay, [blade_side], (120, 145, 165), cv2.LINE_AA)
    cv2.fillPoly(overlay, [guard_front], (40, 190, 255), cv2.LINE_AA)
    cv2.fillPoly(overlay, [grip_front], (55, 40, 28), cv2.LINE_AA)
    cv2.circle(overlay, pommel_center, max(10, int(scale * 0.22)), (70, 210, 255), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    cv2.polylines(frame, [blade_front], True, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.polylines(frame, [blade_side], True, (150, 175, 195), 2, cv2.LINE_AA)
    cv2.polylines(frame, [guard_front], True, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.polylines(frame, [grip_front], True, (170, 130, 90), 2, cv2.LINE_AA)
    cv2.circle(frame, pommel_center, max(10, int(scale * 0.22)), (255, 255, 255), 2, cv2.LINE_AA)

    fuller_start = p(blade_base, along=scale * 0.18)
    fuller_end = p(tip, along=-scale * 0.62)
    cv2.line(frame, fuller_start, fuller_end, (245, 250, 255), 2, cv2.LINE_AA)

    for index in range(4):
        ratio = index / 3 if index else 0
        wrap_center = p(center, along=-grip_length * (0.2 + ratio * 0.62))
        left = p(wrap_center, side=-grip_half * 0.82)
        right = p(wrap_center, side=grip_half * 0.82)
        cv2.line(frame, left, right, (110, 85, 60), 2, cv2.LINE_AA)

    glow = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.line(glow, blade_base, tip, 255, max(10, int(scale * 0.34)), cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), max(8, int(scale * 0.16)))
    blend_mask(frame, (120, 235, 255), glow, 0.18)


def random_range(low: float, high: float) -> float:
    return low + random.random() * (high - low)


def spawn_charge_particle(center: tuple[int, int], charge: float, particles: list[dict], style: dict) -> None:
    angle = random.random() * math.tau
    theme = style["theme"]
    distance_from_center = random_range(60, 160)
    speed = random_range(1.5, 4.0) * (charge / 100 + 0.3)
    particle_type = style["particles"]
    if particle_type == "fire":
        particle_type = "fire" if random.random() > 0.5 else "spark"
    if theme == "toxic":
        distance_from_center = random_range(25, 105)
        speed = random_range(0.25, 1.25) * (charge / 100 + 0.4)
    elif theme == "frost":
        distance_from_center = random_range(70, 190)
        speed = random_range(0.9, 2.2) * (charge / 100 + 0.45)
    elif theme == "plasma":
        speed = random_range(3.0, 6.8) * (charge / 100 + 0.45)

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
            "angle": angle,
            "spin": random_range(-0.12, 0.12),
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
            "theme": style["theme"],
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
            "theme": style["theme"],
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


def blend_mask(frame, color: tuple[int, int, int], mask, opacity: float) -> None:
    alpha = (mask.astype(np.float32) / 255.0) * opacity
    frame[:] = (
        frame.astype(np.float32) * (1 - alpha[..., None])
        + np.array(color, dtype=np.float32) * alpha[..., None]
    ).astype(np.uint8)


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
            size = max(4, int(particle["size"] * life * 2.5))
            flame = np.array(
                [
                    (x, y - size * 2),
                    (x + size, y + size),
                    (x, y + size * 2),
                    (x - size, y + size),
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(frame, [flame], color, cv2.LINE_AA)
            cv2.circle(frame, (x, y), max(1, size // 2), (0, 240, 255), -1, cv2.LINE_AA)
        elif particle["type"] == "ice":
            size = max(8, int(particle["size"] * life * 4))
            angle = particle["angle"] + particle["spin"] * (1 - life) * 20
            tip = (int(x + math.cos(angle) * size), int(y + math.sin(angle) * size))
            left = (int(x + math.cos(angle + 2.55) * size * 0.35), int(y + math.sin(angle + 2.55) * size * 0.35))
            right = (int(x + math.cos(angle - 2.55) * size * 0.35), int(y + math.sin(angle - 2.55) * size * 0.35))
            cv2.fillPoly(frame, [np.array([tip, left, right], dtype=np.int32)], color, cv2.LINE_AA)
            cv2.line(frame, (x, y), tip, (255, 255, 255), 1, cv2.LINE_AA)
        elif particle["type"] == "toxic_cloud":
            radius = max(8, int(particle["size"] * (1.7 - life) * 5))
            overlay = frame.copy()
            cv2.circle(overlay, (x, y), radius, color, -1, cv2.LINE_AA)
            cv2.circle(overlay, (x + radius // 2, y - radius // 3), max(3, radius // 2), color, -1, cv2.LINE_AA)
            cv2.circle(overlay, (x - radius // 2, y + radius // 4), max(3, radius // 2), color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.28 * life, frame, 1 - 0.28 * life, 0, frame)
        else:
            end = (int(x + particle["vx"] * 4), int(y + particle["vy"] * 4))
            mid = (int((x + end[0]) / 2 + random_range(-5, 5)), int((y + end[1]) / 2 + random_range(-5, 5)))
            cv2.polylines(frame, [np.array([(x, y), mid, end], dtype=np.int32)], False, color, max(1, int(particle["size"] * 0.5)), cv2.LINE_AA)


def draw_charge_rings(frame, rings: list[dict]) -> None:
    for ring in rings:
        life = max(0.0, min(ring["life"], 1.0))
        color = scaled_color(ring["color"], 0.45 + 0.55 * life)
        center = (int(ring["x"]), int(ring["y"]))
        radius = int(ring["radius"])
        thickness = max(1, int(ring["width"] * life * 2))
        if ring.get("theme") == "plasma":
            points = regular_polygon(center, radius, 4, math.pi / 4 + time.perf_counter())
            cv2.polylines(frame, [points], True, color, thickness, cv2.LINE_AA)
        elif ring.get("theme") == "frost":
            points = regular_polygon(center, radius, 8, math.pi / 8)
            cv2.polylines(frame, [points], True, color, thickness, cv2.LINE_AA)
        elif ring.get("theme") == "toxic":
            for index in range(10):
                angle = index * math.tau / 10
                bubble = (
                    int(center[0] + math.cos(angle) * radius),
                    int(center[1] + math.sin(angle) * radius),
                )
                cv2.circle(frame, bubble, max(2, thickness * 2), color, 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, center, radius, color, thickness, cv2.LINE_AA)


def draw_release_rings(frame, release_rings: list[dict]) -> None:
    for ring in release_rings:
        life = max(0.0, min(ring["life"], 1.0))
        color = scaled_color(ring["color"], 0.5 + 0.5 * life)
        center = (int(ring["x"]), int(ring["y"]))
        radius = int(ring["radius"])
        thickness = max(1, int(ring["width"] * life))
        if ring.get("theme") == "frost":
            cv2.polylines(frame, [regular_polygon(center, radius, 8, math.pi / 8)], True, color, thickness, cv2.LINE_AA)
        elif ring.get("theme") == "toxic":
            cv2.circle(frame, center, radius, color, max(1, thickness // 2), cv2.LINE_AA)
            cv2.circle(frame, center, int(radius * 0.72), color, 1, cv2.LINE_AA)
        elif ring.get("theme") == "plasma":
            cv2.polylines(frame, [regular_polygon(center, radius, 4, math.pi / 4)], True, color, thickness, cv2.LINE_AA)
        else:
            cv2.circle(frame, center, radius, color, thickness, cv2.LINE_AA)


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


def draw_inferno_flames(frame, center: tuple[int, int], charge: float, style: dict) -> None:
    flame_count = 8
    radius = int(35 + charge * 0.65)
    for index in range(flame_count):
        angle = index * math.tau / flame_count + time.perf_counter() * 0.8
        base = (
            int(center[0] + math.cos(angle) * radius * 0.45),
            int(center[1] + math.sin(angle) * radius * 0.45),
        )
        tip = (
            int(center[0] + math.cos(angle) * radius),
            int(center[1] + math.sin(angle) * radius),
        )
        left = (
            int(base[0] + math.cos(angle + math.pi / 2) * 12),
            int(base[1] + math.sin(angle + math.pi / 2) * 12),
        )
        right = (
            int(base[0] + math.cos(angle - math.pi / 2) * 12),
            int(base[1] + math.sin(angle - math.pi / 2) * 12),
        )
        color = style["impact"] if index % 2 else style["ring"]
        cv2.fillPoly(frame, [np.array([tip, left, right], dtype=np.int32)], color, cv2.LINE_AA)


def draw_plasma_arcs(frame, center: tuple[int, int], charge: float, style: dict) -> None:
    arc_count = 5 + int(charge / 25)
    radius = 45 + charge * 0.75
    for index in range(arc_count):
        angle = index * math.tau / arc_count + time.perf_counter() * 1.6
        points = []
        for step in range(5):
            dist = radius * (0.35 + step * 0.16)
            jitter = random_range(-12, 12)
            points.append(
                (
                    int(center[0] + math.cos(angle) * dist + math.cos(angle + math.pi / 2) * jitter),
                    int(center[1] + math.sin(angle) * dist + math.sin(angle + math.pi / 2) * jitter),
                )
            )
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, style["spark"], 2, cv2.LINE_AA)


def draw_frost_spikes(frame, center: tuple[int, int], charge: float, style: dict) -> None:
    spike_count = 10
    inner = 24 + charge * 0.2
    outer = 65 + charge * 0.9
    for index in range(spike_count):
        angle = index * math.tau / spike_count
        tip = (
            int(center[0] + math.cos(angle) * outer),
            int(center[1] + math.sin(angle) * outer),
        )
        left = (
            int(center[0] + math.cos(angle + 0.12) * inner),
            int(center[1] + math.sin(angle + 0.12) * inner),
        )
        right = (
            int(center[0] + math.cos(angle - 0.12) * inner),
            int(center[1] + math.sin(angle - 0.12) * inner),
        )
        cv2.fillPoly(frame, [np.array([tip, left, right], dtype=np.int32)], style["aura"], cv2.LINE_AA)
        cv2.polylines(frame, [np.array([tip, left, right], dtype=np.int32)], True, style["core"], 1, cv2.LINE_AA)


def draw_toxic_cloud(frame, center: tuple[int, int], charge: float, style: dict) -> None:
    overlay = frame.copy()
    cloud_count = 12
    base_radius = 35 + charge * 0.75
    for index in range(cloud_count):
        angle = index * math.tau / cloud_count + time.perf_counter() * 0.35
        dist = base_radius * random_range(0.25, 0.9)
        radius = int(random_range(18, 42) * (0.55 + charge / 150))
        pos = (
            int(center[0] + math.cos(angle) * dist + random_range(-8, 8)),
            int(center[1] + math.sin(angle) * dist + random_range(-8, 8)),
        )
        cv2.circle(overlay, pos, radius, style["aura"], -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.26 + charge / 500, frame, 0.74 - charge / 500, 0, frame)
    for index in range(5):
        bubble_angle = index * math.tau / 5 + time.perf_counter()
        bubble_pos = (
            int(center[0] + math.cos(bubble_angle) * base_radius * 0.55),
            int(center[1] + math.sin(bubble_angle) * base_radius * 0.55),
        )
        cv2.circle(frame, bubble_pos, 6 + index % 3, style["spark"], 1, cv2.LINE_AA)


def draw_theme_overlay(frame, center: tuple[int, int], charge: float, style: dict) -> None:
    if style["theme"] == "inferno":
        draw_inferno_flames(frame, center, charge, style)
    elif style["theme"] == "plasma":
        draw_plasma_arcs(frame, center, charge, style)
    elif style["theme"] == "frost":
        draw_frost_spikes(frame, center, charge, style)
    elif style["theme"] == "toxic":
        draw_toxic_cloud(frame, center, charge, style)


def draw_power_moon(frame, center: tuple[int, int], size: int, style: dict) -> None:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cut_center = (center[0] + int(size * 0.38), center[1] - int(size * 0.08))
    cv2.circle(mask, center, size, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, cut_center, int(size * 0.92), 0, -1, cv2.LINE_AA)

    glow = cv2.GaussianBlur(mask, (0, 0), max(12, size // 3))
    blend_mask(frame, style["aura"], glow, 0.65)
    blend_mask(frame, style["ring"], mask, 0.95)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, style["core"], 2, cv2.LINE_AA)

    if style["theme"] == "frost":
        for index in range(6):
            angle = index * math.tau / 6 + time.perf_counter() * 0.25
            tip = (
                int(center[0] + math.cos(angle) * size * 1.45),
                int(center[1] + math.sin(angle) * size * 1.45),
            )
            base = (
                int(center[0] + math.cos(angle) * size * 1.05),
                int(center[1] + math.sin(angle) * size * 1.05),
            )
            cv2.line(frame, base, tip, style["core"], 2, cv2.LINE_AA)
    elif style["theme"] == "toxic":
        for index in range(7):
            angle = index * math.tau / 7 + time.perf_counter() * 0.55
            bubble = (
                int(center[0] + math.cos(angle) * size * 1.18),
                int(center[1] + math.sin(angle) * size * 0.95),
            )
            cv2.circle(frame, bubble, 5 + index % 3, style["spark"], 1, cv2.LINE_AA)
    elif style["theme"] == "plasma":
        for index in range(4):
            angle = index * math.tau / 4 + time.perf_counter() * 1.8
            start = (
                int(center[0] + math.cos(angle) * size * 0.75),
                int(center[1] + math.sin(angle) * size * 0.75),
            )
            end = (
                int(center[0] + math.cos(angle) * size * 1.35),
                int(center[1] + math.sin(angle) * size * 1.35),
            )
            cv2.line(frame, start, end, style["spark"], 2, cv2.LINE_AA)
    else:
        for index in range(5):
            angle = index * math.tau / 5 + time.perf_counter() * 0.7
            flame = np.array(
                [
                    (
                        int(center[0] + math.cos(angle) * size * 1.35),
                        int(center[1] + math.sin(angle) * size * 1.35),
                    ),
                    (
                        int(center[0] + math.cos(angle + 0.14) * size),
                        int(center[1] + math.sin(angle + 0.14) * size),
                    ),
                    (
                        int(center[0] + math.cos(angle - 0.14) * size),
                        int(center[1] + math.sin(angle - 0.14) * size),
                    ),
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(frame, [flame], style["impact"], cv2.LINE_AA)


def largest_face_box(face_detector, rgb_frame) -> tuple[int, int, int, int] | None:
    if face_detector.empty():
        return None

    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(90, 90))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda face: face[2] * face[3])


def draw_nerd_face(frame, box: tuple[int, int, int, int]) -> None:
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return

    frame_color = (35, 20, 10)
    lens_color = (55, 185, 230)
    shine_color = (235, 245, 255)
    tooth_color = (245, 245, 230)
    mouth_color = (10, 10, 10)
    acne_color = (40, 70, 210)
    thickness = max(4, int(w * 0.035))

    eye_y = y + int(h * 0.38)
    lens_w = int(w * 0.34)
    lens_h = int(h * 0.19)
    gap = int(w * 0.04)
    left_x = x + int(w * 0.14)
    right_x = left_x + lens_w + gap
    top_y = eye_y - lens_h // 2
    bottom_y = eye_y + lens_h // 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (left_x, top_y), (left_x + lens_w, bottom_y), lens_color, -1, cv2.LINE_AA)
    cv2.rectangle(overlay, (right_x, top_y), (right_x + lens_w, bottom_y), lens_color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.26, frame, 0.74, 0, frame)

    cv2.rectangle(frame, (left_x, top_y), (left_x + lens_w, bottom_y), frame_color, thickness, cv2.LINE_AA)
    cv2.rectangle(frame, (right_x, top_y), (right_x + lens_w, bottom_y), frame_color, thickness, cv2.LINE_AA)
    cv2.line(frame, (left_x + lens_w, eye_y), (right_x, eye_y), frame_color, thickness, cv2.LINE_AA)
    cv2.line(frame, (left_x, top_y + thickness), (max(0, x - int(w * 0.08)), top_y), frame_color, thickness, cv2.LINE_AA)
    cv2.line(frame, (right_x + lens_w, top_y + thickness), (min(frame.shape[1] - 1, x + w + int(w * 0.08)), top_y), frame_color, thickness, cv2.LINE_AA)
    cv2.circle(frame, (left_x + int(lens_w * 0.25), top_y + int(lens_h * 0.25)), max(4, thickness), shine_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (right_x + int(lens_w * 0.25), top_y + int(lens_h * 0.25)), max(4, thickness), shine_color, -1, cv2.LINE_AA)

    mouth_w = int(w * 0.28)
    mouth_h = int(h * 0.08)
    mouth_x = x + w // 2 - mouth_w // 2
    mouth_y = y + int(h * 0.66)
    cv2.rectangle(frame, (mouth_x, mouth_y), (mouth_x + mouth_w, mouth_y + mouth_h), mouth_color, -1, cv2.LINE_AA)

    tooth_w = int(w * 0.22)
    tooth_h = int(h * 0.16)
    tooth_x = x + w // 2 - tooth_w // 2
    tooth_y = mouth_y + int(mouth_h * 0.25)
    cv2.rectangle(frame, (tooth_x, tooth_y), (tooth_x + tooth_w, tooth_y + tooth_h), tooth_color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (tooth_x, tooth_y), (tooth_x + tooth_w, tooth_y + tooth_h), (210, 210, 190), 2, cv2.LINE_AA)
    cv2.line(frame, (tooth_x + tooth_w // 2, tooth_y), (tooth_x + tooth_w // 2, tooth_y + tooth_h), (210, 210, 190), 1, cv2.LINE_AA)

    spots = [
        (x + int(w * 0.21), y + int(h * 0.58)),
        (x + int(w * 0.27), y + int(h * 0.65)),
        (x + int(w * 0.76), y + int(h * 0.59)),
        (x + int(w * 0.83), y + int(h * 0.66)),
    ]
    for spot in spots:
        cv2.circle(frame, spot, max(2, int(w * 0.014)), acne_color, -1, cv2.LINE_AA)


def draw_idea_lamp(frame, index_point: tuple[int, int]) -> None:
    x, y = index_point
    bulb_center = (x, max(45, y - 105))
    glow = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(glow, bulb_center, 58, 255, -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), 18)
    blend_mask(frame, (0, 230, 255), glow, 0.34)

    cv2.circle(frame, bulb_center, 29, (0, 245, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, bulb_center, 29, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (bulb_center[0] - 11, bulb_center[1] + 24), (bulb_center[0] + 11, bulb_center[1] + 42), (95, 95, 95), -1, cv2.LINE_AA)
    cv2.line(frame, (bulb_center[0] - 15, bulb_center[1] + 30), (bulb_center[0] + 15, bulb_center[1] + 30), (220, 220, 220), 2, cv2.LINE_AA)
    cv2.line(frame, (bulb_center[0] - 13, bulb_center[1] + 37), (bulb_center[0] + 13, bulb_center[1] + 37), (220, 220, 220), 2, cv2.LINE_AA)

    for index in range(10):
        angle = index * math.tau / 10
        start = (
            int(bulb_center[0] + math.cos(angle) * 42),
            int(bulb_center[1] + math.sin(angle) * 42),
        )
        end = (
            int(bulb_center[0] + math.cos(angle) * 62),
            int(bulb_center[1] + math.sin(angle) * 62),
        )
        cv2.line(frame, start, end, (0, 245, 255), 2, cv2.LINE_AA)

    cv2.putText(
        frame,
        "Eureka idee!",
        (max(10, bulb_center[0] - 106), max(34, bulb_center[1] - 54)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.74,
        (0, 245, 255),
        3,
        cv2.LINE_AA,
    )


def draw_nerd_effect(frame, face_box, index_point: tuple[int, int]) -> None:
    if face_box is not None:
        draw_nerd_face(frame, face_box)
    draw_idea_lamp(frame, index_point)


def draw_style_label(frame, style: dict) -> None:
    text = f"style = {style['name']}"
    cv2.rectangle(frame, (20, 166), (330, 206), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (35, 194),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        style["ring"],
        2,
        cv2.LINE_AA,
    )


def draw_mode_label(frame, mode: int) -> None:
    text = f"mode = {mode} {GAME_MODES[mode]}"
    cv2.rectangle(frame, (20, 122), (330, 162), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (35, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_charge_effect(frame, center: tuple[int, int], charge: float, particles: list[dict], rings: list[dict], lightning: list[dict], style: dict) -> None:
    draw_charge_aura(frame, center, charge, style)
    draw_theme_overlay(frame, center, charge, style)
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
    style: dict,
) -> None:
    if release_rings:
        center = (int(release_rings[-1]["x"]), int(release_rings[-1]["y"]))
        draw_theme_overlay(frame, center, 70, style)
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
    face_detector = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
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
    cube_size = 120
    current_mode = MODE_DRAW
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
            mode = selected_game_mode()
            if mode != current_mode:
                last_draw_point = None
                was_charging = False
                charge_level = 0.0
                charge_particles.clear()
                charge_rings.clear()
                charge_lightning.clear()
                release_particles.clear()
                release_rings.clear()
                impact_lines.clear()
                release_lightning.clear()
                current_mode = mode

            if mode == MODE_DRAW and clear_requested():
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
            moon_point = None
            nerd_point = None
            fist_hand = None
            control_hand = None
            if results.hand_landmarks:
                hand_count = len(results.hand_landmarks)
                for hand_landmarks in results.hand_landmarks:
                    draw_hand(frame, hand_landmarks, connections)
                    if mode == MODE_DRAW and draw_point is None and only_index_is_up(hand_landmarks):
                        draw_point = point(hand_landmarks[8], frame.shape[1], frame.shape[0])
                    if mode == MODE_POWER and moon_point is None and only_index_is_up(hand_landmarks):
                        moon_point = point(hand_landmarks[8], frame.shape[1], frame.shape[0])
                    if mode == MODE_NERD and nerd_point is None and only_index_is_up(hand_landmarks):
                        nerd_point = point(hand_landmarks[8], frame.shape[1], frame.shape[0])
                    if mode in (MODE_CUBE, MODE_POWER, MODE_SWORD) and fist_hand is None and is_fist(hand_landmarks):
                        fist_hand = hand_landmarks

                if mode == MODE_CUBE and fist_hand is not None:
                    for hand_landmarks in results.hand_landmarks:
                        if hand_landmarks is not fist_hand:
                            control_hand = hand_landmarks
                            break

            if mode == MODE_DRAW and draw_point is not None:
                if last_draw_point is not None:
                    cv2.line(canvas, last_draw_point, draw_point, selected_color(), 8)
                last_draw_point = draw_point
            else:
                last_draw_point = None

            if mode == MODE_DRAW:
                frame = cv2.add(frame, canvas)
            if mode == MODE_CUBE and fist_hand is not None:
                if control_hand is not None:
                    target_size = cube_size_from_pinch(control_hand)
                    cube_size = int(cube_size * 0.8 + target_size * 0.2)
                cube_center = palm_center(fist_hand, frame.shape[1], frame.shape[0])
                draw_cube(frame, cube_center, cube_size)
            elif mode == MODE_SWORD and fist_hand is not None:
                draw_sword(frame, fist_hand)
            elif mode == MODE_POWER and fist_hand is not None:
                charge_level = min(100.0, charge_level + delta * 38)
                charge_center = palm_center(fist_hand, frame.shape[1], frame.shape[0])
                last_charge_center = charge_center
                was_charging = True
                update_charge_effects(charge_center, charge_level, charge_particles, charge_rings, charge_lightning, style)
                draw_charge_effect(frame, charge_center, charge_level, charge_particles, charge_rings, charge_lightning, style)
            else:
                if mode == MODE_POWER and was_charging and last_charge_center is not None:
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

            if mode == MODE_POWER:
                update_release_effects(release_particles, release_rings, impact_lines, release_lightning)
                draw_release_effect(frame, release_particles, release_rings, impact_lines, release_lightning, style)
                if moon_point is not None:
                    draw_power_moon(frame, moon_point, int(58 + charge_level * 0.24), style)
            if mode == MODE_NERD and nerd_point is not None:
                face_box = largest_face_box(face_detector, rgb_frame)
                draw_nerd_effect(frame, face_box, nerd_point)
            draw_status(frame, hand_count, fps)
            draw_mode_label(frame, mode)
            if mode == MODE_POWER:
                draw_style_label(frame, style)
            if mode == MODE_DRAW and time.perf_counter() < clear_feedback_until:
                draw_clear_feedback(frame)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
                cv2.setTrackbarPos("Mode", COLOR_WINDOW_NAME, int(chr(key)))
            if mode == MODE_DRAW and key == ord("c") and canvas is not None:
                clear_feedback_until = clear_canvas(canvas)
                last_draw_point = None

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
