"""Pure observation/action conversions; no robot publishers or model imports."""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

CAMERAS = ("head", "left_wrist", "right_wrist")
ACTION_NAMES = (
    "vx", "vy", "ang_z", "yaw", "pitch", "roll", "height",
    "left_x", "left_y", "left_z", "left_roll", "left_pitch", "left_yaw",
    "right_x", "right_y", "right_z", "right_roll", "right_pitch", "right_yaw",
    "left_trigger", "left_squeeze", "right_trigger", "right_squeeze",
)
NEUTRAL_GRIP = np.array([10.0, 0.0, 10.0, 0.0])


def vector(value, size, name):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected finite ({size},), got {a.shape}")
    return a


def xyzrpy_matrix(pose):
    p = vector(pose, 6, "xyzrpy")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", p[3:], degrees=False).as_matrix()
    transform[:3, 3] = p[:3]
    return transform


def matrix_xyzrpy(transform):
    return np.r_[transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_euler("xyz")]


def unpack_action(action):
    a = vector(action, 23, "action")
    return a[:7].copy(), xyzrpy_matrix(a[7:13]), xyzrpy_matrix(a[13:19]), a[19:].copy()


def read_stats(path):
    """Read the small checkpoint statistics without importing torch/safetensors."""
    data = Path(path).read_bytes()
    header_size = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + header_size])
    result = {}
    for key in ("action.min", "action.max"):
        entry = header[key]
        start, end = entry["data_offsets"]
        dtype = {"F64": "<f8", "F32": "<f4"}[entry["dtype"]]
        result[key] = np.frombuffer(data[8 + header_size + start:8 + header_size + end], dtype=dtype).copy()
        vector(result[key], 23, key)
    return result


def decode_camera(message, config):
    """WBC ImageServer single JPEG: [head stereo | left wrist | right wrist]."""
    import cv2

    header_size = config.get("header_bytes", 0)
    if type(header_size) is not int or header_size < 0:
        raise ValueError("camera.header_bytes must be a nonnegative integer")
    if config.get("timestamp_header", False):
        if header_size:
            raise ValueError("Use either timestamp_header or header_bytes, not both")
        header_size = struct.calcsize("dI")
    if len(message) <= header_size:
        raise ValueError("Incomplete camera header")
    message = message[header_size:]
    if not message.startswith(b"\xff\xd8\xff"):
        raise ValueError("JPEG signature missing at configured camera header offset")
    image = cv2.imdecode(np.frombuffer(message, np.uint8), cv2.IMREAD_COLOR)
    expected = (config["height"], config["total_width"], 3)
    if image is None or image.shape != expected:
        raise ValueError(f"Camera composite must be {expected}; got {None if image is None else image.shape}")
    result = {}
    for name in CAMERAS:
        x0, x1 = config["columns"][name]
        expected_width = 1280 if name == "head" and config.get("head_mode") in ("stereo_squash", "stereo_full") else 640
        if not 0 <= x0 < x1 <= image.shape[1] or x1 - x0 != expected_width or image.shape[0] != 480:
            raise ValueError(f"Invalid 640x480 crop for {name}: {x0}:{x1}")
        crop = image[:, x0:x1]
        if expected_width == 1280 and config.get("head_mode") == "stereo_squash":
            crop = cv2.resize(crop, (640, 480), interpolation=cv2.INTER_LINEAR)
        result[name] = np.ascontiguousarray(crop[:, :, ::-1])  # BGR -> RGB
    return result


def validate_chunk(actions, minimum, maximum, length=32, *, overshoot_fraction=0.0, bounds_mode="reject"):
    a = np.asarray(actions, dtype=np.float64)
    if a.shape != (length, 23) or not np.isfinite(a).all():
        raise ValueError(f"Expected finite ({length}, 23) action chunk, got {a.shape}")
    lo = vector(minimum, 23, "action minimum")
    hi = vector(maximum, 23, "action maximum")
    if np.any(lo > hi):
        raise ValueError("Invalid action bounds")
    if not math.isfinite(overshoot_fraction) or not 0 <= overshoot_fraction <= 1:
        raise ValueError("Invalid overshoot fraction")
    if bounds_mode not in ("reject", "clip"):
        raise ValueError("Invalid action bounds mode")
    if bounds_mode == "clip":
        return np.clip(a, lo, hi)
    tolerance = 1e-4 + overshoot_fraction * (hi - lo)
    mask = (a < lo - tolerance) | (a > hi + tolerance)
    if mask.any():
        details = []
        for j in np.flatnonzero(mask.any(axis=0)):
            details.append(f"{ACTION_NAMES[j]}: output=[{a[:, j].min():.6g}, {a[:, j].max():.6g}], bounds=[{lo[j]:.6g}, {hi[j]:.6g}], tolerance={tolerance[j]:.6g}")
        raise ValueError("Postprocessed actions outside checkpoint min/max; check normalization; " + "; ".join(details))
    return np.clip(a, lo, hi) if overshoot_fraction else a


def limit_gripper_controls(actions, length):
    """Apply the existing Dex1 control ranges before PI05 range validation.

    Saturating these four controls preserves the trigger/squeeze threshold
    decisions in gripper_target. Never mask nonfinite or malformed outputs.
    """
    a = np.asarray(actions, dtype=np.float64)
    if a.shape != (length, 23) or not np.isfinite(a).all():
        raise ValueError(f"Expected finite ({length}, 23) action chunk, got {a.shape}")
    result = a.copy()
    result[:, 19:] = np.clip(result[:, 19:], [0, 0, 0, 0], [10, 1, 10, 1])
    return result


def action_index(captured_at, now, fps, length=32, expiry_enabled=True):
    """Skip actions whose timestamps passed during transport/inference."""
    index = max(0, math.floor((now - captured_at) * fps))
    if not expiry_enabled:
        return min(index, length - 1)
    return index if index < length else None


def slew(previous, desired, max_rate, dt):
    return previous + np.clip(desired - previous, -np.asarray(max_rate) * dt, np.asarray(max_rate) * dt)


def hold_command(command):
    result = {key: list(value) for key, value in command.items()}
    result["base_cmd"][:3] = [0.0, 0.0, 0.0]
    return result


def gripper_target(position, previous, trigger, squeeze):
    """Reference Dex1 trigger semantics: 10=released, 0=pressed."""
    if trigger < 5.0 and squeeze < 0.5:
        return float(np.clip(position - 0.35, 0.0, 4.5))
    if trigger >= 5.0 and squeeze >= 0.5:
        return float(np.clip(position + 0.35, 0.0, 4.5))
    return float(previous)


class CommandLatch:
    """Called under the transport lock. A fault never resumes automatically."""
    def __init__(self, timeout, heartbeat_enabled=True, gripper_mode="controls"):
        if gripper_mode not in ("controls", "position"):
            raise ValueError("Unknown gripper command mode")
        self.gripper_mode = gripper_mode
        self.timeout = timeout
        self.heartbeat_enabled = heartbeat_enabled
        self.command = None
        self.grip = NEUTRAL_GRIP.copy() if gripper_mode == "controls" else np.zeros(2)
        self.updated_at = 0.0
        self.fault = None

    def update(self, command, grip, now):
        if self.fault:
            raise RuntimeError(self.fault)
        validated_grip = vector(grip, 4 if self.gripper_mode == "controls" else 2, "grip").copy()
        validated_command = {
            "arm_q": vector(command["arm_q"], 14, "arm_q").tolist(),
            "arm_tau": vector(command["arm_tau"], 14, "arm_tau").tolist(),
            "base_cmd": vector(command["base_cmd"], 7, "base_cmd").tolist(),
        }
        self.command = validated_command
        self.grip = validated_grip
        self.updated_at = now

    def stop(self, reason):
        self.fault = self.fault or reason

    def sample(self, now, state_fresh, emergency_stop=False):
        if self.command is None:
            return None, self.grip.copy()
        if emergency_stop:
            self.stop("Remote B stop")
        if not state_fresh:
            self.stop("Robot state expired")
        if self.heartbeat_enabled and now - self.updated_at > self.timeout:
            self.stop("Command heartbeat expired")
        if self.fault:
            return hold_command(self.command), (NEUTRAL_GRIP.copy() if self.gripper_mode == "controls" else self.grip.copy())
        return self.command, self.grip.copy()
