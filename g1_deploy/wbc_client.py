"""Camera + G1 observation client and high-level WBC action executor."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
import urllib.request
import urllib.error

import numpy as np
from scipy.spatial.transform import Rotation

from .core import (
    ACTION_NAMES, CAMERAS, NEUTRAL_GRIP, CommandLatch, action_index, decode_camera,
    slew, unpack_action, validate_chunk, vector,
)

LOG = logging.getLogger(__name__)


class IKConvergenceError(RuntimeError):
    """No valid new IK solution; retain the last accepted command."""


class ArmTrackingError(RuntimeError):
    """Reject one candidate command without terminating the control loop."""


def http_json(url, payload=None, timeout=2):
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(8192).decode('utf-8', errors='replace')
        raise RuntimeError(f"Policy HTTP {exc.code}: {detail}") from exc


def encode_payload(images, state, wbc_state, task):
    import cv2

    encoded = {}
    for name in CAMERAS:
        image = np.asarray(images[name])
        valid_shapes = ((480, 640, 3), (480, 1280, 3)) if name == "head" else ((480, 640, 3),)
        if image.shape not in valid_shapes or image.dtype != np.uint8:
            raise ValueError(f"{name} must be RGB uint8 480x640")
        ok, data = cv2.imencode(".jpg", image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        encoded[name] = base64.b64encode(data).decode("ascii")
    payload = {"images": encoded, "state": vector(state, 29, "state").tolist(), "task": task}
    if wbc_state is not None:
        payload["wbc_state"] = vector(wbc_state, 23, "wbc_state").tolist()
    return payload


def validate_schema(schema):
    if (schema.get("protocol") != "g1-wbc-v1" or schema.get("action_shape") not in ([30, 23], [32, 23], [40, 23], [50, 23])
            or schema.get("action_names") != list(ACTION_NAMES) or schema.get("action_fps") != 30):
        raise ValueError("Policy server schema does not match this G1 checkpoint")
    vector(schema["action_min"], 23, "action minimum")
    vector(schema["action_max"], 23, "action maximum")


class CameraReader:
    def __new__(cls, config):
        backend = config.get("backend", "zmq")
        if backend == "dds":
            from .dds_camera import DDSCameraReader
            return DDSCameraReader(config)
        if backend != "zmq":
            raise ValueError(f"Unknown camera backend: {backend}")
        return super().__new__(cls)

    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        self.images = None
        self.timestamp = 0.0
        self.error = None
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self.thread.start()

    def _run(self):
        import zmq

        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt(zmq.RCVTIMEO, 100)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.config["address"])
        try:
            while not self.closed.is_set():
                try:
                    message = socket.recv()
                except zmq.Again:
                    continue
                received = time.monotonic()
                images = decode_camera(message, self.config)
                with self.lock:
                    self.images, self.timestamp = images, received
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
        finally:
            socket.close()
            context.term()

    def snapshot(self, max_age):
        with self.lock:
            if self.error:
                raise RuntimeError(f"Camera: {self.error}")
            if self.images is None or time.monotonic() - self.timestamp > max_age:
                raise RuntimeError("Camera frames missing or stale")
            return {k: v.copy() for k, v in self.images.items()}, self.timestamp

    def close(self):
        self.closed.set()
        self.thread.join(timeout=1)


class Kinematics:
    def __init__(self, repo):
        root = Path(repo).resolve()
        if not (root / "teleop/robot_control/robot_arm_ik.py").is_file():
            raise FileNotFoundError(f"WBC IK not found at {root}")
        sys.path.insert(0, str(root))
        # WBC revisions use either spelling; retain the installed logger backend.
        import logging_mp
        if not hasattr(logging_mp, "get_logger") and hasattr(logging_mp, "getLogger"):
            logging_mp.get_logger = logging_mp.getLogger
        from teleop.robot_control.robot_arm_ik import G1_29_ArmIK

        previous = os.getcwd()
        try:
            # Reference IK loads its URDF/assets relative to cwd.
            os.chdir(root)
            self.ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)
        finally:
            os.chdir(previous)

    def state(self, q):
        return np.concatenate(self.ik.solve_fk(q))

    def command(self, action, q, dq, previous_q, previous_base, config, dt):
        base, left, right, grip = unpack_action(action)
        target_q, _ = self.ik.solve_ik(left, right, q.copy(), dq.copy())
        if not self.ik.opti.stats().get("success", False):
            raise IKConvergenceError("Reference IK did not converge; retaining previous command")
        target_q = vector(target_q, 14, "IK joint target")
        # solve_ik returns a moving average of several previous solutions.
        # Test solver accuracy on the current optimum; the smoothed command
        # intentionally lags changing targets and still goes through slew,
        # measured-state tracking, joint-limit, and torque checks below.
        raw_q = vector(self.ik.opti.value(self.ik.var_q), 14, "raw IK joint target")
        actual_left, actual_right = self.ik.solve_fk(raw_q)
        for arm, actual, target in zip(("left", "right"), (actual_left, actual_right), (left, right)):
            error_m = np.linalg.norm(actual[:3] - target[:3, 3])
            error_rad = (Rotation.from_euler("xyz", actual[3:]).inv()
                         * Rotation.from_matrix(target[:3, :3])).magnitude()
            if error_m > config["max_ik_position_error_m"] or error_rad > config["max_ik_rotation_error_rad"]:
                # Preserve the exact failing command. The reference solver returns
                # a moving-average-filtered solution, which can lag the raw optimum.
                # Diagnostics must never bypass the existing rejection.
                diagnostic_path = None
                try:
                    report = {
                        "residual_stage": "raw_ik_solution",
                        "arm": arm, "position_error_m": float(error_m),
                        "rotation_error_rad": float(error_rad),
                        "action": np.asarray(action).tolist(), "measured_q": np.asarray(q).tolist(),
                        "measured_dq": np.asarray(dq).tolist(), "raw_ik_q": raw_q.tolist(),
                        "filtered_ik_q": target_q.tolist(),
                        "previous_q": np.asarray(previous_q).tolist(),
                        "position_limit_m": config["max_ik_position_error_m"],
                        "rotation_limit_rad": config["max_ik_rotation_error_rad"],
                        "raw_errors": {}, "filtered_errors": {},
                    }
                    for kind, poses in (("raw", (actual_left, actual_right)),
                                        ("filtered", self.ik.solve_fk(target_q))):
                        for name, pose, goal in zip(("left", "right"), poses, (left, right)):
                            report[kind + "_errors"][name] = {
                                "position_m": float(np.linalg.norm(pose[:3] - goal[:3, 3])),
                                "rotation_rad": float((Rotation.from_euler("xyz", pose[3:]).inv()
                                                       * Rotation.from_matrix(goal[:3, :3])).magnitude()),
                            }
                    folder = Path(config.get("ik_failure_dir", "outputs/ik-failures"))
                    folder.mkdir(parents=True, exist_ok=True)
                    diagnostic_path = folder / f"ik-{time.time_ns()}.json"
                    diagnostic_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                    LOG.error("IK failure details saved to %s; raw errors=%s", diagnostic_path, report["raw_errors"])
                except Exception:
                    LOG.exception("Could not save IK failure diagnostics")
                raise RuntimeError(
                    f"{arm} raw IK residual too large: {error_m:.3f} m / {error_rad:.3f} rad "
                    f"(limits {config['max_ik_position_error_m']:.3f} m / "
                    f"{config['max_ik_rotation_error_rad']:.3f} rad; details={diagnostic_path})"
                )
        target_q = slew(previous_q, target_q, config["arm_speed_rad_s"], dt)
        if np.max(np.abs(target_q - q)) > config["max_arm_tracking_error_rad"]:
            raise ArmTrackingError("Arm command too far from measured joint state")
        model = self.ik.reduced_robot.model
        if np.any(target_q < model.lowerPositionLimit) or np.any(target_q > model.upperPositionLimit):
            raise RuntimeError("Arm command outside URDF limits")
        # Recompute gravity compensation at the limited command, not the unslewed IK target.
        import pinocchio as pin

        tau = pin.rnea(model, self.ik.reduced_robot.data, target_q, np.zeros(14), np.zeros(14))
        tau = vector(tau, 14, "arm torque")
        if np.max(np.abs(tau)) > config["max_arm_feedforward_nm"]:
            raise RuntimeError("Arm feedforward torque exceeds configured limit")
        base = np.clip(base, config["base_min"], config["base_max"])
        base = slew(previous_base, base, config["base_slew_per_s"], dt)
        grip = np.clip(grip, [0, 0, 0, 0], [10, 1, 10, 1])
        return {"arm_q": target_q.tolist(), "arm_tau": tau.tolist(), "base_cmd": base.tolist()}, grip


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if type(config.get("command_heartbeat_enabled", True)) is not bool:
        raise ValueError("command_heartbeat_enabled must be a boolean")
    if type(config.get("action_plan_expiry_enabled", True)) is not bool:
        raise ValueError("action_plan_expiry_enabled must be a boolean")
    if type(config.get("robot_state_expiry_enabled", True)) is not bool:
        raise ValueError("robot_state_expiry_enabled must be a boolean")
    if config["action_fps"] != 30 or not 1 <= config["replan_steps"] <= config.get("chunk_size", 32):
        raise ValueError("Invalid action_fps or replan_steps for configured chunk size")
    for name in ("max_state_age_s", "max_camera_age_s", "command_timeout_s", "request_timeout_s",
                 "max_plan_age_s", "startup_timeout_s", "arm_speed_rad_s"):
        if not 0 < config[name] < float("inf"):
            raise ValueError(f"Invalid positive setting: {name}")
    for name in ("initial_base_cmd", "base_min", "base_max", "base_slew_per_s"):
        vector(config[name], 7, name)
    if np.any(np.array(config["base_min"]) > config["base_max"]) or np.any(np.array(config["base_slew_per_s"]) <= 0):
        raise ValueError("Invalid base bounds/rate")
    return config


def run(config, mode, duration, observation_path=None):
    from .transport import RobotTransport

    if config.get("sdk_path"):
        sys.path.insert(0, str(Path(config["sdk_path"]).expanduser().resolve()))
    schema = None
    if mode != "observe":
        schema = http_json(config["policy_url"].rstrip("/") + "/schema", timeout=config["request_timeout_s"])
        validate_schema(schema)
        if config.get("model_key") and schema.get("model_key") != config["model_key"]:
            raise ValueError("Client/server model identity mismatch")
        if schema["action_shape"][0] != config.get("chunk_size", 32):
            raise ValueError("Client/server chunk size mismatch")
        if schema.get("head_mode", "left_eye") != config["camera"].get("head_mode", "left_eye"):
            raise ValueError("Client/server head camera preprocessing mismatch")
    from .recording import RolloutRecorder
    recorder = (RolloutRecorder(config["rollout_dir"], config, schema, mode)
                if config.get("rollout_dir") else None)
    if recorder:
        config = dict(config, ik_failure_dir=str(recorder.path / "ik-failures"))
    robot, camera, pool = None, None, None
    heartbeat_enabled = config.get("command_heartbeat_enabled", True)
    dry_latch = CommandLatch(config["command_timeout_s"], heartbeat_enabled) if mode == "dry-run" else None
    LOG.info("Command heartbeat enabled: %s", heartbeat_enabled)
    plan_expiry_enabled = config.get("action_plan_expiry_enabled", True)
    LOG.info("Action plan expiry enabled: %s", plan_expiry_enabled)
    last_command_at = None
    skipped_tracking_commands = 0
    skipped_ik_commands = 0
    last_tracking_warning = -float("inf")
    timing_max = {"encode_s": 0.0, "ik_s": 0.0, "command_gap_s": 0.0}
    try:
        # IK before connecting DDS; loading assets can be slow.
        kinematics = Kinematics(config["wbc_repo"]) if mode != "observe" else None
        robot = RobotTransport(config, execute=mode == "execute")
        camera = CameraReader(config["camera"])
        robot.wait_ready()
        deadline = time.monotonic() + config["startup_timeout_s"]
        while True:
            try:
                camera.snapshot(config["max_camera_age_s"])
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        LOG.info("DDS and all three cameras ready; mode=%s", mode)
        q, _, _ = robot.snapshot()
        last_q, last_base, last_grip = q[15:29].copy(), np.asarray(config["initial_base_cmd"], float), NEUTRAL_GRIP.copy()
        plan = future = None
        plan_capture = -float("inf")
        request_capture = -float("inf")
        pool = ThreadPoolExecutor(max_workers=1)
        started = last_log = time.monotonic()
        next_tick = started
        saved = False
        request_id = 0
        plan_id = None
        last_recorded_observation = -float("inf")
        while duration <= 0 or time.monotonic() - started < duration:
            now = time.monotonic()
            q, dq, state_time = robot.snapshot()
            images, image_time = camera.snapshot(config["max_camera_age_s"])
            if observation_path and not saved:
                output = Path(observation_path)
                output.parent.mkdir(parents=True, exist_ok=True)
                # Capture only measured signals. No fabricated WBC state in observe mode.
                with output.open("wb") as f:
                    np.savez_compressed(f, state=q.astype(np.float32), **images)
                saved = True
                LOG.info("Saved measured observation to %s", output)
            if recorder and mode == "observe" and now - last_recorded_observation >= 0.2:
                request_id += 1
                recorder.record("observation", request_id=request_id,
                                payload=encode_payload(images, q, None, config["task"]),
                                dq=dq.tolist(), state_time=state_time, camera_time=image_time)
                last_recorded_observation = now
            if mode != "observe":
                if future is not None and future.done():
                    result = future.result()
                    if recorder:
                        recorder.record("policy_response", request_id=request_id, result=result)
                    plan_id = request_id
                    plan = validate_chunk(result["actions"], schema["action_min"], schema["action_max"], schema["action_shape"][0],
                                          overshoot_fraction=config.get("action_overshoot_fraction", 0.10),
                                          bounds_mode=config.get("action_bounds_mode", "reject"))
                    clipped = int(np.count_nonzero(plan != np.asarray(result["actions"])))
                    if clipped:
                        LOG.info("Client clipped %d action values to checkpoint bounds", clipped)
                        if recorder:
                            recorder.record("action_clipped", request_id=request_id, clipped_values=clipped,
                                            bounds_mode=config.get("action_bounds_mode", "reject"),
                                            overshoot_fraction=config.get("action_overshoot_fraction", 0.10),
                                            actions=plan.tolist())
                    plan_capture = request_capture
                    future = None
                    LOG.info("Inference %.3fs; total observation age %.3fs", result.get("inference_s", 0), now - plan_capture)
                if future is None and (plan is None or now - request_capture >= config["replan_steps"] / 30):
                    # Matches reference: commanded pivot + measured FK + last gripper controls.
                    # initial_base_cmd is an explicit starting-command assumption, not sensed base state.
                    wbc_state = np.r_[last_base, kinematics.state(q[15:29]), last_grip]
                    request_capture = min(state_time, image_time)
                    encode_started = time.monotonic()
                    payload = encode_payload(images, q, wbc_state, config["task"])
                    timing_max["encode_s"] = max(timing_max["encode_s"], time.monotonic() - encode_started)
                    request_id += 1
                    if recorder:
                        recorder.record("policy_request", request_id=request_id, payload=payload,
                                        dq=dq.tolist(), state_time=state_time, camera_time=image_time)
                    future = pool.submit(http_json, config["policy_url"].rstrip("/") + "/infer",
                                         payload, config["request_timeout_s"])
                if plan is not None:
                    index = action_index(plan_capture, now, 30, length=len(plan), expiry_enabled=plan_expiry_enabled)
                    if plan_expiry_enabled and (index is None or now - plan_capture > config["max_plan_age_s"]):
                        raise RuntimeError("Action plan expired; no fresh inference available")
                    if recorder:
                        recorder.record("control_attempt", request_id=plan_id, action_index=index,
                                        action=plan[index].tolist(), q=q.tolist(), dq=dq.tolist(),
                                        state_time=state_time, camera_time=image_time)
                    ik_started = time.monotonic()
                    try:
                        command, grip = kinematics.command(
                            plan[index], q[15:29], dq[15:29],
                            q[15:29] if mode == "dry-run" else last_q, last_base, config, 1 / 30,
                        )
                    except (ArmTrackingError, IKConvergenceError) as exc:
                        if recorder:
                            recorder.record("command_skipped", request_id=plan_id, action_index=index, reason=str(exc), error_type=type(exc).__name__)
                        timing_max["ik_s"] = max(timing_max["ik_s"], time.monotonic() - ik_started)
                        if isinstance(exc, IKConvergenceError):
                            skipped_ik_commands += 1
                        else:
                            skipped_tracking_commands += 1
                        if time.monotonic() - last_tracking_warning >= 1.0:
                            LOG.warning("Skipping command: %s; tracking_skipped=%d IK_skipped=%d", exc, skipped_tracking_commands, skipped_ik_commands)
                            last_tracking_warning = time.monotonic()
                        # Do not update the last accepted command or bypass loop pacing.
                        next_tick = max(next_tick + 1 / 30, time.monotonic())
                        time.sleep(max(0.0, next_tick - time.monotonic()))
                        continue
                    ik_elapsed = time.monotonic() - ik_started
                    timing_max["ik_s"] = max(timing_max["ik_s"], ik_elapsed)
                    command_now = time.monotonic()
                    if last_command_at is not None:
                        gap = command_now - last_command_at
                        timing_max["command_gap_s"] = max(timing_max["command_gap_s"], gap)
                        if heartbeat_enabled and gap > config["command_timeout_s"] / 2:
                            LOG.warning("Slow command update: gap=%.3fs IK=%.3fs (limit=%.3fs)",
                                        gap, ik_elapsed, config["command_timeout_s"])
                    if dry_latch is not None:
                        dry_latch.sample(command_now, True)
                        if dry_latch.fault:
                            raise RuntimeError(f"Dry-run watchdog: {dry_latch.fault}")
                    # IK can take longer than one control tick. Never dispatch a
                    # formerly fresh target after a slow solve or lost telemetry.
                    fresh_q, _, _ = robot.snapshot()
                    camera.snapshot(config["max_camera_age_s"])
                    if plan_expiry_enabled and time.monotonic() - plan_capture > config["max_plan_age_s"]:
                        raise RuntimeError("Action plan expired during IK")
                    tracking_error = float(np.max(np.abs(np.array(command["arm_q"]) - fresh_q[15:29])))
                    if tracking_error > config["max_arm_tracking_error_rad"]:
                        if recorder:
                            recorder.record("command_skipped", request_id=plan_id, action_index=index,
                                            reason="post-IK tracking error", error_rad=tracking_error)
                        skipped_tracking_commands += 1
                        if time.monotonic() - last_tracking_warning >= 1.0:
                            LOG.warning("Skipping command after IK: arm tracking error %.3f rad > %.3f rad; skipped=%d",
                                        tracking_error, config["max_arm_tracking_error_rad"], skipped_tracking_commands)
                            last_tracking_warning = time.monotonic()
                    else:
                        robot.set_command(command, grip)
                        if recorder:
                            recorder.record("command_submitted" if mode == "execute" else "dry_run_command",
                                            request_id=plan_id, action_index=index, command=command,
                                            grip=grip.tolist(), measured_q=fresh_q.tolist())
                        last_command_at = time.monotonic()
                        if dry_latch is not None:
                            dry_latch.update(command, grip, last_command_at)
                        last_q, last_base, last_grip = np.array(command["arm_q"]), np.array(command["base_cmd"]), grip
            if now - last_log >= 1:
                LOG.info("%s state_age=%.3f camera_age=%.3f base=%s", mode, now - state_time,
                         now - image_time, np.round(last_base, 3).tolist())
                last_log = now
            next_tick = max(next_tick + 1 / 30, time.monotonic())
            time.sleep(max(0.0, next_tick - time.monotonic()))
    except BaseException as exc:
        if recorder:
            recorder.record("stop", reason=type(exc).__name__, detail=str(exc))
        raise
    else:
        if recorder:
            recorder.record("stop", reason="duration_complete")
    finally:
        LOG.info("Control timing maxima: %s", timing_max)
        LOG.info("Commands skipped after IK tracking check: %d", skipped_tracking_commands)
        LOG.info("Commands skipped after IK non-convergence: %d", skipped_ik_commands)
        if robot is not None:
            robot.close()
        if camera is not None:
            camera.close()
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        if recorder:
            recorder.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/deploy.example.json")
    parser.add_argument("--mode", choices=("observe", "dry-run", "execute"), default="dry-run")
    parser.add_argument("--duration", type=float, default=30, help="Seconds; 0 runs until Ctrl+C")
    parser.add_argument("--save-observation", help="Write one measured observation as NPZ, without pickle")
    parser.add_argument("--rollout-dir", help="Save timestamped rollout under this directory")
    parser.add_argument("--interface", help="Override the G1 DDS network interface in the config")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_config(args.config)
        if args.rollout_dir:
            config["rollout_dir"] = args.rollout_dir
        if args.interface:
            config["interface"] = args.interface
        run(config, args.mode, args.duration, args.save_observation)
    except KeyboardInterrupt:
        LOG.info("Stopped by operator")
    except Exception:
        LOG.exception("Client stopped")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
