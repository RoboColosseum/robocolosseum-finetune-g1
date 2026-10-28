import base64
import copy
import json
from pathlib import Path
import socket
import threading
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from g1_deploy.core import (
    ACTION_NAMES, NEUTRAL_GRIP, CommandLatch, action_index, decode_camera,
    gripper_target, matrix_xyzrpy, read_stats, slew, unpack_action, validate_chunk, xyzrpy_matrix,
)
from g1_deploy.policy_server import decode_observation, make_server
from g1_deploy.wbc_client import IKConvergenceError, ArmTrackingError, CameraReader, Kinematics, encode_payload, http_json, load_config, run, validate_schema

ROOT = Path(__file__).resolve().parents[1]


def require_tcp(test):
    # Some managed execution environments prohibit even loopback TCP. Probe
    # before calling libzmq, which otherwise aborts rather than raising Python.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
    except OSError as exc:
        test.skipTest(f"Loopback sockets unavailable: {exc}")


class Conversions(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "config/deploy.example.json")

    def test_action_order_and_rotation(self):
        a = np.arange(23, dtype=float) / 100
        base, left, right, grip = unpack_action(a)
        np.testing.assert_allclose(base, a[:7])
        np.testing.assert_allclose(left[:3, 3], a[7:10])
        np.testing.assert_allclose(left[:3, :3], Rotation.from_euler("xyz", a[10:13]).as_matrix())
        np.testing.assert_allclose(matrix_xyzrpy(right), a[13:19])
        np.testing.assert_allclose(grip, a[19:])
        np.testing.assert_allclose(xyzrpy_matrix(matrix_xyzrpy(left)), left)

    def test_stereo_left_eye_wrist_crops_and_rgb(self):
        composite = np.zeros((480, 2560, 3), dtype=np.uint8)
        colors_bgr = [(20, 40, 200), (50, 170, 70), (100, 30, 10), (80, 90, 130)]
        for i, color in enumerate(colors_bgr):
            composite[:, i * 640:(i + 1) * 640] = color
        ok, jpg = cv2.imencode(".jpg", composite)
        self.assertTrue(ok)
        camera = dict(self.config["camera"], header_bytes=0)
        images = decode_camera(jpg.tobytes(), camera)
        framed = decode_camera(b"12345678" + jpg.tobytes(), dict(camera, header_bytes=8))
        for name in images:
            np.testing.assert_array_equal(images[name], framed[name])
        with self.assertRaises(ValueError):
            decode_camera(b"12345678" + jpg.tobytes(), camera)
        for name, i in (("head", 0), ("left_wrist", 2), ("right_wrist", 3)):
            np.testing.assert_allclose(images[name][200, 200], colors_bgr[i][::-1], atol=3)
        _, short = cv2.imencode(".jpg", composite[:, :1280])
        with self.assertRaises(ValueError):
            decode_camera(short.tobytes(), camera)

    def test_gripper_neutral_and_thresholds(self):
        self.assertEqual(gripper_target(2, 2.2, 10, 0), 2.2)
        self.assertAlmostEqual(gripper_target(2, 2, 0, 0), 1.65)
        self.assertAlmostEqual(gripper_target(2, 2, 10, 1), 2.35)
        self.assertEqual(gripper_target(0, 0, 0, 0), 0)
        self.assertEqual(gripper_target(4.5, 4.5, 10, 1), 4.5)

    def test_chunk_rejects_bad_shape_nan_and_wrong_normalization(self):
        lo, hi = np.zeros(23), np.ones(23)
        validate_chunk(np.zeros((32, 23)), lo, hi)
        for value in (np.zeros((32, 132)), np.full((32, 23), np.nan), np.full((32, 23), 2)):
            with self.assertRaises(ValueError):
                validate_chunk(value, lo, hi)

    def test_latency_skip_and_expiry(self):
        self.assertEqual(action_index(100, 100.21, 30), 6)
        self.assertIsNone(action_index(100, 101.1, 30))
        self.assertEqual(action_index(100, 101.1, 30, expiry_enabled=False), 31)
        self.assertEqual(action_index(100, 200, 30, expiry_enabled=False), 31)
        self.assertEqual(action_index(100, 100.21, 30, expiry_enabled=False), 6)
        np.testing.assert_allclose(slew(np.zeros(3), np.ones(3), 0.8, 1 / 30), np.ones(3) * 0.8 / 30)

    def test_plate_stereo_keeps_both_eyes_and_wrist_positions(self):
        composite = np.zeros((480, 2560, 3), dtype=np.uint8)
        colors = [(20, 40, 200), (50, 170, 70), (100, 30, 10), (80, 90, 130)]
        for i, color in enumerate(colors):
            composite[:, i * 640:(i + 1) * 640] = color
        _, jpg = cv2.imencode('.jpg', composite)
        config = load_config(ROOT / 'config/plate.example.json')['camera']
        images = decode_camera(b'12345678' + jpg.tobytes(), config)
        self.assertEqual(images['head'].shape, (480, 640, 3))
        np.testing.assert_allclose(images['head'][200, 100], colors[0][::-1], atol=3)
        np.testing.assert_allclose(images['head'][200, 500], colors[1][::-1], atol=3)
        np.testing.assert_allclose(images['left_wrist'][200, 200], colors[2][::-1], atol=3)
        np.testing.assert_allclose(images['right_wrist'][200, 200], colors[3][::-1], atol=3)

    def test_plate_chunk_uses_last_eight_actions(self):
        lo, hi = np.zeros(23), np.ones(23)
        validate_chunk(np.zeros((40, 23)), lo, hi, 40)
        with self.assertRaises(ValueError):
            validate_chunk(np.zeros((32, 23)), lo, hi, 40)
        self.assertEqual(action_index(100, 101.2, 30, length=40), 36)
        self.assertIsNone(action_index(100, 101.4, 30, length=40))
        self.assertEqual(action_index(100, 200, 30, length=40, expiry_enabled=False), 39)

    def test_checkpoint_stats_if_available(self):
        path = ROOT / "models/groot-n17-g1-carry-box-40k/policy_postprocessor_step_0_groot_action_unpack_unnormalize_v2.safetensors"
        if not path.exists():
            self.skipTest("Checkpoint not copied to this machine")
        stats = read_stats(path)
        np.testing.assert_equal(stats["action.min"][19:], [0, 0, 0, 0])
        np.testing.assert_equal(stats["action.max"][19:], [10, 1, 10, 1])


class Watchdog(unittest.TestCase):
    def setUp(self):
        self.command = {"arm_q": [0.2] * 14, "arm_tau": [1.5] * 14,
                        "base_cmd": [0.2, 0.1, 0.3, 0.1, 0.2, 0.1, 0.65]}

    def test_no_command_before_arm(self):
        latch = CommandLatch(0.25)
        self.assertIsNone(latch.sample(100, True)[0])

    def test_state_expiry_option_preserves_initial_state_and_fault_checks(self):
        from g1_deploy.transport import RobotTransport
        robot = RobotTransport.__new__(RobotTransport)
        robot.lock = threading.Lock()
        robot.config = {"max_state_age_s": 0.25}
        robot.latch = CommandLatch(0.25)
        robot.q = robot.dq = np.zeros(29)
        robot.state_time = time.monotonic() - 10
        with self.assertRaisesRegex(RuntimeError, "missing or stale"):
            robot.snapshot()
        robot.config["robot_state_expiry_enabled"] = False
        self.assertEqual(robot.snapshot()[0].shape, (29,))
        robot.q = None
        with self.assertRaisesRegex(RuntimeError, "missing or stale"):
            robot.snapshot()
        robot.q = np.zeros(29)
        robot.latch.stop("Remote B stop")
        with self.assertRaisesRegex(RuntimeError, "Remote B stop"):
            robot.snapshot()

    def test_timeout_holds_pose_height_and_zeroes_velocity(self):
        latch = CommandLatch(0.25)
        latch.update(self.command, [0, 0, 0, 0], 10)
        active, _ = latch.sample(10.1, True)
        self.assertEqual(active, self.command)
        hold, grip = latch.sample(10.3, True)
        self.assertEqual(hold["base_cmd"][:3], [0, 0, 0])
        self.assertEqual(hold["base_cmd"][3:], self.command["base_cmd"][3:])
        self.assertEqual(hold["arm_q"], self.command["arm_q"])
        np.testing.assert_equal(grip, NEUTRAL_GRIP)
        with self.assertRaises(RuntimeError):
            latch.update(self.command, NEUTRAL_GRIP, 11)

    def test_state_loss_and_remote_stop_latch(self):
        for fresh, stop in ((False, False), (True, True)):
            latch = CommandLatch(0.25)
            latch.update(self.command, NEUTRAL_GRIP, 10)
            hold, _ = latch.sample(10.01, fresh, stop)
            self.assertEqual(hold["base_cmd"][:3], [0, 0, 0])
            self.assertIsNotNone(latch.fault)

    def test_disabled_heartbeat_preserves_other_stops(self):
        for fresh, stop in ((True, False), (False, False), (True, True)):
            latch = CommandLatch(0.25, heartbeat_enabled=False)
            latch.update(self.command, NEUTRAL_GRIP, 10)
            command, _ = latch.sample(100, fresh, stop)
            if fresh and not stop:
                self.assertIsNone(latch.fault)
                self.assertEqual(command, self.command)
            else:
                self.assertIsNotNone(latch.fault)
                self.assertEqual(command["base_cmd"][:3], [0, 0, 0])

    def test_dds_write_failure_latches_and_attempts_zero_velocity(self):
        from g1_deploy.transport import RobotTransport

        transport = RobotTransport.__new__(RobotTransport)
        transport.config = {"max_state_age_s": 0.25}
        transport.lock = threading.Lock()
        transport.closed = threading.Event()
        transport.state_time = time.monotonic()
        transport.gripper_time = np.array([transport.state_time] * 2)
        transport.gripper_q = np.array([2.0, 2.0])
        transport.remote_stop = False
        transport.latch = CommandLatch(0.25)
        transport.latch.update(self.command, NEUTRAL_GRIP, time.monotonic())
        transport.String = lambda data: types.SimpleNamespace(data=data)
        messages = []
        class Publisher:
            def Write(self, msg):
                messages.append(json.loads(msg.data))
                return False
        transport.body_pub = Publisher()
        transport._publish()
        self.assertIn("DDS publisher failed", transport.latch.fault)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[-1]["base_cmd"][:3], [0, 0, 0])


class Protocol(unittest.TestCase):
    def test_observation_serialization_without_sockets(self):
        images = {name: np.full((480, 640, 3), [200, 100, 30], np.uint8)
                  for name in ("head", "left_wrist", "right_wrist")}
        payload = encode_payload(images, np.arange(29), np.arange(23), "carrying box")
        decoded = decode_observation(json.loads(json.dumps(payload)))
        np.testing.assert_equal(decoded["observation.state"], np.arange(29))
        np.testing.assert_equal(decoded["observation.state.wbc"], np.arange(23))
        np.testing.assert_allclose(decoded["observation.images.head"][100, 100], [200, 100, 30], atol=3)
        self.assertEqual(decoded["task"], "carrying box")
        payload["state"] = [float("nan")] * 29
        with self.assertRaises(ValueError):
            decode_observation(payload)

    def test_http_round_trip_with_mock_backend(self):
        require_tcp(self)
        # Exercises real HTTP/JPEG/JSON transport; explicitly does not test the model.
        class Backend:
            def schema(self):
                return {"protocol": "g1-wbc-v1", "action_shape": [32, 23],
                        "action_names": list(ACTION_NAMES), "action_fps": 30,
                        "action_min": [-1] * 23, "action_max": [1] * 23}

            def infer(self, payload):
                obs = decode_observation(payload)
                if obs["observation.state"].shape != (29,):
                    raise ValueError("Wrong state")
                return {"actions": np.zeros((32, 23)).tolist(), "inference_s": 0.01}

        server = make_server(("127.0.0.1", 0), Backend())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            self.assertTrue(http_json(url + "/health")["ready"])
            schema = http_json(url + "/schema")
            validate_schema(schema)
            images = {name: np.full((480, 640, 3), 128, np.uint8) for name in ("head", "left_wrist", "right_wrist")}
            payload = encode_payload(images, np.arange(29), np.arange(23), "carrying box")
            decoded = decode_observation(payload)
            self.assertEqual(decoded["task"], "carrying box")
            self.assertEqual(decoded["observation.state.wbc"].shape, (23,))
            result = http_json(url + "/infer", payload)
            validate_chunk(result["actions"], schema["action_min"], schema["action_max"])
            bad = copy.deepcopy(payload)
            bad["state"] = [float("nan")] * 29
            with self.assertRaises(ValueError):
                decode_observation(bad)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

    def test_camera_reader_real_zmq_and_expiry(self):
        require_tcp(self)
        import zmq

        context = zmq.Context()
        pub = context.socket(zmq.PUB)
        port = pub.bind_to_random_port("tcp://127.0.0.1")
        cfg = load_config(ROOT / "config/deploy.example.json")["camera"]
        cfg['backend'] = 'zmq'
        cfg["address"] = f"tcp://127.0.0.1:{port}"
        reader = CameraReader(cfg)
        _, jpg = cv2.imencode(".jpg", np.full((480, 2560, 3), 100, np.uint8))
        try:
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                pub.send(b"12345678" + jpg.tobytes())
                time.sleep(0.03)
                try:
                    images, _ = reader.snapshot(0.2)
                    break
                except RuntimeError:
                    pass
            else:
                self.fail("Camera reader did not receive test frame")
            self.assertEqual(images["head"].shape, (480, 640, 3))
            reader.close()
            time.sleep(0.03)
            with self.assertRaises(RuntimeError):
                reader.snapshot(0.001)
        finally:
            reader.close()
            pub.close(linger=0)
            context.term()

    def test_readonly_transport_never_creates_publishers(self):
        from g1_deploy.transport import RobotTransport

        channel = types.ModuleType("unitree_sdk2py.core.channel")
        channel.ChannelFactoryInitialize = lambda *a: None
        channel.ChannelPublisher = lambda *a: self.fail("Read-only mode created a publisher")
        class Subscriber:
            def __init__(self, *args):
                pass
            def Init(self, *args):
                pass
            def Close(self):
                pass
        channel.ChannelSubscriber = Subscriber
        messages = types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg.dds_")
        messages.LowState_ = object
        modules = {"unitree_sdk2py.core.channel": channel, "unitree_sdk2py.idl.unitree_hg.msg.dds_": messages}
        with patch.dict("sys.modules", modules):
            robot = RobotTransport(load_config(ROOT / "config/deploy.example.json"), execute=False)
            robot.set_command({}, [])  # intentional: must remain a no-op in dry-run
            self.assertIsNone(robot.thread)
            robot.close()

    def test_dry_run_loop_with_mock_robot_camera_and_policy(self):
        instances = []
        class Robot:
            def __init__(self, config, execute=False):
                self.execute, self.closed, self.commands = execute, False, 0
                self.sent = []
                instances.append(self)
            def wait_ready(self):
                pass
            def snapshot(self):
                return np.zeros(29), np.zeros(29), time.monotonic()
            def set_command(self, command, grip):
                self.commands += 1
                self.sent.append(command)
            def close(self):
                self.closed = True
        class Camera:
            def __init__(self, config):
                pass
            def snapshot(self, max_age):
                return {name: np.zeros((480, 640, 3), np.uint8)
                        for name in ("head", "left_wrist", "right_wrist")}, time.monotonic()
            def close(self):
                pass
        class IK:
            def __init__(self, repo):
                pass
            def state(self, q):
                return np.zeros(12)
            def command(self, action, q, dq, previous_q, previous_base, config, dt):
                return {"arm_q": q.tolist(), "arm_tau": [0] * 14,
                        "base_cmd": config["initial_base_cmd"]}, NEUTRAL_GRIP.copy()
        def policy(url, payload=None, timeout=None):
            if url.endswith("/schema"):
                return {"protocol": "g1-wbc-v1", "action_shape": [32, 23], "action_fps": 30,
                        "action_names": list(ACTION_NAMES), "action_min": [-1] * 23, "action_max": [1] * 23}
            obs = decode_observation(payload)
            self.assertEqual(obs["observation.state"].shape, (29,))
            return {"actions": np.zeros((32, 23)).tolist(), "inference_s": 0.001}
        with patch("g1_deploy.transport.RobotTransport", Robot), \
             patch("g1_deploy.wbc_client.CameraReader", Camera), \
             patch("g1_deploy.wbc_client.Kinematics", IK), \
             patch("g1_deploy.wbc_client.http_json", policy):
            with tempfile.TemporaryDirectory() as root:
                config = load_config(ROOT / "config/deploy.example.json")
                config['rollout_dir'] = root
                run(config, "dry-run", duration=0.2)
                folder = next(Path(root).iterdir())
                events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
                responses = {e['request_id'] for e in events if e['event'] == 'policy_response'}
                commands = [e for e in events if e['event'] == 'dry_run_command']
                self.assertTrue(commands)
                self.assertTrue(all(e['request_id'] in responses for e in commands))
                self.assertFalse(any(e['event'] == 'command_submitted' for e in events))
                self.assertEqual(events[-1]['reason'], 'duration_complete')
                self.assertTrue(json.loads((folder / 'summary.json').read_text())['flush_complete'])
        self.assertFalse(instances[0].execute)
        self.assertTrue(instances[0].closed)
        self.assertGreater(instances[0].commands, 0)
        # A dry-run must also detect a command-production stall; it cannot
        # claim readiness solely because it creates no DDS command publisher.
        config = load_config(ROOT / "config/deploy.example.json")
        config["command_timeout_s"] = 0.001  # less than the 30 Hz loop interval
        with patch("g1_deploy.transport.RobotTransport", Robot), \
             patch("g1_deploy.wbc_client.CameraReader", Camera), \
             patch("g1_deploy.wbc_client.Kinematics", IK), \
             patch("g1_deploy.wbc_client.http_json", policy):
            with self.assertRaisesRegex(RuntimeError, "Dry-run watchdog: Command heartbeat expired"):
                run(config, "dry-run", duration=0.3)
        self.assertFalse(instances[-1].execute)
        self.assertTrue(instances[-1].closed)

        class DriftIK(IK):
            calls = 0
            def command(self, *args):
                command, grip = super().command(*args)
                DriftIK.calls += 1
                if DriftIK.calls == 1:
                    command["arm_q"] = [1.0] * 14  # mismatch found by post-IK check
                return command, grip
        with patch("g1_deploy.transport.RobotTransport", Robot), \
             patch("g1_deploy.wbc_client.CameraReader", Camera), \
             patch("g1_deploy.wbc_client.Kinematics", DriftIK), \
             patch("g1_deploy.wbc_client.http_json", policy):
            run(load_config(ROOT / "config/deploy.example.json"), "execute", duration=0.3)
        self.assertGreater(DriftIK.calls, 1)
        self.assertGreater(instances[-1].commands, 0)
        self.assertEqual(instances[-1].commands, DriftIK.calls - 1)
        self.assertTrue(all(command["arm_q"] == [0.0] * 14 for command in instances[-1].sent))
        self.assertTrue(instances[-1].closed)

        class TrackingIK(IK):
            calls = 0
            def command(self, *args):
                TrackingIK.calls += 1
                if TrackingIK.calls == 1:
                    raise ArmTrackingError("Arm command too far from measured joint state")
                return super().command(*args)
        with patch("g1_deploy.transport.RobotTransport", Robot), \
             patch("g1_deploy.wbc_client.CameraReader", Camera), \
             patch("g1_deploy.wbc_client.Kinematics", TrackingIK), \
             patch("g1_deploy.wbc_client.http_json", policy):
            run(load_config(ROOT / "config/deploy.example.json"), "execute", duration=0.3)
        self.assertGreater(instances[-1].commands, 0)
        self.assertEqual(instances[-1].commands, TrackingIK.calls - 1)
        self.assertTrue(instances[-1].closed)

        class NonconvergentIK(IK):
            calls = 0
            def command(self, *args):
                NonconvergentIK.calls += 1
                if NonconvergentIK.calls in (1, 3, 4):
                    raise IKConvergenceError("Reference IK did not converge")
                return super().command(*args)
        with patch("g1_deploy.transport.RobotTransport", Robot), \
             patch("g1_deploy.wbc_client.CameraReader", Camera), \
             patch("g1_deploy.wbc_client.Kinematics", NonconvergentIK), \
             patch("g1_deploy.wbc_client.http_json", policy):
            run(load_config(ROOT / "config/deploy.example.json"), "execute", duration=0.5)
        self.assertGreater(NonconvergentIK.calls, 4)
        self.assertEqual(instances[-1].commands, NonconvergentIK.calls - 3)
        self.assertTrue(instances[-1].closed)

        class BrokenIK(IK):
            def command(self, *args):
                raise RuntimeError("Reference IK did not converge")
        with patch("g1_deploy.transport.RobotTransport", Robot), \
             patch("g1_deploy.wbc_client.CameraReader", Camera), \
             patch("g1_deploy.wbc_client.Kinematics", BrokenIK), \
             patch("g1_deploy.wbc_client.http_json", policy):
            with self.assertRaisesRegex(RuntimeError, "Reference IK did not converge"):
                run(load_config(ROOT / "config/deploy.example.json"), "execute", duration=0.3)
        self.assertEqual(instances[-1].commands, 0)
        self.assertTrue(instances[-1].closed)


if __name__ == "__main__":
    unittest.main()
