"""HTTP GR00T server. Run this module in the existing LeRobot N1.7 environment."""
from __future__ import annotations

import argparse
import dataclasses
import tempfile
import base64
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .core import ACTION_NAMES, CAMERAS, read_stats, vector

LOG = logging.getLogger(__name__)
MAX_BODY = 8 * 1024 * 1024


def decode_observation(payload, head_mode="left_eye"):
    import cv2

    state = vector(payload["state"], 29, "state").astype(np.float32)
    task = payload.get("task", "carrying box")
    if not isinstance(task, str) or not task.strip() or len(task) > 1000:
        raise ValueError("task must be a nonempty string, at most 1000 characters")
    observation = {"observation.state": state, "task": task}
    if "wbc_state" in payload:
        observation["observation.state.wbc"] = vector(payload["wbc_state"], 23, "wbc_state").astype(np.float32)
    for name in CAMERAS:
        raw = base64.b64decode(payload["images"][name], validate=True)
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        width = 1280 if name == "head" and head_mode == "stereo_full" else 640
        if image is None or image.shape != (480, width, 3):
            raise ValueError(f"{name} must decode as 480x{width} RGB")
        observation[f"observation.images.{name}"] = np.ascontiguousarray(image[:, :, ::-1])
    return observation


class GrootBackend:
    def __init__(self, model_path, device, profile="carry-box"):
        if profile not in ("carry-box", "plate-into-rack", "load-clothes-washing-machine"):
            raise ValueError("Unknown checkpoint profile")
        self.profile = profile
        self.length = 40 if profile in ("plate-into-rack", "load-clothes-washing-machine") else 32
        self.head_mode = "stereo_squash" if profile in ("plate-into-rack", "load-clothes-washing-machine") else "left_eye"
        self.action_order = list(range(23))
        if profile in ("plate-into-rack", "load-clothes-washing-machine"):
            # Dataset metadata declares pivot roll,pitch,yaw; WBC uses yaw,pitch,roll.
            self.action_order[3], self.action_order[5] = 5, 3
        import torch
        from lerobot.policies.groot.configuration_groot import GrootConfig
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.torch = torch
        path = Path(model_path).resolve()
        raw = json.loads((path / "config.json").read_text())
        # Adapt newer dtype serialization to the installed LeRobot API, without
        # modifying downloaded checkpoint files or dropping unknown fields.
        if "dtype" in raw and "dtype" not in {f.name for f in dataclasses.fields(GrootConfig)}:
            if raw["dtype"] != "float32":
                raise ValueError("Only float32 checkpoint dtype compatibility is supported")
            raw.pop("dtype")
            raw["model_params_fp32"] = True
        with tempfile.TemporaryDirectory(prefix="g01-config-") as temp:
            (Path(temp) / "config.json").write_text(json.dumps(raw))
            cfg = GrootConfig.from_pretrained(temp)
        if cfg.embodiment_tag != "new_embodiment" or cfg.chunk_size != self.length:
            raise ValueError("Checkpoint embodiment/chunk size does not match selected profile")
        if tuple(cfg.output_features["action"].shape) != (23,):
            raise ValueError("Checkpoint must output 23-dimensional actions")
        expected_inputs = {f"observation.images.{name}": (3, 480, 640) for name in CAMERAS}
        if self.head_mode == "stereo_squash":
            expected_inputs["observation.images.head"] = (3, 480, 1280)
        expected_inputs.update({"observation.state": (29,), "observation.state.wbc": (23,)})
        if {name: tuple(feature.shape) for name, feature in cfg.input_features.items()} != expected_inputs:
            raise ValueError("Checkpoint observation schema does not match this bridge")
        if cfg.use_relative_actions:
            raise ValueError("This bridge expects the inspected non-relative-action checkpoint")
        cfg.device = device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable in this Python environment")
        self.policy = GrootPolicy.from_pretrained(str(path), config=cfg).to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=str(path),
            preprocessor_overrides={
                "device_processor": {"device": device},
                "groot_n1_7_vlm_encode_v1": {"device": device},
            },
        )
        # Upstream creates the Cosmos tokenizer/image/video processors lazily.
        # Resolve them before reporting ready, so missing assets fail at startup
        # rather than during the first /infer request (including offline mode).
        from lerobot.policies.groot.processor_groot import GrootN17VLMEncodeStep
        for step in self.pre.steps:
            if isinstance(step, GrootN17VLMEncodeStep):
                _ = step.proc
        stats = read_stats(path / "policy_postprocessor_step_0_groot_action_unpack_unnormalize_v2.safetensors")
        self.minimum = stats["action.min"][self.action_order]
        self.maximum = stats["action.max"][self.action_order]
        self.lock = threading.Lock()

    def schema(self):
        return {
            "protocol": "g1-wbc-v1", "action_shape": [self.length, 23], "action_fps": 30,
            "action_names": list(ACTION_NAMES),
            "action_min": self.minimum.tolist(), "action_max": self.maximum.tolist(),
            "mapping_status": ("dataset metadata RPY converted to WBC YPR; physical semantics unverified"
                               if self.profile in ("plate-into-rack", "load-clothes-washing-machine") else
                               "inferred from wbc_teleoperation and checkpoint statistics"),
            "profile": self.profile, "head_mode": self.head_mode,
            "state_shape": [29], "cameras": list(CAMERAS),
            "action_bounds_enforcement": "client",
            "action_output": "raw finite postprocessed actions",
        }

    def validate_actions(self, actions):
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.length, 23) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite action chunk ({self.length}, 23), got {actions.shape}")
        return actions

    def image_tensor(self, value):
        return self.torch.from_numpy(value).permute(2, 0, 1).float() / 255.0

    def infer(self, payload):
        obs = decode_observation(payload, self.head_mode)
        if "observation.state.wbc" in obs:
            obs["observation.state.wbc"] = obs["observation.state.wbc"][self.action_order]
        batch = {}
        for key, value in obs.items():
            if key.startswith("observation.images."):
                batch[key] = self.image_tensor(value)
            elif key.startswith("observation.state"):
                batch[key] = self.torch.from_numpy(value)
            else:
                batch[key] = value
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError("Another inference is already running")
        try:
            started = time.monotonic()
            with self.torch.inference_mode():
                processed = self.pre(batch)
                # The inspected upstream processor uses state[29], padded to 132.
                # observation.state.wbc is retained in the payload, never concatenated.
                # Normalize/denormalize statistics in FP32 even if model output is
                # BF16; otherwise min/max endpoints can drift outside physical bounds.
                actions = self.post(self.policy.predict_action_chunk(processed).float())
                actions = actions.detach().float().cpu().numpy()
            if actions.shape != (1, self.length, 23):
                raise ValueError(f"Unexpected model output {actions.shape}")
            actions = self.validate_actions(actions[0][:, self.action_order])
            return {"actions": actions.tolist(), "inference_s": time.monotonic() - started}
        finally:
            self.lock.release()


def make_server(address, backend):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def respond(self, status, data):
            body = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.respond(200, {"ready": True, "protocol": "g1-wbc-v1"})
            elif self.path == "/schema":
                self.respond(200, backend.schema())
            else:
                self.respond(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/infer":
                self.respond(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                self.respond(200, backend.infer(payload))
            except BlockingIOError as exc:
                self.respond(409, {"error": str(exc)})
            except (KeyError, ValueError, TypeError) as exc:
                self.respond(400, {"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                LOG.warning("Client disconnected")
            except Exception:
                LOG.exception("Inference failed")
                self.respond(500, {"error": "Inference failed; inspect policy server log"})

        def log_message(self, format, *args):
            LOG.info(format, *args)

    return ThreadingHTTPServer(address, Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/groot-n17-g1-carry-box-40k")
    parser.add_argument("--profile", choices=["carry-box", "plate-into-rack", "load-clothes-washing-machine"], default="carry-box")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    backend = GrootBackend(args.model, args.device, args.profile)
    server = make_server((args.host, args.port), backend)
    LOG.info("Model loaded. Listening on %s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
