"""Send a measured NPZ observation to the policy server; never opens DDS."""
import argparse
import json
from pathlib import Path

import numpy as np

from .core import CAMERAS, validate_chunk
from .wbc_client import http_json, validate_schema
import base64


def main():
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8008")
    parser.add_argument("--task", default="carrying box")
    parser.add_argument("--output", default="outputs/action_chunk.json")
    parser.add_argument("--timeout", type=float, default=120, help="Allows cold model warm-up")
    parser.add_argument("--config", help="Client configuration with action_overshoot_fraction")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text()) if args.config else {}
    url = args.url.rstrip("/")
    schema = http_json(url + "/schema", timeout=args.timeout)
    validate_schema(schema)
    with np.load(args.observation, allow_pickle=False) as obs:
        images = {}
        for name in CAMERAS:
            image = obs[name]
            width = 1280 if name == "head" and schema.get("head_mode") == "stereo_full" else 640
            if image.shape != (480, width, 3) or image.dtype != np.uint8:
                raise ValueError(f"{name} must be RGB uint8 480x{width}")
            ok, jpg = cv2.imencode(".jpg", image[:, :, ::-1])
            if not ok:
                raise ValueError(f"Cannot encode {name}")
            images[name] = base64.b64encode(jpg).decode()
        payload = {"state": obs["state"].tolist(), "images": images, "task": args.task}
        if "wbc_state" in obs:
            payload["wbc_state"] = obs["wbc_state"].tolist()
    result = http_json(url + "/infer", payload, args.timeout)
    raw = result["actions"]
    bounded = validate_chunk(raw, schema["action_min"], schema["action_max"], schema["action_shape"][0],
                             overshoot_fraction=config.get("action_overshoot_fraction", 0.10))
    result["raw_actions"] = raw
    result["actions"] = bounded.tolist()
    result["clipped_values"] = int(np.count_nonzero(bounded != np.asarray(raw)))
    result["action_overshoot_fraction"] = config.get("action_overshoot_fraction", 0.10)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Validated {schema['action_shape']} action chunk; inference={result['inference_s']:.3f}s; saved {output}")


if __name__ == "__main__":
    main()
