"""Read-only dependency check; does not initialize DDS, cameras, or a robot."""
import argparse
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("policy", "wbc"))
    parser.add_argument("--config", default="config/deploy.example.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config.get("sdk_path"):
        sys.path.insert(0, config["sdk_path"])
    modules = ["numpy", "scipy", "cv2"]
    modules += (["torch", "lerobot.policies.groot.modeling_groot", "lerobot.policies.factory"]
                if args.role == "policy" else ["zmq", "unitree_sdk2py.core.channel", "pinocchio", "casadi", "meshcat", "logging_mp"])
    print(f"Python: {sys.executable} ({sys.version.split()[0]})")
    failed = False
    for name in modules:
        try:
            module = importlib.import_module(name)
            print(f"OK   {name}: {getattr(module, '__file__', '')}")
        except Exception as exc:
            failed = True
            print(f"FAIL {name}: {exc}")
    if args.role == "policy":
        try:
            import torch

            print("LeRobot:", importlib.metadata.version("lerobot"))
            print("CUDA available:", torch.cuda.is_available())
            failed |= not torch.cuda.is_available()
            model = Path(config["model_path"])
            for name in ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json",
                         "policy_preprocessor_step_2_groot_n1_7_pack_inputs_v1.safetensors",
                         "policy_postprocessor_step_0_groot_action_unpack_unnormalize_v2.safetensors"):
                exists = (model / name).is_file()
                print("OK  " if exists else "FAIL", model / name)
                failed |= not exists
        except Exception as exc:
            print("FAIL", exc)
            failed = True
    else:
        for name in ("teleop/robot_control/robot_arm_ik.py", "assets/g1/g1_body29_hand14.urdf"):
            path = Path(config["wbc_repo"]) / name
            print("OK  " if path.is_file() else "FAIL", path)
            failed |= not path.is_file()
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
