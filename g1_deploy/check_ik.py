"""Synthetic FK -> IK check against the original WBC model; no DDS imports."""
import argparse
import json
import time

import numpy as np

from .core import NEUTRAL_GRIP
from .wbc_client import Kinematics, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/deploy.example.json")
    args = parser.parse_args()
    config = load_config(args.config)
    kinematics = Kinematics(config["wbc_repo"])
    q = np.zeros(14)
    q[3] = q[10] = 0.5
    base = np.asarray(config["initial_base_cmd"], dtype=float)
    times = []
    for _ in range(12):
        pose = kinematics.state(q)
        pose[0] += 0.002
        pose[6] += 0.002
        started = time.monotonic()
        command, _ = kinematics.command(np.r_[base, pose, NEUTRAL_GRIP], q, np.zeros(14),
                                         q, base, config, 1 / 30)
        times.append(time.monotonic() - started)
        next_q = np.asarray(command["arm_q"])
        if np.max(np.abs(next_q - q)) > config["arm_speed_rad_s"] / 30 + 1e-9:
            raise RuntimeError("Joint slew limit violated")
        q = next_q
    print(json.dumps({"iterations": 12, "median_s": float(np.median(times)), "max_s": max(times),
                      "validation": "synthetic FK/IK only; no robot, no model inference"}, indent=2))


if __name__ == "__main__":
    main()
