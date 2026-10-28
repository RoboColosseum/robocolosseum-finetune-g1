"""Receive ROS 2 compressed cameras directly over DDS, without rclpy."""
import math
import threading
import time

import numpy as np

from .core import CAMERAS

DEFAULT_TOPICS = {name: f"/camera/{name}/image/compressed" for name in CAMERAS}


def decode_image(data, name, config):
    import cv2

    if not data:
        raise ValueError(f"{name}: empty compressed image")
    image = cv2.imdecode(np.frombuffer(bytes(data), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{name}: cannot decode compressed image")
    mode = config.get("head_mode", "left_eye")
    if name == "head":
        if mode not in ("left_eye", "stereo_squash", "stereo_full"):
            raise ValueError(f"Unknown head_mode: {mode}")
        if image.shape == (480, 1280, 3):
            if mode == "left_eye":
                image = image[:, :640]
            elif mode == "stereo_squash":
                image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
        elif image.shape != (480, 640, 3) or mode != "left_eye":
            raise ValueError(f"head: {mode} requires {'640 or 1280' if mode == 'left_eye' else '1280'}x480 input; got {image.shape}")
    elif image.shape != (480, 640, 3):
        raise ValueError(f"{name}: expected 640x480; got {image.shape}")
    return np.ascontiguousarray(image[:, :, ::-1])


class FrameBuffer:
    """Latest frame per topic; a fresh camera cannot mask another stale camera."""
    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        self.frames = {}
        self.errors = {}
        self.max_skew = float(config.get("max_skew_s", 0.15))
        if not math.isfinite(self.max_skew) or self.max_skew <= 0:
            raise ValueError("camera.max_skew_s must be positive and finite")

    def update(self, name, message, received=None):
        received = time.monotonic() if received is None else received
        try:
            if 'compressedDepth' in message.format:
                raise ValueError("depth image is not an RGB observation")
            image = decode_image(message.data, name, self.config)
            stamp = message.header.stamp.sec * 1000000000 + message.header.stamp.nanosec
            with self.lock:
                previous = self.frames.get(name)
                if previous and stamp > 0 and stamp <= previous[2]:
                    return  # repeated/out-of-order source frames cannot refresh receive age
                self.frames[name] = (image, received, stamp)
                self.errors.pop(name, None)
        except Exception as exc:
            with self.lock:
                self.errors[name] = str(exc)

    def snapshot(self, max_age):
        with self.lock:
            if self.errors:
                raise RuntimeError(f"DDS camera decode: {self.errors}")
            now = time.monotonic()
            missing = [n for n in CAMERAS if n not in self.frames or now - self.frames[n][1] > max_age]
            if missing:
                ages = {n: round(now - self.frames[n][1], 3) if n in self.frames else None for n in CAMERAS}
                raise RuntimeError(f"DDS camera frames missing or stale: {', '.join(missing)}; "
                                   f"max_age_s={max_age}; receive_age_s={ages}")
            times = [self.frames[n][1] for n in CAMERAS]
            if max(times) - min(times) > self.max_skew:
                ages = {n: round((now - self.frames[n][1]) * 1000, 1) for n in CAMERAS}
                stamps = {n: self.frames[n][2] for n in CAMERAS}
                raise RuntimeError(
                    f"DDS camera receive-time skew {(max(times) - min(times)) * 1000:.1f} ms "
                    f"exceeds {self.max_skew * 1000:.1f} ms; receive_age_ms={ages}; source_stamp_ns={stamps}"
                )
            return {n: self.frames[n][0].copy() for n in CAMERAS}, min(times)


class DDSCameraReader(FrameBuffer):
    def __init__(self, config):
        super().__init__(config)
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.topic import Topic
        from cyclonedds.sub import DataReader
        from cyclonedds.qos import Qos, Policy
        from .dds_image_types import CompressedImage

        domain = config.get("domain_id", 0)
        if type(domain) is not int or not 0 <= domain <= 232:
            raise ValueError("camera.domain_id must be an integer from 0 to 232")
        topics = dict(DEFAULT_TOPICS, **config.get("topics", {}))
        if any(not isinstance(topics[n], str) or not topics[n].startswith('/') for n in CAMERAS):
            raise ValueError("DDS camera topics must be absolute ROS topic names")
        if len({topics[n] for n in CAMERAS}) != len(CAMERAS):
            raise ValueError("DDS camera topics must be distinct")
        # Domain 0 inherits the interface selected by RobotTransport before us.
        self.participant = DomainParticipant(domain)
        self.topics = [Topic(self.participant, 'rt' + topics[n], CompressedImage) for n in CAMERAS]
        qos = Qos(Policy.Reliability.BestEffort, Policy.Durability.Volatile,
                  Policy.History.KeepLast(1))
        self.readers = [DataReader(self.participant, topic, qos=qos) for topic in self.topics]
        self.closed = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name="dds-camera", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.closed.is_set():
                for name, reader in zip(CAMERAS, self.readers):
                    samples = reader.take(1)
                    if samples and samples[-1].sample_info.valid_data:
                        self.update(name, samples[-1])
                self.closed.wait(0.002)
        except Exception as exc:
            self.error = str(exc)

    def snapshot(self, max_age):
        if self.error:
            raise RuntimeError(f"DDS camera reader: {self.error}")
        return super().snapshot(max_age)

    def close(self):
        self.closed.set()
        self.thread.join()
        self.readers.clear()
        self.topics.clear()
        self.participant = None


def main():
    """Camera-only diagnostic: no robot state reader or command publisher."""
    import argparse
    import json
    from pathlib import Path
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/robot.example.json')
    parser.add_argument('--interface', default=None)
    parser.add_argument('--duration', type=float, default=10)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error('--duration must be positive and finite')
    config = json.loads(Path(args.config).read_text())
    camera = config['camera']
    ChannelFactoryInitialize(camera.get('domain_id', 0), args.interface or config['interface'])
    reader = DDSCameraReader(camera)
    valid = 0
    last_error = None
    try:
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            try:
                images, stamp = reader.snapshot(min(config['max_camera_age_s'], 0.5))
                valid += 1
                last_error = None
                print(json.dumps({'shapes': {n: list(v.shape) for n, v in images.items()},
                                  'oldest_receive_age_s': time.monotonic() - stamp}))
            except RuntimeError as exc:
                last_error = str(exc)
            time.sleep(0.5)
        print(json.dumps({'valid_snapshots': valid, 'last_error': last_error,
                          'received_topics': list(reader.frames)}))
    finally:
        reader.close()
    if not valid:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
