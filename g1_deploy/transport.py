"""Unitree DDS transport; read-only unless execute=True.

Wire schema and Dex1 trigger behavior follow wbc_teleoperation. This module does
not start the on-robot WBC service or switch the robot into low-level mode.
"""
from __future__ import annotations

import json
import struct
import threading
import time
from collections import deque

import numpy as np

from .core import CommandLatch, gripper_target, vector, slew


class RobotTransport:
    def __init__(self, config, execute=False):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

        self.config = config
        self.execute = execute
        self.lock = threading.Lock()
        self.q = self.dq = None
        self.state_time = 0.0
        self.remote_stop = False
        self.gripper_q = np.full(2, np.nan)
        self.gripper_time = np.zeros(2)
        self.latch = CommandLatch(config["command_timeout_s"],
                                  config.get("command_heartbeat_enabled", True),
                                  config.get("gripper_command_mode", "controls"))
        self.closed = threading.Event()
        self.thread = None
        self.subscribers = []
        self.publishers = []
        ChannelFactoryInitialize(0, config["interface"])
        sub = ChannelSubscriber(config["lowstate_topic"], LowState_)
        self.subscribers.append(sub)
        sub.Init(self._on_state, 10)
        if execute:
            from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

            self.String = String_
            self.body_pub = ChannelPublisher(config["command_topic"], String_)
            self.body_pub.Init()
            self.publishers.append(self.body_pub)
            self.grip_pubs, self.grip_messages = [], []
            for i, side in enumerate(("left", "right")):
                publisher = ChannelPublisher(f"rt/dex1/{side}/cmd", MotorCmds_)
                publisher.Init()
                self.publishers.append(publisher)
                self.grip_pubs.append(publisher)
                message = MotorCmds_()
                message.cmds = [unitree_go_msg_dds__MotorCmd_()]
                cmd = message.cmds[0]
                cmd.dq, cmd.tau, cmd.kp, cmd.kd = 0.0, 0.0, 5.0, 0.05
                self.grip_messages.append(message)
        if execute or config.get("subscribe_gripper_state", False):
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
            for i, side in enumerate(("left", "right")):
                sub = ChannelSubscriber(f"rt/dex1/{side}/state", MotorStates_)
                sub.Init(lambda msg, index=i: self._on_gripper(msg, index), 10)
                self.subscribers.append(sub)

    def _on_state(self, msg):
        q = np.array([m.q for m in msg.motor_state[:29]], dtype=np.float64)
        dq = np.array([m.dq for m in msg.motor_state[:29]], dtype=np.float64)
        if q.shape != (29,) or not np.isfinite(q).all() or not np.isfinite(dq).all():
            return
        remote = bytes(msg.wireless_remote)
        stop = len(remote) >= 4 and bool(struct.unpack("<H", remote[2:4])[0] & (1 << 9))
        with self.lock:
            self.q, self.dq, self.state_time = q, dq, time.monotonic()
            self.remote_stop = bool(stop)
            if stop:
                self.latch.stop("Remote B stop")

    def _on_gripper(self, msg, index):
        if not msg.states or not np.isfinite(msg.states[0].q):
            return
        with self.lock:
            self.gripper_q[index] = msg.states[0].q
            self.gripper_time[index] = time.monotonic()

    def wait_ready(self):
        deadline = time.monotonic() + self.config["startup_timeout_s"]
        while time.monotonic() < deadline:
            try:
                self.snapshot()
                with self.lock:
                    grip_ready = (time.monotonic() - self.gripper_time < self.config["max_state_age_s"]).all()
                    if not self.config.get("robot_state_expiry_enabled", True):
                        grip_ready = bool((self.gripper_time > 0).all())
                if (not self.execute and not self.config.get("subscribe_gripper_state", False)) or grip_ready:
                    return
            except RuntimeError:
                pass
            time.sleep(0.02)
        raise TimeoutError("No fresh DDS robot/gripper state. Check interface, DDS and Dex1 service.")

    def snapshot(self):
        with self.lock:
            if self.latch.fault:
                raise RuntimeError(self.latch.fault)
            if self.q is None or (self.config.get("robot_state_expiry_enabled", True)
                                 and time.monotonic() - self.state_time > self.config["max_state_age_s"]):
                raise RuntimeError("LowState is missing or stale")
            return self.q.copy(), self.dq.copy(), self.state_time

    def set_command(self, command, grip):
        if not self.execute:
            return
        with self.lock:
            self.latch.update(command, grip, time.monotonic())
        if self.thread is None:
            self.thread = threading.Thread(target=self._publish, daemon=True, name="dds-command")
            self.thread.start()

    def snapshot_grippers(self):
        with self.lock:
            if self.latch.fault:
                raise RuntimeError(self.latch.fault)
            if (not np.isfinite(self.gripper_q).all()
                    or np.any(time.monotonic() - self.gripper_time > self.config["max_state_age_s"])):
                raise RuntimeError("Dex1 state is missing or stale")
            return self.gripper_q.copy(), float(np.min(self.gripper_time))

    def _publish(self):
        # Body command: 100 Hz. Dex1 incremental controller: 200 Hz, as in reference.
        targets = None
        history = deque(maxlen=3)
        tick = 0
        deadline = time.monotonic()
        try:
            while not self.closed.is_set():
                now = time.monotonic()
                with self.lock:
                    fresh = now - self.state_time <= self.config["max_state_age_s"]
                    fresh = fresh and bool((now - self.gripper_time <= self.config["max_state_age_s"]).all())
                    if not self.config.get("robot_state_expiry_enabled", True):
                        fresh = self.q is not None and bool((self.gripper_time > 0).all())
                    command, grip = self.latch.sample(now, fresh, self.remote_stop)
                    positions = self.gripper_q.copy()
                    stopped = self.latch.fault is not None
                if command is not None:
                    if tick % 2 == 0:
                        if self.body_pub.Write(self.String(data=json.dumps(command, allow_nan=False))) is False:
                            raise RuntimeError("WBC DDS write failed")
                    if targets is None:
                        targets = positions.copy()
                    if not stopped:
                        if self.config.get("gripper_command_mode", "controls") == "position":
                            desired = np.clip(vector(grip, 2, "Dex1 position"),
                                              self.config["gripper_min"], self.config["gripper_max"])
                            targets = slew(targets, desired, self.config["gripper_speed_per_s"], 0.005)
                        else:
                            targets = np.array([
                                gripper_target(positions[i], targets[i], grip[2 * i], grip[2 * i + 1])
                                for i in range(2)
                            ])
                    if not history or not np.array_equal(history[-1], targets):
                        history.append(targets.copy())
                    filtered = targets if len(history) < 3 else np.array([0.2, 0.3, 0.5]) @ np.array(history)
                    for i in range(2):
                        self.grip_messages[i].cmds[0].q = float(filtered[i])
                        if self.grip_pubs[i].Write(self.grip_messages[i]) is False:
                            raise RuntimeError(f"Dex1 DDS write failed on side {i}")
                tick += 1
                deadline = max(deadline + 0.005, time.monotonic())
                self.closed.wait(max(0.0, deadline - time.monotonic()))
        except Exception as exc:
            with self.lock:
                self.latch.stop(f"DDS publisher failed: {exc}")
                held, _ = self.latch.sample(time.monotonic(), False)
            # A transient writer failure must not leave the last velocity as our
            # final intentional message. This is best effort, not a DDS delivery guarantee.
            if held is not None:
                try:
                    self.body_pub.Write(self.String(data=json.dumps(held, allow_nan=False)))
                except Exception:
                    pass

    def close(self):
        with self.lock:
            self.latch.stop("Client stopped")
        if self.thread is not None:
            # Deliver zero velocity + held arms before stopping publication.
            # Hardware emergency stop / WBC timeout remains the final stop mechanism.
            time.sleep(0.3)
        self.closed.set()
        if self.thread is not None:
            self.thread.join(timeout=1)
        for resource in self.subscribers + self.publishers:
            close = getattr(resource, "Close", None)
            if close:
                close()
