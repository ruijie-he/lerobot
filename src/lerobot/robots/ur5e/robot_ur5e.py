# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from typing import Any

import numpy as np

try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
except ImportError:
    RTDEControlInterface = None
    RTDEReceiveInterface = None

from lerobot.cameras.utils import make_cameras_from_configs

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .configuration_ur5e import UR5eRobotConfig

logger = logging.getLogger(__name__)

# Joint name mapping from LeRobot convention to UR5e joint indices
# UR5e has 6 joints indexed 0-5: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
UR5E_JOINTS = {
    "shoulder_pan.pos": 0,
    "shoulder_lift.pos": 1,
    "elbow.pos": 2,
    "wrist_1.pos": 3,
    "wrist_2.pos": 4,
    "wrist_3.pos": 5,
}


class UR5eRobot(Robot):
    """
    Universal Robots UR5e collaborative robot arm.

    Supports both URSim (simulator) and real UR5e hardware via RTDE (Real-Time Data Exchange).
    Control modes:
    - Joint space: Send target joint positions
    - Cartesian space: Send target TCP (Tool Center Point) poses

    The UR5e has 6 revolute joints with the following approximate ranges:
    - shoulder_pan: ±360° (continuous rotation)
    - shoulder_lift: ±360° (continuous rotation)
    - elbow: ±360° (continuous rotation)
    - wrist_1: ±360° (continuous rotation)
    - wrist_2: ±360° (continuous rotation)
    - wrist_3: ±360° (continuous rotation)

    TCP (Tool Center Point) is the end-effector pose, represented as [x, y, z, rx, ry, rz]:
    - (x, y, z): Position in meters relative to robot base
    - (rx, ry, rz): Orientation as axis-angle (rotation vector) in radians
    """

    config_class = UR5eRobotConfig
    name = "ur5e"

    def __init__(self, config: UR5eRobotConfig):
        if RTDEControlInterface is None or RTDEReceiveInterface is None:
            raise ImportError(
                "ur_rtde library is required for UR5e robot. "
                "Install with: pip install ur-rtde"
            )

        super().__init__(config)
        self.config = config

        # RTDE interfaces (initialized in connect())
        self.rtde_control: RTDEControlInterface | None = None
        self.rtde_receive: RTDEReceiveInterface | None = None

        # Cameras
        self.cameras = make_cameras_from_configs(config.cameras)

        # Logging
        self.logs: dict[str, float] = {}

    @property
    def observation_features(self) -> dict[str, Any]:
        """Define observation space: joint positions + camera images."""
        return {**self.motors_features, **self.camera_features}

    @property
    def action_features(self) -> dict[str, type]:
        """Define action space: joint positions."""
        return self.motors_features

    @property
    def camera_features(self) -> dict[str, tuple[int | None, int | None, int]]:
        """Camera features: {camera_name: (height, width, channels)}."""
        return {cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras}

    @property
    def motors_features(self) -> dict[str, type]:
        """Motor features: all joints return float positions."""
        return {joint_name: float for joint_name in UR5E_JOINTS.keys()}

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected and RTDE interfaces are active."""
        if self.rtde_control is None or self.rtde_receive is None:
            return False
        return self.rtde_control.isConnected() and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        """UR5e robots are factory calibrated, no runtime calibration needed."""
        return True

    def connect(self, calibrate: bool = False) -> None:
        """
        Connect to UR5e robot via RTDE.

        Args:
            calibrate: Ignored for UR5e (always factory calibrated)
        """
        logger.info(f"Connecting to UR5e at {self.config.ip_address}...")

        # Initialize RTDE interfaces
        try:
            self.rtde_control = RTDEControlInterface(self.config.ip_address)
            self.rtde_receive = RTDEReceiveInterface(self.config.ip_address)
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to UR5e at {self.config.ip_address}. "
                f"Make sure URSim is running and accessible. Error: {e}"
            )

        if not self.is_connected:
            raise ConnectionError(
                f"RTDE connection established but verification failed. "
                f"Check that the robot is powered on and in remote control mode."
            )

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        # Configure robot (move to home if requested)
        self.configure()

        logger.info(f"UR5e connected successfully")

    def configure(self) -> None:
        """Apply robot configuration (move to home position if configured)."""
        if self.config.home_on_connect:
            logger.info("Moving to home position...")
            self._move_to_home()
            logger.info("Robot at home position")

    def calibrate(self) -> None:
        """UR5e robots don't require calibration."""
        pass

    def _move_to_home(self) -> None:
        """Move robot to home position using moveJ (blocking movement)."""
        if self.rtde_control is None:
            raise RuntimeError("Robot not connected")

        # Use moveJ for homing (blocking, smooth movement)
        # Parameters: joint_positions, speed, acceleration, asynchronous
        self.rtde_control.moveJ(
            self.config.home_joint_positions,
            speed=self.config.max_joint_velocity / 2,  # Half max speed for safety
            acceleration=self.config.max_cartesian_acceleration,
            asynchronous=False,  # Wait for movement to complete
        )

    def _check_workspace_limits(self, pose: np.ndarray) -> bool:
        """
        Check if a TCP pose is within workspace limits.

        Args:
            pose: TCP pose [x, y, z, rx, ry, rz]

        Returns:
            True if within limits (or no limits configured), False otherwise
        """
        if self.config.workspace_limits is None:
            return True

        x, y, z = pose[0], pose[1], pose[2]
        x_min, x_max, y_min, y_max, z_min, z_max = self.config.workspace_limits

        if not (x_min <= x <= x_max):
            logger.warning(f"X position {x:.3f} outside limits [{x_min:.3f}, {x_max:.3f}]")
            return False
        if not (y_min <= y <= y_max):
            logger.warning(f"Y position {y:.3f} outside limits [{y_min:.3f}, {y_max:.3f}]")
            return False
        if not (z_min <= z <= z_max):
            logger.warning(f"Z position {z:.3f} outside limits [{z_min:.3f}, {z_max:.3f}]")
            return False

        return True

    def get_observation(self) -> dict[str, np.ndarray]:
        """
        Get current robot state.

        Returns:
            Dictionary with joint positions and camera images:
            - "shoulder_pan.pos", "shoulder_lift.pos", etc.: joint positions in radians
            - camera names: RGB images as numpy arrays
        """
        if not self.is_connected:
            raise ConnectionError("Robot is not connected")

        obs_dict: dict[str, Any] = {}

        # Read joint positions
        before_read_t = time.perf_counter()
        joint_positions = self.rtde_receive.getActualQ()  # Returns list of 6 joint angles in radians

        for joint_name, idx in UR5E_JOINTS.items():
            obs_dict[joint_name] = joint_positions[idx]

        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        # Capture camera images
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send action to robot.

        Supports two control modes:
        1. Cartesian control: action contains "cartesian_pose" key
        2. Joint control: action contains joint position keys (e.g., "shoulder_pan.pos")

        Args:
            action: Dictionary with either:
                - "cartesian_pose": np.ndarray of shape (6,) with [x, y, z, rx, ry, rz]
                - Joint positions: e.g., {"shoulder_pan.pos": 0.0, "shoulder_lift.pos": -1.57, ...}

        Returns:
            The action that was sent (may be modified by safety checks)
        """
        if not self.is_connected:
            raise ConnectionError("Robot is not connected")

        before_write_t = time.perf_counter()

        # Check if this is Cartesian control
        if "cartesian_pose" in action:
            # Cartesian space control using servoL
            pose = np.array(action["cartesian_pose"])

            # Check workspace limits
            if not self._check_workspace_limits(pose):
                logger.warning("Target pose outside workspace limits, skipping action")
                return action

            # Get velocity and acceleration (use defaults if not specified)
            velocity = action.get("cartesian_velocity", self.config.max_cartesian_velocity)
            acceleration = action.get("cartesian_acceleration", self.config.max_cartesian_acceleration)

            # Send servo command
            # servoL: Servo to position (linear in tool space)
            # Parameters: pose, velocity, acceleration, dt, lookahead_time, gain
            self.rtde_control.servoL(
                pose.tolist(),
                velocity,
                acceleration,
                self.config.servo_dt,
                self.config.servo_lookahead_time,
                self.config.servo_gain,
            )

        else:
            # Joint space control using servoJ
            goal_positions = {}

            # Extract goal positions from action
            for key, val in action.items():
                if key in UR5E_JOINTS:
                    goal_positions[key] = float(val)

            if not goal_positions:
                logger.warning("No valid joint positions in action")
                return action

            # Apply safety limits if configured
            if self.config.max_relative_target is not None:
                # Get current positions
                current_q = self.rtde_receive.getActualQ()
                current_positions = {
                    joint_name: current_q[idx] for joint_name, idx in UR5E_JOINTS.items()
                }

                # Build goal_present_pos dict for ensure_safe_goal_position
                goal_present_pos = {
                    key: (goal_positions[key], current_positions[key]) for key in goal_positions
                }

                # Apply safety check
                safe_goal_positions = ensure_safe_goal_position(
                    goal_present_pos, float(self.config.max_relative_target)
                )

                # Update goal positions with safe values
                goal_positions = safe_goal_positions

            # Convert to list in joint order
            q = [goal_positions.get(joint_name, 0.0) for joint_name in UR5E_JOINTS.keys()]

            # Send servo command
            # servoJ: Servo to position (linear in joint space)
            # Parameters: q, speed, acceleration, dt, lookahead_time, gain
            self.rtde_control.servoJ(
                q,
                self.config.max_joint_velocity,
                self.config.max_cartesian_acceleration,
                self.config.servo_dt,
                self.config.servo_lookahead_time,
                self.config.servo_gain,
            )

            # Update action with safe values for return
            for joint_name, idx in UR5E_JOINTS.items():
                if joint_name in action:
                    action[joint_name] = q[idx]

        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t

        return action

    def disconnect(self) -> None:
        """Disconnect from robot and cleanup resources."""
        if self.rtde_control is None:
            logger.warning("Robot already disconnected or never connected")
            return

        logger.info("Disconnecting from UR5e...")

        # Stop servo control
        try:
            self.rtde_control.servoStop()
        except Exception as e:
            logger.warning(f"Error stopping servo: {e}")

        # Disconnect cameras
        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting camera: {e}")

        # Disconnect RTDE interfaces
        try:
            if self.rtde_control:
                self.rtde_control.disconnect()
            if self.rtde_receive:
                self.rtde_receive.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting RTDE: {e}")

        self.rtde_control = None
        self.rtde_receive = None

        logger.info("UR5e disconnected")
