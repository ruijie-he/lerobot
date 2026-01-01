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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("ur5e")
@dataclass
class UR5eRobotConfig(RobotConfig):
    """Configuration for UR5e robot arm.

    This config supports both URSim (simulator) and real UR5e hardware.
    Only the ip_address needs to change between them.
    """

    # Connection settings
    # Default IP is for URSim running in Docker with ursim_net network
    # For real robot, change to robot's actual IP (e.g., "192.168.1.10")
    ip_address: str = "192.168.56.101"

    # RTDE servo control parameters
    # These affect the real-time control loop behavior
    servo_dt: float = 0.008  # Control loop timestep (125Hz = 1/125 = 0.008s)
    servo_lookahead_time: float = 0.1  # Lookahead time for trajectory smoothing (seconds)
    servo_gain: int = 300  # Proportional gain for trajectory following

    # Safety limits for Cartesian control
    max_cartesian_velocity: float = 0.25  # Maximum TCP velocity in m/s (conservative default)
    max_cartesian_acceleration: float = 0.5  # Maximum TCP acceleration in m/s²

    # Safety limits for joint control
    max_joint_velocity: float = 1.0  # Maximum joint velocity in rad/s
    max_relative_target: float | None = None  # Maximum relative position change per command (optional)

    # Workspace boundaries [x_min, x_max, y_min, y_max, z_min, z_max] in meters
    # Relative to robot base frame. None means no workspace limits enforced.
    # Example: [-0.5, 0.5, -0.5, 0.5, 0.0, 0.8] restricts to 1m x 1m x 0.8m volume
    workspace_limits: list[float] | None = None

    # Robot behavior on connection
    home_on_connect: bool = True  # Move to home position when connecting
    home_joint_positions: list[float] = field(
        default_factory=lambda: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
    )  # Home joint angles in radians [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]

    # Cameras (optional - can attach external cameras for observation)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        super().__post_init__()

        # Validate workspace limits if provided
        if self.workspace_limits is not None:
            if len(self.workspace_limits) != 6:
                raise ValueError(
                    f"workspace_limits must have exactly 6 values [x_min, x_max, y_min, y_max, z_min, z_max], "
                    f"got {len(self.workspace_limits)}"
                )
            # Check that min < max for each axis
            for i in range(0, 6, 2):
                if self.workspace_limits[i] >= self.workspace_limits[i + 1]:
                    raise ValueError(
                        f"workspace_limits: min must be less than max for each axis. "
                        f"Got {self.workspace_limits[i]} >= {self.workspace_limits[i + 1]} for axis {i//2}"
                    )

        # Validate home position has correct number of joints
        if len(self.home_joint_positions) != 6:
            raise ValueError(
                f"home_joint_positions must have exactly 6 values for UR5e joints, "
                f"got {len(self.home_joint_positions)}"
            )

        # Validate safety limits are positive
        if self.max_cartesian_velocity <= 0:
            raise ValueError(f"max_cartesian_velocity must be positive, got {self.max_cartesian_velocity}")
        if self.max_cartesian_acceleration <= 0:
            raise ValueError(
                f"max_cartesian_acceleration must be positive, got {self.max_cartesian_acceleration}"
            )
        if self.max_joint_velocity <= 0:
            raise ValueError(f"max_joint_velocity must be positive, got {self.max_joint_velocity}")
        if self.servo_dt <= 0:
            raise ValueError(f"servo_dt must be positive, got {self.servo_dt}")
