#!/usr/bin/env python
"""
SpaceMouse teleoperation for UR5e robot (URSim or real hardware).

Requires:
- pyspacemouse: pip install pyspacemouse
- ur_rtde: pip install ur-rtde

Example usage:
    # URSim (default IP)
    python spacemouse_teleop_ur5e.py

    # Real UR5e
    python spacemouse_teleop_ur5e.py --robot-ip 192.168.1.10

    # With workspace limits (safer)
    python spacemouse_teleop_ur5e.py --workspace-limits -0.5,0.5,-0.5,0.5,0.1,0.8

    # Disable rotation control (3-DOF mode)
    python spacemouse_teleop_ur5e.py --disable-rotation
"""

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import pyspacemouse
except ImportError:
    raise ImportError(
        "pyspacemouse is required for spacemouse teleoperation. " "Install it with: pip install pyspacemouse"
    )

from lerobot.robots.ur5e import UR5eRobot, UR5eRobotConfig


def parse_workspace_limits(limits_str: str | None) -> list[float] | None:
    """Parse workspace limits from command line string.

    Args:
        limits_str: String like "x_min,x_max,y_min,y_max,z_min,z_max"

    Returns:
        List of 6 floats or None
    """
    if limits_str is None:
        return None

    try:
        limits = [float(x.strip()) for x in limits_str.split(",")]
        if len(limits) != 6:
            raise ValueError(f"Expected 6 values, got {len(limits)}")
        return limits
    except Exception as e:
        raise ValueError(f"Invalid workspace limits format: {e}")


def main():
    parser = argparse.ArgumentParser(description="SpaceMouse teleoperation for UR5e robot")
    parser.add_argument(
        "--robot-ip",
        type=str,
        default="192.168.56.101",
        help="IP address of UR5e (default: 192.168.56.101 for URSim)",
    )
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=1.0,
        help="Scale factor for translation movements (default: 1.0)",
    )
    parser.add_argument(
        "--rotation-scale",
        type=float,
        default=0.3,
        help="Scale factor for rotation movements (default: 0.3)",
    )
    parser.add_argument(
        "--disable-rotation",
        action="store_true",
        help="Disable rotation control (3-DOF mode, translation only)",
    )
    parser.add_argument(
        "--max-velocity",
        type=float,
        default=0.25,
        help="Maximum Cartesian velocity in m/s (default: 0.25)",
    )
    parser.add_argument(
        "--workspace-limits",
        type=str,
        default=None,
        help='Workspace limits as "x_min,x_max,y_min,y_max,z_min,z_max" in meters',
    )
    parser.add_argument(
        "--no-home",
        action="store_true",
        help="Don't move to home position on connection",
    )
    args = parser.parse_args()

    # Parse workspace limits
    workspace_limits = parse_workspace_limits(args.workspace_limits)

    # Create robot configuration
    config = UR5eRobotConfig(
        ip_address=args.robot_ip,
        max_cartesian_velocity=args.max_velocity,
        workspace_limits=workspace_limits,
        home_on_connect=not args.no_home,
    )

    print("=" * 70)
    print("UR5e SpaceMouse Teleoperation")
    print("=" * 70)
    print(f"Robot IP: {args.robot_ip}")
    print(f"Translation scale: {args.translation_scale}")
    print(f"Rotation scale: {args.rotation_scale if not args.disable_rotation else 'DISABLED'}")
    print(f"Max velocity: {args.max_velocity} m/s")
    if workspace_limits:
        print(f"Workspace limits: {workspace_limits}")
    print("=" * 70)
    print()

    # Initialize robot
    print("Connecting to UR5e robot...")
    robot = UR5eRobot(config)
    robot.connect()
    print("✓ Robot connected successfully")
    print()

    # Initialize SpaceMouse
    print("Initializing SpaceMouse...")
    print("Make sure your SpaceMouse is connected!")
    success = pyspacemouse.open()
    if not success:
        print("ERROR: Could not open SpaceMouse. Please check connection and try again.")
        robot.disconnect()
        return

    print("✓ SpaceMouse initialized")
    print()

    # Print control instructions
    print("=" * 70)
    print("CONTROLS (End-Effector Frame):")
    print("  Move SpaceMouse:")
    print("    - Translation (X/Y/Z): Move the device (relative to tool orientation)")
    if not args.disable_rotation:
        print("    - Rotation (Roll/Pitch/Yaw): Rotate the device (relative to tool orientation)")
    print("  Buttons:")
    print("    - Left button: Reserved (future gripper close)")
    print("    - Right button: Reserved (future gripper open)")
    print("  Keyboard:")
    print("    - Ctrl+C: Exit teleoperation")
    print("=" * 70)
    print()

    print("Starting teleoperation loop...")
    print("(Press Ctrl+C to exit)")
    print()

    try:
        step_count = 0

        while True:
            # Adaptive buffer flushing: read until we get the most recent state
            # This prevents input lag from accumulated buffered commands
            state = pyspacemouse.read()

            # Safety limit to prevent infinite loop if read() never returns None
            for _ in range(100):
                next_state = pyspacemouse.read()
                if next_state is None:
                    break
                state = next_state

            if state is not None:
                # Get current TCP pose
                current_pose = robot.rtde_receive.getActualTCPPose()  # [x, y, z, rx, ry, rz]

                # Calculate deltas from SpaceMouse input in end-effector frame
                # Convert SpaceMouse units to meters and radians with scaling
                # Signs are flipped for more intuitive control
                delta_translation_ee = np.array([
                    state.x * args.translation_scale * 0.001,  # mm to meters
                    -state.y * args.translation_scale * 0.001,
                    -state.z * args.translation_scale * 0.001,
                ])

                # Convert current orientation from axis-angle to rotation matrix
                current_rotvec = np.array(current_pose[3:6])
                R_current = Rotation.from_rotvec(current_rotvec)

                # Transform translation delta from end-effector frame to base frame
                delta_translation_base = R_current.apply(delta_translation_ee)

                if args.disable_rotation:
                    # No rotation change
                    target_rotvec = current_rotvec
                else:
                    # Calculate rotation delta in end-effector frame
                    # Signs are flipped for more intuitive control
                    delta_rotvec_ee = np.array([
                        -state.pitch * args.rotation_scale * 0.01,
                        -state.roll * args.rotation_scale * 0.01,
                        state.yaw * args.rotation_scale * 0.01,
                    ])

                    # Compose rotations: R_target = R_current * R_delta
                    R_delta = Rotation.from_rotvec(delta_rotvec_ee)
                    R_target = R_current * R_delta
                    target_rotvec = R_target.as_rotvec()

                # Compute target pose
                target_position = np.array(current_pose[0:3]) + delta_translation_base
                target_pose = np.concatenate([target_position, target_rotvec])

                # Create action dictionary
                action = {
                    "cartesian_pose": target_pose,
                    "cartesian_velocity": args.max_velocity,
                }

                # Send action to robot
                robot.send_action(action)

                step_count += 1

                # Print status every 100 steps
                if step_count % 100 == 0:
                    print(f"Step {step_count} | TCP: [{target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f}]")

            # Sleep to maintain control loop frequency (125Hz)
            time.sleep(config.servo_dt)

    except KeyboardInterrupt:
        print("\n\nExiting teleoperation...")

    finally:
        # Cleanup
        print("Closing SpaceMouse...")
        pyspacemouse.close()

        print("Disconnecting robot...")
        robot.disconnect()

        print("✓ Cleanup complete")
        print()
        print(f"Total steps: {step_count}")


if __name__ == "__main__":
    main()
