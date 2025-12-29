#!/usr/bin/env python
"""
Keyboard teleoperation for MetaWorld/LIBERO simulation environments.

Note: Episodes will automatically reset when they reach the task's max step limit
(typically 500 steps for MetaWorld). You can continue teleoperating across resets.
"""

import argparse
import numpy as np
from lerobot.envs.factory import make_env
from lerobot.envs.configs import LiberoEnv, MetaworldEnv
import cv2

def main():
    parser = argparse.ArgumentParser(description="Keyboard teleoperation for simulation environments")
    parser.add_argument(
        "--env",
        type=str,
        choices=["metaworld", "libero"],
        default="metaworld",
        help="Environment type (default: metaworld)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task name (default: metaworld-reach-v3 for metaworld, libero_spatial for libero)"
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Override max episode steps (default: uses environment default, typically 500 for MetaWorld)"
    )
    args = parser.parse_args()

    # Choose environment based on command line argument
    if args.env == "metaworld":
        task = args.task if args.task else "metaworld-reach-v3"
        env_cfg = MetaworldEnv(
            task=task,
            obs_type="pixels",
        )
    else:  # libero
        task = args.task if args.task else "libero_spatial"
        env_cfg = LiberoEnv(
            task=task,
            camera_name="agentview_image,robot0_eye_in_hand_image",
            obs_type="pixels",
        )

    print(f"Starting {args.env} environment with task: {task}")

    # Create environment
    envs_dict = make_env(env_cfg, n_envs=1)
    vec_env = list(envs_dict.values())[0][0]  # Get the vectorized environment
    env = vec_env.envs[0]  # Get the first (and only) single environment from vector env

    # Override max episode steps if specified
    if args.max_episode_steps is not None:
        if hasattr(env, '_env') and hasattr(env._env, 'max_path_length'):
            env._env.max_path_length = args.max_episode_steps
            env._max_episode_steps = args.max_episode_steps
            print(f"Overriding max episode steps to: {args.max_episode_steps}")
        else:
            print(f"Warning: Could not override max episode steps for this environment type")

    # Keyboard state tracking
    pressed_keys = set()

    print("Keyboard controls (focus the OpenCV window):")
    print("  Arrow keys: Move end-effector (up/down/left/right)")
    print("  W/S: Move up/down in Z")
    print("  O: Open gripper (continuous)")
    print("  C: Close gripper (continuous)")
    print("  H: Hold gripper position")
    print("  ESC or Q: Exit")
    print("\nStarting teleoperation...")

    try:
        obs, info = env.reset()
        running = True
        terminated = info.get("terminated", False) if isinstance(info, dict) else False
        truncated = info.get("truncated", False) if isinstance(info, dict) else False
        step_count = 0
        episode_count = 1
        gripper_state = 0.0

        print(f"Episode {episode_count} started (max steps: {getattr(env, '_max_episode_steps', 'unknown')})")
        print("Note: Press 'O' to open gripper, 'C' to close gripper, 'H' to hold position")

        while running:
            # Display image first
            if "pixels" in obs:
                img = obs["pixels"]
                if isinstance(img, dict):
                    img = list(img.values())[0]  # Get first camera view
                # Add text overlay with controls and step count
                gripper_text = "CLOSE" if gripper_state > 0.5 else "OPEN" if gripper_state < -0.5 else "HOLD"
                cv2.putText(img, f"Step: {step_count} | Episode: {episode_count} | Gripper: {gripper_text}",
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(img, "Arrow: Move | W/S: Z | O/C: Gripper | ESC: Quit",
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.imshow("Simulation - FOCUS THIS WINDOW", img)

            # Get keyboard input from OpenCV (must call after imshow)
            key = cv2.waitKey(1) & 0xFF

            # Check if episode ended and reset if needed
            if terminated or truncated:
                reason = "task completed" if terminated else "timeout"
                print(f"Episode {episode_count} ended after {step_count} steps ({reason}). Resetting...")
                obs, info = env.reset()
                terminated = False
                truncated = False
                step_count = 0
                episode_count += 1
                print(f"Episode {episode_count} started")
                continue

            # Initialize deltas
            delta_x = 0.0
            delta_y = 0.0
            delta_z = 0.0

            # Map keys to actions
            # Arrow keys
            if key == 82 or key == 0:  # Up arrow
                delta_y = 1.0
            elif key == 84 or key == 1:  # Down arrow
                delta_y = -1.0
            elif key == 81 or key == 2:  # Left arrow
                delta_x = -1.0
            elif key == 83 or key == 3:  # Right arrow
                delta_x = 1.0
            # W/S for Z movement
            elif key == ord('w') or key == ord('W'):
                delta_z = 1.0
            elif key == ord('s') or key == ord('S'):
                delta_z = -1.0
            # Gripper control - toggle state
            elif key == ord('o') or key == ord('O'):
                gripper_state = -1.0  # Open
                print(f"Gripper: OPEN")
            elif key == ord('c') or key == ord('C'):
                gripper_state = 1.0  # Open
                print(f"Gripper: CLOSE")
            elif key == ord('h') or key == ord('H'):
                gripper_state = 0.0  # Hold
                print(f"Gripper: HOLD")
            # Exit
            elif key == 27 or key == ord('q') or key == ord('Q'):  # ESC or Q
                print("Exiting...")
                running = False
                continue

            # Convert deltas to action
            action_dim = env.action_space.shape[0]
            action = np.zeros(action_dim, dtype=np.float32)

            if action_dim == 4:  # MetaWorld
                scale = 0.05  # Adjust this to control movement speed
                action[0] = delta_x * scale
                action[1] = delta_y * scale
                action[2] = delta_z * scale
                action[3] = gripper_state  # Use persistent gripper state

            elif action_dim == 7:  # LIBERO
                scale = 0.05
                action[0] = delta_x * scale
                action[1] = delta_y * scale
                action[2] = delta_z * scale
                action[3:6] = 0.0  # No orientation change
                action[6] = gripper_state  # Use persistent gripper state

            # Clip to action space bounds
            action = np.clip(action, env.action_space.low, env.action_space.high)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()