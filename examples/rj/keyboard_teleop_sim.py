#!/usr/bin/env python
"""Keyboard teleoperation for MetaWorld/LIBERO simulation environments."""

import numpy as np
from lerobot.envs.factory import make_env
from lerobot.envs.configs import LiberoEnv, MetaworldEnv
import cv2

def main():
    # Choose your environment
    # Option 1: MetaWorld (use V3 tasks - see metaworld_config.json for available tasks)
    env_cfg = MetaworldEnv(
        task="metaworld-reach-v3",  # Use V3 tasks: reach-v3, push-v3, pick-place-v3, etc.
        obs_type="pixels",
    )

    # Option 2: LIBERO (uncomment to use instead)
    # env_cfg = LiberoEnv(
    #     task="libero_spatial",
    #     camera_name="top",
    # )

    # Create environment
    envs_dict = make_env(env_cfg, n_envs=1)
    vec_env = list(envs_dict.values())[0][0]  # Get the vectorized environment
    env = vec_env.envs[0]  # Get the first (and only) single environment from vector env

    # Keyboard state tracking
    pressed_keys = set()

    print("Keyboard controls (focus the OpenCV window):")
    print("  Arrow keys: Move end-effector (up/down/left/right)")
    print("  W/S: Move up/down in Z")
    print("  O/C: Open/Close gripper")
    print("  ESC or Q: Exit")
    print("\nStarting teleoperation...")

    try:
        obs, info = env.reset()
        running = True

        while running:
            # Get keyboard input from OpenCV (must call after imshow)
            key = cv2.waitKey(1) & 0xFF

            # Initialize deltas
            delta_x = 0.0
            delta_y = 0.0
            delta_z = 0.0
            gripper_action = 0.0  # 0 = no change

            # Map keys to actions
            # Arrow keys
            if key == 82 or key == 0:  # Up arrow
                delta_y = -1.0
            elif key == 84 or key == 1:  # Down arrow
                delta_y = 1.0
            elif key == 81 or key == 2:  # Left arrow
                delta_x = 1.0
            elif key == 83 or key == 3:  # Right arrow
                delta_x = -1.0
            # W/S for Z movement
            elif key == ord('w') or key == ord('W'):
                delta_z = 1.0
            elif key == ord('s') or key == ord('S'):
                delta_z = -1.0
            # Gripper control
            elif key == ord('o') or key == ord('O'):
                gripper_action = 1.0  # Open
            elif key == ord('c') or key == ord('C'):
                gripper_action = -1.0  # Close
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
                action[3] = gripper_action

            elif action_dim == 7:  # LIBERO
                scale = 0.05
                action[0] = delta_x * scale
                action[1] = delta_y * scale
                action[2] = delta_z * scale
                action[3:6] = 0.0  # No orientation change
                action[6] = gripper_action

            print(f"Action: {action}")
            # Clip to action space bounds
            action = np.clip(action, env.action_space.low, env.action_space.high)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Display image
            if "pixels" in obs:
                img = obs["pixels"]
                if isinstance(img, dict):
                    img = list(img.values())[0]  # Get first camera view
                # Add text overlay with controls
                cv2.putText(img, "Arrow keys: Move | W/S: Z-axis | O/C: Gripper | ESC: Quit",
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.imshow("Simulation - FOCUS THIS WINDOW", img)

            # Reset on episode end
            if terminated or truncated:
                obs, info = env.reset()
                print("Episode ended. Resetting...")

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()