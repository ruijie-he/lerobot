#!/usr/bin/env python
"""Keyboard teleoperation for MetaWorld/LIBERO simulation environments."""

import numpy as np
from lerobot.envs.factory import make_env
from lerobot.envs.configs import LiberoEnv, MetaworldEnv
from lerobot.teleoperators.keyboard import KeyboardEndEffectorTeleop, KeyboardEndEffectorTeleopConfig
import cv2

def main():
    # Choose your environment
    # Option 1: MetaWorld (use V3 tasks - see metaworld_config.json for available tasks)
    # env_cfg = MetaworldEnv(
    #     task="metaworld-reach-v3",  # Use V3 tasks: reach-v3, push-v3, pick-place-v3, etc.
    #     obs_type="pixels",
    # )
    
    # Option 2: LIBERO (uncomment to use instead)
    env_cfg = LiberoEnv(
        task="libero_spatial",
        camera_name="agentview_image",  # Single camera for simplicity
        obs_type="pixels",  # Use pixels for simpler observation structure
    )
    
    # Create environment
    try:
        envs_dict = make_env(env_cfg, n_envs=1)
        vec_env = list(envs_dict.values())[0][0]  # Get the vectorized environment
        env = vec_env.envs[0]  # Get the first (and only) single environment from vector env
    except Exception as e:
        print(f"Error creating environment: {e}")
        print(f"Environment config: {env_cfg}")
        raise
    
    # Create keyboard teleoperator
    keyboard_config = KeyboardEndEffectorTeleopConfig(use_gripper=True)
    keyboard = KeyboardEndEffectorTeleop(keyboard_config)
    keyboard.connect()
    
    print("Keyboard controls:")
    print("  Arrow keys: Move end-effector (up/down/left/right)")
    print("  Shift: Move up/down in Z")
    print("  Ctrl: Control gripper")
    print("  ESC: Exit")
    print("\nStarting teleoperation...")
    
    try:
        obs, info = env.reset()
        
        frame_count = 0
        while True:
            # Get keyboard input (deltas)
            keyboard_action = keyboard.get_action()
            
            # Debug: print keyboard input periodically
            frame_count += 1
            if frame_count % 60 == 0:  # Every 60 frames (~2 seconds at 30fps)
                has_input = any([
                    keyboard_action.get("delta_x", 0.0) != 0.0,
                    keyboard_action.get("delta_y", 0.0) != 0.0,
                    keyboard_action.get("delta_z", 0.0) != 0.0,
                    keyboard_action.get("gripper", 1.0) != 1.0,
                ])
                if has_input:
                    print(f"✓ Keyboard input detected: {keyboard_action}")
                else:
                    print("⚠ No keyboard input. Press arrow keys, shift, or ctrl to test.")
                    # Debug: check what keys are in current_pressed
                    if hasattr(keyboard, 'current_pressed') and keyboard.current_pressed:
                        print(f"  Pressed keys: {list(keyboard.current_pressed.keys())}")
            
            # Convert keyboard deltas to action
            # MetaWorld: 4D [delta_x, delta_y, delta_z, gripper] - uses relative actions
            # LIBERO: 7D [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper] - uses relative actions
            action_dim = env.action_space.shape[0]
            
            # Create action array
            action = np.zeros(action_dim, dtype=np.float32)
            
            if action_dim == 4:  # MetaWorld
                # Scale deltas appropriately (MetaWorld expects actions in [-1, 1])
                scale = 0.1  # Adjust this to control movement speed
                action[0] = keyboard_action.get("delta_x", 0.0) * scale
                action[1] = keyboard_action.get("delta_y", 0.0) * scale
                action[2] = keyboard_action.get("delta_z", 0.0) * scale
                # Map gripper: 0=close, 1=stay, 2=open -> -1=close, 0=stay, 1=open
                gripper_val = keyboard_action.get("gripper", 1.0)
                action[3] = (gripper_val - 1.0)  # Maps 0->-1, 1->0, 2->1
                
            elif action_dim == 7:  # LIBERO
                scale = 0.1
                action[0] = keyboard_action.get("delta_x", 0.0) * scale
                action[1] = keyboard_action.get("delta_y", 0.0) * scale
                action[2] = keyboard_action.get("delta_z", 0.0) * scale
                # Keep orientation deltas at 0 (or you could map additional keys)
                action[3:6] = 0.0
                # Map gripper: 0=close, 1=stay, 2=open -> -1=close, 0=stay, 1=open
                gripper_val = keyboard_action.get("gripper", 1.0)
                action[6] = (gripper_val - 1.0)  # Maps 0->-1, 1->0, 2->1
            
            # Clip to action space bounds
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Display image
            if "pixels" in obs:
                pixels = obs["pixels"]
                if isinstance(pixels, dict):
                    # LIBERO returns dict like {"image": array, "image2": array}
                    # Get the first available image (usually "image" for agentview)
                    img = None
                    for key in ["image", "image2"]:
                        if key in pixels:
                            img = pixels[key]
                            break
                    if img is None:
                        # Fallback to first value
                        img = list(pixels.values())[0]
                else:
                    # MetaWorld returns single image array
                    img = pixels
                
                # Ensure image is numpy array and has correct shape
                if isinstance(img, np.ndarray):
                    # LIBERO images are RGB, OpenCV expects BGR
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        display_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    else:
                        display_img = img
                    cv2.imshow("Simulation", display_img)
                    cv2.waitKey(1)
                else:
                    print(f"Warning: Unexpected image type: {type(img)}, shape: {getattr(img, 'shape', 'N/A')}")
            
            # Reset on episode end
            if terminated or truncated:
                obs, info = env.reset()
                print("Episode ended. Resetting...")
            
            # Check for exit
            if not keyboard.is_connected:
                break
                
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        keyboard.disconnect()
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()