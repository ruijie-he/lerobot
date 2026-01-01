#!/usr/bin/env python
"""
SpaceMouse teleoperation for MetaWorld/LIBERO simulation environments.

Requires pyspacemouse: pip install pyspacemouse

Note: Episodes will automatically reset when they reach the task's max step limit
(typically 10000 steps for MetaWorld). You can continue teleoperating across resets.
"""

import argparse
import time

import cv2
import numpy as np

try:
    import pyspacemouse
except ImportError:
    raise ImportError(
        "pyspacemouse is required for spacemouse teleoperation. "
        "Install it with: pip install pyspacemouse"
    )

from lerobot.envs.configs import LiberoEnv, MetaworldEnv
from lerobot.envs.factory import make_env
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.processor import DataProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_STR
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="SpaceMouse teleoperation for simulation environments")
    parser.add_argument(
        "--env",
        type=str,
        choices=["metaworld", "libero"],
        default="metaworld",
        help="Environment type (default: metaworld)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task name (default: metaworld-reach-v3 for metaworld, libero_spatial for libero)",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=10000,
        help="Override max episode steps (default: uses environment default, typically 10000 for MetaWorld)",
    )
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=1.5,
        help="Scale factor for translation movements (default: 1.5)",
    )
    parser.add_argument(
        "--rotation-scale",
        type=float,
        default=0.3,
        help="Scale factor for rotation movements (default: 0.3)",
    )
    # Dataset collection arguments
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default=None,
        help="Dataset repository ID for saving episodes (e.g., 'username/my-dataset'). If not provided, no data will be saved.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data",
        help="Root directory for dataset storage (default: data)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to collect before exiting (default: 10). Set to 0 for unlimited.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for dataset (default: 30)",
    )
    parser.add_argument(
        "--task-description",
        type=str,
        default=None,
        help="Task description for the dataset (e.g., 'Reach the red cube')",
    )
    args = parser.parse_args()

    # Choose environment based on command line argument
    if args.env == "metaworld":
        task = args.task if args.task else "metaworld-reach-v3"
        env_cfg = MetaworldEnv(
            task=task,
            obs_type="pixels_agent_pos",
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
        if hasattr(env, "_env") and hasattr(env._env, "max_path_length"):
            env._env.max_path_length = args.max_episode_steps
            env._max_episode_steps = args.max_episode_steps
            print(f"Overriding max episode steps to: {args.max_episode_steps}")
        else:
            print("Warning: Could not override max episode steps for this environment type")

    # Initialize dataset for recording if repo_id is provided
    dataset = None
    if args.dataset_repo_id is not None:
        print(f"\nInitializing dataset: {args.dataset_repo_id}")
        task_description = args.task_description if args.task_description else f"Teleoperate {task}"

        # Get a sample observation to determine structure
        sample_obs, _ = env.reset()

        # Flatten observation structure for features
        flat_obs = {}
        for key, value in sample_obs.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_obs[f"{key}.{subkey}"] = subvalue
            else:
                flat_obs[key] = value

        # Build observation features following LeRobot convention
        obs_features = {}
        for key, value in flat_obs.items():
            if isinstance(value, np.ndarray):
                if len(value.shape) == 3:  # Image
                    obs_features[key] = value.shape
                elif len(value.shape) == 1:  # State vector - decompose
                    for i in range(value.shape[0]):
                        obs_features[f"{key}.{i}"] = float
                else:
                    flat_value = value.flatten()
                    for i in range(len(flat_value)):
                        obs_features[f"{key}.{i}"] = float
            else:
                obs_features[key] = type(value)

        # Build action features - decompose action vector
        action_dim = env.action_space.shape[0]
        action_features = {f"action.{i}": float for i in range(action_dim)}

        # Create dataset features using empty pipeline (acts as identity transformation)
        identity_pipeline = DataProcessorPipeline([])
        features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=identity_pipeline,
                initial_features=create_initial_features(action=action_features),
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=identity_pipeline,
                initial_features=create_initial_features(observation=obs_features),
                use_videos=True,
            ),
        )

        # Create the dataset
        dataset_root = Path(args.dataset_root) / args.dataset_repo_id.replace("/", "_") if args.dataset_root else None
        dataset = LeRobotDataset.create(
            repo_id=args.dataset_repo_id,
            fps=args.fps,
            root=dataset_root,
            robot_type=f"{args.env}_sim",
            features=features,
            use_videos=True,
        )

        print(f"Dataset initialized. Will collect {args.num_episodes if args.num_episodes > 0 else 'unlimited'} episodes")
        print(f"Task: {task_description}")

    # Initialize SpaceMouse
    print("\nInitializing SpaceMouse...")
    print("Make sure your SpaceMouse is connected!")
    success = pyspacemouse.open()
    if not success:
        print("ERROR: Could not open SpaceMouse. Please check connection and try again.")
        print("On Linux, you may need to add udev rules. See pyspacemouse documentation.")
        return

    print("\nSpaceMouse controls:")
    print("  Translation (X/Y/Z): Move the SpaceMouse")
    print("  Rotation (if supported): Rotate the SpaceMouse")
    print("  Left button: Close gripper")
    print("  Right button: Open gripper")
    print("  Press Ctrl+C to exit")
    print(f"\nTranslation scale: {args.translation_scale}")
    print(f"Rotation scale: {args.rotation_scale}")
    print("\nStarting teleoperation...")

    try:
        obs, info = env.reset()
        running = True
        terminated = info.get("terminated", False) if isinstance(info, dict) else False
        truncated = info.get("truncated", False) if isinstance(info, dict) else False
        step_count = 0
        episode_count = 1
        gripper_state = 0.0
        task_description = args.task_description if args.task_description else f"Teleoperate {task}"

        print(f"Episode {episode_count} started (max steps: {getattr(env, '_max_episode_steps', 'unknown')})")

        if dataset is not None:
            print("Recording data to dataset...")

        while running:
            # Flush SpaceMouse buffer by reading until no more data is available
            # This prevents accumulated lag by discarding old buffered inputs
            state = pyspacemouse.read()
            # Safety limit to prevent infinite loop if read() never returns None
            for _ in range(100):
                next_state = pyspacemouse.read()
                if next_state is None:
                    break
                state = next_state

            # Display image
            if "pixels" in obs:
                img = obs["pixels"]
                if isinstance(img, dict):
                    img = list(img.values())[0]  # Get first camera view

                # Add text overlay with controls and step count
                gripper_text = "CLOSE" if gripper_state > 0.5 else "OPEN" if gripper_state < -0.5 else "HOLD"
                cv2.putText(
                    img,
                    f"Step: {step_count} | Episode: {episode_count} | Gripper: {gripper_text}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    img,
                    "SpaceMouse: Move/Rotate | Buttons: Gripper | Ctrl+C: Quit",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                cv2.imshow("Simulation - SpaceMouse Control", img)

            # Check for OpenCV window close or ESC key
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("Exiting...")
                running = False
                break

            # Check if episode ended and reset if needed
            if terminated or truncated:
                reason = "task completed" if terminated else "timeout"
                print(f"Episode {episode_count} ended after {step_count} steps ({reason}).")

                # Save episode to dataset if enabled
                if dataset is not None:
                    print(f"Saving episode {episode_count} to dataset...")
                    dataset.save_episode()
                    print(f"Episode {episode_count} saved. Total frames: {dataset.meta.total_frames}")

                # Check if we've collected enough episodes
                if args.num_episodes > 0 and episode_count >= args.num_episodes:
                    print(f"\nCollected {args.num_episodes} episodes. Exiting...")
                    running = False
                    break

                print("Resetting...")
                obs, info = env.reset()
                terminated = False
                truncated = False
                step_count = 0
                episode_count += 1
                print(f"Episode {episode_count} started")
                continue

            # Extract SpaceMouse data
            if state is not None:
                # Get translation values (normalized to approximately -1 to 1)
                delta_x = state.x * args.translation_scale
                delta_y = state.y * args.translation_scale
                delta_z = state.z * args.translation_scale

                # Get rotation values (for future use if needed)
                # roll = state.roll * args.rotation_scale
                # pitch = state.pitch * args.rotation_scale
                # yaw = state.yaw * args.rotation_scale

                # Handle gripper buttons
                if state.buttons[0]:  # Left button - close gripper
                    gripper_state = 1.0
                elif state.buttons[1]:  # Right button - open gripper
                    gripper_state = -1.0
                # If no button pressed, maintain current gripper state
            else:
                # No input from spacemouse, use zero deltas
                delta_x = 0.0
                delta_y = 0.0
                delta_z = 0.0

            # Convert deltas to action
            action_dim = env.action_space.shape[0]
            action = np.zeros(action_dim, dtype=np.float32)

            if action_dim == 4:  # MetaWorld
                # Scale down the movements for better control
                scale = 0.05
                action[0] = delta_x * scale
                action[1] = delta_y * scale
                action[2] = delta_z * scale
                action[3] = gripper_state

            elif action_dim == 7:  # LIBERO
                scale = 0.05
                action[0] = delta_x * scale
                action[1] = delta_y * scale
                action[2] = delta_z * scale
                action[3:6] = 0.0  # No orientation change for now
                action[6] = gripper_state

            # Clip to action space bounds
            action = np.clip(action, env.action_space.low, env.action_space.high)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            # Record data to dataset if enabled
            if dataset is not None:
                # Flatten observations and decompose arrays
                flat_obs = {}
                for key, value in obs.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            flat_key = f"{key}.{subkey}"
                            if isinstance(subvalue, np.ndarray):
                                if len(subvalue.shape) == 3:  # Image - convert RGB to BGR
                                    flat_obs[flat_key] = cv2.cvtColor(subvalue, cv2.COLOR_RGB2BGR)
                                elif len(subvalue.shape) == 1:  # State vector - decompose
                                    for i in range(subvalue.shape[0]):
                                        flat_obs[f"{flat_key}.{i}"] = float(subvalue[i])
                                else:
                                    flat_value = subvalue.flatten()
                                    for i in range(len(flat_value)):
                                        flat_obs[f"{flat_key}.{i}"] = float(flat_value[i])
                            else:
                                flat_obs[flat_key] = subvalue
                    else:
                        if isinstance(value, np.ndarray):
                            if len(value.shape) == 3:  # Image - convert RGB to BGR
                                flat_obs[key] = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
                            elif len(value.shape) == 1:  # State vector - decompose
                                for i in range(value.shape[0]):
                                    flat_obs[f"{key}.{i}"] = float(value[i])
                            else:
                                flat_value = value.flatten()
                                for i in range(len(flat_value)):
                                    flat_obs[f"{key}.{i}"] = float(flat_value[i])
                        else:
                            flat_obs[key] = value

                # Decompose action vector
                action_values = {str(i): float(action[i]) for i in range(len(action))}

                # Build frames and combine
                observation_frame = build_dataset_frame(dataset.features, flat_obs, prefix=OBS_STR)
                action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
                frame = {**observation_frame, **action_frame, "task": task_description}

                dataset.add_frame(frame)

            # Small sleep to control loop rate
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Save any partial episode before exiting
        if dataset is not None and step_count > 0:
            print(f"\nSaving partial episode {episode_count} (Step count: {step_count})...")
            try:
                dataset.save_episode()
                print(f"Partial episode {episode_count} saved.")
            except Exception as e:
                print(f"Warning: Could not save partial episode: {e}")

        # Save dataset and push to hub
        if dataset is not None:
            print(f"\nDataset saved to: {dataset.root}")
            print(f"Total frames collected: {dataset.meta.total_frames}")
            print(f"Total episodes collected: {dataset.meta.total_episodes}")

            # Ask user if they want to push to HuggingFace Hub
            try:
                response = input("\nPush dataset to HuggingFace Hub? (yes/no): ")
                if response.lower() == "yes":
                    print(f"Pushing dataset to {args.dataset_repo_id}...")
                    dataset.push_to_hub()
                    print(f"Dataset pushed successfully!")
                    print(f"View at: https://huggingface.co/datasets/{args.dataset_repo_id}")
            except EOFError:
                # Input not available (e.g., in background), skip push
                print("Skipping HuggingFace Hub push (no interactive terminal)")

        pyspacemouse.close()
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
