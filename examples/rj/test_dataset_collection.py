#!/usr/bin/env python
"""
Test script to verify dataset collection functionality without requiring SpaceMouse hardware.
"""

import numpy as np
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.envs.configs import MetaworldEnv
from lerobot.envs.factory import make_env
from lerobot.processor import DataProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_STR


def test_dataset_creation():
    """Test creating a dataset and collecting a few frames."""
    print("Testing dataset creation and data collection...")

    # Create environment
    print("\n1. Creating MetaWorld environment...")
    env_cfg = MetaworldEnv(
        task="metaworld-reach-v3",
        obs_type="pixels_agent_pos",
    )
    envs_dict = make_env(env_cfg, n_envs=1)
    vec_env = list(envs_dict.values())[0][0]
    env = vec_env.envs[0]

    # Get sample observation
    print("2. Getting sample observation...")
    sample_obs, _ = env.reset()

    # Flatten observation structure
    print("3. Building features...")
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

    # Create features using empty pipeline (acts as identity transformation)
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

    # Create dataset
    print("4. Creating dataset...")
    dataset_path = Path("data/test_spacemouse_metaworld")
    if dataset_path.exists():
        import shutil
        shutil.rmtree(dataset_path)

    dataset = LeRobotDataset.create(
        repo_id="test/spacemouse_metaworld",
        fps=30,
        root=dataset_path,  # Pass the full path to the dataset directory
        robot_type="metaworld_sim",
        features=features,
        use_videos=True,
    )

    print(f"Dataset created at: {dataset.root}")
    print(f"Features: {list(dataset.features.keys())[:10]}...")  # Show first 10 features

    # Collect a few frames
    print("\n5. Collecting test frames...")
    obs, _ = env.reset()
    task_description = "Reach the target"

    for step in range(5):
        # Random action
        action = env.action_space.sample()

        # Prepare observation values: flatten nested dicts and decompose arrays
        obs_values = {}
        for key, value in obs.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    full_key = f"{key}.{subkey}"
                    if isinstance(subvalue, np.ndarray) and len(subvalue.shape) == 3:
                        obs_values[full_key] = subvalue
                    elif isinstance(subvalue, np.ndarray) and len(subvalue.shape) == 1:
                        for i in range(subvalue.shape[0]):
                            obs_values[f"{full_key}.{i}"] = subvalue[i]
                    else:
                        obs_values[full_key] = subvalue
            elif isinstance(value, np.ndarray):
                if len(value.shape) == 3:
                    obs_values[key] = value
                elif len(value.shape) == 1:
                    for i in range(value.shape[0]):
                        obs_values[f"{key}.{i}"] = value[i]
                else:
                    flat_value = value.flatten()
                    for i in range(len(flat_value)):
                        obs_values[f"{key}.{i}"] = flat_value[i]
            else:
                obs_values[key] = value

        # Prepare action values: decompose action array
        # Note: feature names are just indices ('0', '1', ...), not full keys
        action_values = {str(i): action[i] for i in range(len(action))}

        # Build frames
        observation_frame = build_dataset_frame(dataset.features, obs_values, prefix=OBS_STR)
        action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
        frame = {**observation_frame, **action_frame, "task": task_description}

        # Add to dataset
        dataset.add_frame(frame)
        print(f"  Added frame {step + 1}")

        # Step environment
        obs, _, _, _, _ = env.step(action)

    # Save episode
    print("\n6. Saving episode...")
    dataset.save_episode()
    print(f"Episode saved! Total episodes: {len(dataset)}")

    # Cleanup
    print("\n7. Cleaning up...")
    env.close()

    print("\n✓ Test passed! Dataset collection functionality works correctly.")
    print(f"Dataset location: {dataset.root}")


if __name__ == "__main__":
    test_dataset_creation()
