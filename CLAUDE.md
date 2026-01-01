# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeRobot is a hardware-agnostic robotics framework built on PyTorch that provides:

- **Unified Robot Interface**: Abstract `Robot` and `Teleoperator` classes for hardware-agnostic control
- **LeRobotDataset Format**: Standardized Parquet + MP4/images for efficient dataset storage and streaming
- **State-of-the-Art Policies**: Imitation learning (ACT, Diffusion, VQ-BeT), RL (TDMPC, SAC), and VLA models (Pi0, GR00T, SmolVLA, XVLA)
- **Simulation Environments**: Support for LIBERO and MetaWorld benchmarks

This fork includes custom work on UR5e robot integration with SpaceMouse and keyboard teleoperation.

## Common Commands

### Installation & Setup

```bash
# Install from source (for development)
pip install -e .

# Install with specific extras
pip install -e ".[dev]"           # Development tools
pip install -e ".[test]"          # Testing dependencies
pip install -e ".[aloha]"         # ALOHA simulation
pip install -e ".[metaworld]"     # MetaWorld simulation
pip install -e ".[libero]"        # LIBERO simulation

# Install git-lfs for test artifacts
git lfs install
git lfs pull
```

### Code Quality & Testing

```bash
# Install pre-commit hooks
pre-commit install

# Run all pre-commit checks manually
pre-commit run --all-files

# Run full test suite
pytest -sv ./tests

# Run specific test file
pytest -sv tests/test_specific_feature.py

# Run end-to-end tests for specific policy
make DEVICE=cpu test-act-ete-train
make DEVICE=cpu test-diffusion-ete-train
```

### Training & Evaluation

```bash
# Train a policy
lerobot-train \
  --policy=act \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet \
  --batch_size=8 \
  --steps=10000

# Evaluate a policy
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10

# Resume training from checkpoint
lerobot-train \
  --config_path=outputs/act/checkpoints/000002/pretrained_model/train_config.json \
  --resume=true
```

### Dataset Management

```bash
# Visualize dataset
lerobot-dataset-viz --repo-id=lerobot/aloha_mobile_cabinet

# Edit dataset (delete episodes, split, etc)
lerobot-edit-dataset --help

# Visualize image transforms
lerobot-imgtransform-viz --help
```

### Robot Control & Data Collection

```bash
# Find connected cameras
lerobot-find-cameras

# Find motor ports
lerobot-find-port

# Setup and calibrate motors
lerobot-setup-motors

# Calibrate robot
lerobot-calibrate --robot-type=so100

# Teleoperate robot
lerobot-teleoperate --robot-type=so100 --teleop-type=gamepad

# Record data
lerobot-record \
  --robot-type=so100 \
  --teleop-type=gamepad \
  --repo-id=username/dataset_name

# Replay recorded episode
lerobot-replay --episode=0 --repo-id=username/dataset_name
```

## Code Architecture

### Core Abstractions

**Robot Interface** (`src/lerobot/robots/robot.py`):

- Abstract base class that all robot implementations inherit from
- Key methods: `connect()`, `disconnect()`, `get_observation()`, `send_action()`
- Properties: `observation_features`, `action_features` (define data shapes/types)
- Each robot subclass lives in `src/lerobot/robots/<robot_name>/`
- Configuration defined via dataclasses in `configuration_<robot>.py`

**Teleoperator Interface** (`src/lerobot/teleoperators/teleoperator.py`):

- Abstract base class for teleoperation devices (gamepad, keyboard, SpaceMouse, leader arms)
- Key methods: `connect()`, `disconnect()`, `get_action()`, `send_feedback()`
- Properties: `action_features`, `feedback_features`
- Each teleoperator lives in `src/lerobot/teleoperators/<teleop_name>/`

**LeRobotDataset** (`src/lerobot/datasets/lerobot_dataset.py`):

- Unified dataset format: Parquet files for state/action + MP4/images for vision
- Integrates with HuggingFace Hub for easy sharing and streaming
- Version: v3.0 (see `CODEBASE_VERSION` constant)
- Supports delta timestamps, episode management, and video encoding/decoding

### Policy Architecture

**Policy Factory** (`src/lerobot/policies/factory.py`):

- `get_policy_class(name)`: Dynamic import to load policy classes
- Each policy has a config class (inherits from `PreTrainedConfig`) and model class (inherits from `PreTrainedPolicy`)
- Policy structure: `src/lerobot/policies/<policy_name>/`
  - `configuration_<policy>.py`: Config dataclass
  - `modeling_<policy>.py`: Model implementation
  - `README.md`: Policy-specific documentation

**Supported Policies**:

- Imitation Learning: ACT, Diffusion, VQ-BeT
- Reinforcement Learning: TDMPC, SAC, HIL-SERL
- Vision-Language-Action: Pi0, Pi0.5, GR00T, SmolVLA, XVLA, SARM

### Environment System

**Simulation Environments** (`src/lerobot/envs/`):

- `factory.py`: `make_env(env_config)` creates environment instances
- `configs.py`: Environment configuration dataclasses (MetaworldEnv, LiberoEnv, etc.)
- Environments implement gymnasium-style interface
- Each environment in separate file: `metaworld.py`, `libero.py`, etc.

### Training Pipeline

**Training Flow**:

1. `lerobot-train` script → `src/lerobot/scripts/lerobot_train.py`
2. Policy created via `make_policy()` in `policies/factory.py`
3. Dataset loaded via `LeRobotDataset` from HF Hub or local path
4. Training loop uses PyTorch + Accelerate for distributed training
5. Checkpoints saved to `outputs/<policy>/checkpoints/`
6. Metrics logged to wandb (if enabled) and Rerun (visualization)

**Evaluation Flow**:

1. `lerobot-eval` script → `src/lerobot/scripts/lerobot_eval.py`
2. Load pretrained policy from HF Hub or local checkpoint
3. Environment created via `make_env()`
4. Rollout policy for N episodes, collect success metrics

### Configuration System

**Draccus-based Configs** (`src/lerobot/configs/`):

- Uses `draccus` library for type-safe, composable configs
- Base classes: `PolicyConfig`, `EnvConfig`, `RobotConfig`, `TeleoperatorConfig`
- CLI args automatically generated from dataclass fields
- Supports config inheritance and field overrides

## Custom Extensions (This Fork)

### UR5e Robot Integration

- Location: `src/lerobot/robots/ur5e/`
- Custom teleoperation scripts in `examples/rj/`:
  - `spacemouse_teleop_ur5e.py`: SpaceMouse control for real UR5e robot
  - `spacemouse_teleop_sim.py`: SpaceMouse control for MetaWorld/LIBERO
  - `keyboard_teleop_sim.py`: Keyboard control for MetaWorld/LIBERO
- URSim Docker setup: `examples/rj/setup_ursim_docker.sh`

### Recent Changes

- End-effector frame control for SpaceMouse teleoperation
- UR5e robot support with URSim integration
- Improved SpaceMouse buffer flushing
- Keyboard teleoperation with CLI args

## Motor Systems

**Motor Bus Architecture** (`src/lerobot/motors/`):

- Abstract `MotorBus` class for motor communication
- Implementations: `dynamixel/`, `feetech/`
- Motor calibration stored in `~/.cache/huggingface/lerobot/calibration/`
- Use `lerobot-setup-motors` to configure motor IDs, models, and calibration

## Development Guidelines

### Adding a New Robot

1. Create directory: `src/lerobot/robots/<robot_name>/`
2. Implement:
   - `configuration_<robot>.py`: Robot config dataclass
   - `robot_<robot>.py`: Robot class inheriting from `Robot`
   - `__init__.py`: Export robot class and config
3. Register in `src/lerobot/robots/__init__.py`
4. Define `observation_features` and `action_features` properties
5. Implement required methods: `connect()`, `disconnect()`, `get_observation()`, `send_action()`

### Adding a New Policy

1. Create directory: `src/lerobot/policies/<policy_name>/`
2. Implement:
   - `configuration_<policy>.py`: Config dataclass inheriting from `PreTrainedConfig`
   - `modeling_<policy>.py`: Policy class inheriting from `PreTrainedPolicy`
   - `README.md`: Documentation
3. Add to `get_policy_class()` in `src/lerobot/policies/factory.py`
4. Implement required methods: `forward()`, `select_action()`, `update()`

### Code Style

- Python 3.10+
- Type hints enforced via mypy (gradual rollout in progress)
- Ruff for linting and formatting (replaces black, isort, flake8)
- Google-style docstrings
- Pre-commit hooks run: ruff, typos, prettier, bandit, mypy

### Testing Strategy

- Unit tests in `tests/` mirror `src/lerobot/` structure
- End-to-end tests defined in `Makefile` (train + eval pipelines)
- Use `pytest -sv` for verbose output
- Mock hardware using `mock-serial` for motor tests
