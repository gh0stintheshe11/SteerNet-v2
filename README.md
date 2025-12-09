# SteerNet-v2

A highway driving agent implementation using the highway-env Gymnasium environment.

## Baseline Agent

The baseline agent implements a rule-based driving strategy that maintains safe highway driving behavior.

### Action Space

The agent can perform 5 discrete actions:
- `0: LANE_LEFT` - Change to the left lane
- `1: IDLE` - Maintain current speed and lane
- `2: LANE_RIGHT` - Change to the right lane
- `3: FASTER` - Accelerate
- `4: SLOWER` - Decelerate

### Observation Format

The observation is a 5×5 matrix:
- **Row 0**: Ego car (the agent's vehicle)
- **Rows 1-4**: Other cars on the highway
- **Columns**:
  - Col 0: Presence (1 if car exists, 0 otherwise)
  - Col 1: X-offset (longitudinal position, positive means ahead, relative to the ego car)
  - Col 2: Y-offset (lateral position, absolute value, lanes are split by 0.25 apart)
  - Col 3: X-velocity (forward velocity, relative to ego car)
  - Col 4: Y-velocity (lateral velocity, relative to ego car)

### Baseline Strategy

The baseline agent follows a simple driving strategy:

1. **No car ahead**: Speed up until reaching the target velocity (default 1.0)

2. **Car ahead detected**:
   - If too close (distance < 0.2):
     - Check left lane - if empty, change to left lane
     - Otherwise, check right lane - if empty, change to right lane
   
   - Otherwise follow the car at front:
     - If distance < 0.1: Slow down (too close)
     - If distance > 0.2: Speed up (too far)
     - Otherwise: Match the velocity of the car ahead

## Running the Baseline Agent

To run the baseline agent:

```bash
python highway_baseline.py
```

### Environment Configuration

- Environment: `highway-v0`
- Duration: 100 steps (modified from default 40)
- Render mode: `rgb_array`

## DQN Agent

The DQN (Deep Q-Network) agent implements a reinforcement learning approach to learn optimal driving behavior using deep neural networks.

### Model Architecture

The DQN uses a 3-layer fully connected neural network:
- **Input Layer**: 25 features (flattened 5×5 kinematic observation)
- **Hidden Layer 1**: 128 neurons with ReLU activation
- **Hidden Layer 2**: 24 neurons with ReLU activation
- **Output Layer**: 5 neurons (one Q-value per action)

### Key Features

- **Experience Replay**: Stores and samples past experiences to break temporal correlations
- **Target Network**: Separate target network updated periodically for stable learning
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation during training
- **Gradient Clipping**: Prevents exploding gradients during training

### Hyperparameters

Default training configuration:
- Learning Rate: `1e-4`
- Discount Factor (γ): `0.99`
- Epsilon Start: `1.0`
- Epsilon End: `0.025`
- Epsilon Decay: `0.998`
- Batch Size: `64`
- Replay Buffer Capacity: `6000`
- Target Network Update Frequency: `1000` steps
- Training Episodes: `1000`

## Running the DQN Agent

### Command Line Arguments

The kinematic DQN agent supports several command line arguments for flexible training and evaluation:

- `--train`: Enable training mode
- `--evaluate`: Enable evaluation mode
- `--episodes N`: Set number of episodes (default: 1000 for training, 30 for evaluation)
- `--render`: Enable rendering during evaluation
- `--model_path PATH`: Specify model save/load path (default: `highway_dqn_kinematic.pth`)

### Usage Examples

#### Training

```bash
# Train for 1000 episodes (default)
python highway_dqn_kinematic.py --train

# Train for 500 episodes
python highway_dqn_kinematic.py --train --episodes 500

# Train with custom model path
python highway_dqn_kinematic.py --train --model_path my_model.pth
```

#### Evaluation

```bash
# Evaluate with 30 episodes (no rendering)
python highway_dqn_kinematic.py --evaluate

# Evaluate with rendering enabled
python highway_dqn_kinematic.py --evaluate --render

# Evaluate 10 episodes with rendering
python highway_dqn_kinematic.py --evaluate --episodes 10 --render

# Evaluate using a custom model
python highway_dqn_kinematic.py --evaluate --model_path my_model.pth --render
```

#### Combined Training and Evaluation

```bash
# Train and then evaluate
python highway_dqn_kinematic.py --train --evaluate --episodes 100
```

**Note**: If you don't specify `--train` or `--evaluate`, the script defaults to training mode.

## DQN Agent with Grayscale Observations

The grayscale DQN agent uses CNN-based deep learning to process visual observations from the highway environment.

### Model Architecture

The grayscale DQN uses a convolutional neural network:
- **Input**: 4 stacked grayscale frames (4 × 240 × 64)
- **Conv Layer 1**: 32 filters, kernel=(8,4), stride=(4,4) with ReLU
- **Conv Layer 2**: 64 filters, kernel=(4,3), stride=(2,2) with ReLU
- **Conv Layer 3**: 64 filters, kernel=(3,3), stride=(1,1) with ReLU
- **FC Layer 1**: 512 neurons with ReLU
- **FC Layer 2**: 256 neurons with ReLU
- **Output Layer**: 5 neurons (one Q-value per action)

### Command Line Arguments

The grayscale DQN agent supports several command line arguments for flexible training and evaluation:

- `--train`: Enable training mode
- `--evaluate`: Enable evaluation mode
- `--episodes N`: Set number of episodes (default: 2000 for training, 30 for evaluation)
- `--render`: Enable rendering during evaluation
- `--model_path PATH`: Specify model save/load path (default: `highway_dqn_grayscale.pth`)

### Usage Examples

#### Training

```bash
# Train for 2000 episodes (default)
python highway_dqn_grayscale.py --train

# Train for 500 episodes
python highway_dqn_grayscale.py --train --episodes 500

# Train with custom model path
python highway_dqn_grayscale.py --train --model_path my_model.pth
```

#### Evaluation

```bash
# Evaluate with 30 episodes (no rendering)
python highway_dqn_grayscale.py --evaluate

# Evaluate with rendering enabled
python highway_dqn_grayscale.py --evaluate --render

# Evaluate 10 episodes with rendering
python highway_dqn_grayscale.py --evaluate --episodes 10 --render

# Evaluate using a custom model
python highway_dqn_grayscale.py --evaluate --model_path my_model.pth --render
```

#### Combined Training and Evaluation

```bash
# Train and then evaluate
python highway_dqn_grayscale.py --train --evaluate --episodes 100
```

**Note**: If you don't specify `--train` or `--evaluate`, the script defaults to training mode.

## Files

- `highway_baseline.py` - Rule-based baseline agent implementation
- `highway_dqn_kinematic.py` - DQN agent implementation with kinematic observations
- `highway_dqn_kinematic.pth` - Trained DQN model weights (generated after training)
- `highway_dqn_grayscale.py` - DQN agent implementation with grayscale (visual) observations
- `highway_dqn_grayscale.pth` - Trained grayscale DQN model weights (generated after training)
- `training_progress_dqn_kinematic.png` - Kinematic DQN training progress visualization (generated after training)
- `training_progress_dqn_grayscale.png` - Grayscale DQN training progress visualization (generated after training)