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

### Training a New Model

To train a new DQN agent from scratch:

```bash
python highway_dqn_kinematic.py
```

This will:
- Train for 1000 episodes
- Save the trained model to `highway_dqn_kinematic.pth`
- Generate a training progress plot: `training_progress_dqn_kinematic.png`
- Evaluate the trained model for 10 episodes

### Visualizing Performance

To render episodes with the trained model, set `render_mode="human"` or `render_mode="rgb_array"`:

```python
agent = HighwayKinematicDQNAgent(render_mode="human")
agent.load_model("highway_dqn_kinematic.pth")
agent.render_episode(num_episodes=5)
agent.close()
```

## Files

- `highway_baseline.py` - Rule-based baseline agent implementation
- `highway_dqn_kinematic.py` - DQN agent implementation with kinematic observations
- `highway_dqn_kinematic.pth` - Trained DQN model weights (generated after training)
- `training_progress_dqn_kinematic.png` - Training progress visualization (generated after training)