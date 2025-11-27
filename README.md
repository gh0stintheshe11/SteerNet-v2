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