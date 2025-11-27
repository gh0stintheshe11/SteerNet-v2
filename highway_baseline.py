import gymnasium
import highway_env
from matplotlib import pyplot as plt
import numpy as np
from time import sleep
import random

class BaselineAgent:
    """
    Baseline driving agent that follows the car in front or maintains target velocity.
    
    Action space:
    0: LANE_LEFT
    1: IDLE
    2: LANE_RIGHT
    3: FASTER
    4: SLOWER
    """
    
    def __init__(self, target_velocity=1.0, same_lane_threshold=0.2, velocity_threshold=0.05,
                 lane_separation=0.25, adjacent_lane_range=0.05):
        """
        Args:
            target_velocity: Target x-velocity (forward speed) when no car ahead
            same_lane_threshold: Max y-offset to consider a car in the same lane
            velocity_threshold: Velocity difference threshold for speed matching
            lane_separation: Distance between lane centers (default 0.25)
            adjacent_lane_range: Range around lane_separation to detect cars (default 0.05)


        Strategy:
        - If the lane ahead is empty, speed up, till max speed of 1.0.
        - Check if the lane to the left or right is empty. If empty, switch to that lane.
        - If there is a car ahead in the same lane, follow it till speed matches.
        - Maintain distance from the car ahead, if too close slow down, if too far speed up.
        """
        self.target_velocity = target_velocity
        self.same_lane_threshold = same_lane_threshold
        self.velocity_threshold = velocity_threshold
        self.lane_separation = lane_separation
        self.adjacent_lane_range = adjacent_lane_range
    
    def get_action(self, obs):
        """
        Select action based on observation.
        
        Observation format (5x5 matrix):
        - Row 0: Ego car
        - Rows 1-4: Other cars
        - Col 0: Presence (1 if car exists, 0 otherwise)
        - Col 1: X-offset (longitudinal position, positive means ahead)
        - Col 2: Y-offset (lateral position, 0 means same lane)
        - Col 3: X-velocity (forward velocity)
        - Col 4: Y-velocity (lateral velocity)
        """
        # Extract ego car information
        ego_vx = obs[0, 3]  # Ego x-velocity (forward speed)
        ego_vy = obs[0, 4]  # Ego y-velocity (lateral velocity)
        ego_x = obs[0, 1]
        ego_y = obs[0, 2]
        # Find car in front in the same lane
        car_ahead = None
        min_distance = float('inf')
        
        # Check adjacent lanes
        left_lane_occupied = False
        right_lane_occupied = False
        
        for i in range(1, 5):  # Check other cars (rows 1-4)
            presence = obs[i, 0]
            x_offset = obs[i, 1]
            y_offset = obs[i, 2]
            if presence == 0:
                continue
            
            # Check if car exists, is in same lane, and is ahead
            if abs(y_offset) < self.same_lane_threshold and x_offset > 0:
                # Found a car in front in the same lane
                #print(f"Car ahead found at x_offset: {x_offset:.3f}")
                if x_offset < min_distance:
                    min_distance = x_offset
                    car_ahead = i
            
            # Check left lane (y_offset around -0.25, range -0.3 to -0.2)
            left_lane_min = -self.lane_separation - self.adjacent_lane_range
            left_lane_max = -self.lane_separation + self.adjacent_lane_range
            if left_lane_min <= y_offset <= left_lane_max and x_offset > -0.1:
                left_lane_occupied = True
                #print(f"Car in left lane at y_offset: {y_offset:.3f}, x_offset: {x_offset:.3f}")
            
            # Check right lane (y_offset around +0.25, range 0.2 to 0.3)
            right_lane_min = self.lane_separation - self.adjacent_lane_range
            right_lane_max = self.lane_separation + self.adjacent_lane_range
            if right_lane_min <= y_offset <= right_lane_max and x_offset > -0.1:
                right_lane_occupied = True
                #print(f"Car in right lane at y_offset: {y_offset:.3f}, x_offset: {x_offset:.3f}")
        
        # Print lane status
        #print(f"Left lane: {'OCCUPIED' if left_lane_occupied else 'EMPTY'}, "
        #      f"Right lane: {'OCCUPIED' if right_lane_occupied else 'EMPTY'}")
        
        # Decision making
        if car_ahead is not None:
            # There's a car ahead in our lane
            distance_diff = obs[car_ahead, 1]
            
            # If too close or car ahead is slow, consider changing lanes
            if distance_diff < 0.2:
                if not left_lane_occupied and ego_y >= 0.25:
                    #print(f"Changing to LEFT lane (empty), ego_y: {ego_y:.3f}")
                    return 0  # LANE_LEFT
                elif not right_lane_occupied and ego_y <= 0.5:
                    #print(f"Changing to RIGHT lane (empty), ego_y: {ego_y:.3f}")
                    return 2  # LANE_RIGHT
            
            # Follow the car ahead - match its velocity and distance
            car_ahead_vx = obs[car_ahead, 3]
            velocity_diff = ego_vx - car_ahead_vx
            
            if distance_diff < 0.1:
                #print("Too close to the car ahead ", obs[car_ahead, 3])
                return 4  # SLOWER
            elif distance_diff > 0.2:
                #print("Too far from the car ahead")
                return 3  # FASTER
            else:
                if velocity_diff > self.velocity_threshold:
                    # We're going too fast, slow down
                    return 4  # SLOWER
                elif velocity_diff < -self.velocity_threshold:
                    # We're going too slow, speed up
                    return 3  # FASTER
                else:
                    # Speed is matched, maintain
                    return 1  # IDLE
        else:
            # No car ahead in current lane
            #print("No car ahead, maintaining speed")
            # No car ahead, try to reach target velocity
            if ego_vx < self.target_velocity - self.velocity_threshold:
                # Speed up to reach target velocity
                return 3  # FASTER
            elif ego_vx > self.target_velocity + self.velocity_threshold:
                # Slow down to target velocity
                return 4  # SLOWER
            else:
                # At target velocity
                return 1  # IDLE

    def evaluate(self, env, num_episodes=10):
        """Evaluate the baseline agent"""
        print(f"Evaluating baseline agent for {num_episodes} episodes...")
        eval_rewards = []
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 1000:
                action = self.get_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                state = next_state
                episode_reward += reward
                steps += 1
                if done or truncated:
                    break
            eval_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
        avg_reward = np.mean(eval_rewards)
        print(f"Average evaluation reward: {avg_reward:.2f} over {num_episodes} episodes")
        return eval_rewards

# Test the baseline agent
env = gymnasium.make('highway-v0', render_mode='rgb_array')
config = {
    "simulation_frequency": 10,  # Hz (lower = faster, default is 15)
    #"policy_frequency": 1,  # Hz (higher = fewer steps per episode, default is 1)
    "duration": 100,  # Shorter episodes (default is 40)
    "vehicles_count": 20,  # Fewer vehicles to simulate (default is 50)
    #"lanes_count": 4,  # Fewer lanes (default is 4)
    #"offscreen_rendering": False,
    #"real_time_rendering": False,
}
env.unwrapped.configure(config)
agent = BaselineAgent(target_velocity=1.0)

agent.evaluate(env, num_episodes=10)
#--------------------------------
# Render the baseline agent for 1 episode
#--------------------------------
seed = random.randint(0, 1000000)
print(f"Seed: {seed}")
#obs, info = env.reset(seed=763936)
obs, info = env.reset()
total_reward = 0

print("Running baseline agent...")
print("=" * 60)

for step in range(1000):
    # Get action from baseline agent
    action = agent.get_action(obs)
    
    # Execute action
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    
    # Print information
    action_names = ['LANE_LEFT', 'IDLE', 'LANE_RIGHT', 'FASTER', 'SLOWER']
    print(f"Step {step + 1}:")
    print(f"  Action: {action_names[action]}")
    print(f"  Ego velocity (forward, lateral): ({obs[0, 3]:.3f}, {obs[0, 4]:.3f})")
    print(f"  Reward: {reward:.2f}, Total: {total_reward:.2f}")
    print()
    
    env.render()
    
    if done or truncated:
        print(f"Episode ended at step {step + 1}")
        break

print("=" * 60)
print(f"Final total reward: {total_reward:.2f}")

# Show final frame
plt.imshow(env.render())
plt.title(f"Final Frame - Total Reward: {total_reward:.2f}")
plt.axis('off')
plt.tight_layout()
plt.show()