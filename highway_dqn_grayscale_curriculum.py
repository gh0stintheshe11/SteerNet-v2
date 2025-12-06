import gymnasium
import highway_env
from matplotlib import pyplot as plt
import numpy as np
from time import sleep
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
import pickle

# Disable audio
os.environ["SDL_AUDIODRIVER"] = "dummy"


class FrameStackResetWrapper(gymnasium.Wrapper):
    """Fix black frames on reset"""
    def __init__(self, env):
        super().__init__(env)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if len(obs.shape) == 3 and obs.shape[0] >= 4:
            obs[0] = obs[3]
            obs[1] = obs[3]
            obs[2] = obs[3]
        return obs, info
    
    def step(self, action):
        return self.env.step(action)


class SpeedRewardWrapper(gymnasium.Wrapper):
    """Custom reward function with crash penalty"""
    def __init__(self, env, absolute_min_speed=0.0, cutoff_speed=20.0, max_speed=30.0, crash_reward=-50.0):
        super().__init__(env)
        self.absolute_min_speed = absolute_min_speed
        self.cutoff_speed = cutoff_speed
        self.max_speed = max_speed
        self.crash_reward = crash_reward
        
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        if hasattr(self.env.unwrapped, 'controlled_vehicles') and len(self.env.unwrapped.controlled_vehicles) > 0:
            ego_speed = self.env.unwrapped.controlled_vehicles[0].speed

            crashed = False
            if 'crashed' in info:
                crashed = info['crashed']
            else:
                crashed = getattr(self.env.unwrapped.controlled_vehicles[0], "crashed", False)
            
            info['speed'] = ego_speed
            
            if crashed:
                reward = self.crash_reward
            else:
                reward = (ego_speed - self.absolute_min_speed) / (self.max_speed - self.absolute_min_speed)
                if ego_speed > 20.0:
                    reward = reward + ego_speed / 20.0
                if action == 0 or action == 2:
                    reward = reward - 0.5
                    reward = max(reward, 0.0)

        return obs, reward, done, truncated, info


class DQN_CNN(nn.Module):
    """CNN-based DQN with speed input"""
    def __init__(self, input_channels=4, num_actions=5, input_height=240, input_width=64):
        super(DQN_CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(8, 4), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 3), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        
        conv_output_size = self._calculate_conv_output_size(input_height, input_width)
        
        self.fc1 = nn.Linear(conv_output_size + 1, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_actions)
    
    def _calculate_conv_output_size(self, height, width):
        height = (height - 8) // 4 + 1
        width = (width - 4) // 4 + 1
        height = (height - 4) // 2 + 1
        width = (width - 3) // 2 + 1
        height = (height - 3) // 1 + 1
        width = (width - 3) // 1 + 1
        print(f"conv_output_size: {64 * height * width}")
        return 64 * height * width
        
    def forward(self, x, speed):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, speed], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PrioritizedReplayBuffer:
    """PER buffer"""
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6
    
    def push(self, state, speed, action, reward, next_state, next_speed, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        experience = (state, speed, action, reward, next_state, next_speed, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = max_priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = [self.buffer[idx] for idx in indices]
        states, speeds, actions, rewards, next_states, next_speeds, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(speeds, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(next_speeds, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32)
        )
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + self.epsilon
    
    def __len__(self):
        return len(self.buffer)


class HighwayGrayscaleDQNAgent:
    """DQN Agent with Curriculum Learning and Simulated Annealing Epsilon"""
    
    def __init__(
        self,
        env,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=0.99,
        batch_size=32,
        target_update_freq=1000,
        buffer_capacity=10000,
        stack_size=4,
        per_alpha=0.6,
        per_beta=0.4,
        per_beta_increment=0.001
    ):
        self.env = env
        self.num_actions = self.env.action_space.n
        print(f"num_actions: {self.num_actions}")
        print(f"observation_space: {self.env.observation_space.shape}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.online_net = DQN_CNN(
            input_channels=stack_size, 
            num_actions=self.num_actions, 
            input_height=240, 
            input_width=64
        ).to(self.device)
        
        self.target_net = DQN_CNN(
            input_channels=stack_size, 
            num_actions=self.num_actions, 
            input_height=240, 
            input_width=64
        ).to(self.device)
        
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            alpha=per_alpha,
            beta=per_beta,
            beta_increment=per_beta_increment
        )
        
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_speeds = []
        self.episode_collisions = []
        self.episode_levels = []
        
    def preprocess_state(self, state):
        return state.astype('float32') / 255.0
    
    def select_action(self, state, speed, training=True):
        if training and random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            speed_tensor = torch.FloatTensor([[speed / 30.0]]).to(self.device)
            q_values = self.online_net(state_tensor, speed_tensor)
            return q_values.argmax(1).item()
    
    def update_network(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, speeds, actions, rewards, next_states, next_speeds, dones, indices, weights = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        speeds = torch.FloatTensor(speeds).unsqueeze(1).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        next_speeds = torch.FloatTensor(next_speeds).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        current_q_values = self.online_net(states, speeds).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states, next_speeds).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        td_errors = (current_q_values - target_q_values).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)
        
        loss = (weights * (current_q_values - target_q_values) ** 2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10)
        self.optimizer.step()
        
        return loss.item()

    def train(self, num_episodes=3000, max_steps_per_episode=1000, print_every=10):
        """Train with Curriculum Learning and Simulated Annealing Epsilon"""
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Observation shape: {self.env.observation_space.shape}")
        
        # ==================== CURRICULUM SETUP ====================
        curriculum_levels = [
            {"vehicles_count": 5,  "vehicles_density": 0.5},   # Level 0: Easy
            {"vehicles_count": 10, "vehicles_density": 0.75},  # Level 1: Medium
            {"vehicles_count": 15, "vehicles_density": 1.0},   # Level 2: Hard
            {"vehicles_count": 20, "vehicles_density": 1.25},  # Level 3: Full
        ]
        
        # Per-level epsilon settings (simulated annealing)
        # Same floor (0.02), different ceilings
        level_epsilon_start = [1.0, 0.5, 0.3, 0.15]
        level_epsilon_end = [0.02, 0.02, 0.02, 0.02]
        level_epsilon_decay = [0.99, 0.99, 0.99, 0.99]
        
        # Initialize level 0
        current_level = 0
        self.env.unwrapped.config.update(curriculum_levels[current_level])
        self.epsilon = level_epsilon_start[current_level]
        self.epsilon_end = level_epsilon_end[current_level]
        self.epsilon_decay = level_epsilon_decay[current_level]
        
        print(f"Starting at Level {current_level}: {curriculum_levels[current_level]}")
        print(f"Epsilon: {self.epsilon} -> {self.epsilon_end} (decay: {self.epsilon_decay})")
        
        # Level up requirements
        min_episodes_per_level = 200
        crash_rate_threshold = 0.20
        min_avg_speed = 22.0
        
        # Per-level tracking (reset on level up)
        level_collisions = []
        level_speeds = []
        episodes_at_current_level = 0
        
        # ==================== TRAINING LOOP ====================
        training_start_time = time.time()
        interval_start_time = time.time()
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = self.preprocess_state(state)
            
            current_speed = self.env.unwrapped.controlled_vehicles[0].speed
            current_speed_normalized = current_speed / 30.0

            episode_reward = 0
            episode_loss = 0
            loss_count = 0
            episode_speed_sum = current_speed
            episode_crashed = False
            
            for step in range(max_steps_per_episode):
                action = self.select_action(state, current_speed_normalized, training=True)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                next_state = self.preprocess_state(next_state)
                
                next_speed = info.get('speed', current_speed)
                next_speed_normalized = next_speed / 30.0
                
                episode_speed_sum += next_speed
                if 'crashed' in info and info['crashed']:
                    episode_crashed = True
                
                self.replay_buffer.push(
                    state, current_speed_normalized, action, reward, 
                    next_state, next_speed_normalized, done
                )
                
                state = next_state
                current_speed_normalized = next_speed_normalized
                episode_reward += reward
                self.steps += 1
                
                loss = self.update_network()
                if loss is not None:
                    episode_loss += loss
                    loss_count += 1
                
                if self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())
                
                if done:
                    break
            
            # Decay epsilon (within current level bounds)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Store global statistics (for plotting)
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step + 1)
            avg_speed = episode_speed_sum / (step + 2)
            self.episode_speeds.append(avg_speed)
            self.episode_collisions.append(1 if episode_crashed else 0)
            self.episode_levels.append(current_level)
            
            # Store per-level statistics (for level up check)
            level_collisions.append(1 if episode_crashed else 0)
            level_speeds.append(avg_speed)
            episodes_at_current_level += 1
            
            # ==================== CHECK LEVEL UP ====================
            if (episode + 1) % 50 == 0 and episodes_at_current_level >= min_episodes_per_level:
                if len(level_collisions) >= 100:
                    # Use only current level's last 100 episodes
                    recent_crash_rate = np.mean(level_collisions[-100:])
                    recent_avg_speed = np.mean(level_speeds[-100:])
                    
                    # Must meet ALL criteria
                    if (recent_crash_rate < crash_rate_threshold and 
                        recent_avg_speed > min_avg_speed and 
                        self.epsilon <= self.epsilon_end + 0.01 and
                        current_level < len(curriculum_levels) - 1):
                        
                        # LEVEL UP!
                        current_level += 1
                        
                        # Reset per-level tracking
                        level_collisions = []
                        level_speeds = []
                        episodes_at_current_level = 0
                        
                        # Update environment
                        self.env.unwrapped.config.update(curriculum_levels[current_level])
                        
                        # Reset epsilon (simulated annealing)
                        self.epsilon = level_epsilon_start[current_level]
                        self.epsilon_end = level_epsilon_end[current_level]
                        self.epsilon_decay = level_epsilon_decay[current_level]
                        
                        print(f"\n{'='*60}")
                        print(f"LEVEL UP! Now at Level {current_level}: {curriculum_levels[current_level]}")
                        print(f"Previous stats: Crash rate {recent_crash_rate:.1%}, Speed {recent_avg_speed:.1f}")
                        print(f"New epsilon: {self.epsilon} -> {self.epsilon_end}")
                        print(f"Buffer size: {len(self.replay_buffer)}")
                        print(f"{'='*60}\n")
            
            # ==================== PRINT PROGRESS ====================
            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(self.episode_rewards[-print_every:])
                avg_length = np.mean(self.episode_lengths[-print_every:])
                avg_loss = episode_loss / max(loss_count, 1)
                
                # Show per-level stats
                if len(level_collisions) >= 100:
                    recent_crash_rate = np.mean(level_collisions[-100:])
                    recent_avg_speed = np.mean(level_speeds[-100:])
                elif len(level_collisions) > 0:
                    recent_crash_rate = np.mean(level_collisions)
                    recent_avg_speed = np.mean(level_speeds)
                else:
                    recent_crash_rate = 0.0
                    recent_avg_speed = 0.0
                
                interval_time = time.time() - interval_start_time
                total_elapsed = time.time() - training_start_time
                avg_episode_time = interval_time / print_every
                
                if total_elapsed < 60:
                    elapsed_str = f"{total_elapsed:.0f}s"
                elif total_elapsed < 3600:
                    elapsed_str = f"{total_elapsed/60:.1f}m"
                else:
                    elapsed_str = f"{total_elapsed/3600:.1f}h"
                
                # Show progress toward level up
                level_ep_str = f"({episodes_at_current_level}/{min_episodes_per_level})"
                
                print(f"Ep {episode + 1}/{num_episodes} | "
                      f"Lv: {current_level} {level_ep_str} | "
                      f"Reward: {avg_reward:.1f} | "
                      f"Crash: {recent_crash_rate:.0%} | "
                      f"Speed: {recent_avg_speed:.1f} | "
                      f"Len: {avg_length:.0f} | "
                      f"Eps: {self.epsilon:.3f} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"{avg_episode_time:.2f}s/ep | "
                      f"{elapsed_str}")
                
                interval_start_time = time.time()
        
        print(f"\nTraining completed! Final level: {current_level}")

    def evaluate(self, num_episodes=10):
        """Evaluate trained agent"""
        print(f"\nEvaluating agent for {num_episodes} episodes...")
        eval_rewards = []
        eval_avg_speeds = []
        eval_crashes = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = self.preprocess_state(state)
            
            current_speed = self.env.unwrapped.controlled_vehicles[0].speed
            current_speed_normalized = current_speed / 30.0

            episode_reward = 0
            speeds = [current_speed]
            crashed = False
            done = False
            steps = 0

            while not done and steps < 1000:
                action = self.select_action(state, current_speed_normalized, training=False)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                ego_speed = info.get("speed", current_speed)
                speeds.append(ego_speed)
                current_speed_normalized = ego_speed / 30.0
                
                if 'crashed' in info and info['crashed']:
                    crashed = True

                done = terminated or truncated
                next_state = self.preprocess_state(next_state)
                state = next_state
                episode_reward += reward
                steps += 1

            avg_speed = np.mean(speeds)
            eval_rewards.append(episode_reward)
            eval_avg_speeds.append(avg_speed)
            eval_crashes.append(1 if crashed else 0)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
                  f"Steps = {steps}, Speed = {avg_speed:.1f}, Crashed = {crashed}")

        print(f"\n{'='*40}")
        print(f"Average Reward: {np.mean(eval_rewards):.2f}")
        print(f"Average Speed: {np.mean(eval_avg_speeds):.2f}")
        print(f"Crash Rate: {np.mean(eval_crashes):.0%}")
        print(f"{'='*40}")
        return eval_rewards, eval_avg_speeds

    def plot_training_progress(self, save_path="training_progress_curriculum.png"):
        """Plot training progress with level indicators"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Find level transition points for vertical lines
        level_changes = []
        if len(self.episode_levels) > 0:
            for i in range(1, len(self.episode_levels)):
                if self.episode_levels[i] != self.episode_levels[i-1]:
                    level_changes.append(i)
        
        # Rewards
        ax1.plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
        if len(self.episode_rewards) >= 50:
            window = 50
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(self.episode_rewards)), moving_avg, 'r-', linewidth=2, label='50-ep Moving Avg')
        for lc in level_changes:
            ax1.axvline(x=lc, color='g', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards (Curriculum Learning)')
        ax1.legend()
        ax1.grid(True)
        
        # Episode lengths
        ax2.plot(self.episode_lengths, alpha=0.6, label='Episode Length')
        if len(self.episode_lengths) >= 50:
            window = 50
            moving_avg = np.convolve(self.episode_lengths, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(self.episode_lengths)), moving_avg, 'r-', linewidth=2, label='50-ep Moving Avg')
        for lc in level_changes:
            ax2.axvline(x=lc, color='g', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Lengths (Curriculum Learning)')
        ax2.legend()
        ax2.grid(True)
        
        # Speeds
        ax3.plot(self.episode_speeds, alpha=0.6, label='Average Speed')
        if len(self.episode_speeds) >= 50:
            window = 50
            moving_avg = np.convolve(self.episode_speeds, np.ones(window)/window, mode='valid')
            ax3.plot(range(window-1, len(self.episode_speeds)), moving_avg, 'r-', linewidth=2, label='50-ep Moving Avg')
        ax3.axhline(y=22.0, color='orange', linestyle='--', label='Level-up threshold (22)')
        for lc in level_changes:
            ax3.axvline(x=lc, color='g', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Average Speed')
        ax3.set_title('Average Speed (Curriculum Learning)')
        ax3.legend()
        ax3.grid(True)
        
        # Collision rate with level indicators
        ax4.plot(self.episode_collisions, alpha=0.4, label='Collision (0/1)')
        if len(self.episode_collisions) >= 50:
            window = 50
            moving_avg = np.convolve(self.episode_collisions, np.ones(window)/window, mode='valid')
            ax4.plot(range(window-1, len(self.episode_collisions)), moving_avg, 'r-', linewidth=2, label='50-ep Crash Rate')
        ax4.axhline(y=0.2, color='orange', linestyle='--', label='20% Threshold')
        for lc in level_changes:
            ax4.axvline(x=lc, color='g', linestyle='--', alpha=0.7, label='Level Up' if lc == level_changes[0] else '')
        
        # Add level as secondary y-axis
        if len(self.episode_levels) > 0:
            ax4_twin = ax4.twinx()
            ax4_twin.plot(self.episode_levels, 'b-', alpha=0.3, linewidth=2)
            ax4_twin.set_ylabel('Level', color='b')
            ax4_twin.set_ylim(-0.5, 4.5)
            ax4_twin.set_yticks([0, 1, 2, 3])
        
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Collision Rate')
        ax4.set_title('Collision Rate with Level Progress')
        ax4.legend(loc='upper right')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training progress plot saved to {save_path}")
        plt.close()
    
    def save_model(self, path="highway_dqn_curriculum.pth"):
        torch.save({
            'online_net_state_dict': self.online_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_speeds': self.episode_speeds,
            'episode_collisions': self.episode_collisions,
            'episode_levels': self.episode_levels
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path="highway_dqn_curriculum.pth"):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint['online_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.episode_speeds = checkpoint.get('episode_speeds', [])
        self.episode_collisions = checkpoint.get('episode_collisions', [])
        self.episode_levels = checkpoint.get('episode_levels', [])
        print(f"Model loaded from {path}")
    
    def close(self):
        self.env.close()


def main():
    """Main training function"""
    
    stack_size = 4
    config = {
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (240, 64),
            "stack_size": stack_size,
            "weights": [0.2989, 0.5870, 0.1140],
            "scaling": 2.0,
        },
        "simulation_frequency": 15,
        "duration": 100,
        "vehicles_count": 5,
        "vehicles_density": 0.5,
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": np.linspace(10, 30, 5).tolist(),
        },
    }
    
    env = gymnasium.make('highway-v0', config=config, render_mode=None)
    env = FrameStackResetWrapper(env)
    env = SpeedRewardWrapper(env, crash_reward=-50.0)
    
    agent = HighwayGrayscaleDQNAgent(
        env=env,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=0.99,
        batch_size=512,
        target_update_freq=2000,
        buffer_capacity=100000,
        stack_size=stack_size,
        per_alpha=0.6,
        per_beta=0.4,
        per_beta_increment=0.001
    )
    
    # Train
    agent.train(num_episodes=3000, max_steps_per_episode=1000, print_every=10)
    
    # Save
    agent.plot_training_progress("training_progress_curriculum.png")
    agent.save_model("highway_dqn_curriculum.pth")
    
    # Evaluate
    agent.load_model("highway_dqn_curriculum.pth")
    agent.evaluate(num_episodes=30)
    
    agent.close()


if __name__ == "__main__":
    main()