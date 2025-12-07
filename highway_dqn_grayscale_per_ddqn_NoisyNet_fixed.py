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
    """Custom reward function"""
    def __init__(self, env, absolute_min_speed=0.0, cutoff_speed=20.0, max_speed=30.0, crash_reward=0.0):
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


class NoisyLinear(nn.Module):
    """Noisy Linear layer for exploration (Fortunato et al., 2017)
    
    Instead of epsilon-greedy, adds learnable noise to weights.
    Network learns WHEN to explore based on uncertainty.
    """
    
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable parameters (mu - the mean)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        
        # Learnable noise scale (sigma)
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Noise buffers (not learnable, resampled each step)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize mu and sigma parameters"""
        bound = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """Sample new noise - call this each step for fresh exploration"""
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()
    
    def forward(self, x):
        if self.training:
            # Training: add noise to weights
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # Evaluation: use clean weights (no exploration)
            weight = self.weight_mu
            bias = self.bias_mu
        
        return nn.functional.linear(x, weight, bias)


class DQN_CNN(nn.Module):
    """CNN-based DQN with NoisyNet FC layers"""
    
    def __init__(self, input_channels=4, num_actions=5, input_height=240, input_width=64, sigma_init=0.5):
        super(DQN_CNN, self).__init__()
        
        # CNN layers (no noise - only FC layers are noisy)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(8, 4), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 3), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        
        conv_output_size = self._calculate_conv_output_size(input_height, input_width)
        
        # NoisyNet FC layers
        self.fc1 = NoisyLinear(conv_output_size + 1, 512, sigma_init)
        self.fc2 = NoisyLinear(512, 256, sigma_init)
        self.fc3 = NoisyLinear(256, num_actions, sigma_init)
    
    def _calculate_conv_output_size(self, height, width):
        height = (height - 8) // 4 + 1
        width = (width - 4) // 4 + 1
        height = (height - 4) // 2 + 1
        width = (width - 3) // 2 + 1
        height = (height - 3) // 1 + 1
        width = (width - 3) // 1 + 1
        print(f"conv_output_size: {64 * height * width}")
        return 64 * height * width
    
    def reset_noise(self):
        """Reset noise for all NoisyLinear layers"""
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()
        
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
    """Prioritized Experience Replay buffer"""
    
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


class HighwayDDQNNoisyAgent:
    """Double DQN Agent with PER and NoisyNet
    
    Combines three improvements:
    1. Double DQN - reduces Q-value overestimation
    2. PER - samples important experiences more often
    3. NoisyNet - learned exploration (no epsilon needed)
    """
    
    def __init__(
        self,
        env,
        learning_rate=1e-4,
        gamma=0.99,
        batch_size=128,
        target_update_freq=1000,
        buffer_capacity=100000,
        stack_size=4,
        per_alpha=0.6,
        per_beta=0.4,
        per_beta_increment=0.001,
        sigma_init=0.5
    ):
        self.env = env
        self.num_actions = self.env.action_space.n
        print(f"num_actions: {self.num_actions}")
        print(f"observation_space: {self.env.observation_space.shape}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Networks with NoisyNet
        self.online_net = DQN_CNN(
            input_channels=stack_size, 
            num_actions=self.num_actions, 
            input_height=240, 
            input_width=64,
            sigma_init=sigma_init
        ).to(self.device)
        
        self.target_net = DQN_CNN(
            input_channels=stack_size, 
            num_actions=self.num_actions, 
            input_height=240, 
            input_width=64,
            sigma_init=sigma_init
        ).to(self.device)
        
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        
        # PER buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            alpha=per_alpha,
            beta=per_beta,
            beta_increment=per_beta_increment
        )
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Statistics
        self.steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_speeds = []
        self.episode_collisions = []
        
    def preprocess_state(self, state):
        return state.astype('float32') / 255.0
    
    def select_action(self, state, speed, training=True):
        """Action selection - NoisyNet handles exploration automatically"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            speed_tensor = torch.FloatTensor([[speed / 30.0]]).to(self.device)
            
            if training:
                self.online_net.train()  # Noise active
            else:
                self.online_net.eval()   # No noise (greedy)
            
            q_values = self.online_net(state_tensor, speed_tensor)
            
        return q_values.argmax(1).item()
    
    def update_network(self):
        """Double DQN update with PER and NoisyNet"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from PER
        states, speeds, actions, rewards, next_states, next_speeds, dones, indices, weights = \
            self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        speeds = torch.FloatTensor(speeds).unsqueeze(1).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        next_speeds = torch.FloatTensor(next_speeds).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Set to train mode (noise active)
        self.online_net.train()
        
        # Current Q values
        current_q_values = self.online_net(states, speeds).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # ==================== DOUBLE DQN ====================
        with torch.no_grad():
            # Step 1: ONLINE network selects best actions
            # Keep train mode - consistent noise for selection
            best_actions = self.online_net(next_states, next_speeds).argmax(1)
            
            # Step 2: TARGET network evaluates those actions
            next_q_values = self.target_net(next_states, next_speeds).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            
            # Calculate target
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        # ====================================================
        
        # Calculate TD errors for PER priority update
        td_errors = (current_q_values - target_q_values).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Weighted loss (importance sampling correction for PER bias)
        loss = (weights * (current_q_values - target_q_values) ** 2).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10)
        self.optimizer.step()
        
        # Reset noise after each update (fresh exploration)
        self.online_net.reset_noise()
        self.target_net.reset_noise()
        
        return loss.item()

    def train(self, num_episodes=3000, max_steps_per_episode=1000, print_every=10):
        """Train the agent"""
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Using: Double DQN + PER + NoisyNet (no epsilon!)")
        print(f"Observation shape: {self.env.observation_space.shape}")
        print(f"Batch size: {self.batch_size}")
        
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
            
            # Reset noise at episode start (new exploration pattern)
            self.online_net.reset_noise()
            self.online_net.train()  # Ensure train mode
            
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
            
            # Store statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step + 1)
            avg_speed = episode_speed_sum / (step + 2)
            self.episode_speeds.append(avg_speed)
            self.episode_collisions.append(1 if episode_crashed else 0)
            
            # Print progress
            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(self.episode_rewards[-print_every:])
                avg_length = np.mean(self.episode_lengths[-print_every:])
                avg_loss = episode_loss / max(loss_count, 1)
                
                # Calculate crash rate
                if len(self.episode_collisions) >= 100:
                    crash_rate = np.mean(self.episode_collisions[-100:])
                else:
                    crash_rate = np.mean(self.episode_collisions)
                
                avg_speed_recent = np.mean(self.episode_speeds[-print_every:])
                
                interval_time = time.time() - interval_start_time
                total_elapsed = time.time() - training_start_time
                avg_episode_time = interval_time / print_every
                
                if total_elapsed < 60:
                    elapsed_str = f"{total_elapsed:.0f}s"
                elif total_elapsed < 3600:
                    elapsed_str = f"{total_elapsed/60:.1f}m"
                else:
                    elapsed_str = f"{total_elapsed/3600:.1f}h"
                
                print(f"Ep {episode + 1}/{num_episodes} | "
                      f"Reward: {avg_reward:.1f} | "
                      f"Crash: {crash_rate:.0%} | "
                      f"Speed: {avg_speed_recent:.1f} | "
                      f"Len: {avg_length:.0f} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Buffer: {len(self.replay_buffer)} | "
                      f"{avg_episode_time:.2f}s/ep | "
                      f"{elapsed_str}")
                
                interval_start_time = time.time()
        
        print("Training completed!")

    def evaluate(self, num_episodes=10):
        """Evaluate trained agent (no noise)"""
        print(f"\nEvaluating agent for {num_episodes} episodes...")
        eval_rewards = []
        eval_avg_speeds = []
        eval_crashes = []
        
        # Ensure eval mode (no noise)
        self.online_net.eval()

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

    def plot_training_progress(self, save_path="training_progress_ddqn_noisy.png"):
        """Plot training progress"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Rewards
        ax1.plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
        if len(self.episode_rewards) >= 50:
            window = 50
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(self.episode_rewards)), moving_avg, 'r-', linewidth=2, label='50-ep Moving Avg')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards (Double DQN + PER + NoisyNet)')
        ax1.legend()
        ax1.grid(True)
        
        # Episode lengths
        ax2.plot(self.episode_lengths, alpha=0.6, label='Episode Length')
        if len(self.episode_lengths) >= 50:
            window = 50
            moving_avg = np.convolve(self.episode_lengths, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(self.episode_lengths)), moving_avg, 'r-', linewidth=2, label='50-ep Moving Avg')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Lengths (Double DQN + PER + NoisyNet)')
        ax2.legend()
        ax2.grid(True)
        
        # Speeds
        ax3.plot(self.episode_speeds, alpha=0.6, label='Average Speed')
        if len(self.episode_speeds) >= 50:
            window = 50
            moving_avg = np.convolve(self.episode_speeds, np.ones(window)/window, mode='valid')
            ax3.plot(range(window-1, len(self.episode_speeds)), moving_avg, 'r-', linewidth=2, label='50-ep Moving Avg')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Average Speed')
        ax3.set_title('Average Speed (Double DQN + PER + NoisyNet)')
        ax3.legend()
        ax3.grid(True)
        
        # Collision rate
        ax4.plot(self.episode_collisions, alpha=0.4, label='Collision (0/1)')
        if len(self.episode_collisions) >= 50:
            window = 50
            moving_avg = np.convolve(self.episode_collisions, np.ones(window)/window, mode='valid')
            ax4.plot(range(window-1, len(self.episode_collisions)), moving_avg, 'r-', linewidth=2, label='50-ep Crash Rate')
        ax4.axhline(y=0.15, color='g', linestyle='--', label='15% Target')
        ax4.axhline(y=0.20, color='orange', linestyle='--', label='20% Threshold')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Collision Rate')
        ax4.set_title('Collision Rate (Double DQN + PER + NoisyNet)')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training progress plot saved to {save_path}")
        plt.close()
    
    def save_model(self, path="highway_ddqn_noisy.pth"):
        torch.save({
            'online_net_state_dict': self.online_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_speeds': self.episode_speeds,
            'episode_collisions': self.episode_collisions
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path="highway_ddqn_noisy.pth"):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint['online_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.episode_speeds = checkpoint.get('episode_speeds', [])
        self.episode_collisions = checkpoint.get('episode_collisions', [])
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
        "vehicles_count": 20,
        "vehicles_density": 1.25,
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": np.linspace(10, 30, 5).tolist(),
        },
    }
    
    env = gymnasium.make('highway-v0', config=config, render_mode=None)
    env = FrameStackResetWrapper(env)
    env = SpeedRewardWrapper(env, crash_reward=0.0)
    
    agent = HighwayDDQNNoisyAgent(
        env=env,
        learning_rate=1e-4,
        gamma=0.99,
        batch_size=256,              # 4090
        target_update_freq=2000,
        buffer_capacity=100000,
        stack_size=stack_size,
        per_alpha=0.6,
        per_beta=0.4,
        per_beta_increment=0.001,
        sigma_init=0.3               # NoisyNet noise scale
    )
    
    # Train
    agent.train(num_episodes=5000, max_steps_per_episode=1000, print_every=10)
    
    # Save
    agent.plot_training_progress("training_progress_per_ddqn_NoisyNet_fixed.png")
    agent.save_model("highway_per_ddqn_NoisyNet_fixed.pth")
    
    # Evaluate
    agent.load_model("highway_per_ddqn_NoisyNet_fixed.pth")
    agent.evaluate(num_episodes=30)
    
    agent.close()


if __name__ == "__main__":
    main()