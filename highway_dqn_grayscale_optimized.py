"""
Fixed version of highway_dqn_grayscale_optimized.py

Key fixes:
1. Added input normalization (x / 255.0) - CRITICAL
2. Changed epsilon decay from per-episode to per-step (linear)
3. Removed BatchNorm (optional: can keep but use eval() mode during action selection)
4. Added warmup period
5. Added Double DQN
"""

import gymnasium
import highway_env
from matplotlib import pyplot as plt
import numpy as np
from time import sleep
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
import pickle

os.environ["SDL_AUDIODRIVER"] = "dummy"


class FrameStackResetWrapper(gymnasium.Wrapper):
    """
    Wrapper to fix the black frames issue in highway-env during reset.
    """
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


class DQN_CNN_Fixed(nn.Module):
    """
    Fixed CNN-based Deep Q-Network
    
    Changes:
    1. REMOVED BatchNorm - causes issues with non-stationary RL data
    2. Added input normalization in forward()
    3. Using Huber loss compatible architecture
    """

    def __init__(self, input_channels=4, num_actions=5, input_height=240, input_width=64):
        super(DQN_CNN_Fixed, self).__init__()

        # CNN layers with asymmetric kernels (NO BatchNorm)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(8, 4), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 3), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))

        conv_output_size = self._calculate_conv_output_size(input_height, input_width)

        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_actions)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def _calculate_conv_output_size(self, height, width):
        height = (height - 8) // 4 + 1
        width = (width - 4) // 4 + 1
        height = (height - 4) // 2 + 1
        width = (width - 3) // 2 + 1
        height = (height - 3) // 1 + 1
        width = (width - 3) // 1 + 1
        print(f"Conv output size: {64 * height * width}")
        return 64 * height * width

    def forward(self, x):
        # FIX #1: Normalize input to [0, 1]
        x = x / 255.0
        
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    """Experience replay buffer"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class HighwayGrayscaleDQNFixed:
    """
    Fixed DQN Agent with:
    1. Input normalization
    2. Linear epsilon decay (per-step, not per-episode)
    3. Double DQN
    4. Warmup period
    5. Huber loss
    """

    def __init__(
        self,
        env,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay_steps=100000,  # FIX #2: Linear decay over N steps
        batch_size=256,
        target_update_freq=1000,
        buffer_capacity=100000,  # Larger buffer
        stack_size=4,
        update_every=4,
        warmup_steps=5000,  # FIX #3: Warmup period
        use_double_dqn=True  # FIX #4: Double DQN
    ):
        self.env = env
        self.num_actions = self.env.action_space.n
        self.use_double_dqn = use_double_dqn
        self.warmup_steps = warmup_steps
        
        print("num_actions:", self.num_actions)
        print("observation_space:", self.env.observation_space.shape)
        print(f"Using Double DQN: {use_double_dqn}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Using fixed architecture (no BatchNorm)
        self.online_net = DQN_CNN_Fixed(
            input_channels=stack_size,
            num_actions=self.num_actions,
            input_height=240,
            input_width=64
        ).to(self.device)

        self.target_net = DQN_CNN_Fixed(
            input_channels=stack_size,
            num_actions=self.num_actions,
            input_height=240,
            input_width=64
        ).to(self.device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        # FIX #5: Huber loss instead of MSE
        self.loss_function = nn.SmoothL1Loss()
        
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_every = update_every

        self.steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []

    def preprocess_state(self, state):
        """Preprocess state - keep as float32"""
        return state.astype(np.float32)

    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return self.env.action_space.sample()

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_tensor)
            return q_values.argmax(1).item()

    def update_epsilon(self):
        """Linear epsilon decay per step"""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.steps / self.epsilon_decay_steps)
        )

    def update_network(self):
        """Update network using Double DQN"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: online net selects action, target net evaluates
                next_actions = self.online_net(next_states).argmax(1)
                next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q_values = self.target_net(next_states).max(1)[0]

            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_function(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10)
        self.optimizer.step()

        return loss.item()

    def train(self, num_episodes=5000, max_steps_per_episode=1000, print_every=50):
        """Train the DQN agent"""
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Warmup: {self.warmup_steps} steps, Epsilon decay over: {self.epsilon_decay_steps} steps")
        
        best_avg_reward = float('-inf')

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = self.preprocess_state(state)

            episode_reward = 0
            episode_loss = 0
            loss_count = 0

            for step in range(max_steps_per_episode):
                action = self.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self.preprocess_state(next_state)

                self.replay_buffer.push(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                self.steps += 1

                # Update epsilon per step (not per episode!)
                self.update_epsilon()

                # Only learn after warmup and every N steps
                if self.steps > self.warmup_steps and self.steps % self.update_every == 0:
                    loss = self.update_network()
                    if loss is not None:
                        episode_loss += loss
                        loss_count += 1
                        self.losses.append(loss)

                if self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                if done:
                    break

            # NO per-episode epsilon decay - it's now per-step

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step + 1)

            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(self.episode_rewards[-print_every:])
                avg_length = np.mean(self.episode_lengths[-print_every:])
                avg_loss = episode_loss / max(loss_count, 1)
                
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    self.save_model("highway_dqn_grayscale_fixed_best.pth")
                
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.0f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Buffer: {len(self.replay_buffer)} | "
                      f"Steps: {self.steps}")

        print(f"\nTraining completed! Best avg reward: {best_avg_reward:.2f}")

    def evaluate(self, num_episodes=10):
        """Evaluate the trained agent"""
        print(f"\nEvaluating for {num_episodes} episodes...")
        eval_rewards = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = self.preprocess_state(state)
            episode_reward = 0
            done = False
            steps = 0

            while not done and steps < 1000:
                action = self.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = self.preprocess_state(next_state)
                episode_reward += reward
                steps += 1

            eval_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")

        avg_reward = np.mean(eval_rewards)
        print(f"\nAverage Evaluation Reward: {avg_reward:.2f} ± {np.std(eval_rewards):.2f}")
        return eval_rewards

    def plot_training_progress(self, save_path="training_progress_grayscale_fixed.png"):
        """Plot training progress"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Rewards
        ax1 = axes[0]
        ax1.plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
        if len(self.episode_rewards) >= 50:
            window = 50
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(self.episode_rewards)), moving_avg, 'r-', linewidth=2, 
                    label=f'{window}-Episode Moving Avg')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards - Fixed DQN (Normalized + Linear ε-decay + Double DQN)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Episode lengths
        ax2 = axes[1]
        ax2.plot(self.episode_lengths, alpha=0.6, label='Episode Length')
        if len(self.episode_lengths) >= 50:
            window = 50
            moving_avg = np.convolve(self.episode_lengths, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(self.episode_lengths)), moving_avg, 'r-', linewidth=2,
                    label=f'{window}-Episode Moving Avg')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Episode Lengths Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Loss
        ax3 = axes[2]
        if self.losses:
            ax3.plot(self.losses, alpha=0.3, label='Loss')
            if len(self.losses) >= 100:
                window = 100
                moving_avg = np.convolve(self.losses, np.ones(window)/window, mode='valid')
                ax3.plot(range(window-1, len(self.losses)), moving_avg, 'r-', linewidth=2,
                        label=f'{window}-Step Moving Avg')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
        plt.close()

    def save_model(self, path="highway_dqn_grayscale_fixed.pth"):
        torch.save({
            'online_net_state_dict': self.online_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path="highway_dqn_grayscale_fixed.pth"):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint['online_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint.get('steps', 0)
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.losses = checkpoint.get('losses', [])
        print(f"Model loaded from {path}")

    def close(self):
        self.env.close()


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Environment config (same as yours)
    stack_size = 4
    config = {
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (240, 64),
            "stack_size": stack_size,
            "weights": [0.2989, 0.5870, 0.1140],
            "scaling": 2.0,
        },
        "collision_reward": -10.0,
        "simulation_frequency": 10,
        "duration": 100,
        "vehicles_count": 20,
    }
    env = gymnasium.make('highway-v0', config=config, render_mode=None)
    env = FrameStackResetWrapper(env)

    # Create fixed agent
    agent = HighwayGrayscaleDQNFixed(
        env=env,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay_steps=100000,  # Linear decay over 100k steps
        batch_size=256,
        target_update_freq=1000,
        buffer_capacity=100000,
        stack_size=stack_size,
        update_every=4,
        warmup_steps=5000,
        use_double_dqn=True
    )

    # Train longer
    agent.train(num_episodes=5000, max_steps_per_episode=1000, print_every=50)

    agent.plot_training_progress("training_progress_grayscale_fixed.png")
    agent.save_model("highway_dqn_grayscale_fixed_final.pth")

    # Evaluate best model
    agent.load_model("highway_dqn_grayscale_fixed_best.pth")
    rewards = agent.evaluate(num_episodes=20)
    print(f"\nFinal average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

    agent.close()


if __name__ == "__main__":
    main()