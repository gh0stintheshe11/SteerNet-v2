"""
Highway DQN with RGB Observation
Uses same camera as GrayscaleObservation but keeps full color
"""

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
from gymnasium import spaces
from highway_env.envs.common.observation import ObservationType
from highway_env.envs.common.graphics import EnvViewer

# Disable audio and video display
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["SDL_VIDEODRIVER"] = "dummy"


class RGBObservation(ObservationType):
    """
    RGB version of GrayscaleObservation.
    Uses the same bird's-eye camera but keeps full RGB color.
    Observation shape: (stack_size, height, width, 3)
    """

    def __init__(self, env, observation_shape, stack_size, scaling=None, centering_position=None, **kwargs):
        super().__init__(env)
        self.observation_shape = observation_shape
        self.stack_size = stack_size
        self.shape = (stack_size,) + self.observation_shape + (3,)
        self.obs = np.zeros(self.shape, dtype=np.uint8)

        viewer_config = env.config.copy()
        viewer_config.update(
            {
                "offscreen_rendering": True,
                "screen_width": self.observation_shape[0],
                "screen_height": self.observation_shape[1],
                "scaling": scaling or viewer_config.get("scaling", 2.0),
                "centering_position": centering_position or viewer_config.get("centering_position", [0.3, 0.5]),
            }
        )
        self.viewer = EnvViewer(env, config=viewer_config)

    def space(self):
        return spaces.Box(shape=self.shape, low=0, high=255, dtype=np.uint8)

    def observe(self):
        new_obs = self._render_to_rgb()
        self.obs = np.roll(self.obs, -1, axis=0)
        self.obs[-1, :, :, :] = new_obs
        return self.obs.copy()

    def _render_to_rgb(self):
        self.viewer.observer_vehicle = self.observer_vehicle
        self.viewer.display()
        raw_rgb = self.viewer.get_image()
        raw_rgb = np.moveaxis(raw_rgb, 0, 1)
        return raw_rgb[:, :, :3].astype(np.uint8)


class RGBObservationWrapper(gymnasium.Wrapper):
    """
    Wrapper that replaces the observation type with RGBObservation.
    Also fixes black frames on reset.
    """

    def __init__(self, env, observation_shape=(240, 64), stack_size=4, scaling=2.0):
        super().__init__(env)
        self.observation_shape = observation_shape
        self.stack_size = stack_size
        self.scaling = scaling

        self.rgb_obs = RGBObservation(
            env.unwrapped,
            observation_shape=observation_shape,
            stack_size=stack_size,
            scaling=scaling,
        )

        self.observation_space = self.rgb_obs.space()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        rgb_obs = self.rgb_obs.observe()
        for i in range(self.stack_size - 1):
            self.rgb_obs.obs[i] = self.rgb_obs.obs[-1]
        return self.rgb_obs.obs.copy(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        rgb_obs = self.rgb_obs.observe()
        return rgb_obs, reward, done, truncated, info


class SpeedRewardWrapper(gymnasium.Wrapper):
    """
    Wrapper to enforce minimum speed threshold for rewards.
    """

    def __init__(self, env, absolute_min_speed=0.0, cutoff_speed=20.0, max_speed=30.0, crash_reward=0.0):
        super().__init__(env)
        self.absolute_min_speed = absolute_min_speed
        self.cutoff_speed = cutoff_speed
        self.max_speed = max_speed
        self.crash_reward = crash_reward

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        if hasattr(self.env.unwrapped, "controlled_vehicles") and len(self.env.unwrapped.controlled_vehicles) > 0:
            ego_speed = self.env.unwrapped.controlled_vehicles[0].speed

            crashed = False
            if "crashed" in info:
                crashed = info["crashed"]
            else:
                crashed = getattr(self.env.unwrapped.controlled_vehicles[0], "crashed", False)

            info["speed"] = ego_speed

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
    """CNN for RGB frame stacks with speed input

    Input: (batch, 4, 240, 64, 3) -> reshape to (batch, 12, 240, 64)
    """

    def __init__(self, stack_size=4, num_actions=5, input_height=240, input_width=64):
        super(DQN_CNN, self).__init__()

        input_channels = stack_size * 3  # 4 frames * 3 RGB = 12 channels

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
        print(f"CNN output: {height}x{width} = {64 * height * width} features")
        return 64 * height * width

    def forward(self, x, speed):
        batch_size = x.size(0)
        x = x.permute(0, 1, 4, 2, 3)
        x = x.reshape(batch_size, -1, x.size(3), x.size(4))

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)

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
        probs = priorities**self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        batch = [self.buffer[idx] for idx in indices]
        states, speeds, actions, rewards, next_states, next_speeds, dones = zip(*batch)

        return (np.array(states), np.array(speeds, dtype=np.float32), np.array(actions), np.array(rewards, dtype=np.float32), np.array(next_states), np.array(next_speeds, dtype=np.float32), np.array(dones, dtype=np.float32), indices, np.array(weights, dtype=np.float32))

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + self.epsilon

    def __len__(self):
        return len(self.buffer)


class HighwayRGBDQNAgent:
    """DQN Agent for Highway with RGB Observation"""

    def __init__(self, env, learning_rate=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, batch_size=32, target_update_freq=1000, buffer_capacity=10000, stack_size=4, per_alpha=0.6, per_beta=0.4, per_beta_increment=0.001):
        self.env = env
        self.num_actions = self.env.action_space.n
        self.stack_size = stack_size

        print(f"num_actions: {self.num_actions}")
        print(f"observation_space: {self.env.observation_space.shape}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.online_net = DQN_CNN(stack_size=stack_size, num_actions=self.num_actions, input_height=240, input_width=64).to(self.device)

        self.target_net = DQN_CNN(stack_size=stack_size, num_actions=self.num_actions, input_height=240, input_width=64).to(self.device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)

        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity, alpha=per_alpha, beta=per_beta, beta_increment=per_beta_increment)

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

    def preprocess_state(self, state):
        return state.astype("float32") / 255.0

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

    def train(self, num_episodes=500, max_steps_per_episode=1000, print_every=10):
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Observation shape: {self.env.observation_space.shape}")

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

                next_speed = info.get("speed", current_speed)
                next_speed_normalized = next_speed / 30.0

                episode_speed_sum += next_speed
                if "crashed" in info and info["crashed"]:
                    episode_crashed = True

                self.replay_buffer.push(state, current_speed_normalized, action, reward, next_state, next_speed_normalized, done)

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

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step + 1)
            avg_speed = episode_speed_sum / (step + 2)
            self.episode_speeds.append(avg_speed)
            self.episode_collisions.append(1 if episode_crashed else 0)

            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(self.episode_rewards[-print_every:])
                avg_length = np.mean(self.episode_lengths[-print_every:])
                avg_loss = episode_loss / max(loss_count, 1)

                interval_time = time.time() - interval_start_time
                total_elapsed = time.time() - training_start_time
                avg_episode_time = interval_time / print_every

                if total_elapsed < 60:
                    elapsed_str = f"{total_elapsed:.0f}s"
                elif total_elapsed < 3600:
                    elapsed_str = f"{total_elapsed / 60:.1f}m"
                else:
                    elapsed_str = f"{total_elapsed / 3600:.1f}h"

                print(f"Episode {episode + 1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.0f} | Epsilon: {self.epsilon:.3f} | Loss: {avg_loss:.4f} | Buffer: {len(self.replay_buffer)} | {avg_episode_time:.2f}s/ep | Elapsed: {elapsed_str}")

                interval_start_time = time.time()

        print("Training completed!")

    def evaluate(self, num_episodes=10):
        print(f"\nEvaluating agent for {num_episodes} episodes...")
        eval_rewards = []
        eval_avg_speeds = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = self.preprocess_state(state)

            current_speed = self.env.unwrapped.controlled_vehicles[0].speed
            current_speed_normalized = current_speed / 30.0

            episode_reward = 0
            speeds = [current_speed]
            done = False
            steps = 0

            while not done and steps < 1000:
                action = self.select_action(state, current_speed_normalized, training=False)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                ego_speed = info.get("speed", current_speed)
                speeds.append(ego_speed)
                current_speed_normalized = ego_speed / 30.0

                done = terminated or truncated
                next_state = self.preprocess_state(next_state)
                state = next_state
                episode_reward += reward
                steps += 1

            avg_speed = np.mean(speeds) if speeds else 0.0
            eval_rewards.append(episode_reward)
            eval_avg_speeds.append(avg_speed)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}, Avg Speed = {avg_speed:.2f}")

        avg_reward = np.mean(eval_rewards)
        avg_speed_overall = np.mean(eval_avg_speeds)
        print(f"\nAverage Evaluation Reward: {avg_reward:.2f}")
        print(f"Average Evaluation Speed: {avg_speed_overall:.2f}")
        return eval_rewards, eval_avg_speeds

    def plot_training_progress(self, save_path="training_progress_dqn_rgb.png"):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        ax1.plot(self.episode_rewards, alpha=0.6, label="Episode Reward")
        if len(self.episode_rewards) >= 10:
            window = 10
            moving_avg = np.convolve(self.episode_rewards, np.ones(window) / window, mode="valid")
            ax1.plot(range(window - 1, len(self.episode_rewards)), moving_avg, "r-", linewidth=2)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.set_title("Training Rewards Over Time DQN RGB")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.episode_lengths, alpha=0.6, label="Episode Length")
        if len(self.episode_lengths) >= 10:
            window = 10
            moving_avg = np.convolve(self.episode_lengths, np.ones(window) / window, mode="valid")
            ax2.plot(range(window - 1, len(self.episode_lengths)), moving_avg, "r-", linewidth=2)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Episodes Length")
        ax2.set_title("Episode Lengths Over Time DQN RGB")
        ax2.legend()
        ax2.grid(True)

        ax3.plot(self.episode_speeds, alpha=0.6, label="Average Speed")
        if len(self.episode_speeds) >= 20:
            window = 20
            moving_avg = np.convolve(self.episode_speeds, np.ones(window) / window, mode="valid")
            ax3.plot(range(window - 1, len(self.episode_speeds)), moving_avg, "r-", linewidth=2)
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Average Speed")
        ax3.set_title("Average Speed Over Time DQN RGB")
        ax3.legend()
        ax3.grid(True)

        ax4.plot(self.episode_collisions, alpha=0.6, label="Collision")
        if len(self.episode_collisions) >= 20:
            window = 20
            moving_avg = np.convolve(self.episode_collisions, np.ones(window) / window, mode="valid")
            ax4.plot(range(window - 1, len(self.episode_collisions)), moving_avg, "r-", linewidth=2)
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Collision Rate")
        ax4.set_title("Collision Rate Over Time DQN RGB")
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training progress plot saved to {save_path}")
        plt.close()

    def save_model(self, path="highway_dqn_rgb.pth"):
        torch.save(
            {
                "online_net_state_dict": self.online_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "episode_rewards": self.episode_rewards,
                "episode_lengths": self.episode_lengths,
                "episode_speeds": self.episode_speeds,
                "episode_collisions": self.episode_collisions,
            },
            path,
        )
        print(f"Model saved to {path}")

    def load_model(self, path="highway_dqn_rgb.pth"):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint["online_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.episode_rewards = checkpoint["episode_rewards"]
        self.episode_lengths = checkpoint["episode_lengths"]
        self.episode_speeds = checkpoint.get("episode_speeds", [])
        self.episode_collisions = checkpoint.get("episode_collisions", [])
        print(f"Model loaded from {path}")

    def close(self):
        self.env.close()


def main():
    stack_size = 4

    config = {
        "simulation_frequency": 15,
        "duration": 100,
        "vehicles_count": 20,
        "vehicles_density": 1.25,
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": np.linspace(10, 30, 5).tolist(),
        },
    }

    env = gymnasium.make("highway-v0", config=config)
    env = RGBObservationWrapper(env, observation_shape=(240, 64), stack_size=stack_size, scaling=2.0)
    env = SpeedRewardWrapper(env, absolute_min_speed=0.0, cutoff_speed=20.0, max_speed=30.0, crash_reward=0.0)

    agent = HighwayRGBDQNAgent(env=env, learning_rate=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.020, epsilon_decay=0.998, batch_size=512, target_update_freq=2000, buffer_capacity=100000, stack_size=stack_size, per_alpha=0.6, per_beta=0.4, per_beta_increment=0.001)

    agent.train(num_episodes=3000, max_steps_per_episode=1000, print_every=10)

    agent.plot_training_progress("training_progress_dqn_rgb_per.png")
    agent.save_model("highway_dqn_rgb_per.pth")

    agent.load_model("highway_dqn_rgb_per.pth")
    rewards, speeds = agent.evaluate(num_episodes=30)
    print(f"Average reward: {np.mean(rewards):.2f}")

    agent.close()


if __name__ == "__main__":
    main()
