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

#disable audio to avoid weird warnings
os.environ["SDL_AUDIODRIVER"] = "dummy"

class FrameStackResetWrapper(gymnasium.Wrapper):
    """
    Wrapper to fix the black frames issue in highway-env during reset.
    On reset, the first 3 frames are black, only the 4th frame has content.
    This wrapper copies the 4th frame to frames 1-3 to provide meaningful initial state.
    """
    def __init__(self, env):
        super().__init__(env)
        
    def reset(self, **kwargs):
        """Reset the environment and fix the black frames issue"""
        obs, info = self.env.reset(**kwargs)
        
        # obs shape: (stack_size, height, width) = (4, 240, 64)
        # The first 3 frames (indices 0, 1, 2) are black
        # The 4th frame (index 3) has actual content
        # Copy frame 3 to frames 0, 1, 2
        if len(obs.shape) == 3 and obs.shape[0] >= 4:
            obs[0] = obs[3]
            obs[1] = obs[3]
            obs[2] = obs[3]
        
        return obs, info
    
    def step(self, action):
        """Step function passes through normally"""
        return self.env.step(action)

class DQN_CNN(nn.Module):
    """CNN-based Deep Q-Network for processing frame stacks"""
    
    def __init__(self, input_channels=4, num_actions=5, input_height=240, input_width=64):
        super(DQN_CNN, self).__init__()
        
        # CNN layers with asymmetric kernels for highway view (tall 240px, narrow 64px)
        # Kernel format: (height, width) - larger for height (240px), smaller for width (64px)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(8, 4), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 3), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        
        # Dynamically calculate the size after conv layers
        conv_output_size = self._calculate_conv_output_size(input_height, input_width)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 354)
        self.fc2 = nn.Linear(354, 128)
        self.fc3 = nn.Linear(128, num_actions)
    
    def _calculate_conv_output_size(self, height, width):
        """Calculate the flattened size after all conv layers
        
        For 240x64 input (highway env - tall view):
        - After conv1: 59 x 31
        - After conv2: 28 x 15
        - After conv3: 26 x 13
        - Final: 64 * 26 * 13 = 21,632 features
        """
        # Conv1: kernel=(8, 4), stride=(4, 2)
        height = (height - 8) // 4 + 1
        width = (width - 4) // 4 + 1
        
        # Conv2: kernel=(4, 3), stride=(2, 2)
        height = (height - 4) // 2 + 1
        width = (width - 3) // 2 + 1
        
        # Conv3: kernel=(3, 3), stride=(1, 1)
        height = (height - 3) // 1 + 1
        width = (width - 3) // 1 + 1
        
        #print("CNN output height: ", height)
        #print("CNN output width: ", width)
        print("conv_output_size: ", 64 * height * width)
        # Final size: 64 channels * height * width
        return 64 * height * width
        
    def forward(self, x):
        # x shape: (batch_size, input_channels, height, width)
        # Expected: (batch, 4, 240, 64) for highway env
        x = torch.relu(self.conv1(x))
        #print("conv1 x shape: ", x.shape)
        x = torch.relu(self.conv2(x))
        #print("conv2 x shape: ", x.shape)
        x = torch.relu(self.conv3(x))
        #print("conv3 x shape: ", x.shape)
        x = x.view(x.size(0), -1)  # Flatten
        #print("flatten x shape: ", x.shape)
        x = torch.relu(self.fc1(x))
        #print("fc1 x shape: ", x.shape)
        x = torch.relu(self.fc2(x))
        #print("fc2 x shape: ", x.shape)
        x = self.fc3(x)
        return x


class ReplayBuffer:
    """Experience replay buffer for storing experiences"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, path="replay_buffer.pkl"):
        """Save the replay buffer to a file
        
        Args:
            path: Path to save the replay buffer (default: replay_buffer.pkl)
        """
        # Convert deque to list for pickling
        buffer_list = list(self.buffer)
        
        # Save buffer data along with capacity info
        data = {
            'buffer': buffer_list,
            'capacity': self.buffer.maxlen,
            'size': len(self.buffer)
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Replay buffer saved to {path} ({len(self.buffer)} experiences)")
    
    def load(self, path="replay_buffer.pkl"):
        """Load the replay buffer from a file
        
        Args:
            path: Path to load the replay buffer from (default: replay_buffer.pkl)
        """
        if not os.path.exists(path):
            print(f"Warning: Replay buffer file {path} not found. Starting with empty buffer.")
            return
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Restore capacity if it was saved
        if 'capacity' in data:
            self.buffer = deque(data['buffer'], maxlen=data['capacity'])
        else:
            # Fallback for older saved buffers
            self.buffer = deque(data['buffer'], maxlen=self.buffer.maxlen)
        
        print(f"Replay buffer loaded from {path} ({len(self.buffer)} experiences)")

class HighwayGrayscaleDQNAgent:
    """DQN Agent for Highway with Grayscale Observation"""
    
    def __init__(
        self,
        env,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=32,
        target_update_freq=1000,
        buffer_capacity=10000,
        stack_size=4
    ):
        self.env = env
        self.num_actions = self.env.action_space.n
        print ("num_actions: ", self.num_actions)
        print ("observation_space: ", self.env.observation_space.shape)
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = "cpu" #disable GPU for now, doesn't seem to help at all
        print(f"Using device: {self.device}")

        # Networks
        self.online_net = DQN_CNN(input_channels=stack_size, num_actions=self.num_actions, input_height=240, input_width=64).to(self.device)
        self.target_net = DQN_CNN(input_channels=stack_size, num_actions=self.num_actions, input_height=240, input_width=64).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.loss_function = nn.MSELoss()
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Statistics
        self.steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
    def preprocess_state(self, state):
        """Preprocess state from grayscale observation"""
        # Rescale state values to [0, 1] by dividing by 255
        state = state.astype('float32') / 255.0
        return state
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            #print("state_tensor shape: ", state_tensor.shape)
            q_values = self.online_net(state_tensor)
            return q_values.argmax(1).item()
    
    def update_network(self):
        """Update the policy network using a batch from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.loss_function(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 5)
        self.optimizer.step()
        
        return loss.item()

    def train(self, num_episodes=500, max_steps_per_episode=1000, print_every=10):
        """Train the DQN agent"""
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Observation shape: {self.env.observation_space.shape}")
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            #print("state shape: ", state.shape)
            #print("state: ", state[3])
            #exit()
            #print ("starting preprocess_state")
            state = self.preprocess_state(state)

            episode_reward = 0
            episode_loss = 0
            loss_count = 0
            #exit()
            for step in range(max_steps_per_episode):
                # Select and perform action
                action = self.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self.preprocess_state(next_state)
                
                # Store transition
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                # Update state and reward
                state = next_state
                episode_reward += reward
                self.steps += 1
                
                # Train network
                loss = self.update_network()
                if loss is not None:
                    episode_loss += loss
                    loss_count += 1
                
                # Update target network
                if self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())
                
                if done:
                    break
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Store statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step + 1)
            
            # Print progress
            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(self.episode_rewards[-print_every:])
                avg_length = np.mean(self.episode_lengths[-print_every:])
                avg_loss = episode_loss / max(loss_count, 1)
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.0f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Buffer: {len(self.replay_buffer)}")
        
        print("Training completed!")
    
    def train_offline(self, num_steps=10000, print_every=1000):
        """Train the DQN agent using only existing replay buffer experiences
        
        Args:
            num_steps: Number of training steps (gradient updates) to perform
            print_every: Print progress every N steps
        """
        if len(self.replay_buffer) < self.batch_size:
            print(f"Error: Replay buffer has only {len(self.replay_buffer)} experiences, "
                  f"but batch_size is {self.batch_size}. Cannot train offline.")
            return
        
        print(f"Starting offline training for {num_steps} steps...")
        print(f"Replay buffer size: {len(self.replay_buffer)}")
        
        losses = []
        
        for step in range(num_steps):
            # Train network
            loss = self.update_network()
            if loss is not None:
                losses.append(loss)
            
            # Update target network
            if (step + 1) % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())
            
            # Print progress
            if (step + 1) % print_every == 0:
                avg_loss = np.mean(losses[-print_every:]) if losses else 0
                print(f"Step {step + 1}/{num_steps} | "
                      f"Avg Loss: {avg_loss:.4f} | "
                      f"Epsilon: {self.epsilon:.3f}")
        
        # Final statistics
        avg_loss = np.mean(losses) if losses else 0
        print(f"Offline training completed!")
        print(f"Average loss: {avg_loss:.4f}")
        print(f"Total training steps: {num_steps}")

    def evaluate(self, num_episodes=10, render=False):
        """Evaluate the trained agent"""
        print(f"\nEvaluating agent for {num_episodes} episodes...")
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
                next_state = self.preprocess_state(next_state)
                
                state = next_state
                episode_reward += reward
                steps += 1
            
            eval_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
        
        avg_reward = np.mean(eval_rewards)
        print(f"\nAverage Evaluation Reward: {avg_reward:.2f}")
        return eval_rewards

    def plot_training_progress(self, save_path="training_progress_dqn_grayscale.png"):
        """Plot training progress"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot rewards
        ax1.plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
        if len(self.episode_rewards) >= 10:
            window = 10
            moving_avg = np.convolve(self.episode_rewards, 
                                     np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(self.episode_rewards)), 
                    moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Moving Avg')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards Over Time DQN Grayscale')
        ax1.legend()
        ax1.grid(True)
        
        # Plot episode lengths
        ax2.plot(self.episode_lengths, alpha=0.6, label='Episode Length')
        if len(self.episode_lengths) >= 10:
            window = 10
            moving_avg = np.convolve(self.episode_lengths, 
                                     np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(self.episode_lengths)), 
                    moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Moving Avg')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Episode Lengths Over Time DQN Grayscale')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training progress plot saved to {save_path}")
        plt.close()
    
    def save_model(self, path="highway_dqn_grayscale.pth"):
        """Save the trained model"""
        torch.save({
            'online_net_state_dict': self.online_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path="highway_dqn_grayscale.pth"):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint['online_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        print(f"Model loaded from {path}")
    
    def save_replay_buffer(self, path="replay_buffer.pkl"):
        """Save the replay buffer to a file
        
        Args:
            path: Path to save the replay buffer (default: replay_buffer.pkl)
        """
        self.replay_buffer.save(path)
    
    def load_replay_buffer(self, path="replay_buffer.pkl"):
        """Load the replay buffer from a file
        
        Args:
            path: Path to load the replay buffer from (default: replay_buffer.pkl)
        """
        self.replay_buffer.load(path)

    def render_episode(self, num_episodes=5):
        """Render a single episode"""
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = self.preprocess_state(state)
            done = False
            steps = 0
            total_reward = 0
            while not done and steps < 1000:
                action = self.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self.preprocess_state(next_state)
                state = next_state
                steps += 1
                total_reward += reward
                if done or truncated:
                    break
            print ("Episode ended with reward: ", total_reward)
                #sleep(0.1)
        return
    
    def close(self):
        """Close the environment"""
        self.env.close()


def main():
    """Main training function"""
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create environment
    stack_size = 4
    config = {
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (240, 64),
            "stack_size": stack_size,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 2.0,
        },
        "collision_reward" : -10.0,
        "simulation_frequency": 10,  # Hz (lower = faster, default is 15)
        #"policy_frequency": 1,  # Hz (higher = fewer steps per episode, default is 1)
        "duration": 100,  # Shorter episodes (default is 40)
        "vehicles_count": 20,  # Fewer vehicles to simulate (default is 50)
        #"lanes_count": 4,  # Fewer lanes (default is 4)
        #"offscreen_rendering": False,
        #"real_time_rendering": False,
    }
    env = gymnasium.make('highway-v0', config=config, render_mode=None)
    env = FrameStackResetWrapper(env)
    
    # Create agent
    agent = HighwayGrayscaleDQNAgent(
        env=env,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.020,
        epsilon_decay=0.998,
        batch_size=128,
        target_update_freq=2000,
        buffer_capacity=100000,
        stack_size=stack_size
    )
    #agent.load_model("highway_dqn_grayscale.pth")
    # Train agent
    agent.train(num_episodes=2000, max_steps_per_episode=1000, print_every=10)
    
    # Train offline using pre-generated replay buffer
    #agent.load_replay_buffer("replay_buffer.pkl")
    #agent.train_offline(num_steps=10000, print_every=1000)

    # Plot progress
    agent.plot_training_progress("training_progress_dqn_grayscale.png")
    
    # Save model
    agent.save_model("highway_dqn_grayscale.pth")
    agent.save_replay_buffer("replay_buffer.pkl")

    # Evaluate agent
    agent.load_model("highway_dqn_grayscale.pth")
    rewards = agent.evaluate(num_episodes=30)
    print(f"Average reward: {np.mean(rewards):.2f}")

    
    #steps = agent.render_episode(num_episodes=5)
    # Close environment
    agent.close()


if __name__ == "__main__":
    main()
