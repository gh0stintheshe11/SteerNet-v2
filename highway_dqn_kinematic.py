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

class DQN(nn.Module):
    """
    The Deep Q-Network (DQN) model 
    Implement the MLP described in the assignment 
    """
    
    def __init__(self, num_actions, feature_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(feature_size, 128)
        self.fc2 = nn.Linear(128, 24)
        self.out = nn.Linear(24, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.out(x)
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

class HighwayKinematicDQNAgent:
    """DQN Agent for Highway with Kinematic Observation"""
    
    def __init__(
        self,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=32,
        target_update_freq=1000,
        buffer_capacity=10000,
        fast_skip = False,
        render_mode="rgb_array"
    ):
        self.env = gymnasium.make('highway-v0', render_mode=render_mode)
        config = {
            "simulation_frequency": 10,  # Hz (lower = faster, default is 15)
            #"policy_frequency": 1,  # Hz (higher = fewer steps per episode, default is 1)
            "duration": 100,  # Shorter episodes (default is 40)
            "vehicles_count": 20,  # Fewer vehicles to simulate (default is 50)
            #"lanes_count": 4,  # Fewer lanes (default is 4)
            #"offscreen_rendering": False,
            #"real_time_rendering": False,
        }
        self.env.unwrapped.configure(config)

        self.num_actions = self.env.action_space.n
        print ("num_actions: ", self.num_actions)
        print ("observation_space: ", self.env.observation_space.shape)
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu" #disable GPU for now, doesn't seem to help at all
        print(f"Using device: {self.device}")
        
        # Networks
        self.online_net = DQN(num_actions=self.num_actions, feature_size=5*5).to(self.device)
        self.target_net = DQN(num_actions=self.num_actions, feature_size=5*5).to(self.device)
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
        """Preprocess state from kinematic observation"""
        # Flatten the 5x5 kinematic observation to a 25-element vector
        state = state.flatten()
        return state
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
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
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10)
        self.optimizer.step()
        
        return loss.item()

    def train(self, num_episodes=500, max_steps_per_episode=1000, print_every=10):
        """Train the DQN agent"""
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Observation shape: {self.env.observation_space.shape}")
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = self.preprocess_state(state)

            episode_reward = 0
            episode_loss = 0
            loss_count = 0
            
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

    def plot_training_progress(self, save_path="training_progress.png"):
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
        ax1.set_title('Training Rewards Over Time')
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
        ax2.set_title('Episode Lengths Over Time')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training progress plot saved to {save_path}")
        plt.close()
    
    def save_model(self, path="highway_dqn_kinematic.pth"):
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
    
    def load_model(self, path="highway_dqn_kinematic.pth"):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint['online_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        print(f"Model loaded from {path}")
    
    def close(self):
        """Close the environment"""
        self.env.close()


def main():
    """Main training function"""
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create agent
    agent = HighwayKinematicDQNAgent(
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        target_update_freq=1000,
        buffer_capacity=6000,
        render_mode=None
    )
    
    # Train agent
    agent.train(num_episodes=1000, max_steps_per_episode=1000, print_every=10)
    
    # Plot progress
    agent.plot_training_progress("training_progress.png")
    
    # Save model
    agent.save_model("highway_dqn_kinematic.pth")
    
    # Evaluate agent
    agent.load_model("highway_dqn_kinematic.pth")
    agent.evaluate(num_episodes=3)
    
    # Close environment
    agent.close()


if __name__ == "__main__":
    main()
