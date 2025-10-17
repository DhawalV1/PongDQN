# --- START OF FILE agent.py ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from utils import ReplayBuffer, FrameProcessor


class QNetwork(nn.Module):
    """
    Deep Q-Network following the Nature DQN architecture.
    Input: Stacked grayscale frames (4, 84, 84)
    Output: 6 Q-values (one for each action)
    """
    def __init__(self, input_channels=4, output_size=6):
        super(QNetwork, self).__init__()
        
        # Convolutional layers (Nature DQN architecture)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(in_features=64*7*7, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=output_size)
        )

    def forward(self, x):
        """Forward pass through the network"""
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class StudentAgent:
    """
    DQN Agent for PettingZoo Pong following the Nature DQN implementation.
    
    This agent implements:
    - Experience replay
    - Target network
    - Frame stacking (4 frames)
    - Epsilon-greedy exploration
    - Double DQN (optional)
    """
    
    def __init__(self, agent_id, action_space, use_double_dqn=True):
        """
        Initialize the DQN agent
        
        Args:
            agent_id: Agent identifier (0 or 1)
            action_space: Gym action space
            use_double_dqn: Whether to use Double DQN
        """
        self.agent_id = agent_id
        self.action_space = action_space
        self.n_actions = 6  # PettingZoo Pong has 6 actions
        
        # Hyperparameters (matching the original DQN implementation)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.9995
        self.epsilon = self.epsilon_start
        self.lr = 0.00025
        self.target_update_freq = 10000
        self.learning_starts = 10000
        self.use_double_dqn = use_double_dqn
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Networks
        self.policy_network = QNetwork(input_channels=4, output_size=self.n_actions).to(self.device)
        self.target_network = QNetwork(input_channels=4, output_size=self.n_actions).to(self.device)
        self.update_target_network()
        self.target_network.eval()
        
        # Optimizer (using RMSprop as in original implementation)
        self.optimizer = torch.optim.RMSprop(self.policy_network.parameters(), lr=self.lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(size=100000)
        
        # Frame processor for stacking frames
        self.frame_processor = FrameProcessor(stack_size=4)
        
        # Training step counter
        self.steps = 0
        
    def act(self, obs):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            obs: RGB image observation of shape (210, 160, 3)
            
        Returns:
            action: An integer (0-5) representing the action to take
        """
        # Process observation (convert to grayscale and stack frames)
        processed_obs = self.frame_processor.get_stacked_frames(obs)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            # Convert to tensor and add batch dimension
            state = torch.from_numpy(processed_obs).float().unsqueeze(0).to(self.device) / 255.0
            
            with torch.no_grad():
                q_values = self.policy_network(state)
                action = q_values.max(1)[1].item()
                
            return action
    
    def update(self, obs, action, reward, next_obs, done):
        """
        Update the agent using a transition.
        
        Args:
            obs: Current observation (RGB image)
            action: Action taken
            reward: Reward received
            next_obs: Next observation (RGB image)
            done: Whether episode is finished
        """
        # Process observations
        processed_obs = self.frame_processor.get_stacked_frames(obs)
        processed_next_obs = self.frame_processor.get_stacked_frames(next_obs)
        
        # Store transition in replay buffer
        self.memory.add(processed_obs, action, reward, processed_next_obs, done)
        
        self.steps += 1
        
        # Start learning after collecting enough samples
        if self.steps < self.learning_starts:
            return
        
        # Perform learning update
        if len(self.memory) >= self.batch_size:
            self._optimize_td_loss()
        
        # Update target network periodically
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()
        
        # Reset frame processor on episode end
        if done:
            self.frame_processor.reset()
    
    def _optimize_td_loss(self):
        """
        Optimize the TD-error over a single minibatch of transitions
        """
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Normalize states
        states = np.array(states) / 255.0
        next_states = np.array(next_states) / 255.0
        
        # Convert to tensors
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        
        # Compute target Q-values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use policy network to select actions, target network to evaluate
                _, max_next_action = self.policy_network(next_states).max(1)
                max_next_q_values = self.target_network(next_states).gather(1, max_next_action.unsqueeze(1)).squeeze()
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states)
                max_next_q_values, _ = next_q_values.max(1)
            
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # Compute current Q-values
        input_q_values = self.policy_network(states)
        input_q_values = input_q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Compute loss (Huber loss / smooth L1 loss)
        loss = F.smooth_l1_loss(input_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clean up
        del states
        del next_states
    
    def update_target_network(self):
        """
        Update the target Q-network by copying weights from the policy network
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())
    
    def update_epsilon(self):
        """
        Decay epsilon for exploration
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_epsilon(self):
        """Get current epsilon value"""
        return self.epsilon
    
    def reset_epsilon(self):
        """Reset epsilon to start value"""
        self.epsilon = self.epsilon_start
    
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filepath)

        # print(f"Model saved to {filepath}")
    
    def load_model(self, model_data):
        """
        Load a trained model.
        
        Args:
            model_data: The loaded model data (either state dict or checkpoint)
        """
        if isinstance(model_data, dict):
            if 'policy_network' in model_data:
                # Full checkpoint with optimizer state
                self.policy_network.load_state_dict(model_data['policy_network'])
                self.target_network.load_state_dict(model_data['target_network'])
                if 'optimizer' in model_data:
                    self.optimizer.load_state_dict(model_data['optimizer'])
                if 'epsilon' in model_data:
                    self.epsilon = model_data['epsilon']
                if 'steps' in model_data:
                    self.steps = model_data['steps']
            else:
                # Just state dict
                self.policy_network.load_state_dict(model_data)
                self.target_network.load_state_dict(model_data)
        else:
            # PyTorch model object
            if hasattr(model_data, 'state_dict'):
                state_dict = model_data.state_dict()
                self.policy_network.load_state_dict(state_dict)
                self.target_network.load_state_dict(state_dict)
            else:
                self.policy_network.load_state_dict(model_data)
                self.target_network.load_state_dict(model_data)
        
        self.policy_network.eval()
        self.target_network.eval()
        print("Model loaded successfully")