import numpy as np
from gymnasium import spaces
from pettingzoo.atari import pong_v3
import cv2

class PettingZooPongWrapper:
    """
    Multi-agent wrapper for PettingZoo Atari Pong environment.

    Specifications:
    - Uses pong_v3 environment from PettingZoo
    - RGB image observations (210, 160, 3)
    - 6 discrete actions: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT, 4=FIRE+RIGHT, 5=FIRE+LEFT
    - Same reward structure as original Pong (+1 for scoring, -1 for conceding)
    - 60 FPS frame rate
    """

    def __init__(self, render_mode=None):
        # Create the underlying PettingZoo environment
        self.env = pong_v3.env(render_mode=render_mode)

        # Number of agents
        self.n_agents = 2
        self.agents = None

        # Action space: 6 discrete actions per agent
        self.action_space = spaces.Tuple([
            spaces.Discrete(6),  # Agent 0: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT, 4=FIRE+RIGHT, 5=FIRE+LEFT
            spaces.Discrete(6)   # Agent 1: Same actions
        ])

        # Observation space: RGB images (210, 160, 3)
        self.observation_space = spaces.Tuple([
            spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=np.uint8),
            spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=np.uint8)
        ])

        # Track scores and game state
        self.score = [0, 0]
        self.prev_score = [0, 0]

        # Agent mapping (PettingZoo uses different agent names)
        self.agent_mapping = {}

    def _preprocess_observation(self, obs):
        """
        Preprocess PettingZoo observation to ensure correct format.

        Args:
            obs: Raw observation from PettingZoo

        Returns:
            Preprocessed observation as numpy array (210, 160, 3)
        """
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)

        # Ensure correct shape
        if obs.shape != (210, 160, 3):
            # Resize if necessary
            obs = cv2.resize(obs, (160, 210))
            if len(obs.shape) == 2:
                obs = cv2.cvtColor(obs, cv2.COLOR_GRAY2RGB)
            elif obs.shape[2] == 3:
                obs = obs  # Already RGB
            else:
                # Handle other cases by taking first 3 channels
                obs = obs[:, :, :3]

        return obs

    def _get_current_agent_id(self):
        """Get the current agent index (0 or 1)"""
        current_agent = self.env.agent_selection
        return self.agent_mapping.get(current_agent, 0)

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        self.env.reset(seed=seed)

        # Initialize agent mapping
        self.agents = list(self.env.agents)
        self.agent_mapping = {agent: i for i, agent in enumerate(self.agents)}

        # Reset scores
        self.score = [0, 0]
        self.prev_score = [0, 0]

        # Get initial observations for both agents
        observations = []
        for agent in self.agents:
            self.env.agent_selection = agent
            current_obs, _, _, _, _ = self.env.last()
            observations.append(self._preprocess_observation(current_obs))

        return observations, {}

    def step(self, actions):
        """
        Execute one step in the environment using PettingZoo's agent iteration.

        Args:
            actions: List of [agent0_action, agent1_action]

        Returns:
            observations, rewards, done, truncated, info
        """
        rewards = [0, 0]
        done = False
        truncated = False
        observations = [None, None]  # Store observations for both agents

        # Process each agent individually to ensure both get handled
        agents_processed = set()

        for agent in self.agents:
            if agent not in self.env.agents:
                continue

            # Set the agent selection manually
            self.env.agent_selection = agent
            agent_idx = self.agent_mapping[agent]
            agents_processed.add(agent_idx)

            # Get current observation, reward, termination, truncation, info
            obs, reward, termination, truncation, info = self.env.last()
            obs = self._preprocess_observation(obs)
            observations[agent_idx] = obs  # Store observation

            # Determine action for this agent
            if termination or truncation:
                action = None  # Required by PettingZoo for terminated agents
            else:
                # Use the provided action for this agent
                action = actions[agent_idx]

                # Validate action
                if not (0 <= action < 6):
                    print(f"⚠️  Agent {agent_idx} produced invalid action {action}, using NOOP")
                    action = 0  # NOOP

            # Step the environment for this agent
            self.env.step(action)

            # Accumulate rewards
            rewards[agent_idx] += reward

            # Check if game should end
            if termination or truncation:
                done = True

        # print(f"Debug: Processed agents: {agents_processed}, Expected: {set(range(len(self.agents)))}")

        # Update scores based on environment state
        if hasattr(self.env.env, 'score'):
            current_score = [self.env.env.score, 0]
            if current_score[0] > self.prev_score[0]:
                self.score[0] += 1
            elif current_score[0] < self.prev_score[0]:
                self.score[1] += 1
            self.prev_score = current_score.copy()

        # Check if game is over (one player reaches 21 points)
        if max(self.score) >= 21:
            done = True

        return observations, rewards, done, truncated, {}

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()

    def get_score(self):
        """Get current scores."""
        return self.score.copy()