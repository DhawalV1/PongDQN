"""
Utility classes for DQN implementation
Contains ReplayBuffer and image preprocessing utilities
"""

import numpy as np
import cv2
from collections import deque

cv2.ocl.setUseOpenCL(False)


class ReplayBuffer:
    """
    Simple storage for transitions from an environment.
    """

    def __init__(self, size):
        """
        Initialise a buffer of a given size for storing transitions
        :param size: the maximum number of transitions that can be stored
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer. Old transitions will be overwritten if the buffer is full.
        :param state: the agent's initial state
        :param action: the action taken by the agent
        :param reward: the reward the agent received
        :param next_state: the subsequent state
        :param done: whether the episode terminated
        """
        data = (state, action, reward, next_state, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, indices):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            data = self._storage[i]
            state, action, reward, next_state, done = data
            states.append(np.array(state, copy=False))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from the buffer.
        :param batch_size: the number of transitions to sample
        :return: a mini-batch of sampled transitions
        """
        indices = np.random.randint(0, len(self._storage), size=batch_size)
        return self._encode_sample(indices)


class FrameProcessor:
    """
    Processes frames similar to the wrappers in the original code.
    Converts RGB to grayscale, resizes to 84x84, and stacks frames.
    """
    
    def __init__(self, stack_size=4):
        """
        Initialize frame processor
        :param stack_size: number of frames to stack
        """
        self.stack_size = stack_size
        self.frames = deque([], maxlen=stack_size)
        self.width = 84
        self.height = 84
        
    def reset(self):
        """Reset the frame stack"""
        self.frames.clear()
        
    def process_frame(self, frame):
        """
        Process a single frame: convert to grayscale and resize to 84x84
        :param frame: RGB frame of shape (210, 160, 3)
        :return: processed frame of shape (84, 84, 1)
        """
        # Convert RGB to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize to 84x84
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]  # Add channel dimension
    
    def get_stacked_frames(self, frame):
        """
        Process and stack frames
        :param frame: new frame to add
        :return: stacked frames of shape (stack_size, 84, 84)
        """
        processed = self.process_frame(frame)
        
        # If first frame, replicate it stack_size times
        if len(self.frames) == 0:
            for _ in range(self.stack_size):
                self.frames.append(processed)
        else:
            self.frames.append(processed)
            
        # Stack frames along first dimension
        stacked = np.concatenate(list(self.frames), axis=2)  # (84, 84, stack_size)
        stacked = np.transpose(stacked, (2, 0, 1))  # (stack_size, 84, 84)
        
        return stacked


class LazyFrames(object):
    """
    This object ensures that common frames between the observations are only stored once.
    It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay buffers.
    """
    def __init__(self, frames):
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, i):
        return self._frames[i]