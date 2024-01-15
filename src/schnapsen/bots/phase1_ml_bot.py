from schnapsen.game import Bot, PlayerPerspective, SchnapsenDeckGenerator, Move, Trick, GamePhase
from typing import Optional, cast, Literal
from schnapsen.deck import Suit, Rank, Card
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import time
import pathlib
import random
import tensorflow as tf
from tensorflow.python.keras import layers
import numpy as np

class DQN(tf.keras.Model):
    """Deep Q-Network, a neural network model for estimating Q-values of actions."""
    def __init__(self, action_size: int):
        """Initialises the network, with a specified number of layers and constructs the class"""
        super(DQN, self).__init__() # Calling the constructor of the tf.keras.Model class
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(action_size, activation='linear')
    def call(self, state: np.ndarray):
        """Performs a forward pass."""
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_layer(x)

class Agent(PlayerPerspective):
    """Reinforcement Learning agent that uses the DQN algorithm to select actions."""
    def __init__(self, action_size: int):
        super().__init__()
        self.model = DQN(action_size)
        self.epsilon = 1.0 # Initial exploration rate
        self.epsilon_min = 0.01 # Minimum exploration rate
        self.epsilon_decay = 0.995 # Decay rate for exploring probabilities
        self.memory = [] # Replay memory
        self.gamma = 0.95 # Discount rate for future rewards
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    def memorize(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Store past experiences in the replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state: np.ndarray) -> int:
        """Selects an action to play based on the current state using the epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.model.output_layer.units) # Explore: select a random action
        state = np.reshape(state, [1, -1]) # Reshape state for the neural network
        act_values = self.model(state)
        return np.argmax(act_values[0]) # Exploit: select the action with the highest Q-value
    def replay(self, batch_size: int):
        """Trains the model using a batch of experiences from the replay memory."""
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = np.reshape(next_state, [1, -1])
                target = reward + self.gamma * np.amax(self.model(next_state)[0])
            state = np.reshape(state, [1, -1])
            target_f = self.model(state)
            target_f[0][action] = target
            self.model.train_on_batch(state, target_f)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Decrease epsilon