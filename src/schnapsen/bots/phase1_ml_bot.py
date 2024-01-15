from random import Random
from schnapsen.game import Bot, BotState, GamePlayEngine, GameState, Hand, PlayerPerspective, SchnapsenDeckGenerator, Move, Score, Trick, GamePhase
from typing import Optional, cast, Literal
from schnapsen.deck import CardCollection, Suit, Rank, Card
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import time
import pathlib
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers

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
        self.model = DQN(action_size)
        self.epsilon = 1.0 # Initial exploration rate
        self.epsilon_min = 0.01 # Minimum exploration rate
        self.epsilon_decay = 0.995 # Decay rate for exploring probabilities
        self.memory = [] # Replay memory
        self.gamma = 0.95 # Discount rate for future rewards
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    def get_game_history(self) -> list[tuple[PlayerPerspective, Trick | None]]:
        return super().get_game_history()
    def get_trump_suit(self) -> Suit:
        return super().get_trump_suit()
    def get_trump_card(self) -> Card | None:
        return super().get_trump_card()
    def get_talon_size(self) -> int:
        return super().get_talon_size()
    def get_phase(self) -> GamePhase:
        return super().get_phase()
    def __get_opponent_bot_state(self) -> BotState:
        return super().__get_opponent_bot_state()
    def __get_own_bot_state(self) -> BotState:
        return super().__get_own_bot_state()
    def seen_cards(self, leader_move: Move | None) -> CardCollection:
        return super().seen_cards(leader_move)
    def __past_tricks_cards(self) -> set[Card]:
        return super().__past_tricks_cards()
    def get_known_cards_of_opponent_hand(self) -> CardCollection:
        return super().get_known_cards_of_opponent_hand()
    def get_engine(self) -> GamePlayEngine:
        return super().get_engine()
    def get_state_in_phase_two(self) -> GameState:
        return super().get_state_in_phase_two()
    def make_assumption(self, leader_move: Move | None, rand: Random) -> GameState:
        return super().make_assumption(leader_move, rand)
    
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

Agent(PlayerPerspective())