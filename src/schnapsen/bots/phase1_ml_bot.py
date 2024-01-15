from random import Random
from schnapsen.game import Bot, BotState, GamePlayEngine, GameState, Hand, PlayerPerspective, SchnapsenDeckGenerator, Move, Score, Trick, GamePhase, RegularMove
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

class Agent:
    """Reinforcement Learning agent that uses the DQN algorithm to select actions."""
    def __init__(self, action_size: int):
        self.model = DQN(action_size)
        self.epsilon = 1.0 # Initial exploration rate
        self.epsilon_min = 0.01 # Minimum exploration rate
        self.epsilon_decay = 0.995 # Decay rate for exploring probabilities
        self.memory = [] # Replay memory
        self.gamma = 0.95 # Discount rate for future rewards
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    def encode_suits(self, card_suit: Card) -> list[int]:
        card_suit_one_hot: list[int]
        if card_suit == Suit.HEARTS:
            card_suit_one_hot = [0, 0, 0, 1]
        elif card_suit == Suit.CLUBS:
            card_suit_one_hot = [0, 0, 1, 0]
        elif card_suit == Suit.SPADES:
            card_suit_one_hot = [0, 1, 0, 0]
        elif card_suit == Suit.DIAMONDS:
            card_suit_one_hot = [1, 0, 0, 0]
        else:
            raise ValueError("Suit of card was not found!")
        return card_suit_one_hot
    def encode_ranks(self, card_rank):
        card_rank_one_hot: list[int]
        if card_rank == Rank.ACE:
            card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        elif card_rank == Rank.TWO:
            card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif card_rank == Rank.THREE:
            card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif card_rank == Rank.FOUR:
            card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif card_rank == Rank.FIVE:
            card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif card_rank == Rank.SIX:
            card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif card_rank == Rank.SEVEN:
            card_rank_one_hot = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif card_rank == Rank.EIGHT:
            card_rank_one_hot = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif card_rank == Rank.NINE:
            card_rank_one_hot = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif card_rank == Rank.TEN:
            card_rank_one_hot = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif card_rank == Rank.JACK:
            card_rank_one_hot = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif card_rank == Rank.QUEEN:
            card_rank_one_hot = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif card_rank == Rank.KING:
            card_rank_one_hot = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            raise AssertionError("Provided card Rank does not exist!")
        return card_rank_one_hot
    def get_feature_vector(self, move: Move) -> list[int]:
        if move is None:
            move_type_arr_numpy = [0, 0, 0]
            rank_encoding_arr_numpy = [0, 0, 0, 0]
            suit_encoding_arr_numpy = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            if move.is_marriage():
                move_type_arr = [0, 0, 1]
                card = move.queen_card
            elif move.is_trump_exchange():
                move_type_arr = [0, 1, 0]
                card = move.jack
            else:
                move_type_arr = [1, 0, 0]
                card = move.cards[0]
        move_type_arr_numpy = move_type_arr
        rank_encoding_arr_numpy = self.encode_ranks(card.rank)
        suit_encoding_arr_numpy = self.encode_suits(card.suit)

        return np.array(move_type_arr_numpy + rank_encoding_arr_numpy + suit_encoding_arr_numpy)
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

agent = Agent(5)
move = RegularMove(Card.KING_DIAMONDS)
print (agent.get_feature_vector(move))