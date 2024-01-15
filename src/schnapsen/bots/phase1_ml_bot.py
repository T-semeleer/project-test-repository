from schnapsen.game import Bot, PlayerPerspective, SchnapsenDeckGenerator, Move, Trick, GamePhase
from typing import Optional, cast, Literal
from schnapsen.deck import Suit, Rank, Card
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import time
import pathlib
import tensorflow as tf
from tensorflow.python.keras import layers
import numpy as np

class DQN(tf.keras.Model):
    
    def __init__(self, action_size: int):
        """Deep Q-Network, a neural network model for estimating Q-values of actions."""
        super(DQN, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(action_size, activation='linear')
    def call(self, state: np.ndarray):
        """Performs a forward pass."""
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_layer(x)