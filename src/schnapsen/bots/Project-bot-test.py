from schnapsen.game import Bot, PlayerPerspective, SchnapsenDeckGenerator, Move, Trick, GamePhase
from typing import Optional, cast, Literal
from schnapsen.deck import Suit, Rank, Card
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import time
import pathlib
import tensorflow as tf
from tensorflow.python.keras import layers as layers


# git config pull.rebase false

class project_test_bot(Bot):
    def __init__(self, model_location: pathlib.Path, name: Optional[str] = None) -> None:
        """
        Create a new MLPlayingBot which uses the model stored in the mofel_location.

        :param model_location: The file containing the model.
        """
        super().__init__(name)
        model_location = model_location
        assert model_location.exists(), f"Model could not be found at: {model_location}"
        # load model
        self.__model = tf.keras.Model.Sequential()