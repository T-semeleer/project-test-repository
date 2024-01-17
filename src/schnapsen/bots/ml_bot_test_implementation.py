from schnapsen.game import SchnapsenGamePlayEngine, Move, RegularMove, Marriage, Bot
from schnapsen.bots import RandBot, RdeepBot
from ml_bot import MLDataBot
import tensorflow as tf
from tensorflow.python.keras import layers, models
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
import numpy as np
from random import Random
import random
import pathlib
from typing import List, Optional
from schnapsen.game import Card, Suit, Rank, Move

class SchnapsenNNBot:
    def __init__(self):
        # Initialize the neural network
        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(20,)), Dropout(0.5), Dense(256, activation='relu'),
            Dropout(0.5), Dense(128, activation='relu'), Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    def encode_suits(self, card_suit: Card) -> list[int]:
        """One-hot encodes the value of the suits"""
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
        """One-hot encodes the ranks of each of the cards"""
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
        """Returns the full one-hot encoded value of the move"""
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

    def train(self, x_train, y_train, epochs=10, batch_size=32):
        # directory where the replay memory is saved
        replay_memory_filename: str = "random_random_10k_games.txt"
        # filename of replay memory within that directory
        replay_memories_directory: str = "ML_replay_memories"
        # Whether to train a complicated Neural Network model or a simple one.
        # Tips: a neural network usually requires bigger datasets to be trained on, and to play with the parameters of the model.
        # Feel free to play with the hyperparameters of the model in file 'ml_bot.py', function 'train_ML_model',
        # under the code of body of the if statement 'if use_neural_network:'
        replay_memory_location = (
            pathlib.Path(replay_memories_directory) / replay_memory_filename
        )
        model_name: str = "simple_model"
        model_dir: str = "ML_models"
        model_location = pathlib.Path(model_dir) / model_name
        overwrite: bool = True

        model_name: str = "simple_model"
        model_dir: str = "ML_models"
        model_location = pathlib.Path(model_dir) / model_name
        overwrite: bool = True

        if overwrite and model_location.exists():
            print(
                f"Model at {model_location} exists already and will be overwritten as selected."
            )
            model_location.unlink()
        
        # Write code to train model here

    def predict(self, move: Move) -> float:
        """Predicts the probability of a winning move"""
        feature_vector = self.get_feature_vector(move)
        prediction = self.model.predict(feature_vector.reshape(1, -1))
        return prediction[0,0]
    def get_move(self, move: Move):
        move = self.predict()

def create_replay_memory_dataset(bot1: Bot, bot2: Bot) -> None:
    """Create offline dataset for training a ML bot.

    Args:
        bot1, bot2: the bot of your choice.

    """
    # define replay memory database creation parameters
    num_of_games: int = 10000
    replay_memory_dir: str = "ML_replay_memories"
    replay_memory_filename: str = "random_random_10k_games.txt"
    replay_memory_location = pathlib.Path(replay_memory_dir) / replay_memory_filename

    delete_existing_older_dataset = False

    # check if needed to delete any older versions of the dataset
    if delete_existing_older_dataset and replay_memory_location.exists():
        print(
            f"An existing dataset was found at location '{replay_memory_location}', which will be deleted as selected."
        )
        replay_memory_location.unlink()

    # in any case make sure the directory exists
    replay_memory_location.parent.mkdir(parents=True, exist_ok=True)

    # create new replay memory dataset, according to the behaviour of the provided bots and the provided random seed
    engine = SchnapsenGamePlayEngine()
    replay_memory_recording_bot_1 = MLDataBot(
        bot1, replay_memory_location=replay_memory_location
    )
    replay_memory_recording_bot_2 = MLDataBot(
        bot2, replay_memory_location=replay_memory_location
    )
    for i in range(1, num_of_games + 1):
        if i % 500 == 0:
            print(f"Progress: {i}/{num_of_games}")
        engine.play_game(
            replay_memory_recording_bot_1,
            replay_memory_recording_bot_2,
            random.Random(i),
        )
    print(
        f"Replay memory dataset recorder for {num_of_games} games.\nDataset is stored at: {replay_memory_location}"
    )

create_replay_memory_dataset(RandBot(rand=random.Random(), name='bot1'), RandBot(rand=random.Random(), name='bot2'))
