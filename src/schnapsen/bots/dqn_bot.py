from random import Random
from schnapsen.game import Bot, BotState, GamePlayEngine, GameState, Hand, PlayerPerspective, SchnapsenDeckGenerator, Move, Score, Trick, GamePhase, RegularMove, SchnapsenGamePlayEngine, Marriage
from schnapsen.bots import RdeepBot, AlphaBetaBot
from typing import Optional, cast, Literal
from schnapsen.deck import CardCollection, Suit, Rank, Card
import joblib
import time
import pathlib
import random
import numpy as np
import tensorflow as tf
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
import gym

class TrainBot:
    def __init__(self):
        # Initialize the neural network
        self.model = Sequential([
            Input(shape=(173,)),  # I think the input shape should be 173, because of the size of the feature vector
            Dense(128, activation='relu'),
            Dropout(0.35),
            #BatchNormalization(),
            Dense(256, activation='relu'),
            Dropout(0.35),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
       
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])
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
        
        return move_type_arr_numpy + rank_encoding_arr_numpy + suit_encoding_arr_numpy

    def train(self, replay_memory_filename, epochs=10, batch_size=32):
        # directory where the replay memory is saved
        # filename of replay memory within that directory
        replay_memories_directory: str = "src/schnapsen/bots/ML_replay_memories"
        # Whether to train a complicated Neural Network model or a simple one.
        # Tips: a neural network usually requires bigger datasets to be trained on, and to play with the parameters of the model.
        # Feel free to play with the hyperparameters of the model in file 'ml_bot.py', function 'train_ML_model',
        # under the code of body of the if statement 'if use_neural_network:'
        replay_memory_location = pathlib.Path(replay_memories_directory) / replay_memory_filename
        model_name: str = "random_100k_nobatch_0.35_10epochs"
        model_dir: str = "src/schnapsen/bots/ML_models/rohan_models"
        model_location = pathlib.Path(model_dir) / model_name
        overwrite: bool = True

        data = []
        targets = []

        if overwrite and model_location.exists():
            print(
                f"Model at {model_location} exists already and will be overwritten as selected."
            )
            model_location.unlink()
        with open(file=replay_memory_location, mode="r") as replay_memory_file:
            for line in replay_memory_file:
                feature_string, won_label_str = line.split("||")
                feature_list_strings: list[str] = feature_string.split(",")
                feature_list = [int(feature) for feature in feature_list_strings]
                won_label = int(won_label_str)
                data.append(feature_list)
                targets.append(won_label)
        print("Dataset Statistics:")
        samples_of_wins = sum(targets)
        samples_of_losses = len(targets) - samples_of_wins
        print("Samples of wins:", samples_of_wins)
        print("Samples of losses:", samples_of_losses)
        
        model_location = str(model_location) + '.keras'
        start = time.time()
        print("Starting training phase...")
        try:
            self.model.fit(x=data, y=targets, batch_size=32, epochs=10)
        except Exception as e:
            print (f'An error occured during training {e}')
        self.model.save(model_location, overwrite=True)
        #keras.models.save_model(self.model, 'src/schnapsen/bots/ML_models/tf_model_10k.keras', overwrite=True)
        end = time.time()
        print('The model was trained in ', (end - start) / 60, 'minutes.')

def get_one_hot_encoding_of_card_suit(card_suit: Suit) -> list[int]:
    """
    Translating the suit of a card into one hot vector encoding of size 4.
    """
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


def get_one_hot_encoding_of_card_rank(card_rank: Rank) -> list[int]:
    """
    Translating the rank of a card into one hot vector encoding of size 13.
    """
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

def get_move_feature_vector(move: Optional[Move]) -> list[int]:
    """
        In case there isn't any move provided move to encode, we still need to create a "padding"-"meaningless" vector of the same size,
        filled with 0s, since the ML models need to receive input of the same dimensionality always.
        Otherwise, we create all the information of the move i) move type, ii) played card rank and iii) played card suit
        translate this information into one-hot vectors respectively, and concatenate these vectors into one move feature representation vector
    """

    if move is None:
        move_type_one_hot_encoding_numpy_array = [0, 0, 0]
        card_rank_one_hot_encoding_numpy_array = [0, 0, 0, 0]
        card_suit_one_hot_encoding_numpy_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    else:
        move_type_one_hot_encoding: list[int]
        # in case the move is a marriage move
        if move.is_marriage():
            move_type_one_hot_encoding = [0, 0, 1]
            card = move.queen_card
        #  in case the move is a trump exchange move
        elif move.is_trump_exchange():
            move_type_one_hot_encoding = [0, 1, 0]
            card = move.jack
        #  in case it is a regular move
        else:
            move_type_one_hot_encoding = [1, 0, 0]
            card = move.card
        move_type_one_hot_encoding_numpy_array = move_type_one_hot_encoding
        card_rank_one_hot_encoding_numpy_array = get_one_hot_encoding_of_card_rank(card.rank)
        card_suit_one_hot_encoding_numpy_array = get_one_hot_encoding_of_card_suit(card.suit)

    return move_type_one_hot_encoding_numpy_array + card_rank_one_hot_encoding_numpy_array + card_suit_one_hot_encoding_numpy_array


def get_state_feature_vector(perspective: PlayerPerspective) -> list[int]:
    """
        This function gathers all subjective information that this bot has access to, that can be used to decide its next move, including:
        - points of this player (int)
        - points of the opponent (int)
        - pending points of this player (int)
        - pending points of opponent (int)
        - the trump suit (1-hot encoding)
        - phase of game (1-hoy encoding)
        - talon size (int)
        - if this player is leader (1-hot encoding)
        - What is the status of each card of the deck (where it is, or if its location is unknown)

        Important: This function should not include the move of this agent.
        It should only include any earlier actions of other agents (so the action of the other agent in case that is the leader)
    """
    # a list of all the features that consist the state feature set, of type np.ndarray
    state_feature_list: list[int] = []

    player_score = perspective.get_my_score()
    # - points of this player (int)
    player_points = player_score.direct_points
    # - pending points of this player (int)
    player_pending_points = player_score.pending_points

    # add the features to the feature set
    state_feature_list += [player_points]
    state_feature_list += [player_pending_points]

    opponents_score = perspective.get_opponent_score()
    # - points of the opponent (int)
    opponents_points = opponents_score.direct_points
    # - pending points of opponent (int)
    opponents_pending_points = opponents_score.pending_points

    # add the features to the feature set
    state_feature_list += [opponents_points]
    state_feature_list += [opponents_pending_points]

    # - the trump suit (1-hot encoding)
    trump_suit = perspective.get_trump_suit()
    trump_suit_one_hot = get_one_hot_encoding_of_card_suit(trump_suit)
    # add this features to the feature set
    state_feature_list += trump_suit_one_hot

    # - phase of game (1-hot encoding)
    game_phase_encoded = [1, 0] if perspective.get_phase() == GamePhase.TWO else [0, 1]
    # add this features to the feature set
    state_feature_list += game_phase_encoded

    # - talon size (int)
    talon_size = perspective.get_talon_size()
    # add this features to the feature set
    state_feature_list += [talon_size]

    # - if this player is leader (1-hot encoding)
    i_am_leader = [0, 1] if perspective.am_i_leader() else [1, 0]
    # add this features to the feature set
    state_feature_list += i_am_leader

    # gather all known deck information
    hand_cards = perspective.get_hand().cards
    trump_card = perspective.get_trump_card()
    won_cards = perspective.get_won_cards().get_cards()
    opponent_won_cards = perspective.get_opponent_won_cards().get_cards()
    opponent_known_cards = perspective.get_known_cards_of_opponent_hand().get_cards()
    # each card can either be i) on player's hand, ii) on player's won cards, iii) on opponent's hand, iv) on opponent's won cards
    # v) be the trump card or vi) in an unknown position -> either on the talon or on the opponent's hand
    # There are all different cases regarding card's knowledge, and we represent these 6 cases using one hot encoding vectors as seen bellow.

    deck_knowledge_in_consecutive_one_hot_encodings: list[int] = []

    for card in SchnapsenDeckGenerator().get_initial_deck():
        card_knowledge_in_one_hot_encoding: list[int]
        # i) on player's hand
        if card in hand_cards:
            card_knowledge_in_one_hot_encoding = [0, 0, 0, 0, 0, 1]
        # ii) on player's won cards
        elif card in won_cards:
            card_knowledge_in_one_hot_encoding = [0, 0, 0, 0, 1, 0]
        # iii) on opponent's hand
        elif card in opponent_known_cards:
            card_knowledge_in_one_hot_encoding = [0, 0, 0, 1, 0, 0]
        # iv) on opponent's won cards
        elif card in opponent_won_cards:
            card_knowledge_in_one_hot_encoding = [0, 0, 1, 0, 0, 0]
        # v) be the trump card
        elif card == trump_card:
            card_knowledge_in_one_hot_encoding = [0, 1, 0, 0, 0, 0]
        # vi) in an unknown position as it is invisible to this player. Thus, it is either on the talon or on the opponent's hand
        else:
            card_knowledge_in_one_hot_encoding = [1, 0, 0, 0, 0, 0]
        # This list eventually develops to one long 1-dimensional numpy array of shape (120,)
        deck_knowledge_in_consecutive_one_hot_encodings += card_knowledge_in_one_hot_encoding
    # deck_knowledge_flattened: np.ndarray = np.concatenate(tuple(deck_knowledge_in_one_hot_encoding), axis=0)

    # add this features to the feature set
    state_feature_list += deck_knowledge_in_consecutive_one_hot_encodings

    return state_feature_list

class PlayBot(Bot):
    '''
    Loads the trained model and plays moves
    '''
    def __init__(self, model_location, name: Optional[str] = None):
        super().__init__(name)

        self.model = load_model(model_location)
        self.minimax_model = AlphaBetaBot()
    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        if perspective.get_phase() == GamePhase.TWO:
            return self.minimax_model.get_move(perspective, leader_move)
        else:
            state_representation = get_state_feature_vector(perspective)
            # get the leader's move representation, even if it is None
            leader_move_representation = get_move_feature_vector(leader_move)
            # get all my valid moves
            my_valid_moves = perspective.valid_moves()
            # get the feature representations for all my valid moves
            my_move_representations: list[list[int]] = []
            for my_move in my_valid_moves:
                my_move_representations.append(get_move_feature_vector(my_move))
            action_state_representations: list[list[int]] = []

            if perspective.am_i_leader():
                follower_move_representation = get_move_feature_vector(None)
                for my_move_representation in my_move_representations:
                    action_state_representations.append(
                        (state_representation + my_move_representation + follower_move_representation))
            else:
                for my_move_representation in my_move_representations:
                    action_state_representations.append(
                        (state_representation + leader_move_representation + my_move_representation))
            best_score = -np.inf
            best_move_index = -1
            for index, move in enumerate(action_state_representations):
                move_input = np.array([move])
                score = self.model.predict(move_input)
                if score > best_score:
                    best_score = score
                    best_move_index = index

            return my_valid_moves[best_move_index]
class DQN(tf.keras.Model):
    """Deep Q-Network, a neural network model for estimating Q-values of actions."""
    def __init__(self, action_size: int):
        """Initialises the network, with a specified number of layers and constructs the class"""
        super(DQN, self).__init__() # Calling the constructor of the tf.keras.Model class
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(128, activation='relu')
        self.output_layer = Dense(action_size, activation='linear')
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
    
    def memorize(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Store past experiences in the replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> int:
        """Returns the action to be taken given the current state"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.model.output_layer.units) # Explore: select a random action
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]) # Exploit: select the action with the highest Q-value
    
    def replay(self, batch_size: int):
        """Trains the model using a batch of experiences from the replay memory."""
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
    def update_epsilon(self):
        """Updates the exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def save_model(self):
        """Saves the model"""
        self.model.save('src/schnapsen/bots')
    def load_model(self):
        """Loads model from stored location"""
        model = load_model('src/schnapsen/bots')
        return model

class ReplayBufer():
    """
    Creates a replay memory system
    """
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_counter = 0 # Keeps track of first saved memory, used to insert memory to the replay buffer

        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeroes(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)

    def store_transition(self, state, action, reward, state_, done):
        """
        Stores the addition and transition of each state to the replay memory
        """
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_counter += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = Sequential([
        Dense(fc1_dims, activation='relu'),
        Dense(fc2_dims, activation='relu'),
        Dense(n_actions, activation=None)])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    
    return model

class AgentYT:
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims, 
                 epsilon_dec = 1e-3, epsilon_end=0.01, mem_size = 1000000, fname='dqn_model.keras'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = epsilon_end
        self.eps_dec = epsilon_dec
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBufer(mem_size, input_dims)
        self.q_eval = build_dqn(lr, n_actions, input_dims, 256, 256)
    
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[str] = None):
        valid_moves = perspective.valid_moves()
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state_representation = get_state_feature_vector(perspective)
            # get the leader's move representation, even if it is None
            leader_move_representation = get_move_feature_vector(leader_move)
            # get the feature representations for all my valid moves
            my_move_representations: list[list[int]] = []
            for my_move in valid_moves:
                my_move_representations.append(get_move_feature_vector(my_move))
            action_state_representations: list[list[int]] = []

            if perspective.am_i_leader():
                follower_move_representation = get_move_feature_vector(None)
                for my_move_representation in my_move_representations:
                    action_state_representations.append(
                        (state_representation + my_move_representation + follower_move_representation))
            else:
                for my_move_representation in my_move_representations:
                    action_state_representations.append(
                        (state_representation + leader_move_representation + my_move_representation))
            best_score = -np.inf
            best_move_index = -1
            for index, move in enumerate(action_state_representations):
                move_input = np.array([move])
                score = self.q_eval.predict(move_input)
                if score > best_score:
                    best_score = score
                    best_move_index = index

            return valid_moves[best_move_index]
    
    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return 
        
        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)

        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(states_)

        q_target = np.copy(q_eval)

        batch_index = np.arrange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1)*dones

        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_model(self):
        self.q_eval.save(self.model_file)
    
    def load_model(self):
        self.q_eval = load_model(self.model_file)

env = SchnapsenGamePlayEngine()
lr = 0.001
n_games = 100
agent = AgentYT(gamma=0.99, epsilon=1, lr=lr, input_dims=env.play_game(
        PlayBot('src/schnapsen/bots/ML_models/rohan_models/random_100k_nobatch_0.35_10epochs.keras'), 
        PlayBot('src/schnapsen/bots/ML_models/rohan_models/mixed_nobatch_0.35_10epochs.keras'),
        random.Random()
    ), n_actions=5, mem_size=1000000, batch_size=32, epsilon_end=0.01)

agent.learn()