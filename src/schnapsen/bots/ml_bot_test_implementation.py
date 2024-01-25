from schnapsen.game import SchnapsenGamePlayEngine, Move, RegularMove, Marriage, Bot, PlayerPerspective, SchnapsenDeckGenerator, GamePhase, GamePlayEngine
from ml_bot import MLDataBot, MLPlayingBot
import tensorflow as tf
from schnapsen.bots import RandBot, AlphaBetaBot, RdeepBot
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Input, BatchNormalization, Activation
from keras.metrics import Precision, Recall
import numpy as np
from random import Random
import random
import pathlib
import joblib
import time
from typing import List, Optional
from schnapsen.game import Card, Suit, Rank, Move

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

def create_replay_memory_dataset() -> None:
    # define replay memory database creation parameters
    num_of_games: int = 10000
    replay_memory_dir: str = 'src/schnapsen/bots/ML_replay_memories'
    replay_memory_filename: str = 'random100k_mixed_10k_games.txt'
    replay_memory_location = pathlib.Path(replay_memory_dir) / replay_memory_filename

    #bot_1_behaviour: Bot = RandBot(random.Random(5234243))
    model_location = pathlib.Path('src/schnapsen/bots/ML_models/og_mlbot_10k')
    #bot_1_behaviour: Bot = MLPlayingBot(model_location)
    bot_1_behaviour = PlayBot('src/schnapsen/bots/ML_models/rohan_models/random_100k_nobatch_0.35_10epochs.keras')
    #bot_2_behaviour: Bot = RdeepBot(num_samples=4, depth=4, rand=random.Random(4564654644))
    bot_2_behaviour: Bot = PlayBot('src/schnapsen/bots/ML_models/rohan_models/mixed_nobatch_0.35.keras')
    # bot_2_behaviour: Bot = RdeepBot(num_samples=4, depth=4, rand=random.Random(68438))
    delete_existing_older_dataset = False

    # check if needed to delete any older versions of the dataset
    if delete_existing_older_dataset and replay_memory_location.exists():
        print(f"An existing dataset was found at location '{replay_memory_location}', which will be deleted as selected.")
        replay_memory_location.unlink()

    # in any case make sure the directory exists
    replay_memory_location.parent.mkdir(parents=True, exist_ok=True)

    # create new replay memory dataset, according to the behaviour of the provided bots and the provided random seed
    engine = SchnapsenGamePlayEngine()
    replay_memory_recording_bot_1 = MLDataBot(bot_1_behaviour, replay_memory_location=replay_memory_location)
    replay_memory_recording_bot_2 = MLDataBot(bot_2_behaviour, replay_memory_location=replay_memory_location)
    for i in range(1, num_of_games + 1):
        if i % 50 == 0:
            print(f"\n\n\nProgress: {i}/{num_of_games}\n\n\n")
        engine.play_game(replay_memory_recording_bot_1, replay_memory_recording_bot_2, random.Random(i))
    print(f"Replay memory dataset recorder for {num_of_games} games.\nDataset is stored at: {replay_memory_location}")

def create_state_and_actions_vector_representation(perspective: PlayerPerspective, leader_move: Optional[Move],
                                                   follower_move: Optional[Move]) -> list[int]:
    """
    This function takes as input a PlayerPerspective variable, and the two moves of leader and follower,
    and returns a list of complete feature representation that contains all information
    """
    player_game_state_representation = get_state_feature_vector(perspective)
    leader_move_representation = get_move_feature_vector(leader_move)
    follower_move_representation = get_move_feature_vector(follower_move)

    return player_game_state_representation + leader_move_representation + follower_move_representation


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

def play_games_and_return_stats(engine: GamePlayEngine, bot1: Bot, bot2: Bot, number_of_games: int) -> int:
    """
    Play number_of_games games between bot1 and bot2, using the SchnapsenGamePlayEngine, and return how often bot1 won.
    Prints progress.
    """
    bot1_wins: int = 0
    lead, follower = bot1, bot2
    for i in range(1, number_of_games + 1):
        if i % 2 == 0:
            # swap bots so both start the same number of times
            lead, follower = follower, lead
        winner, _, _ = engine.play_game(lead, follower, random.Random(i))
        if winner == bot1:
            bot1_wins += 1
        if i % 10 == 0:
            print(f"\n\n\nProgress: {i}/{number_of_games}\n\n\n")
    return bot1_wins

def try_bot_game() -> None:
    engine = SchnapsenGamePlayEngine()
    model_dir: str = 'src/schnapsen/bots/ML_models'
    model_name: str = 'og_mlbot_10k'
    #model_name: str = "10k, 70perc.keras" # Idk if its just me that gets a pikling error with this
    time1 = time.time()
    model_location = pathlib.Path(model_dir) / model_name
    #bot1: Bot = MLPlayingBot(model_location=model_location)
    bot1 = PlayBot('src/schnapsen/bots/ML_models/rohan_models/random_100k_nobatch_0.35_10epochs.keras')
    bot2: Bot = RdeepBot(num_samples=4, depth=4, rand=random.Random())
    #bot2: Bot = RandBot(random.Random(464566))
    #bot2: Bot = PlayBot('src/schnapsen/bots/ML_models/rohan_models/mixed_nobatch_0.35_10epochs.keras')
    #bot2: Bot = MLPlayingBot(model_location)
    number_of_games: int = 100

    # play games with altering leader position on first rounds
    time1 = time.time()
    ml_bot_wins_against_random = play_games_and_return_stats(engine=engine, bot1=bot1, bot2=bot2, number_of_games=number_of_games)
    time2 = time.time()
    print(f"The ML bot with name {model_name}, won {ml_bot_wins_against_random} times out of {number_of_games} games played.")
    print (f'It took {(time2-time1)/60} minutes to play {number_of_games} games.')

try_bot_game()
# TrainBot().train('test_replay_memory')
#create_replay_memory_dataset()