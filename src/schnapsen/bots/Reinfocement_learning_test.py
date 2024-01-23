from schnapsen.game import SchnapsenGamePlayEngine, Move, RegularMove, Marriage, Bot, PlayerPerspective, SchnapsenDeckGenerator, GamePhase, GamePlayEngine
from ml_bot import MLDataBot, MLPlayingBot
import tensorflow as tf
from schnapsen.bots import RandBot
from typing import List, Optional
from schnapsen.game import Card, Suit, Rank, Move

class RLBot(Bot):
    def __init__(self):
        super()
        pass
