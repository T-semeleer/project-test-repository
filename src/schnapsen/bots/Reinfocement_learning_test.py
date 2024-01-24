import gym
from schnapsen.game import SchnapsenGamePlayEngine, Move, RegularMove, Marriage, Bot, PlayerPerspective, SchnapsenDeckGenerator, GamePhase, GamePlayEngine
from ml_bot import MLDataBot, MLPlayingBot
import tensorflow as tf
from schnapsen.bots import RandBot, AlphaBetaBot
from typing import List, Optional
from schnapsen.game import Card, Suit, Rank, Move

class RLBot(Bot):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.engine = SchnapsenGamePlayEngine()
        self.minimax_model = AlphaBetaBot()
    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        valid_moves = perspective.valid_moves()
        if perspective.get_phase() == GamePhase.TWO:
            return self.minimax_model.get_move(perspective, leader_move)
        else:
            
