import random
from schnapsen.bots import SchnapsenServer
from schnapsen.bots import RandBot, PlayBot, RdeepBot
from schnapsen.game import SchnapsenGamePlayEngine


if __name__ == "__main__":
    engine = SchnapsenGamePlayEngine()
    with SchnapsenServer() as s:
        bot1 = PlayBot('src/schnapsen/bots/ML_models/rohan_models/early_stop/loss_0.001_actual_full_datasetv3_nobatch_0.35_10epochs.keras')
        #bot1 = RdeepBot(num_samples=4, depth=4, rand=random.Random())
        #bot1 = RandBot(random.Random(12))
        # bot1 = s.make_gui_bot(name="mybot1")
        bot2 = s.make_gui_bot(name="NNBot")
        engine.play_game(bot1, bot2, random.Random())