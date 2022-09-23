from rl.dreamer.config import DreamerConfig
from rl.dreamer.trainer import Trainer
from rl.env import PCGEnv
from rl.wrappers import ChannelFirstObservation


def main():
    config = DreamerConfig()
    env = ChannelFirstObservation(PCGEnv())
    trainer = Trainer(env, config)
    trainer.train()


if __name__ == "__main__":
    main()
