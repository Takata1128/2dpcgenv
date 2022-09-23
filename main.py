from rl.ppo.config import Config
from rl.ppo.agent import Agent


def main():
    config = Config()
    agent = Agent(
        config
    )
    agent.train(
        num_env_steps=config.steps
    )


if __name__ == "__main__":
    main()
