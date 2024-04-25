import typing
from env import BattleSnakeEnv


class PPO:
    def __init__(self):
        pass

    def step(self, env: BattleSnakeEnv):
        pass


if __name__ == "__main__":
    from server import run_server

    snake = PPO()
    run_server({"move": snake.step})
