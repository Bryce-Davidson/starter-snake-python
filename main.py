from env import BattleSnakeEnv
import typing


class Snake:
    def __init__(self):
        pass

    def step(self, env: BattleSnakeEnv):
        return {"move": "up"}


if __name__ == "__main__":
    from server import run_server

    snake = Snake()
    run_server({"move": snake.step})
