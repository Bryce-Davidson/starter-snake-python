from models.PPO import PPO
from env import BattleSnakeEnv


class Snake:
    def __init__(self, model):
        self.model = model

    def step(self, env: BattleSnakeEnv):
        state, reward = env.state, env.reward

        return {"move": "up"}


if __name__ == "__main__":
    from server import start_server

    model = PPO()
    snake = Snake(model)

    start_server({"step": snake.step})
