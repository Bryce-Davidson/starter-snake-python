import torch
import torch.nn as nn
from env import BattleSnakeEnv
import typing


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()

    def forward(self, env: BattleSnakeEnv):
        pass


class Snake:
    def __init__(self):
        self.model = PPO()

    def step(self, env: BattleSnakeEnv):
        observation, reward, terminated, info, done = env.step()
        return self.model(env)


if __name__ == "__main__":
    from server import run_server

    snake = Snake()
    run_server({"move": snake.step})

    BattleSnakeEnv.reset()
