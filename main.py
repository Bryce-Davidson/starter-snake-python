import torch
import torch.nn as nn
from env import BattleSnakeEnv
import typing


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()

    def forward(self, env: BattleSnakeEnv):
        pass


if __name__ == "__main__":
    from server import run_server

    snake = PPO()
    run_server({"move": snake.forward})

    BattleSnakeEnv.reset()
