import torch
import time
import torch.nn as nn
from env import BattleSnakeEnv


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()

    def forward(self, env: BattleSnakeEnv):
        pass


class Snake:
    def __init__(self, model):
        self.model = model

        self.states = []
        self.actions = []
        self.rewards = []
        self.old_log_probs = []

    def store(self, state, reward):
        self.states.append(state)
        self.rewards.append(reward)

    def start(self, env: BattleSnakeEnv):
        self.store(env.observe())

    def move(self, env: BattleSnakeEnv):
        state, reward = env.observe()
        self.store(state, reward)

        return {"move": "right"}

    def end(self, env: BattleSnakeEnv):
        self.store(env.observe())


if __name__ == "__main__":
    from server import start_server

    model = PPO()
    snake = Snake(model)

    start_server({"move": snake.move, "end": snake.end})
