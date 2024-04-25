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

        self.states = []
        self.actions = []
        self.rewards = []
        self.old_log_probs = []

    def move(self, env: BattleSnakeEnv):
        state, reward = env.state, env.reward

        self.states.append(state)
        self.rewards.append(reward)

        return self.model(env)

    def end(self, env):
        pass


if __name__ == "__main__":
    from server import run_server

    snake = Snake()
    run_server({"move": snake.move, "end": snake.end})

    for episode in range(100):
        # BattleSnakeEnv.reset()
        pass
