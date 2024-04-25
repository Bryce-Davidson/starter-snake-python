import torch
import torch.nn as nn
from env import BattleSnakeEnv
from multiprocessing import Process


def reset_env():
    BattleSnakeEnv.reset()


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

    def move(self, env: BattleSnakeEnv):
        state, reward = env.state, env.reward

        self.states.append(state)
        self.rewards.append(reward)

        # action = self.model(env)
        print(env)

        return {"move": "right"}

    def end(self, env: BattleSnakeEnv):
        state, reward = env.state, env.reward
        print(env)


if __name__ == "__main__":
    from server import run_server

    model = PPO()
    snake = Snake(model)

    # Start the reset function in a separate process
    reset_process = Process(target=reset_env)
    reset_process.start()

    # Start the Flask server
    run_server({"move": snake.move, "end": snake.end})

    # Wait for the reset process to finish
    reset_process.join()
