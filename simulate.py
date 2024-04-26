import os
import json
import numpy as np
from multiprocessing import Process
from battlesnake_gym.snake_gym import BattlesnakeGym


def reset_env():
    print("Resetting game...")
    args = [
        "./battlesnake",
        "play",
        "--name",
        "Python Starter Project",
        "--url",
        "http://localhost:8000",
        "-g",
        "solo",
        # "--browser",
        "--board-url",
        "http://localhost:5173/",
        "--foodSpawnChance",
        "0",
        "--minimumFood",
        "0",
    ]

    os.system(" ".join(args))


env = BattlesnakeGym(map_size=(20, 20), number_of_snakes=1)

if __name__ == "__main__":
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step([0])
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
