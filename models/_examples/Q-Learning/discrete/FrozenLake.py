import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

params = {
    "id": "FrozenLake-v1",
    "map_name": "8x8",
    "is_slippery": False,
}

env = gym.make(**params)

Q = np.zeros([env.observation_space.n, env.action_space.n])

alpha = 0.5
gamma = 0.9
epsilon = 1.0
decay = 0.99999
episodes = 10000
max_steps = 1000

for i in range(episodes):
    state, prob = env.reset()

    R = 0
    terminated, truncated = False, False

    steps = 0
    while not terminated and not truncated and steps < max_steps:

        action = (
            np.argmax(Q[state, :])
            if np.random.rand() > epsilon
            else env.action_space.sample()
        )

        next_state, reward, terminated, truncated, info = env.step(action)

        Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
        )

        R += reward
        state = next_state

        steps += 1

    epsilon *= decay
    print(f"Episode: {i}, Reward: {R}")


# -------------------------------------------------

env = gym.make(**params, render_mode="human")

state, prob = env.reset()

terminated, truncated = False, False
while not terminated and not truncated:
    action = np.argmax(Q[state, :])
    print(f"Action: {action}")
    state, reward, terminated, truncated, info = env.step(action)
