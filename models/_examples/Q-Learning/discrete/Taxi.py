import gymnasium as gym
import numpy as np


env = gym.make("Taxi-v3")

Q = np.zeros([env.observation_space.n, env.action_space.n])

alpha = 0.5
gamma = 0.9
epsilon = 1
decay = 0.9999
episodes = 5000

for i in range(episodes):
    state, prob = env.reset()

    R = 0
    terminated, truncated = False, False

    max_steps = 1000
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
    print(f"Episode: {i}, Reward: {R}, Steps: {steps}, Epsilon: {epsilon}")


env = gym.make("Taxi-v3", render_mode="human")

for i in range(episodes):
    state, prob = env.reset()
    terminated, truncated = False, False
    while not terminated and not truncated:
        action = np.argmax(Q[state, :])
        state, reward, terminated, truncated, info = env.step(action)
