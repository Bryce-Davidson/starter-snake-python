import gymnasium as gym
import numpy as np

env = gym.make("CliffWalking-v0", render_mode="human")

Q = np.zeros([env.observation_space.n, env.action_space.n])

lr = 0.8
gamma = 0.95

terminated = False
truncated = False

for i in range(100):
    state, prob = env.reset()

    R = 0
    while not terminated and not truncated:

        action = np.argmax(Q[state, :])

        next_state, reward, terminated, truncated, info = env.step(action)

        Q[state, action] = Q[state, action] + lr * (
            reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
        )

        R += reward
        state = next_state

        if i % 10 == 0:
            print(f"Episode: {i}, Reward: {R}")

    terminated = False
    truncated = False
