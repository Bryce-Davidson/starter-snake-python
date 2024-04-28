import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.autograd import Variable
import numpy as np


class DQN(nn.Module):
    def __init__(self, action_shape):
        super(DQN, self).__init__()
        self.action_shape = action_shape

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(
                self.features(torch.zeros(1, 3, 96, 96)).view(1, -1).size(1), 512
            ),
            nn.ReLU(),
            nn.Linear(512, self.action_shape),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def act(self, state, epsilon):
        return (
            np.random.randint(0, self.action_shape)
            if np.random.rand() < epsilon
            else self.forward(state).max(1)[1].view(1, 1).item()
        )


# -------------------------------------------------

env = gym.make("CarRacing-v2", continuous=False, domain_randomize=True)
model = DQN(env.action_space.n)
memory = []

optimizer = optim.Adam(model.parameters(), lr=0.0001)

epsilon = 1.0
decay = 0.99
gamma = 0.9
episodes = 100
max_steps = 100

for i in range(episodes):
    state, info = env.reset(options={"randomize": False})

    R = 0
    terminated, truncated = False, False

    steps = 0
    while not terminated and not truncated and steps < max_steps:
        state = torch.tensor(state).permute(2, 0, 1).unsqueeze(0).float()
        action = model.act(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        memory.append((state, action, reward, next_state, terminated))

        R += reward
        state = next_state

        steps += 1

    epsilon *= decay

    print(f"Episode: {i}, Reward: {R}")

    if len(memory) > 1000:
        batch = random.sample(memory, 100)
        for state, action, reward, next_state, terminated in batch:
            state = torch.tensor(state).permute(2, 0, 1).unsqueeze(0).float()
            next_state = torch.tensor(next_state).permute(2, 0, 1).unsqueeze(0).float()

            Qt = model(state)
            Qt1 = model(next_state)

            loss = torch.pow(reward + gamma * torch.max(Qt1) - torch.max(Qt), 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# -------------------------------------------------

env = gym.make(
    "CarRacing-v2", continuous=False, domain_randomize=True, render_mode="human"
)

for i in range(episodes):
    state, info = env.reset(options={"randomize": False})

    terminated, truncated = False, False
    while not terminated and not truncated and steps < max_steps:
        state = torch.tensor(state).permute(2, 0, 1).unsqueeze(0).float()
        action = model.act(state, 0)
        state, reward, terminated, truncated, info = env.step(action)
