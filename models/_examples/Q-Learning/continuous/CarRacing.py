import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# ---------------------------------------------------------
# This is not a very good example.

# To implement something which works other methods need to be used.

# References:

# - https://dspace.cvut.cz/bitstream/handle/10467/108761/F3-BP-2023-Sykora-Vojtech-PPO_Thesis_Sykora.pdf?sequence=-1&isAllowed=y

# ---------------------------------------------------------


# Set the seed for reproducibility
seed = 9999
torch.manual_seed(seed)

actions = {
    0: "do nothing",
    1: "steer left",
    2: "steer right",
    3: "gas",
    4: "brake",
}


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, state, action, reward, next_state, terminated):
        if len(self.memory) > self.capacity:
            self.memory.pop(0)

        self.memory.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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
            nn.Linear(512, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, self.action_shape),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def act(self, state, epsilon):
        if np.random.rand() < epsilon:
            action = np.random.randint(0, self.action_shape)
            # print(f"Random Action: {actions[action]}, Epsilon: {epsilon}")
        else:
            action = torch.argmax(self.forward(state)).item()
            # print(f"Model Action: {actions[action]}, {epsilon}")

        return action


# # -------------------------------------------------

env = gym.make(
    "CarRacing-v2",
    continuous=False,
    render_mode="human",
)

EPISODES = 10
EPISODE_MAX_STEPS = 2000
MEMORY_CAPACITY = 2000
MEMORY_SAMPLE_SIZE = 1000

EPSILON = 0.5
EPSILON_DECAY = 0.9
GAMMA = 0.9

model = DQN(env.action_space.n)
memory = ReplayBuffer(capacity=MEMORY_CAPACITY)

optimizer = optim.Adam(model.parameters(), lr=0.00001)

for i in range(EPISODES):
    state, info = env.reset(options={"randomize": False})

    R = 0
    terminated, truncated = False, False

    steps = 0
    while not terminated and not truncated and steps < EPISODE_MAX_STEPS:
        state = torch.tensor(state).permute(2, 0, 1).unsqueeze(0).float()
        action = model.act(state, EPSILON)

        next_state, reward, terminated, truncated, info = env.step(action)

        memory.push(state, action, reward, next_state, terminated)

        R += reward
        state = next_state

        steps += 1

    EPSILON *= EPSILON_DECAY

    print(f"Episode: {i}, Reward: {R}, Epsilon: {EPSILON}, Memory: {len(memory)}")

    if len(memory) > MEMORY_CAPACITY:
        batch = memory.sample(MEMORY_SAMPLE_SIZE)
        for state, action, reward, next_state, terminated in batch:
            next_state = torch.tensor(next_state).permute(2, 0, 1).unsqueeze(0).float()

            Qt = model(state).squeeze(0)[action]
            Qt1 = model(next_state).max().item()

            target = reward + (1 - terminated) * GAMMA * Qt1
            loss = torch.pow(target - Qt, 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if i % 10 == 0:
        torch.save(model.state_dict(), "model.pth")

# -------------------------------------------------

env = gym.make("CarRacing-v2", continuous=False, render_mode="human")

model.load_state_dict(torch.load("model.pth"))

for i in range(EPISODES):
    state, info = env.reset(options={"randomize": False})

    terminated, truncated = False, False

    while not terminated and not truncated:
        state = torch.tensor(state).permute(2, 0, 1).unsqueeze(0).float()
        action = model.act(state, 0)
        print(f"Action: {action}")
        state, reward, terminated, truncated, info = env.step(action)
