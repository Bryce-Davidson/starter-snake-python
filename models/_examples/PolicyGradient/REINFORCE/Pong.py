import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# set seed
torch.manual_seed(0)

actions = {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT", 4: "RIGHTFIRE", 5: "LEFTFIRE"}


class Policy(nn.Module):
    def __init__(self, action_dim):
        super(Policy, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        o = self.conv(torch.zeros(1, *[3, 210, 160]))
        conv_out_size = int(np.prod(o.size()))

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, action_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze()


def policy_gradient(env, policy, episodes, lr):
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for i in range(episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        state = state.permute(2, 0, 1).unsqueeze(0)

        log_probs = []
        rewards = []
        done = False

        while not done:
            probs = policy(state)
            print(probs)
            action = torch.multinomial(probs, 1)

            # print(actions[action.item()])

            next_state, reward, terminated, truncated, info = env.step(action.item())
            next_state = torch.tensor(next_state, dtype=torch.float32)
            next_state = next_state.permute(2, 0, 1).unsqueeze(0)

            rewards.append(reward)

            log_prob = torch.log(probs[action])
            log_probs.append(log_prob)

            state = next_state

            done = terminated or truncated or reward == -1

        print(f"Episode: {i}, Rewards: {sum(rewards)}")

        optimizer.zero_grad()
        log_probs = torch.stack(log_probs)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        loss = -torch.mean(log_probs * rewards)
        loss.backward()
        optimizer.step()

        # Save model
        torch.save(policy.state_dict(), "pong.pth")

    return policy


env = gym.make("Pong-v4", mode=1, obs_type="rgb", render_mode="human")
state, info = env.reset()  # state.shape = (210, 160, 3)

policy = Policy(env.action_space.n)

if os.path.exists("pong.pth"):
    policy.load_state_dict(torch.load("pong.pth"))

policy_gradient(env, policy, episodes=1000, lr=1e-4)
