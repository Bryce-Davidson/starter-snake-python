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
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        o = self.conv(torch.zeros(1, *[3, 210, 160]))
        conv_out_size = int(np.prod(o.size()))

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Softmax(dim=0),
        )

    def forward(self, x):
        x = x.permute(2, 0, 1).unsqueeze(
            0
        )  # Rearrange dimensions to (C, H, W) and add a batch dimension
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x.squeeze()  # Remove the batch dimension


def policy_gradient(env, policy, episodes, gamma, lr):
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for i in range(episodes):
        state, info = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            state = torch.tensor(state, dtype=torch.float32)

            probs = policy(state)
            action = torch.multinomial(probs, 1)

            # print(actions[action.item()])

            next_state, reward, terminated, truncated, info = env.step(action.item())

            log_prob = torch.log(probs[action])
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

            done = terminated or truncated or reward == -1

        print(f"Episode: {i}, Reward: {sum(rewards)}")

        optimizer.zero_grad()
        loss = -torch.sum(torch.stack(log_probs) * torch.tensor(rewards))
        loss.backward()
        optimizer.step()

    return policy


# Example usage
env = gym.make("Pong-v4", mode=1, obs_type="rgb", render_mode="human")
state, info = env.reset()  # state.shape = (210, 160, 3)

policy = Policy(env.action_space.n)
policy_gradient(env, policy, episodes=1000, gamma=0.99, lr=1)
