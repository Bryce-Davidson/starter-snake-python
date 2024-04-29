import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 10000)
        self.fc2 = nn.Linear(10000, 5000)
        self.fc3 = nn.Linear(5000, action_dim)

    def forward(self, x):
        x = torch.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.softmax(x, dim=0)

        exit()

        return x


def policy_gradient(env, policy, episodes, gamma, lr):
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for _ in range(episodes):
        state, info = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            state = torch.tensor(state, dtype=torch.float32)
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            log_prob = torch.log(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        optimizer.zero_grad()
        loss = -torch.sum(torch.stack(log_probs) * torch.tensor(rewards))
        loss.backward()
        optimizer.step()

    return policy


# Example usage
env = gym.make("Pong-v4", obs_type="grayscale")
state, info = env.reset()  # state.shape = (210, 160) -> (33600,)

policy = Policy(state_dim=33600, action_dim=env.action_space.n)
policy_gradient(env, policy, episodes=1000, gamma=0.99, lr=0.01)
