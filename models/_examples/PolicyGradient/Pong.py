import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# set seed
torch.manual_seed(0)


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 10000),
            nn.ReLU(),
            nn.Linear(10000, 5000),
            nn.ReLU(),
            nn.Linear(5000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, action_dim),
            nn.Softmax(dim=0),
        )

    def forward(self, x):
        x = torch.flatten(x)
        x = self.network(x)

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

            next_state, reward, terminated, truncated, info = env.step(
                torch.argmax(action).item()
            )

            done = terminated or truncated
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
env = gym.make("Pong-v4", obs_type="grayscale", render_mode="human")
state, info = env.reset()  # state.shape = (210, 160) -> (33600,)

policy = Policy(state_dim=33600, action_dim=env.action_space.n)
policy_gradient(env, policy, episodes=1000, gamma=0.99, lr=0.01)
