import os
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt

weight_path = "cartpole.pth"

# set seed
torch.manual_seed(0)


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Linear(256, 256),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=0),
        )

    def forward(self, x):
        x = self.fc(x)
        return x.squeeze()


def policy_gradient(env, policy, episodes, lr):
    optimizer = optim.Adam(policy.parameters(), lr=lr, amsgrad=True)

    rewards_per_episode = []  # Store rewards for each episode

    # Set up plot
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    (line,) = ax.plot(rewards_per_episode)
    plt.show()

    for i in range(episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        log_probs = []
        rewards = []
        done = False

        while not done:
            probs = policy(state)
            print(probs)
            action = torch.multinomial(probs, 1).item()

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated or reward == -1

            next_state = torch.tensor(next_state, dtype=torch.float32)
            reward = torch.tensor([reward])
            reward = reward if not done else torch.tensor([0])

            log_probs.append(probs[action].log())
            rewards.append(reward)
            state = next_state

        total_reward = sum(rewards)
        rewards_per_episode.append(total_reward.item())

        print(f"Episode: {i}, Rewards: {total_reward}")

        optimizer.zero_grad()
        log_probs = torch.stack(log_probs).unsqueeze(1)
        rewards = torch.stack(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)
        loss = -torch.mean(log_probs * rewards)
        loss.backward()
        optimizer.step()

        # Update plot
        line.set_ydata(rewards_per_episode)
        line.set_xdata(range(len(rewards_per_episode)))
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)

        # Save model
        torch.save(policy.state_dict(), weight_path)

    return policy


env = gym.make("CartPole-v1", render_mode="human")
state, info = env.reset()

policy = Policy(len(state), env.action_space.n)

# if os.path.exists(weight_path):
# policy.load_state_dict(torch.load(weight_path))

policy_gradient(env, policy, episodes=1000, lr=1e-5)
