import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


torch.manual_seed(0)

operations = {
    0: "NOOP",
    1: "FIRE",
    2: "RIGHT",
    3: "LEFT",
    4: "RIGHTFIRE",
    5: "LEFTFIRE",
}


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
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
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Actor(nn.Module):
    def __init__(self, action_dim):
        super(Actor, self).__init__()
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


def actor_critic(env, actor, critic, gamma, episodes, alr=1e-5, clr=1e-2):
    actor_optimizer = optim.Adam(actor.parameters(), lr=alr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=clr)

    for i in range(episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        state = state.permute(2, 0, 1).unsqueeze(0)
        done = False

        states = []
        actions = []
        rewards = []
        next_states = []

        while not done:
            probs = actor(state)
            action = torch.multinomial(probs, 1)

            next_state, reward, terminated, truncated, info = env.step(action.item())

            next_state = torch.tensor(next_state, dtype=torch.float32)
            next_state = next_state.permute(2, 0, 1).unsqueeze(0)
            done = terminated or truncated or reward == -1

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)

            state = next_state

        states = torch.cat(states)
        actions = torch.stack(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.cat(next_states)

        # print(states.shape)
        # print(actions.shape)
        # print(rewards.shape)
        # print(next_states.shape)

        values = critic(states)
        next_values = critic(next_states)

        td_targets = rewards.detach() + gamma * next_values * (1 - done)
        advantages = td_targets.detach() - values
        log_probs = torch.log(actor(states)).gather(1, actions)

        critic_optimizer.zero_grad()
        critic_loss = advantages.pow(2).mean()
        critic_loss.backward()
        critic_optimizer.step()

        actor_optimizer.zero_grad()
        actor_loss = -torch.mean(log_probs * advantages.detach())
        actor_loss.backward()
        actor_optimizer.step()

        # Save models
        torch.save(actor.state_dict(), "actor-pong.pth")
        torch.save(critic.state_dict(), "critic-pong.pth")

        print(f"Episode: {i}, Rewards: {sum(rewards).item()}")

    return actor


# env = gym.make("Pong-v4", mode=1, obs_type="rgb", render_mode="human")
env = gym.make("Pong-v4", mode=1, obs_type="rgb")
state, info = env.reset()  # state.shape = (210, 160, 3)

actor = Actor(env.action_space.n)
if os.path.exists("pong.pth"):
    actor.load_state_dict(torch.load("pong.pth"))

critic = Critic()

actor_critic(env, actor, critic, gamma=0.999, episodes=1000, alr=1e-5, clr=1e-2)
