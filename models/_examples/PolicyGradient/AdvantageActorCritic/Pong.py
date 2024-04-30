import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


torch.manual_seed(0)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
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
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = x.permute(2, 0, 1).unsqueeze(0)
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
        x = x.permute(2, 0, 1).unsqueeze(0)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x.squeeze()


def actor_critic(env, actor, critic, gamma, episodes, lr):
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)

    for i in range(episodes):
        state, info = env.reset()
        done = False

        R = []
        while not done:
            if type(state) != torch.Tensor:
                state = torch.tensor(state, dtype=torch.float32)

            probs = actor(state)
            action = torch.multinomial(probs, 1)

            next_state, reward, terminated, truncated, info = env.step(action.item())
            next_state = torch.tensor(next_state, dtype=torch.float32)
            done = terminated or truncated or reward == -1

            R.append(reward)

            log_prob = torch.log(probs[action])

            value = critic(state)
            next_value = critic(next_state) * (1 - done)

            td_target = reward + gamma * next_value * (1 - done)
            advantage = td_target - value

            critic_optimizer.zero_grad()
            critic_loss = advantage.pow(2).mean()
            critic_loss.backward()
            critic_optimizer.step()

            actor_optimizer.zero_grad()
            actor_loss = -log_prob * advantage.detach()
            actor_loss.backward()
            actor_optimizer.step()

            state = next_state

        # Save model
        torch.save(actor.state_dict(), "pong.pth")

        print(f"Episode: {i}, Rewards: {sum(R)}")

    return actor


env = gym.make("Pong-v4", mode=1, obs_type="rgb")
state, info = env.reset()  # state.shape = (210, 160, 3)

actor = Actor(env.action_space.n)
critic = Critic()

actor_critic(env, actor, critic, gamma=0.999, episodes=1000, lr=0.1)
