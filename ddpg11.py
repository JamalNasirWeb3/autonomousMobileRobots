from env13 import CoppeliaSimEnv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot  as  plt
import sys
import select
import msvcrt
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, next_state, reward):
        self.buffer.append((state, action, next_state, reward))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, next_state, reward = map(np.stack, zip(*batch))
        return state, action, next_state, reward

    def __len__(self):
        return len(self.buffer)

class DDPG:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer(max_size=100000)
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.001

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy()[0]
        return action

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, next_state, reward = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor(reward)

        # Critic update
        next_action = self.actor_target(next_state)
        target_q = self.critic_target(next_state, next_action)
        target_q = reward + (self.gamma * target_q).detach()
        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


def train_ddpg(env):
    state_dim = env.state_space
    action_dim = 2  # Two-dimensional action space for left and right motor speeds
    agent = DDPG(state_dim, action_dim)
    max_episodes = 100  # Increase the number of episodes for a better plot
    max_steps = 50  # Increase the number of steps per episode
    rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            keypress = check_for_keypress()
            if keypress == 'x':
                env.plot_cumulative_behaviors()
                env.plot_distance_graph()
                continue_prompt = input("Press 'y' to continue simulation: ")
                while continue_prompt.lower() != 'y':
                    continue_prompt = input("Invalid input. Press 'y' to continue simulation: ")

            action = agent.select_action(state)
            try:
                next_state, reward, done, _ = env.step(action)
            except ValueError as e:
                print(f"Error in env.step: {e}")
                break
            
            agent.replay_buffer.push(state, action, next_state, reward)
            episode_reward += reward

            agent.train()

            if done:
                break

            state = next_state

        rewards.append(episode_reward)
        print(f"Episode {episode}, Total Reward: {episode_reward}")

        env.accumulate_episode_data()
        #env.plot_distance_graph()

    env.close()
    
    return rewards

    
# def check_for_exit():
#     # Check if 'x' key was pressed
#     if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
#         user_input = sys.stdin.read(1)
#         if user_input == 'x':
#             print("Exiting due to 'x' key press.")
#             return True
#     return False

    import msvcrt

def check_for_keypress():
    # Check if 'x' or 'y' key was pressed
    if msvcrt.kbhit():
        user_input = msvcrt.getch().decode('utf-8')
        if user_input == 'x':
            return 'x'
        elif user_input == 'y':
            return 'y'
    return None


if __name__ == "__main__":
    #from env import CoppeliaSimEnv  # Assuming the environment is in a file named env.py

    try:
        env = CoppeliaSimEnv(num_obstacles=3)
       
        rewards = train_ddpg(env)
        
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Rewards over Episodes')
        plt.show()
        env.plot_distance_graph()
            
        
    except Exception as e:
        print(f"Error during DDPG training: {e}")
    finally:
        #env.close()
        print("DDPG training complete.")
