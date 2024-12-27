import gymnasium as gym
import minari
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import imageio.v3 as iio
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import os



# Define the Actor and Critic Networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.model(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


def preprocess_observation(obs):
    achieved_goal = obs['achieved_goal']
    desired_goal = obs['desired_goal']
    observation = obs['observation']
    return np.concatenate([observation, achieved_goal, desired_goal], axis=-1)

# TD3+BC Algorithm
class TD3_BC:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).cuda()
        self.actor_target = Actor(state_dim, action_dim, max_action).cuda()
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).cuda()
        self.critic_target = Critic(state_dim, action_dim).cuda()
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.tau = 0.005
        self.gamma = 0.99
        self.bc_alpha = 2.5

    def train(self, replay_buffer, batch_size=256):
        # Sample from buffer
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)
        state, action, next_state, reward, done = state.cuda(), action.cuda(), next_state.cuda(), reward.cuda(), done.cuda()

        # Critic update
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = reward + self.gamma * (1 - done) * torch.min(target_q1, target_q2)

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        current_action = self.actor(state)
        actor_q = self.critic(state, current_action)[0]

        bc_loss = ((current_action - action) ** 2).mean()
        actor_loss = -actor_q.mean() + self.bc_alpha * bc_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, dataset):
        self.transitions = []
        for episode in dataset.iterate_episodes():
            # Ensure consistent lengths across attributes
            min_length = min(
                len(episode.observations['observation']) - 1,  # Truncate observations
                len(episode.actions),
                len(episode.rewards),
                len(episode.terminations),
                len(episode.truncations),
            )

            # Process attributes
            observations = np.array([
                preprocess_observation(
                    {'achieved_goal': episode.observations['achieved_goal'][i],
                     'desired_goal': episode.observations['desired_goal'][i],
                     'observation': episode.observations['observation'][i]}
                ) for i in range(min_length)
            ])
            actions = np.array(episode.actions[:min_length])
            rewards = np.array(episode.rewards[:min_length])
            terminations = np.array(episode.terminations[:min_length])
            truncations = np.array(episode.truncations[:min_length])

            # Combine terminations and truncations to derive terminals
            terminals = np.logical_or(terminations, truncations).astype(np.float32)

            # Compute next_observations
            next_observations = np.array([
                preprocess_observation(
                    {'achieved_goal': episode.observations['achieved_goal'][i + 1],
                     'desired_goal': episode.observations['desired_goal'][i + 1],
                     'observation': episode.observations['observation'][i + 1]}
                ) for i in range(min_length)
            ])

            for i in range(min_length):
                transition = {
                    "observations": observations[i],
                    "actions": actions[i],
                    "rewards": rewards[i],
                    "next_observations": next_observations[i],
                    "terminals": terminals[i],
                }
                self.transitions.append(transition)

    def sample(self, batch_size):
        batch = np.random.choice(self.transitions, size=batch_size)
        states = torch.tensor([t['observations'] for t in batch], dtype=torch.float32)
        actions = torch.tensor([t['actions'] for t in batch], dtype=torch.float32)
        rewards = torch.tensor([t['rewards'] for t in batch], dtype=torch.float32).unsqueeze(-1)
        next_states = torch.tensor([t['next_observations'] for t in batch], dtype=torch.float32)
        dones = torch.tensor([t['terminals'] for t in batch], dtype=torch.float32).unsqueeze(-1)
        return states, actions, next_states, rewards, dones

def select_action(policy, state):
    """Select an action using the trained policy."""
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).cuda()
    with torch.no_grad():
        action = policy(state).cpu().numpy().squeeze(0)
    return action

def evaluate_policy(policy, env, num_episodes=10):
    """Evaluate the policy and return average reward."""
    total_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = preprocess_observation(obs)
        episode_reward = 0
        done = False

        while not done:
            action = select_action(policy, state)
            obs, reward, terminated, truncated, info = env.step(action)
            state = preprocess_observation(obs)
            episode_reward += reward

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward

def render_and_save_video(actor, eval_env, filename= "rendered_video.mp4"):
    """
    Renders the evaluation environment and saves it as a video.
    """
    frames = []
    state, _ = eval_env.reset()
    
    for _ in range(100000):  # Render for 1000 steps
        # Convert state to tensor and pass through actor to get action
        action = actor(torch.tensor(preprocess_observation(state), dtype=torch.float32).cuda().unsqueeze(0)).detach().cpu().numpy().squeeze(0)
        
        # Step the environment
        state, reward, terminated, truncated, info = eval_env.step(action)
        
        # Render and collect frames
        frame = eval_env.render()  # Collect the frame directly
        if frame is not None:
            frames.append(frame)
        
        # Reset environment if episode ends
        if terminated or truncated:
            state, _ = eval_env.reset()

    print(f"Current working directory: {os.getcwd()}")
    iio.imwrite(filename, frames, fps=30, plugin="FFMPEG")
    print(f"Video saved as {filename}")


if __name__ == "__main__":
    # Specify the dataset name
    dataset_name = 'D4RL/pointmaze/medium-dense-v2'
    
    # Load the Minari dataset (download if not available locally)
    try:
        dataset = minari.load_dataset(dataset_name)
    except FileNotFoundError:
        print(f"Dataset {dataset_name} not found locally. Downloading...")
        dataset = minari.load_dataset(dataset_name, download=True)
        print(f"Dataset {dataset_name} downloaded successfully.")

    #print("Dataset attributes:", dir(dataset))
    
    #print(f"Dataset type: {type(dataset)}")
    #print(f"Dataset attributes: {dir(dataset)}")
    #print(torch.__version__)
    #print(torch.cuda.is_available())
    # Extract environment information
    env  = dataset.recover_environment()
    obs_example, _ = env.reset()
    state_dim = preprocess_observation(obs_example).shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize the replay buffer
    replay_buffer = ReplayBuffer(dataset)

    # Initialize TD3+BC
    td3_bc = TD3_BC(state_dim, action_dim, max_action)

    # Train the agent
    for epoch in range(10000):
        td3_bc.train(replay_buffer)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} completed.")

    torch.save(td3_bc.actor.state_dict(), "trained_actor_10mins.pt")
    print("Trained actor saved.")

    actor = Actor(state_dim, action_dim, max_action).cuda()

    # Load the trained weights into the actor
    actor.load_state_dict(torch.load("trained_actor_10mins.pt",weights_only=True))
    actor.eval()

    eval_env = dataset.recover_environment()
    
    eval_env = gym.make(eval_env.spec.id, render_mode="human")
    eval_env.reset()
    #print(dir(eval_env))

    obs, _ = eval_env.reset()
    state = preprocess_observation(obs)

    #for _ in range(100):
    #    obs, rew, terminated, truncated, info = eval_env.step(eval_env.action_space.sample())
    #    if terminated or truncated:
    #        eval_env.reset()

    evaluate_policy(actor, eval_env)
    #render_and_save_video(actor, eval_env)
    env.close()
    eval_env.close()
