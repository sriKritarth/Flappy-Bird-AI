import flappy_bird_gymnasium
import gymnasium as gym
import torch
import random

if torch.cuda.is_available():
    device = "cuda"

else:
    device = "cpu"


def run(self , is_Training = True , render = False):
    env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

    num_states = env.observation_space.shape[0]
    num_action = env.action_space.n

    policy = DQN(num_states , num_action).to(device)
    state, _ = env.reset()
    while True:
        # Next action:
        # (feed the observation to your agent here)
        action = env.action_space.sample()

        # Processing:
        next_state, reward, terminated, _, _ = env.step(action)
        
        # Checking if the player is still alive
        if terminated:
            break

    env.close()