import flappy_bird_gymnasium
import gymnasium as gym
import torch
import random
from experience_replay import ReplayMemory
import itertools
import yaml
import torch.nn as nn
import torch.optim as optim
from dqn import DQN
import argparse
import os

if torch.cuda.is_available():
    device = "cuda"

else:
    device = "cpu"


RUNS_DIR = "runs"
os.makedirs(RUNS_DIR , exist_ok= True)
class Agent :

    def __init__(self , params_set):

        self.params_set = params_set

        with open("parameters.yaml" , "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[params_set]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_dacay = params["epsilon_dacay"]
        self.replay_memory_size = params["replay_memory_size"]
        self.min_batch_size = params["min_batch_size"]
        self.network_sync_rate = params["network_sync_rate"]
        self.reward_threshold = params["reward_threshold"]


        self.loss_fn = nn.MSELoss()
        self.optim = None

        self.LOG_FILE = os.path.join(RUNS_DIR , f"{self.params_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR , f"{self.params_set}.pt")


    def run(self , is_Training = True , render = False):
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_action = env.action_space.n

        policy = DQN(num_states , num_action).to(device)
        

        if is_Training:
            memory =ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states , num_action).to(device)
            target_dqn.load_state_dict(policy.state_dict())

            steps = 0
            self.optim = optim.Adam(policy.parameters() , lr = self.alpha)
            
            best_reward = float("-inf")

        else:
            policy.load_state_dict(torch.load(self.MODEL_FILE))
            policy.eval()

        for episode in itertools.count():

            state, _ = env.reset()
            state = torch.tensor(state , dtype=torch.float , device=device)
            terminated = False
            ep_rewards = 0

            while (not terminated and ep_rewards < self.reward_threshold):
                # Next action:
                # (feed the observation to your agent here)

                if is_Training and random.random < epsilon :
                    action = env.action_space.sample() #explore
                    action = torch.tensor(action , dtype=torch.long , device= device)

                else:
                    with torch.no_grad():
                        action = policy(state.unsqueeze(dim=0)).squeeze().argmax() #exploit

                # Processing:
                next_state, reward, terminated, _, _ = env.step(action.item())
                next_state = torch.tensor(next_state , dtype=torch.float , device=device)
                ep_rewards += reward
                reward = torch.tensor(reward , dtype=torch.float , device=device)

                if is_Training:
                    memory.append((state , action , next_state , reward , terminated))
                    steps += 1 
                
                state = next_state
                

            print(f"Episode = {episode + 1} with total_rewards = {ep_rewards} & epsilon = {epsilon}")

            if is_Training :
                epsilon = max(epsilon * self.epsilon_dacay , self.epsilon_min)
                # Checking if the player is still alive

                if ep_rewards  > best_reward :
                    log_msg = f"best reward = {ep_rewards} for episode = {episode + 1}"

                    with open(self.LOG_FILE , "a") as f:
                        f.write(log_msg + "\n")

                    torch.save(policy.state_dict() , self.MODEL_FILE)
                    best_reward = ep_rewards

            if is_Training and len(memory) > self.min_batch_size:
                mini_batch = memory.sample(self.min_batch_size)

                self.optimize(mini_batch , policy , target_dqn)

            if steps > self.network_sync_rate:
                target_dqn.load_state_dict(policy.state_dict())
                steps = 0


    def optimize(self , mini_batch , policy , target_dqn):
        state , action , next_state , reward , termination = zip(*mini_batch)

        state = torch.stack(state)
        action = torch.stack(action)
        next_state = torch.stack(next_state)
        reward = torch.stack(reward)
        termination = torch.tensor(termination).float().to(device)

        with torch.no_grad():
            target_q = reward + (1 - termination) * self.gamma * target_dqn(next_state).max(dim = 1)[0]



            current_q = policy(state).gather(dim = 1 , index = action.unsqueeze(dim = 1)).squeeze()

            loss = self.loss_fn(current_q , target_q)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()




    # env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test model.")
    parser.add_argument('hyperparameters' , help= '')
    parser.add_argument("--train" , help='Training mode' , action='store_true')
    args =parser.parse_args()


    dql = Agent(params_set= args.hyperparameters)

    if args.train :
        dql.run(is_Training= True )

    else:
        dql.run(is_Training=False ,render=True)
