import torch
import gymnasium
from vizdoom import gymnasium_wrapper
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml

device = 'gpu' if torch.cuda.is_available() else 'cpu'

class Agent:
    
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        
    
    def run(self, training=False, render=False):
        env = gymnasium.make("VizdoomCorridor-v1", render_mode="human" if render else None)   
        num_states = env.observation_space.shape
        num_actions = env.action_space
        
        policy = DQN(num_states, num_actions).to_device(device)
        
        rewards_per_episode = []
        
        if training:
            replay_experience = ReplayMemory(self.replay_memory_size)
        
        
        for episode in itertools.count():
            state, info = env.reset()
            terminated = truncated = False
            episode_reward = 0.0
            
            while not terminated and not truncated:
                action = env.action_space.sample() 
                new_state, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                
                if training:
                    replay_experience.append((state, action, new_state, reward, terminated))
                
                state = new_state
             
            rewards_per_episode.append(episode_reward) 
              
        env.close()

if __name__ == "__main__":
    print([name for name in gymnasium.registry.keys() if "VizdoomCorridor" in name])

    A = Agent()
    A.run(False, True)