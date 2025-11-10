import torch
import gymnasium
from vizdoom import gymnasium_wrapper
from dqn import DQN

device = 'gpu' if torch.cuda.is_available() else 'cpu'

class Agent:
    def run(self, training=False, render=False):
        env = gymnasium.make("VizdoomDeadlyCorridor-v0", render_mode="human" if render else None)   
        
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        
        policy = DQN(num_states, num_actions).to_device(device)
        
        obs, info = env.reset()
        
        for _ in range(1000):
            action = env.action_space.sample() 
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        env.close()
