from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from tomas.environment import QubitAllocationEnv
from src.tomas.fe import FeatureExtractor


class QubitAllocator:
    def __init__(self, env):
        self.env = env

    def train(self, total_timesteps=10000, policy_kwargs=None):

        model = PPO("MultiInputPolicy", 
                    self.env, 
                    verbose=1, 
                    policy_kwargs=policy_kwargs)
        
        model.learn(total_timesteps=total_timesteps)

        return model

