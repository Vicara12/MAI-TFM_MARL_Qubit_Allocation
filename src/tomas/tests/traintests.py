from fgproee.alg.fgp import FineGrainedPartitionerROEE
from sampler.randomcircuit import RandomCircuit
from tomas.environment import QubitAllocationEnv, RandomCircuitEnv
from tomas.fe import FeatureExtractor
from tomas.ppo import QubitAllocator
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy




def basicTrainParams():
  return dict(
    num_lq = 8,
    num_slices = 2,
    num_cores = 2,
    use_extended = False,
    batch_size = 1,
    N_max = 50,
    alpha = 0.1,
    beta = 0.1,
    gamma = 0.9,
    initial_capacity = 4,
    gnn = "gcn",
    heads = 4,
    hidden_dim = 16,
    total_timesteps = 10000
  )



def trainDemo(num_lq: int,
              num_slices: int,
              num_cores: int,
              initial_capacity: int,
              batch_size: int,
              use_extended: bool = False,
              N_max: int = 50,
              alpha: float = 3,
              beta: float = 0.1,
              gamma: float = 0.9,
              gnn: str = "gcn",
              heads: int = 4,
              hidden_dim: int = 16,
              total_timesteps: int = 10000
            ):
    

    base_env = QubitAllocationEnv(N_max=N_max,
                 C=num_cores,
                 use_extended=use_extended,
                 alpha=alpha,
                 beta=beta,
                 gamma=gamma,
                 initial_capacity=initial_capacity)
    
    # This wrapper provides a random circuit sampler
    env = RandomCircuitEnv(base_env, num_lq=num_lq, num_slices=num_slices)

    policy_kwargs = dict(
            features_extractor_class=FeatureExtractor,
            features_extractor_kwargs=dict(
                gnn=gnn,
                hidden_dim=hidden_dim,
                heads=heads
            )
        )

    # We assume that PPO hyperparameters were set to default
    model = PPO("MultiInputPolicy", 
                    env, 
                    verbose=1,
                    gamma=1, # Since the method is episodic
                    policy_kwargs=policy_kwargs)
        
    model.learn(total_timesteps=total_timesteps)

    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)