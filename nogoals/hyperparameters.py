import optuna
import numpy as np
from gimbal import gimbal
#from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, MlpLnPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.gail import ExpertDataset, generate_expert_traj
from stable_baselines import DDPG, PPO2, GAIL

def objective(trial):
    # Hyper-parameters to adjust
    policy = trial.suggest_categorical('policy', ['MlpPolicy', 'MlpLnPolicy', 'MlpLstmPolicy', 'MlpLnLstmPolicy'])
    gamma = trial.suggest_uniform('gamma', 0.10, 1.0)
    ent_coef = trial.suggest_uniform('ent_coef', 0.01, 0.10)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    vf_coef = trial.suggest_uniform('vf_coef', 0.10, 1.0)
    lam = trial.suggest_uniform('lam', 0.01, 0.95)

    if policy == 'MlpPolicy':
        policy = MlpPolicy
    elif policy == 'MlpLnPolicy':
        policy = MlpLnPolicy
    elif policy == 'MlpLstmPolicy':
        policy = MlpLstmPolicy
    elif policy == 'MlpLnLstmPolicy':
        policy = MlpLnLstmPolicy

    # Train with those hyper-parameters
    n_cpu = 4
    env = SubprocVecEnv([lambda: gimbal(5, 500) for i in range(n_cpu)])
    model = PPO2(policy=policy, env=env, gamma=gamma, n_steps=100, ent_coef=ent_coef, learning_rate=learning_rate, 
                vf_coef=vf_coef, max_grad_norm=0.5, lam=lam, nminibatches=4, noptepochs=4, cliprange=0.2, 
                verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False)
    model.learn(total_timesteps=250000, callback=None, seed=None, log_interval=1, tb_log_name='PPO2', reset_num_timesteps=True)

    # Calculate worth
    env = gimbal(5, 500)
    MAX_episodes = 25
    reward_avg = 0
    for episodes in range(MAX_episodes):
        obs = env.reset()
        r = 0
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            r += rewards
            #env.render()
            if dones:
                reward_avg += r
                break
    return - (reward_avg / MAX_episodes)

def optimize:
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    print("The best, found params:\n", study.best_params)
