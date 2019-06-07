import numpy as np
from gimbal import gimbal
#from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.gail import ExpertDataset, generate_expert_traj
from stable_baselines import DDPG, PPO2, GAIL

# DDPG
def train_ddpg():
    env = gimbal(5, 500)
    env = DummyVecEnv([lambda: env])
    eval_env = gimbal(5, 500)
    eval_env = DummyVecEnv([lambda: eval_env])

    # the noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = None

    model = DDPG(policy=MlpPolicy, env=env, gamma=0.99, memory_policy=None, eval_env=eval_env, 
                nb_train_steps=500, nb_rollout_steps=500, nb_eval_steps=500, param_noise=param_noise, 
                action_noise=action_noise, normalize_observations=False, tau=0.001, batch_size=128, 
                param_noise_adaption_interval=50, normalize_returns=False, enable_popart=False, 
                observation_range=(-5000.0, 5000.0), critic_l2_reg=0.0, return_range=(-inf, inf), 
                actor_lr=0.0001, critic_lr=0.001, clip_norm=None, reward_scale=1.0, render=False, 
                render_eval=False, memory_limit=50000, verbose=1, tensorboard_log="./logs", 
                _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False)
    #model = DDPG.load("./models/baseline_ddpg_t2")
    #model.set_env(env)
    model.learn(total_timesteps=1000000, callback=None, seed=None, log_interval=100, tb_log_name='DDPG', reset_num_timesteps=True)
    model.save("./models/baseline_ddpg_t2")

def view_ddpg():
    env = gimbal(5, 500)
    model = DDPG.load("./models/baseline_ddpg_t2")
    success_rate = 0
    reward_avg = 0
    for episodes in range(50):
        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()
            if dones:
                if rewards > -100:
                    success_rate += 1
                    reward_avg += rewards
                break
    print("Success rate: ", success_rate, "Avg rewards: ", reward_avg / success_rate)
# DDPG



# PPO2
def train_ppo2_mlplstm():
    env = DummyVecEnv([lambda: gimbal(5, 500)])
    #model = PPO2(policy=MlpLstmPolicy, env=env, gamma=0.99, n_steps=500, ent_coef=0.01, learning_rate=0.00025, 
    #            vf_coef=0.5, max_grad_norm=0.5, lam=0.95, nminibatches=1, noptepochs=4, cliprange=0.2, 
    #            verbose=1, tensorboard_log="./logs", _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False)
    model = PPO2.load("./models/baseline_ppo2_t6_dynamicR", env=env)
    model.learn(total_timesteps=500000, callback=None, seed=None, log_interval=1, tb_log_name='PPO2', reset_num_timesteps=True)
    model.save("./models/baseline_ppo2_t6_dynamicR")

def train_ppo2_mlp():
    n_cpu = 4
    env = SubprocVecEnv([lambda: gimbal(5, 500) for i in range(n_cpu)])
    model = PPO2(policy=MlpPolicy, env=env, gamma=0.99, n_steps=100, ent_coef=0.01, learning_rate=0.00025, 
                vf_coef=0.5, max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, 
                verbose=1, tensorboard_log="./logs", _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False)
    model.learn(total_timesteps=1000000, callback=None, seed=None, log_interval=1, tb_log_name='PPO2', reset_num_timesteps=True)
    model.save("./models/baseline_ppo2_t11_camshifted")

def view_ppo2_mlp():
    env = gimbal(5, 500)
    model = PPO2.load("./models/baseline_ppo2_t10_camshifted")
    success_rate = 0
    reward_avg = 0
    for episodes in range(50):
        obs = env.reset()
        r = 0
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            r += rewards
            #env.render()
            #env.target_ctrl()
            if dones:
                if r > -100:
                    success_rate += 1
                    reward_avg += r
                break
    print("Success rate: ", success_rate, "Avg rewards: ", (reward_avg / success_rate))

def view_ppo2_mlplstm():
    env = gimbal(5, 500)
    env = DummyVecEnv([lambda: gimbal(5, 500)])
    model = PPO2.load("./models/baseline_ppo2_t6_dynamicR")
    success_rate = 0
    reward_avg = 0
    for episodes in range(50):
        obs = env.reset()
        state = None
        done = [False]
        r = 0
        while True:
            action, state = model.predict(obs, state=state, mask=done)
            obs, rewards , done, _ = env.step(action)
            r += rewards
            env.render()
            if done:
                if r > -100:
                    success_rate += 1
                    reward_avg += r
                break
    print("Success rate: ", success_rate, "Avg rewards: ", (reward_avg / success_rate))
# PPO2



# GAIL
def train_gail_withppo2():
    env = gimbal(5, 500)
    env = DummyVecEnv([lambda: env])
    model = PPO2.load("./models/baseline_ppo2_t1")
    generate_expert_traj(model, './models/baseline_expert_t1', env, n_timesteps=0, n_episodes=100)
    dataset = ExpertDataset(expert_path='./models/baseline_expert_t1.npz', traj_limitation=-1, verbose=1)
    model = GAIL("MlpPolicy", env, dataset, verbose=1)
    model.learn(total_timesteps=500000)
    model.save("./models/baseline_gail_ppo2_t1")
# GAIL



# HER
def train_her():
    env = gimbal(5, 500)
    n_sampled_goal = 4
    model = HER('MlpPolicy', env, SAC, n_sampled_goal=n_sampled_goal,
            goal_selection_strategy='future',
            verbose=1, buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95, batch_size=256,
            policy_kwargs=dict(layers=[256, 256, 256]))
# HER



def main():
    train_ppo2_mlp()
if __name__ == "__main__":
    main()