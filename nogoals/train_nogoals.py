from agent_nogoals import Memory, DDPGAgent
from gimbal import gimbal
from test_nogoals import conduct_test
import numpy as np
import time

def train(model, frame_skip, episode_length, actions, features, high, low, actor_lr, critic_lr, rewardDecay, priorityA, variance, varianceDecay, rewardcap, checkTime, success_rate):
    # Initialize
    env = gimbal(frame_skip, episode_length)
    agent = DDPGAgent(env, n_actions=actions, n_features=features, featurize=False, action_high=high, action_low=low, actor_learning_rate=actor_lr,
                    critic_learning_rate=critic_lr, reward_decay=rewardDecay, priority_alpha=priorityA)
    agent.variance = variance

    # Training parameters
    MAX_episodes = 1000000000
    i_episode = 0

    # Reward handling
    rewards = []
    max_reward = -999999999999999999
    reward_cap = rewardcap

    # agent.saver.restore(agent.sess, "model/model3d_ddpg.ckpt")
    while round(agent.variance, 1) != 0.0  and i_episode < MAX_episodes:
        state = agent.env.reset()
        r = 0
        while True:
            action = agent.choose_action([state], agent.variance, agent.action_low, agent.action_high)
            next_state, reward, done, info = agent.env.step(action)
            r += reward
            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) == agent.memory_size:
                agent.variance *= varianceDecay
                agent.replay(agent.batch_size)
            state = next_state
            if done:
                i_episode += 1
                max_reward = max(max_reward, r)
                print("episode:", i_episode, "rewards: %.2f" % r, "explore var: %.2f" % agent.variance, "max reward: ", max_reward, end="\r")
                rewards += [r]
                break
        if i_episode % checkTime == 0:
            improvement, avg_reward = conduct_test(agent, 10, frame_skip, episode_length, rewardcap, success_rate)
            if improvement:
                agent.saver.save(agent.sess, model + str("_BEST.ckpt"))
                rewardcap = avg_reward
    print("\n")
    print("Finished training!")
    agent.saver.save(agent.sess, model + str("_LAST.ckpt"))

train(model="./models/gimbal_nogoals_0505_t5", frame_skip=5, episode_length=500, actions=2, 
        features=11, high=20, low=-20, actor_lr=0.0001, critic_lr=0.0002, rewardDecay=0.98, priorityA=0, 
        variance=5, varianceDecay=0.999995, rewardcap=-100, checkTime=10, success_rate=9)

def ultimate_train():
    # tries different hyper-parameters
    episodeLength = 500
