from agent_nogoals_test import Memory, DDPGAgent
from gimbal import gimbal
import numpy as np

def conduct_test(agent, n_episodes, frame_skip, episode_length, success_threshold, success_rate):
    env = gimbal(frame_skip, episode_length)
    successes = 0
    rewards_sum = 0
    for _ in range(n_episodes):
        state = env.reset()
        r = 0
        while True:
            action = agent.choose_action([state], 0, agent.action_low, agent.action_high)
            next_state, reward, done, info = env.step(action)
            r += reward
            state = next_state
            if done:
                if r >= success_threshold:
                    successes += 1
                    rewards_sum += r
                break
    env.close()
    if successes >= success_rate:
        return True, rewards_sum / successes
    else:
        return False, 0

def view(n_episodes, frame_skip, episode_length, success_threshold, model, actions, features, high, low, actor_lr, critic_lr, rewardDecay, priorityA):
    test_rewards = []
    dones_anyreward = 0
    dones_goodreward = 0
    env = gimbal(frame_skip, episode_length)
    agent = DDPGAgent(env, n_actions=actions, n_features=features, featurize=False, action_high=high, action_low=low, actor_learning_rate=actor_lr,
                    critic_learning_rate=critic_lr, reward_decay=rewardDecay, priority_alpha=priorityA)
    agent.saver.restore(agent.sess, model + str("_BEST.ckpt"))
    for i_episode in range(n_episodes):
        state = agent.env.reset()
        r = 0
        while True:
            agent.env.render()
            action = agent.choose_action([state], 0, agent.action_low, agent.action_high)
            next_state, reward, done, info = agent.env.step(action)
            r += reward
            state = next_state
            if done:
                if agent.env.timestep < agent.env.MAX_timestep:
                    dones_anyreward += 1
                    if r > success_threshold:
                        dones_goodreward += 1
                print("episode:", i_episode + 1, "rewards: %.2f" % r, end="\r")
                test_rewards += [r]
                break
    print("\n")
    print("finished testing! Average reward: ", np.sum(test_rewards) / n_episodes, "Dones (any reward): ", dones_anyreward, "Dones (good reward)", dones_goodreward)

#view(n_episodes = 50, frame_skip=5, episode_length=500, success_threshold=-100, model="./models/gimbal_nogoals_0505_t4", actions=2, 
        #features=11, high=20, low=-20, actor_lr=0.0001, critic_lr=0.0002, rewardDecay=0.98, priorityA=0)
