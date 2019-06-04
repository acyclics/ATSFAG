from agent import DDPGAgent, Episode_experience
from gimbal import gimbal
import numpy as np
import time

num_epochs = 100
num_episodes = 50
episode_length = 500
env = gimbal(5, episode_length)
agent = DDPGAgent(7, 2, 2, action_low=0, action_high=6.18, actor_learning_rate=1e-4, critic_learning_rate=3e-4, tau=0.1)
variance = 6.18
use_her = False # use hindsight experience replay or not
optimization_steps = 40
K = 10 # number of random future states

a_losses = []
c_losses = []
ep_mean_r = []
success_rate = []
MAX_success_rate = -1

ep_experience = Episode_experience()
ep_experience_her = Episode_experience()

start = time.clock()
total_step = 0
for i in range(num_epochs):
    successes = 0
    ep_total_r = 0
    for n in range(num_episodes):
        state = env.reset()
        goal = env.get_goal()
        for ep_step in range(episode_length):
            total_step += 1
            action = agent.choose_action([state], [goal], variance)
            next_state, reward, done, info = env.step(action)
            ep_total_r += reward
            ep_experience.add(state, action, reward, next_state, done, goal)
            state = next_state
            if total_step % 200 == 0 or done:
                if use_her: # The strategy can be changed here
                    for t in range(len(ep_experience.memory)):
                        for _ in range(K):
                            future = np.random.randint(t, len(ep_experience.memory))
                            goal_ = ep_experience.memory[future][3][6:9] # next_state of future
                            goal_data = ep_experience.memory[future][3]
                            state_ = ep_experience.memory[t][0]
                            action_ = ep_experience.memory[t][1]
                            next_state_ = ep_experience.memory[t][3]
                            done_, reward_ = env.reward_func(next_state_, goal_, goal_data)
                            ep_experience_her.add(state_, action_, reward_, next_state_, done_, goal_)
                    agent.remember(ep_experience_her)
                    ep_experience_her.clear()
                agent.remember(ep_experience)
                ep_experience.clear()
                variance *= 0.9995
                a_loss, c_loss = agent.replay(optimization_steps)
                a_losses += [a_loss]
                c_losses += [c_loss]
                agent.update_target_net()
            if done:
                break
        successes += reward>=0 and done
    success_rate.append(successes/num_episodes)
    ep_mean_r.append(ep_total_r/num_episodes)
    MAX_success_rate = max(MAX_success_rate, success_rate[-1])
    print("\repoch", i+1, "success rate", success_rate[-1], "ep_mean_r %.2f"%ep_mean_r[-1], 'exploration %.2f'%variance, end=' '*10)

print("Training time : %.2f"%(time.clock()-start), "s")
print("Max success rate: ", MAX_success_rate)
agent.saver.save(agent.sess, "./models/gimbal_New_t3.ckpt")
