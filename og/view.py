from agent import DDPGAgent, Episode_experience
from gimbal import gimbal

def view():
    episode_length = 500
    env = gimbal(5, episode_length)
    agent = DDPGAgent(11, 2, 2, action_low=-6.18, action_high=6.18, actor_learning_rate=1e-4, critic_learning_rate=3e-4, tau=0.1)
    agent.saver.restore(agent.sess, "./models/gimbal_New_t2.ckpt")
    total_avg_reward = 0
    dones = 0
    for _ in range(50):
        state = env.reset()
        goal = env.get_goal()
        r = 0
        ts = 0
        for _ in range(episode_length):
            env.render()
            action = agent.choose_action([state], [goal], 0)
            ts += 1
            next_state, reward, done, info = env.step(action)
            r += reward
            state = next_state
            if done:
                if ts < episode_length:
                    dones += 1
                break
        print("reward : %06.2f"%r, " success :", reward==0)
        total_avg_reward += r
    total_avg_reward /= 50.0
    print("total_avg_reward : %06.2f"%total_avg_reward, " dones :", dones)

view()
