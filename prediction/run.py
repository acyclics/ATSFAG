from gimbal import gimbal
import numpy as np
import mujoco_py

class test:
    def run(self):
        gm = gimbal(5, 1000)
        obs = gm.reset()
        action = [5, 0]
        for _ in range(50000):
            gm.render()
            #action = np.random.uniform(low=-6.13, high=6.13, size=2)
            obs, rwd, done, info = gm.step(action)
            if done:
                obs = gm.reset()
        gm.close()

debug = test()
debug.run()
