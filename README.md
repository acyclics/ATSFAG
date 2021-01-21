# ATSFAG
The premise of this project is to train an agent via reinforcement learning to track an object with on-board camera. Below is a demo showing this in-action.

![](rl_demo.mp4)
![RL Demo](https://github.com/acyclics/ATSFAG/blob/master/rl_demo.gif)

The simulation is done with Mujoco and the agent was trained with PPO. The files for training the agent are located in the "TRACKING_train" folder. The remaining files are the code for deploying the agent onto the corresponding microcontrollers.

Done:
1. Local coordinate tracking trained. Agent transfer to hardware successful

To-do:
1. Prediction
2. Projectile
3. Self-play
