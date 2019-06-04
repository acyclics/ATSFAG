import mujoco_py as mjpy
import numpy as np
from gym.utils import seeding
from gym import spaces
import os

# CHANGABLE VARIABLES
DEFAULT_SIZE = 200

class gimbal:
    ''' Mujoco initialization '''
    def __init__(self, frame_skip, MAX_timestep, rgb_rendering_tracking=True):
         # Model specific
        xml_path = 'gimbal.xml'
        self.model = mjpy.load_model_from_path(xml_path)
        self.sim = mjpy.MjSim(self.model)
        self.rgb_rendering_tracking = rgb_rendering_tracking
        # Data specific
        self.frame_skip = frame_skip
        self.data = self.sim.data
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        self.PB = [0, 0, 0, 1]
        self.lookup = dict()
        self.seed()
        self.viewer = None
        self._viewers = {}
        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.timestep = 0
        self.MAX_timestep = MAX_timestep
        # Action specific data
        observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
        self.obs_dim = observation.size
        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    ''' END '''




    ''' Model upper-level '''
    def pid_fst(self, a):
        a = a[0] % (2 * np.pi)
        angle = self.data.sensordata[0] % (2 * np.pi)
        if abs(a - angle) <= abs(a - angle + (2 * np.pi)):
            error = a - angle
        else:
            error = a - angle + (2 * np.pi)
        p = 0.2
        return p * error
    def pid_snd(self, a):
        a = a[1] % (2 * np.pi)
        angle = self.data.sensordata[1] % (2 * np.pi)
        p = 25
        if abs(a - angle) <= abs(a - angle + (2 * np.pi)):
            error = a - angle
        else:
            error = a - angle + (2 * np.pi)
        return p * error
    def transform(self, theta, phi):
        U = self.get_body_com("gun_platform")
        S = self.get_body_com("base")
        A = self.get_body_com("head_of_barrel")
        B = self.get_body_com("target")
        R_SU1 = [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ]
        R_W = [
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi), 0],
            [0, 0, 1]
        ]
        R_SU2 = np.matmul(R_W, R_SU1)
        P_SU2 = abs(U - S)
        T_U2S = [
            np.append(np.transpose(R_SU2)[0], -np.matmul(np.transpose(R_SU2), P_SU2)[0]),
            np.append(np.transpose(R_SU2)[1], -np.matmul(np.transpose(R_SU2), P_SU2)[1]),
            np.append(np.transpose(R_SU2)[2], -np.matmul(np.transpose(R_SU2), P_SU2)[2]),
            [0, 0, 0, 1]
        ]
        P_U2A = abs(A - U)
        T_AU2 = [
            [1, 0, 0, -P_U2A[0]],
            [0, 1, 0, -P_U2A[1]],
            [0, 0, 1, -P_U2A[2]],
            [0, 0, 0, 1]
        ]
        T_AS = np.matmul(T_AU2, T_U2S)
        P_SB = abs(B - S)
        T_SB = [
            [1, 0, 0, P_SB[0]],
            [0, 1, 0, P_SB[1]],
            [0, 0, 1, P_SB[2]],
            [0, 0, 0, 1]
        ]
        T_AB = np.matmul(T_AS, T_SB)
        return T_AB
    def transform_TAS(self, obs):
        theta = obs[4]
        phi = obs[3]
        U = obs[9:12]
        S = [0, 0, 0]
        A = obs[12:15]
        R_SU1 = [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ]
        R_W = [
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi), 0],
            [0, 0, 1]
        ]
        R_SU2 = np.matmul(R_W, R_SU1)
        P_SU2 = abs(U - S)
        T_U2S = [
            np.append(np.transpose(R_SU2)[0], -np.matmul(np.transpose(R_SU2), P_SU2)[0]),
            np.append(np.transpose(R_SU2)[1], -np.matmul(np.transpose(R_SU2), P_SU2)[1]),
            np.append(np.transpose(R_SU2)[2], -np.matmul(np.transpose(R_SU2), P_SU2)[2]),
            [0, 0, 0, 1]
        ]
        P_U2A = abs(A - U)
        T_AU2 = [
            [1, 0, 0, -P_U2A[0]],
            [0, 1, 0, -P_U2A[1]],
            [0, 0, 1, -P_U2A[2]],
            [0, 0, 0, 1]
        ]
        T_AS = np.matmul(T_AU2, T_U2S)
        return T_AS
    def transform_TSA2(self, obs):
        theta = obs[4]
        phi = obs[3]
        U = obs[9:12]
        S = [0, 0, 0]
        A = obs[12:15]
        R_SU1 = [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ]
        R_W = [
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi), 0],
            [0, 0, 1]
        ]
        R_SU2 = np.matmul(R_W, R_SU1)
        P_SU2 = abs(U - S)
        T_SU2 = [
            np.append(R_SU2[0], P_SU2[0]),
            np.append(R_SU2[1], P_SU2[1]),
            np.append(R_SU2[2], P_SU2[2]),
            [0, 0, 0, 1]
        ]
        P_U2A = abs(A - U)
        T_U2A = [
            [1, 0, 0, P_U2A[0]],
            [0, 1, 0, P_U2A[1]],
            [0, 0, 1, P_U2A[2]],
            [0, 0, 0, 1]
        ]
        T_SA = np.matmul(T_SU2, T_U2A)
        return T_SA
    def step(self, a):
        done, reward = self.reward_func(0, 0, 0)
        self.do_simulation([self.pid_fst(a), self.pid_snd(a)], self.frame_skip)
        self.timestep += 1
        ob = self._get_obs()
        if self.timestep >= self.MAX_timestep:
            self.timestep = 0
            done = True
        return ob, reward, done, dict(goal=self.PB)
    def reward_func(self, state, goal, goal_info):
        ep = 0.1
        if isinstance(state, int):
            angles = self.data.sensordata[0:2] % (2 * np.pi)
            P_AB = np.matmul(self.transform(angles[1], angles[0]), self.PB)
            angle_diff = abs(np.arccos (np.dot(P_AB[0:3], [0, 1, 0]) / (np.linalg.norm(P_AB[0:3]) * 1)))
            if angle_diff <= ep:
                return True, 1
            else:
                return False, -1
        else:
            P_Y = np.matmul(np.matmul(self.transform_TAS(state), self.transform_TSA2(goal_info)), [0, 1, 0, 1])
            angle_diff = abs(np.arccos (np.dot(P_Y[0:3], [0, 1, 0]) / (np.linalg.norm(P_Y[0:3]) * 1)))
            if angle_diff <= ep:
                return True, 1
            else:
                return False, -1
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 3.0
    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.PB[0] = -0.5#np.random.uniform(low=0.2, high=0.8)
        self.PB[1] = 0
        self.PB[2] = 0.5#np.random.uniform(low=0.2, high=0.8)
        self.PB[3] = 1
        self.set_state(qpos, qvel)
        return self._get_obs()
    def get_goal(self):
        return self.PB[0:3]
    def _get_obs(self):
        angles = self.data.sensordata[0:2] % (2 * np.pi)
        P_AB = np.matmul(self.transform(angles[1], angles[0]), self.PB)[0:3]
        angle_diff = np.arccos (np.dot(P_AB[0:3], [0, 1, 0]) / (np.linalg.norm(P_AB[0:3]) * 1))
        return np.concatenate([
            P_AB,
            angles,
            [angle_diff],
            self.get_goal(),
            self.get_body_com("gun_platform"),
            self.get_body_com("head_of_barrel"),
        ])
    ''' END '''




    ''' Model lower-level '''
    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        return ob
    def set_state(self, qpos, qvel):
        old_state = self.sim.get_state()
        new_state = mjpy.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()
    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip
    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()
    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        if mode == 'rgb_array':
            camera_id = None 
            camera_name = 'track'
            if self.rgb_rendering_tracking and camera_name in self.model.camera_names:
                camera_id = self.model.camera_name2id(camera_name)
            self._get_viewer(mode).render(width, height, camera_id=camera_id)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'depth_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == 'human':
            self._get_viewer(mode).render()
    def close(self):
        if self.viewer is not None:
            self.viewer = None
            self._viewers = {}
    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mjpy.MjViewer(self.sim)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mjpy.MjRenderContextOffscreen(self.sim, -1)
                
            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer
    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)
    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])
    ''' END '''
