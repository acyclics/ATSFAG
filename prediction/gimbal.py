import mujoco_py as mjpy
import numpy as np
import cv2
from gym.utils import seeding
from gym import spaces
import os

# CHANGABLE VARIABLES
DEFAULT_SIZE = 200

class gimbal:
    ''' Mujoco initialization '''
    def __init__(self, frame_skip, MAX_timestep, rgb_rendering_tracking=True):
         # Mujoco specific
        xml_path = 'gimbal.xml'
        self.model = mjpy.load_model_from_path(xml_path)
        self.sim = mjpy.MjSim(self.model)
        self.rgb_rendering_tracking = rgb_rendering_tracking
        # Camera
        self.initCam1(500, 500)
        self.cam1 = [0, 0 ,0]
        # Model specific
        self.frame_skip = frame_skip
        self.data = self.sim.data
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        self.seed()
        self.viewer = None
        self._viewers = {}
        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        # Data specific
        self.obs_dim = 6
        low = -1.0
        high = 1.0
        self.action_space = spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32)
        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.timestep = 0
        self.MAX_timestep = MAX_timestep
        # Control specific
        self.integratedError0 = 0
        self.integratedError1 = 0
        self.p0 = 0.3
        self.p1 = 0.3
        self.i0 = 0.1
        self.i1 = 0.1
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    ''' END '''




    ''' Model functions '''
    def pid_control(self, a):
        error0 = a[0] - self.sim.data.qvel[0]
        error1 = a[1] - self.sim.data.qvel[1]
        if error0 == 0:
            self.integratedError0 = 0
        else:
            self.integratedError0 += error0
        if error1 == 0:
            self.integratedError1 = 0
        else:
            self.integratedError1 += error1
        return [self.p0 * error0 + self.i0 * self.integratedError0, self.p1 * error1 + self.i1 * self.integratedError1]
    def findPixel_centroid(self):
        img = self.getCam1Data()
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        thresh, blackAndWhiteImage = cv2.threshold(gray_image, 0, 230, cv2.THRESH_BINARY)
        M = cv2.moments(blackAndWhiteImage)
        if M["m00"] == 0 or M["m00"] == 0:
            return [-self.cam1_w, 0]
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cX = cX - (self.cam1_w/2.0)
        cY = (self.cam1_h/2.0) - cY
        norm_pt = [cX / (self.cam1_w/2), cY / (self.cam1_h/2)]
        return norm_pt
    def XZ_angle_posX(self):
        if self.XZ[1] == 0 and self.XZ[0] == 0:
            return 2.0 * np.pi
        if self.XZ[1] == 0 and self.XZ[0] > 0:
            return 0.0
        elif self.XZ[1] == 0 and self.XZ[0] < 0:
            return np.pi
        elif self.XZ[0] == 0 and self.XZ[1] > 0:
            return np.pi / 2.0
        elif self.XZ[0] == 0 and self.XZ[1] < 0:
            return (np.pi / 2.0) * 3.0
        elif self.XZ[0] > 0 and self.XZ[1] > 0:
            return np.arctan(self.XZ[1] / self.XZ[0])
        elif self.XZ[0] < 0 and self.XZ[1] > 0:
            return np.pi + np.arctan(self.XZ[1] / self.XZ[0])
        elif self.XZ[0] < 0 and self.XZ[1] < 0:
            return np.pi + np.arctan(self.XZ[1] / self.XZ[0])
        elif self.XZ[0] > 0 and self.XZ[1] < 0:
            return 2.0 * np.pi + np.arctan(self.XZ[1] / self.XZ[0])
    def forward_timestep(self, frames):
        for _ in range(frames):
            self.sim.step()
    ''' END '''




    ''' Model upper-level '''
    def step(self, a):
        done, reward = self.reward_func(a)
        self.do_simulation(self.frame_skip)
        self.timestep += 1
        ob = self._get_obs()
        if self.timestep >= self.MAX_timestep:
            done = True
        return ob, reward, done, dict(goal=0)
    def reward_func(self, prediction):
        self.forward_timestep(20)
        self.XZ = self.findPixel_centroid()
        reward = -np.linalg.norm([self.XZ[0] - prediction[0], self.XZ[1] - prediction[1]])
        if self.XZ[0] == -self.cam1_w and self.XZ[1] == 0:
            return True, reward
        else:
            return False, reward
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 3.0
    def reset_model(self):
        self.timestep = 0
        qpos = self.init_qpos
        qvel = self.init_qvel
        '''
        Camera range
        x: -0.50 to 0.68
        z: -0.50 to 0.68
        '''
        qvel[-1] = np.random.uniform(low=-5, high=5)
        qpos[-1] = np.random.uniform(low=-0.5, high=0.5)
        qvel[-3] = np.random.uniform(low=0, high=5)
        qpos[-3] = np.random.uniform(low=-0.1, high=0.5)
        self.set_state(qpos, qvel)
        self.sim.data.qfrc_applied[-3] = self.sim.data.qfrc_bias[-3]    # no gravity for target
        self.XZ = self.findPixel_centroid()
        return self._get_obs()
    def _get_obs(self):
        dist = np.linalg.norm(self.XZ)
        XZ_angle = self.XZ_angle_posX()
        vec_a = [-1 - self.XZ[0], 1 - self.XZ[1]]
        vec_b = [1 - self.XZ[0], 1 - self.XZ[1]]
        vec_c = [-1 - self.XZ[0], -1 - self.XZ[1]]
        vec_d = [1 - self.XZ[0], -1 - self.XZ[1]]
        return np.concatenate([
            self.XZ,
            [self.sim.data.qvel[-1], self.sim.data.qvel[-3]],
            [XZ_angle, dist]
            #[XZ_angle, dist, np.linalg.norm(vec_a), np.linalg.norm(vec_b), np.linalg.norm(vec_c), np.linalg.norm(vec_d)]
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
    def initCam1(self, WIDTH, HEIGHT):
        self.cam1_w = WIDTH
        self.cam1_h = HEIGHT
        self.cam1_id = self.model.camera_name2id("cam1")
        self.cam1Viewer = mjpy.MjRenderContextOffscreen(self.sim, self.cam1_id)
    def getCam1Data(self):
        self.cam1Viewer.render(self.cam1_w, self.cam1_h, self.cam1_id)
        data = self.cam1Viewer.read_pixels(self.cam1_w, self.cam1_h, depth=False)
        return data[::-1, :, :]
    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip
    def do_simulation(self, n_frames):
        for _ in range(n_frames):
            self.sim.step()
    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        if mode == 'rgb_array':
            camera_id = None 
            camera_name = 'cam1'
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

'''
Mujoco info (http://mujoco.org/book/APIreference.html#tyPrimitive)
In general, stuff are ordered by the order they were initialzied in XML

BODY
self.model.body_pos[15] = "name='camera'"
self.model.body_mass[10] = "name='barrel'"

GEOM
self.model.geom_size[0] = "name='base'" -> geom(Cylinder)
'''
