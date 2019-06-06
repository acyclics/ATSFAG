import mujoco_py as mjpy
import numpy as np
import cv2
from gym.utils import seeding
from gym import spaces
import os
import keyboard

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
        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
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
        # Dynamic randomization specific
        self.storeInitParams()
        # Annealing
        self.gamma = 1.0
        self.decay = 0.98
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
    ''' END '''




    ''' Model dynamic randomization '''
    def storeInitParams(self):
        self.initParams = [
            self.frame_skip,                                                # frame skip
            self.p0,                                                        # motor0 p of pid
            self.p1,                                                        # motor1 p of pid
            self.i0,                                                        # motor0 i of pid
            self.i1,                                                        # motor1 i of pid
            self.model.body_pos[15],                                        # position of camera
            self.model.body_mass[10],                                       # mass of barrel
            self.model.jnt_stiffness[0],                                    # stiffness of "motor1" joint
            self.model.jnt_stiffness[1],                                    # stiffness of "motor2" joint
            self.model.dof_frictionloss[0],                                 # frictionloss of "motor1" joint
            self.model.dof_frictionloss[1],                                 # frictionloss of "motor2" joint
            self.model.dof_damping[0],                                      # damping of "motor1" joint
            self.model.dof_damping[1],                                      # damping of "motor2" joint
        ]
    def dRandomize(self):
        self.frame_skip = np.random.randint(low=1, high=10)
        self.p0 = np.random.uniform(low=0.5, high=2) * self.initParams[1]
        self.p1 = np.random.uniform(low=0.5, high=2) * self.initParams[2]
        self.i0 = np.random.uniform(low=0.5, high=2) * self.initParams[3]
        self.i1 = np.random.uniform(low=0.5, high=2) * self.initParams[4]
        self.model.body_pos[15] = [0., np.random.uniform(low=0.10, high=0.20), 0.]
        self.model.body_mass[10] = np.random.uniform(low=0.01, high=0.10)
        self.model.jnt_stiffness[0] = np.random.uniform(low=0, high=0.10)
        self.model.jnt_stiffness[1] = np.random.uniform(low=0, high=0.10)
        self.model.dof_frictionloss[0] = np.random.uniform(low=0, high=0.10)
        self.model.dof_frictionloss[1] = np.random.uniform(low=0, high=0.10)
        self.model.dof_damping[0] = np.random.uniform(low=0, high=0.10)
        self.model.dof_damping[1] = np.random.uniform(low=0, high=0.10)
    ''' END '''




    ''' Model upper-level '''
    def step(self, a):
        done, reward = self.reward_func(0, 0, 0)
        action = self.pid_control(a)
        if not np.isfinite(action[0]) or not np.isfinite(action[1]):
            done = True
            reward = -100000
            action = [0, 0]
        self.do_simulation(action, self.frame_skip)
        self.timestep += 1
        ob = self._get_obs()
        if self.timestep >= self.MAX_timestep:
            done = True
        return ob, reward, done, dict(goal=0)
    def reward_func(self, state, goal, goal_info):
        self.XZ = self.findPixel_centroid()
        if isinstance(state, int):
            eps = 0.01
            dist = np.linalg.norm(self.XZ)
            if self.gamma != 0:
                if self.XZ[0] == -self.cam1_w and self.XZ[1] == 0:
                    return True, -100000
                elif dist <= eps:
                    return True, 0
                else:
                    return False, -dist
            else:
                if dist <= eps:
                    return True, 0
                else:
                    return False, -1
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 3.0
    def reset_model(self):
        self.timestep = 0
        self.gamma *= self.decay
        qpos = self.init_qpos
        qvel = self.init_qvel
        '''
        Camera range
        x: -0.50 to 0.68
        z: -0.50 to 0.68
        '''
        qpos[-2] = np.random.uniform(low=-0.1, high=0.2)
        qpos[-1] = np.random.uniform(low=-0.5, high=0.2)
        qvel[-1] = np.random.uniform(low=-5, high=5)
        qvel[-2] = np.random.uniform(low=0, high=3)
        self.set_state(qpos, qvel)
        self.sim.data.qfrc_applied[-2] = self.sim.data.qfrc_bias[-2]    # no gravity for target
        #self.dRandomize()
        self.XZ = self.findPixel_centroid()
        return self._get_obs()
    def _get_obs(self):
        #angular = [np.random.normal(self.sim.data.qvel[0], 0.005), np.random.normal(self.sim.data.qvel[1], 0.005)]  # noise
        angular = self.sim.data.qvel[0:2]
        dist = np.linalg.norm(self.XZ)
        XZ_angle = self.XZ_angle_posX()
        vec_a = [-1 - self.XZ[0], 1 - self.XZ[1]]
        vec_b = [1 - self.XZ[0], 1 - self.XZ[1]]
        vec_c = [-1 - self.XZ[0], -1 - self.XZ[1]]
        vec_d = [1 - self.XZ[0], -1 - self.XZ[1]]
        return np.concatenate([
            angular,
            self.XZ,
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
    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
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




    ''' Target control suite '''
    def target_up(self):
        self.sim.data.qvel[-2] = 2
    def target_down(self):
        self.sim.data.qvel[-2] = -2
    def target_left(self):
        self.sim.data.qvel[-1] = -2
    def target_right(self):
        self.sim.data.qvel[-1] = 2
    def target_ctrl(self):
        if keyboard.is_pressed('up'):
            self.target_up()
        elif keyboard.is_pressed('down'):
            self.target_down()
        else:
            self.sim.data.qvel[-2] = 0
        if keyboard.is_pressed('left'):
            self.target_left()
        elif keyboard.is_pressed('right'):
            self.target_right()
        else:
            self.sim.data.qvel[-1] = 0
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
