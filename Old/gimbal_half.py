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
    def findPixel(self):
        img = self.getCam1Data()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        pt = cv2.findNonZero(gray)
        if pt is None:
            return (-100, 0)
        else:
            return pt[0][0]
    def transform(self, theta, phi):
        P_SU2 = self.P_SU2
        P_U2A = self.P_U2A
        P_SB = self.P_SB
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
        T_U2S = [
            np.append(np.transpose(R_SU2)[0], -np.matmul(np.transpose(R_SU2), P_SU2)[0]),
            np.append(np.transpose(R_SU2)[1], -np.matmul(np.transpose(R_SU2), P_SU2)[1]),
            np.append(np.transpose(R_SU2)[2], -np.matmul(np.transpose(R_SU2), P_SU2)[2]),
            [0, 0, 0, 1]
        ]
        T_AU2 = [
            [1, 0, 0, -P_U2A[0]],
            [0, 1, 0, -P_U2A[1]],
            [0, 0, 1, -P_U2A[2]],
            [0, 0, 0, 1]
        ]
        T_AS = np.matmul(T_AU2, T_U2S)
        T_SB = [
            [1, 0, 0, P_SB[0]],
            [0, 1, 0, P_SB[1]],
            [0, 0, 1, P_SB[2]],
            [0, 0, 0, 1]
        ]
        T_AB = np.matmul(T_AS, T_SB)
        return T_AB
    def step(self, a):
        done, reward = self.reward_func(0, 0, 0)
        self.do_simulation([self.pid_fst(a), self.pid_snd(a)], self.frame_skip)
        self.timestep += 1
        ob = self._get_obs()
        if self.timestep >= self.MAX_timestep:
            self.timestep = 0
            done = True
        return ob, reward, done, dict(goal=self.get_goal())
    def reward_func(self, state, goal, goal_info):
        self.XZ = self.findPixel()
        if isinstance(state, int):
            '''
            if self.XZ[0] == -100 and self.XZ[1] == 0:
                return False, -1
            else:
                return True, 0
            '''
            if self.data.sensordata[2] >= 0:
                return True, 0
            else:
                return False, -1
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 3.0
    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        qpos[-3] = np.random.uniform(low=-1, high=1)
        qpos[-1] = np.random.uniform(low=-1, high=1)
        self.set_state(qpos, qvel)
        self.sim.data.qfrc_applied[-2] = self.sim.data.qfrc_bias[-2]    # no gravity for target
        U = self.get_body_com("gun_platform")
        S = self.get_body_com("base")
        A = self.get_body_com("head_of_barrel")
        B = self.get_body_com("target")
        self.P_SU2 = U - S
        self.P_U2A = A - U
        self.P_SB = B - S
        self.XZ = self.findPixel()
        return self._get_obs()
    def get_goal(self):
        return self.XZ
    def _get_obs(self):
        angles = self.data.sensordata[0:2] % (2 * np.pi)
        dist = np.linalg.norm(self.XZ)
        return np.concatenate([
            angles,
            self.XZ,
            [dist],
            self.get_body_com("gun_platform"),
            self.get_body_com("head_of_barrel")
        ])

''' Model upper-level '''
    def pid_fst(self, a):
        a = a[0] % (2 * np.pi)
        angle = self.data.sensordata[0] % (2 * np.pi)
        if abs(a - angle) <= abs((2 * np.pi) - abs(a - angle)):
            error = a - angle
        else:
            error = -( (2 * np.pi) - abs(a - angle) ) * (a - angle) / abs(a - angle)
        p = 0.3
        return p * error
    def pid_snd(self, a):
        a = a[1] % (2 * np.pi)
        angle = self.data.sensordata[1] % (2 * np.pi)
        p = 1.0
        if abs(a - angle) <= abs(a - angle + (2 * np.pi)):
            error = a - angle
        else:
            error = a - angle + (2 * np.pi)
        return p * error
    def findPixel(self):
        img = self.getCam1Data()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        pt = cv2.findNonZero(gray)
        if pt is None:
            return [-self.cam1_w, 0]
        else:
            pt[0][0][0] = pt[0][0][0] - self.cam1_w/2
            pt[0][0][1] = self.cam1_h/2 - pt[0][0][1]
            norm_pt = [pt[0][0][0] / self.cam1_w/2, pt[0][0][1] / self.cam1_h/2]
            return norm_pt
    def step(self, a):
        done, reward = self.reward_func(0, 0, 0)
        self.do_simulation([self.pid_fst(a), self.pid_snd(a)], self.frame_skip)
        self.timestep += 1
        ob = self._get_obs()
        if self.timestep >= self.MAX_timestep:
            self.timestep = 0
            done = True
        return ob, reward, done, dict(goal=self.get_goal())
    def reward_func(self, state, goal, goal_info):
        self.XZ = self.findPixel()
        if isinstance(state, int):
            '''
            if self.XZ[0] == -self.cam1_w and self.XZ[1] == 0:
                return False, -100000000
            elif self.data.sensordata[2] >= 0:
                return True, 0
            else:
                dist = np.linalg.norm(self.XZ)
                return False, -dist
            '''
            if self.data.sensordata[2] >= 0:
                return True, 0
            else:
                return False, -1
            
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 3.0
    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        '''
        Camera range
        x: -0.50 to 0.68
        z: -0.50 to 0.68
        '''
        qpos[-2] = 0 #np.random.uniform(low=-0.5, high=0.5)
        qpos[-1] = 0.5 #np.random.uniform(low=-0.5, high=0.5)
        self.set_state(qpos, qvel)
        self.sim.data.qfrc_applied[-2] = self.sim.data.qfrc_bias[-2]    # no gravity for target
        self.XZ = self.findPixel()
        self.initial_goal = self.XZ
        return self._get_obs()
    def get_goal(self):
        return self.initial_goal
    def _get_obs(self):
        angles = self.data.sensordata[0:2] % (2 * np.pi)
        dist = np.linalg.norm(self.XZ)
        return np.concatenate([
            angles,
            self.initial_goal,
            self.XZ,
            [dist]
        ])
    ''' END '''