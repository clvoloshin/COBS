import math

import numpy as np

import gym
from gym import spaces
from gym.envs.classic_control import MountainCarEnv
from gym.envs.classic_control.rendering import Viewer
from gym.utils import seeding
from skimage import draw
from skimage.transform import rescale, resize, downscale_local_mean
from copy import deepcopy
import matplotlib.pyplot as plt

class ModifiedMountainCarEnv(MountainCarEnv):
    def __init__(self, deterministic_start = None, seed=0, frameskip=1, frameheight=1, *args, **kw):
        self.start = deterministic_start
        super(ModifiedMountainCarEnv, self).__init__(*args, **kw)
        self.background = None
        self.n_actions = 3
        self.n_dim = 1
        self.reward_model = None
        self.T = 0
        self.frameskip = frameskip
        self.frameheight = frameheight
        self.seed(seed)
        self.reset()
        self.render()

    def overwrite_rewards(self, new_r):
        self.reward_model = new_r

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.T += 1

        position, velocity = self.state
        done = bool(position >= self.goal_position)
        velocity += (action-1)*self.force + math.cos(3*position)*(-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position==self.min_position and velocity<0): velocity = 0



        # if self.done:
        #     reward = 0
        # else:
        # if action == 0:
        #     reward = -1
        # elif action == 1:
        #     reward = -.01
        # else:
        #     reward = -.5

        # reward = -1 if action != 1 else -.5
        # reward = np.random.randn()# 0 if not done else 1
        # reward = 0 if self.done else -1
        # reward = -1
        if self.reward_model is not None:
            reward = -self.reward_model.predict(np.atleast_2d(np.hstack([*self.last_state, self.state, np.eye(3)[action]])))[0][0]
        else:
            reward = min(0, position - self.goal_position) # penalize you from being far away

        self.state = (position, velocity)
        if (self.T % self.frameskip) == 0:
            self.last_state.pop(0)
            self.last_state.append(np.array(deepcopy(self.state)))

        # if done:
        #     self.done = True
        #     self.state = self.start_state

        return np.array(self.state), reward, done, {}


    def initial_states(self):
        return np.array([[x,x] for x in self.start])

    def reset(self):
        self.done = False
        self.T = 0
        if self.start is not None:
            self.state = np.array([self.np_random.choice(self.start), 0])
            self.start_state = deepcopy(self.state)
            self.last_state = [deepcopy(self.state)] * self.frameheight
            return np.array(self.state)
        else:
            self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
            self.start_state = deepcopy(self.state)
            self.last_state = [deepcopy(self.state)] * self.frameheight
            return np.array(self.state)

    @staticmethod
    def line_to_thickline(pt0, pt1,thickness=1):
        (x1,y1),(x2,y2) = pt0, pt1
        if x2 != x1:
            a = math.atan((y2-y1)/(x2-x1))
            sin = math.sin(a)
            cos = math.cos(a)
        else:
            sin = 1.
            cos = 0.
        xdelta = sin * thickness / 2.0
        ydelta = cos * thickness / 2.0
        xx1 = x1 - xdelta
        yy1 = y1 + ydelta
        xx2 = x1 + xdelta
        yy2 = y1 - ydelta
        xx3 = x2 + xdelta
        yy3 = y2 - ydelta
        xx4 = x2 - xdelta
        yy4 = y2 + ydelta
        return [xx1,xx2,xx3,xx4], [yy1,yy2,yy3,yy4]

    def render(self, mode='rgb_array'):
        self.downscale = 1
        self.circle_radius = 8
        self.screen_width = 120 #75
        self.screen_height = 80#50
        self.line_thickness = 2

        world_width = self.max_position - self.min_position
        self.scale = self.screen_width/world_width
        # import pdb; pdb.set_trace()
        carwidth=40
        carheight=20

        if self.background is None:
            self.background = np.ones((self.screen_width, self.screen_height))
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*self.scale, ys*self.scale))

            # Track
            for pt0, pt1 in (zip(xys[:-1], xys[1:])):
                poly_xs, poly_ys = self.line_to_thickline(pt0, pt1, self.line_thickness)
                poly_xs, poly_ys = np.array(poly_xs).astype(int), np.array(poly_ys).astype(int)
                poly_xs, poly_ys = np.minimum(self.screen_width, poly_xs), np.minimum(self.screen_height, poly_ys)
                rr, cc = draw.polygon(poly_xs, poly_ys)
                self.background[rr, cc] = .5

            # Flagpole
            flagx = (self.goal_position-self.min_position)*self.scale
            flagy1 = self._height(self.goal_position)*self.scale
            flagy2 = flagy1 + 50 * (self.screen_width/600)
            poly_xs, poly_ys = self.line_to_thickline((flagx, flagy1), (flagx, flagy2), self.line_thickness)
            poly_xs, poly_ys = np.array(poly_xs).astype(int), np.array(poly_ys).astype(int)
            poly_xs, poly_ys = np.minimum(self.screen_width, poly_xs), np.minimum(self.screen_height, poly_ys)
            rr, cc = draw.polygon(poly_xs, poly_ys)
            self.background[rr, cc] = .3

            # Flag
            poly_xs, poly_ys = [flagx, flagx, flagx+25*(self.screen_width/600)],[flagy2, flagy2-10*(self.screen_width/600),flagy2-5*(self.screen_width/600)]
            poly_xs, poly_ys = np.array(poly_xs).astype(int), np.array(poly_ys).astype(int)
            poly_xs, poly_ys = np.minimum(self.screen_width, poly_xs), np.minimum(self.screen_height, poly_ys)
            rr, cc = draw.polygon(poly_xs, poly_ys)
            self.background[rr, cc] = .1

            self.background = self.background.T[::-1]

            self.background = downscale_local_mean(self.background, (self.downscale,self.downscale))
            self.rr, self.cc = draw.circle(500,500, radius=self.circle_radius, shape=(1000,1000))

        if False and (self.viewer is None) and (self.background is None):
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*self.scale, ys*self.scale))

            # self.track = rendering.make_polyline(xys)
            # self.track.set_linewidth(30) # 4 -> 16
            # self.viewer.add_geom(self.track)

            # clearance = 10 // self.downscale # 10 -> 20

            # l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            # car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            # car.add_attr(rendering.Transform(translation=(0, clearance)))
            # self.cartrans = rendering.Transform()
            # car.add_attr(self.cartrans)
            # self.viewer.add_geom(car)
            # frontwheel = rendering.make_circle(carheight/2.5)
            # frontwheel.set_color(.5, .5, .5)
            # frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            # frontwheel.add_attr(self.cartrans)
            # self.viewer.add_geom(frontwheel)
            # backwheel = rendering.make_circle(carheight/2.5)
            # backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            # backwheel.add_attr(self.cartrans)
            # backwheel.set_color(.5, .5, .5)
            # self.viewer.add_geom(backwheel)


            flagx = (self.goal_position-self.min_position)*self.scale
            flagy1 = self._height(self.goal_position)*self.scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)


            pos = self.state[0]
            x, y = (pos-self.min_position)*self.scale, self._height(pos)*self.scale
            # self.cartrans.set_translation(x, y)

            self.background = self.to_gray(self.viewer.render(return_rgb_array = True))
            # import pdb; pdb.set_trace()
            self.background = downscale_local_mean(self.background, (self.downscale,self.downscale))
            self.viewer.close()


        pos = self.state[0]
        x, y = (pos-self.min_position)*self.scale, (self.screen_height-self._height(pos)*self.scale)
        # self.cartrans.set_translation(x/2, -y/2+self.screen_height)
        # self.cartrans.set_rotation(math.cos(3 * pos))
        # import pdb; pdb.set_trace()

        arr = deepcopy(self.background)
        normal = self._normal(pos)
        normal = normal * self.circle_radius
        arr[np.minimum(self.screen_height // self.downscale - 1,(self.rr -500 + y + normal[1]).astype(int) // self.downscale),
            np.minimum(self.screen_width // self.downscale - 1, (self.cc -500 + x - normal[0]).astype(int) // self.downscale)] = [0.]

        # plt.imshow(arr, cmap='gray')
        # plt.show()
        return arr

    def pos_to_image(self, positions, w_velocity=False):
        out = []
        assert len(positions) > 0
        if w_velocity: positions = positions[...,[0]].reshape(-1, 2)
        for position in positions:
            arrs = []
            for pos in position:
                x, y = (pos-self.min_position)*self.scale, (self.screen_height-self._height(pos)*self.scale)

                arr = deepcopy(self.background)
                normal = self._normal(pos)
                normal = normal * self.circle_radius
                arr[np.minimum(self.screen_height // self.downscale - 1,(self.rr -500 + y + normal[1]).astype(int) // self.downscale),
                    np.minimum(self.screen_width // self.downscale - 1, (self.cc -500 + x - normal[0]).astype(int) // self.downscale)] = [0.]

                arrs.append(arr)
            out.append(arrs)

        return np.array(out)


    def _normal(self, x):

        normal = np.array([.45*3*np.cos(3 * x), -1])
        unit_normal = normal/np.sqrt(sum(normal**2))

        return unit_normal

    @staticmethod
    def to_gray(arr):
        return np.dot(arr[...,:3]/255. , [0.299, 0.587, 0.114])

