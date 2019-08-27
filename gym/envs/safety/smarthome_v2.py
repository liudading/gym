import os, time, pickle
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import gym
from collections import deque

""" Deferrable Appliances """
class Stove(object):
    def __init__(self, init_time, dt, rnd=np.random.RandomState()):

        self.state = [0.0, 0.0, 0.0] # status, progress, attribute
        self.action = 0.0

        self.MIN_POWER = 0.0 #kW
        self.MAX_POWER = 1.9 #kW
        self.DEMAND = 3 # time slots

        self.init_time = init_time
        self.dt = dt
        self._rnd = rnd
        self._uniform = lambda x: int(np.round(self._rnd.uniform(x[0],x[1])))

        self.alpha_dist = dict(
            {"left": (pd.datetime(2016,1,1,14,00) - self.init_time).seconds // self.dt.seconds,
             "right": (pd.datetime(2016,1,1,15,00) - self.init_time).seconds // self.dt.seconds,
             })
        self.beta_dist = dict(
            {"left": (pd.datetime(2016,1,1,17,00) - self.init_time).seconds // self.dt.seconds,
             "right": (pd.datetime(2016,1,1,17,30) - self.init_time).seconds // self.dt.seconds,
             })

    def step(self, action, time):
        assert action in [0, 1]
        status, progress, remain_slots = self.state
        if time >= self.alpha and time <= self.beta:
            status = 1.0
            progress += action
            remain_slots = (self.beta - time) / self.dt
        else:
            status = 0.0
            progress = 0.0
            remain_slots = 0.0

        self.state = [status, progress, remain_slots]

    def reset(self, rnd):
        self.state = [0.0, 0.0, 0.0]
        self.action = 0.0
        self._rnd = rnd
        self._uniform = lambda x: int(np.round(self._rnd.uniform(x[0],x[1])))

        self.alpha = self.init_time + self.dt * self._uniform(list(self.alpha_dist.values()))
        self.beta = self.init_time + self.dt * self._uniform(list(self.beta_dist.values()))

    def feasible_action(self):
        status, progress, remain_slots = self.state
        if status == 1.0:
            if progress == 0.0:
                if remain_slots > self.DEMAND:
                    action = np.array([0.0, 1.0], dtype=np.float32)
                else:
                    action = np.array([1.0, 1.0], dtype=np.float32)
            elif progress == self.DEMAND:
                action = np.array([0.0, 0.0], dtype=np.float32)
            else:
                action = np.array([1.0, 1.0], dtype=np.float32)
        else:
            action = np.array([0.0, 0.0], dtype=np.float32)

        return action

class Dishwasher(object):
    def __init__(self, init_time, dt, rnd=np.random.RandomState()):

        self.state = [0.0, 0.0, 0.0] # status, progress, attribute
        self.action = 0.0

        self.MIN_POWER = 0.0 #kW
        self.MAX_POWER = 0.6 #kW
        self.DEMAND = 2 # time slots

        self.init_time = init_time
        self.dt = dt
        self._rnd = rnd
        self._uniform = lambda x: int(np.round(self._rnd.uniform(x[0],x[1])))

        self.alpha_dist = dict(
            {"left": (pd.datetime(2016,1,1,8,00) - self.init_time).seconds // self.dt.seconds,
             "right": (pd.datetime(2016,1,1,9,00) - self.init_time).seconds // self.dt.seconds,
             })
        self.beta_dist = dict(
            {"left": (pd.datetime(2016,1,1,16,00) - self.init_time).seconds // self.dt.seconds,
             "right": (pd.datetime(2016,1,1,17,00) - self.init_time).seconds // self.dt.seconds,
             })

    def step(self, action, time):
        assert action in [0, 1]
        status, progress, remain_slots = self.state
        if time >= self.alpha and time <= self.beta:
            status = 1.0
            progress += action
            remain_slots = (self.beta - time) / self.dt
        else:
            status = 0.0
            progress = 0.0
            remain_slots = 0.0

        self.state = [status, progress, remain_slots]

    def reset(self, rnd):
        self.state = [0.0, 0.0, 0.0]
        self.action = 0.0
        self._rnd = rnd
        self._uniform = lambda x: int(np.round(self._rnd.uniform(x[0],x[1])))

        self.alpha = self.init_time + self.dt * self._uniform(list(self.alpha_dist.values()))
        self.beta = self.init_time + self.dt * self._uniform(list(self.beta_dist.values()))

    def feasible_action(self):
        status, progress, remain_slots = self.state
        if status == 1.0:
            if progress == 0.0:
                if remain_slots > self.DEMAND:
                    action = np.array([0.0, 1.0], dtype=np.float32)
                else:
                    action = np.array([1.0, 1.0], dtype=np.float32)
            elif progress == self.DEMAND:
                action = np.array([0.0, 0.0], dtype=np.float32)
            else:
                action = np.array([1.0, 1.0], dtype=np.float32)
        else:
            action = np.array([0.0, 0.0], dtype=np.float32)

        return action

class WashingMachine(object):
    def __init__(self, init_time, dt, rnd=np.random.RandomState()):

        self.state = [0.0, 0.0, 0.0] # status, progress, attribute
        self.action = 0.0

        self.MIN_POWER = 0.0 #kW
        self.MAX_POWER = 0.38 #kW
        self.DEMAND = 4 # time slots

        self.init_time = init_time
        self.dt = dt
        self._rnd = rnd
        self._uniform = lambda x: int(np.round(self._rnd.uniform(x[0],x[1])))

        self.alpha_dist = dict(
            {"left": (pd.datetime(2016,1,1,9,00) - self.init_time).seconds // self.dt.seconds,
             "right": (pd.datetime(2016,1,1,10,00) - self.init_time).seconds // self.dt.seconds,
             })
        self.beta_dist = dict(
            {"left": (pd.datetime(2016,1,1,17,00) - self.init_time).seconds // self.dt.seconds,
             "right": (pd.datetime(2016,1,1,18,00) - self.init_time).seconds // self.dt.seconds,
             })

    def step(self, action, time):
        assert action in [0, 1]
        status, progress, remain_slots = self.state
        if time >= self.alpha and time <= self.beta:
            status = 1.0
            progress += action
            remain_slots = (self.beta - time) / self.dt
        else:
            status = 0.0
            progress = 0.0
            remain_slots = 0.0

        self.state = [status, progress, remain_slots]

    def reset(self, rnd):
        self.state = [0.0, 0.0, 0.0]
        self.action = 0.0
        self._rnd = rnd

        self.alpha = self.init_time + self.dt * self._uniform(list(self.alpha_dist.values()))
        self.beta = self.init_time + self.dt * self._uniform(list(self.beta_dist.values()))

    def feasible_action(self):
        status, progress, remain_slots = self.state
        if status == 1.0:
            if progress == 0.0:
                if remain_slots > self.DEMAND:
                    action = np.array([0.0, 1.0], dtype=np.float32)
                else:
                    action = np.array([1.0, 1.0], dtype=np.float32)
            elif progress == self.DEMAND:
                action = np.array([0.0, 0.0], dtype=np.float32)
            else:
                action = np.array([1.0, 1.0], dtype=np.float32)
        else:
            action = np.array([0.0, 0.0], dtype=np.float32)

        return action


class ClothDryer(object):
    def __init__(self, init_time, dt, rnd=np.random.RandomState()):

        self.state = [0.0, 0.0, 0.0] # status, progress, attribute
        self.action = 0.0

        self.MIN_POWER = 0.0 #kW
        self.MAX_POWER = 1.2 #kW
        self.DEMAND = 4 # time slots

        self.init_time = init_time
        self.dt = dt
        self._rnd = rnd
        self._uniform = lambda x: int(np.round(self._rnd.uniform(x[0],x[1])))

        self.alpha_dist = dict(
            {"left": (pd.datetime(2016,1,1,18,00) - self.init_time).seconds // self.dt.seconds,
             "right": (pd.datetime(2016,1,1,19,00) - self.init_time).seconds // self.dt.seconds,
             })
        self.beta_dist = dict(
            {"left": (pd.datetime(2016,1,1,23,00) - self.init_time).seconds // self.dt.seconds,
             "right": (pd.datetime(2016,1,2,0,00) - self.init_time).seconds // self.dt.seconds,
             })

    def step(self, action, time):
        assert action in [0, 1]
        status, progress, remain_slots = self.state
        if time >= self.alpha and time <= self.beta:
            status = 1.0
            progress += action
            remain_slots = (self.beta - time) / self.dt
        else:
            status = 0.0
            progress = 0.0
            remain_slots = 0.0

        self.state = [status, progress, remain_slots]

    def reset(self, rnd):
        self.state = [0.0, 0.0, 0.0]
        self.action = 0.0
        self._rnd = rnd
        self._uniform = lambda x: int(np.round(self._rnd.uniform(x[0],x[1])))

        self.alpha = self.init_time + self.dt * self._uniform(list(self.alpha_dist.values()))
        self.beta = self.init_time + self.dt * self._uniform(list(self.beta_dist.values()))

    def feasible_action(self):
        status, progress, remain_slots = self.state
        if status == 1.0:
            if progress == 0.0:
                if remain_slots > self.DEMAND:
                    action = np.array([0.0, 1.0], dtype=np.float32)
                else:
                    action = np.array([1.0, 1.0], dtype=np.float32)
            elif progress == self.DEMAND:
                action = np.array([0.0, 0.0], dtype=np.float32)
            else:
                action = np.array([1.0, 1.0], dtype=np.float32)
        else:
            action = np.array([0.0, 0.0], dtype=np.float32)

        return action

""" Critical Appliances """
class Refrigerator(object):
    def __init__(self, init_time, dt, rnd=np.random.RandomState()):

        self.state = [0.0, 0.0, 0.0] # status, progress, attribute

        self.MIN_POWER = 0.0 #kW
        self.MAX_POWER = 0.2 #kW

        self.init_time = init_time
        self.dt = dt
        self._rnd = rnd
        self._truncnorm = lambda x: int(np.round(np.clip(self._rnd.normal(x[0],x[1]),x[2],x[3])))

        self.alpha_dist = dict(
            {"mean": (pd.datetime(2016,1,1,8,0) - self.init_time).seconds // self.dt.seconds,
             "std": pd.Timedelta(minutes=0) // self.dt,
             "low": (pd.datetime(2016,1,1,8,0) - self.init_time).seconds // self.dt.seconds,
             "high": (pd.datetime(2016,1,1,8,0) - self.init_time).seconds // self.dt.seconds,
             })
        self.beta_dist = dict(
            {"mean": (pd.datetime(2016,1,2,7,50) - self.init_time).seconds // self.dt.seconds,
             "std": pd.Timedelta(minutes=0) // self.dt,
             "low": (pd.datetime(2016,1,2,7,50) - self.init_time).seconds // self.dt.seconds,
             "high": (pd.datetime(2016,1,2,7,50) - self.init_time).seconds // self.dt.seconds,
             })

    def step(self, time):
        status, progress, start_time = self.state
        status = 1.0
        progress = (time - self.alpha) / self.dt
        start_time = 0.0

        self.state = [status, progress, start_time]

    def reset(self, rnd):
        self.state = [0.0, 0.0, 0.0]
        self.action = 0.0
        self._rnd = rnd
        self._truncnorm = lambda x: int(np.round(np.clip(self._rnd.normal(x[0],x[1]),x[2],x[3])))

        self.alpha = self.init_time + self.dt * self._truncnorm(list(self.alpha_dist.values()))
        self.beta = self.init_time + self.dt * self._truncnorm(list(self.beta_dist.values()))


class TV(object):
    def __init__(self, init_time, dt, rnd=np.random.RandomState()):

        self.state = [0.0, 0.0, 0.0] # status, progress, attribute

        self.MIN_POWER = 0.0 #kW
        self.MAX_POWER = 0.1 #kW

        self.init_time = init_time
        self.dt = dt
        self._rnd = rnd
        self._uniform = lambda x: int(np.round(self._rnd.uniform(x[0],x[1])))

        self.alpha_dist = dict(
            {"left": (pd.datetime(2016,1,1,18,00) - self.init_time).seconds // self.dt.seconds,
             "right": (pd.datetime(2016,1,1,19,00) - self.init_time).seconds // self.dt.seconds,
             })
        self.beta_dist = dict(
            {"left": (pd.datetime(2016,1,1,22,00) - self.init_time).seconds // self.dt.seconds,
             "right": (pd.datetime(2016,1,1,23,00) - self.init_time).seconds // self.dt.seconds,
             })

    def step(self, time):
        status, progress, start_time = self.state
        if time >= self.alpha and time <= self.beta:
            status = 1.0
            progress = (time - self.alpha) / self.dt
            start_time = 0.0
        else:
            status = 0.0
            progress = 0.0
            start_time = 0.0

        self.state = [status, progress, start_time]

    def reset(self, rnd):
        self.state = [0.0, 0.0, 0.0]
        self.action = 0.0
        self._rnd = rnd
        self._uniform = lambda x: int(np.round(self._rnd.uniform(x[0],x[1])))

        self.alpha = self.init_time + self.dt * self._uniform(list(self.alpha_dist.values()))
        self.beta = self.init_time + self.dt * self._uniform(list(self.beta_dist.values()))

class Lights(object):
    def __init__(self, init_time, dt, rnd=np.random.RandomState()):

        self.state = [0.0, 0.0, 0.0] # status, progress, attribute

        self.MIN_POWER = 0.0 #kW
        self.MAX_POWER = 0.2 #kW

        self.init_time = init_time
        self.dt = dt
        self._rnd = rnd
        self._uniform = lambda x: int(np.round(self._rnd.uniform(x[0],x[1])))

        self.alpha_dist = dict(
            {"left": (pd.datetime(2016,1,1,17,00) - self.init_time).seconds // self.dt.seconds,
             "right": (pd.datetime(2016,1,1,18,00) - self.init_time).seconds // self.dt.seconds,
             })
        self.beta_dist = dict(
            {"left": (pd.datetime(2016,1,1,22,00) - self.init_time).seconds // self.dt.seconds,
             "right": (pd.datetime(2016,1,1,23,00) - self.init_time).seconds // self.dt.seconds,
             })

    def step(self, time):
        status, progress, start_time = self.state
        if time >= self.alpha and time <= self.beta:
            status = 1.0
            progress = (time - self.alpha) / self.dt
            start_time = 0.0
        else:
            status = 0.0
            progress = 0.0
            start_time = 0.0

        self.state = [status, progress, start_time]

    def reset(self, rnd):
        self.state = [0.0, 0.0, 0.0]
        self.action = 0.0
        self._rnd = rnd
        self._uniform = lambda x: int(np.round(self._rnd.uniform(x[0],x[1])))

        self.alpha = self.init_time + self.dt * self._uniform(list(self.alpha_dist.values()))
        self.beta = self.init_time + self.dt * self._uniform(list(self.beta_dist.values()))

class EV(object):
    def __init__(self, init_time, dt, rnd=np.random.RandomState()):
        self.state = [0.0, 0.0, 0.0] # status, progress, attribute
        self.action = 0.0

        self.MIN_POWER = -6.0
        self.MAX_POWER = 6.0
        self.MIN_SOC = 0.1
        self.MAX_SOC = 1.0
        self.CAPCITY = 24.0

        self.init_time = init_time
        self.dt = dt
        self._rnd = rnd
        self._uniform = lambda x: int(np.round(self._rnd.uniform(x[0],x[1])))

        self.alpha_dist = dict(
            {"left": (pd.datetime(2016,1,1,17,00) - self.init_time).seconds // self.dt.seconds,
             "right": (pd.datetime(2016,1,1,19,00) - self.init_time).seconds // self.dt.seconds,
             })
        self.beta_dist = dict(
            {"left": (pd.datetime(2016,1,2,7,00) - self.init_time).seconds // self.dt.seconds,
             "right": (pd.datetime(2016,1,2,7,45) - self.init_time).seconds // self.dt.seconds,
             })

    def step(self, action, time):
        status, soc, slots = self.state
        if time == self.alpha:
            soc = self.init_soc

        if time >= self.alpha and time <= self.beta:
            status = 1.0
            soc += (self.dt.seconds/3600) * action/self.CAPCITY
            slots = (time - self.init_time) / self.dt
        else:
            status = 0.0
            soc = 0.0
            slots = 0.0
    
        self.state = [status, soc, slots]

    def feasible_action(self):
        if self.state[0] == 0:
            action = [0.0, 0.0]
        else:
            action = [self.MIN_POWER, self.MAX_POWER]

        return action

    def reset(self, rnd):
        self.state = [0.0, 0.0, 0.0]
        self.action = 0.0
        self._rnd = rnd
        self._uniform = lambda x: int(np.round(self._rnd.uniform(x[0],x[1])))

        self.alpha = self.init_time + self.dt * self._uniform(list(self.alpha_dist.values()))
        self.beta = self.init_time + self.dt * self._uniform(list(self.beta_dist.values()))
        self.init_soc = np.clip(self._rnd.normal(0.5,0.1),0.4,0.6)

class SmartHome_v2(gym.Env):
    """A smart home environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, seed=None, train=True):

        super(SmartHome_v2, self).__init__()
        self.seed(seed)

        self.t = 0
        self.T = 96
        self.dt = pd.Timedelta(minutes=15)
        self.init_time = pd.datetime(2016,1,1,8,0)
        self.one_day = pd.Timedelta(days=1)

        # Deferrable appliances
        self.st = Stove(self.init_time, self.dt, rnd=self.rnd)
        self.dw = Dishwasher(self.init_time, self.dt, rnd=self.rnd)
        self.wm = WashingMachine(self.init_time, self.dt, rnd=self.rnd)
        self.cd = ClothDryer(self.init_time, self.dt, rnd=self.rnd)
        # Regulatable appliances
        self.ev = EV(self.init_time, self.dt, rnd=self.rnd)
        # Critical appliances
        self.fg = Refrigerator(self.init_time, self.dt, rnd=self.rnd)
        self.tv = TV(self.init_time, self.dt, rnd=self.rnd)
        self.lg = Lights(self.init_time, self.dt, rnd=self.rnd)

        self._load_data(train)
        self._build_monitor()

        self.reward_range = (-1000.0, 0.0)

        self.action_space = gym.spaces.Tuple([
            gym.spaces.MultiBinary(4),
            gym.spaces.Box(low=np.array([self.ev.MIN_POWER]), 
                           high=np.array([self.ev.MAX_POWER]), 
                           dtype=np.float32),
        ])

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(106,), dtype=np.float32)

    def seed(self, seed=None):
        self.rnd = np.random.RandomState(seed)
        return [seed]

    def step(self, action):
        action = action.ravel()

        # Step
        self.st.step(action[0], self.time)
        self.dw.step(action[1], self.time)
        self.wm.step(action[2], self.time)
        self.cd.step(action[3], self.time)
        self.ev.step(action[4], self.time)
        self.fg.step(self.time)
        self.tv.step(self.time)
        self.lg.step(self.time)
        self._update_monitor(action)

        # Get observation
        ob = self._get_obs()

        # Calculate the reward
        reward = self._get_reward(action)

        # Update time
        self.t += 1
        self.time += self.dt

        # Check done
        if self.time == self.init_time + self.one_day:
            done = True
        else:
            done = False

        return ob, reward, done, {}

    def reset(self, seed=None, day=None):
        # render the env every 50 minutes
        if time.time() - self._render_time > 50 * 60:
            self.render()
            self._render_time = time.time()

        if seed is not None:
            self.seed(seed)

        if day is None:
            self.t = self.rnd.randint(2,364) * self.T
        else:
            self.t = day * self.T
        self.init_time = pd.datetime(2016,1,1,8,0)
        self.time = self.init_time

        # reset the appliances
        self.st.reset(self.rnd)
        self.dw.reset(self.rnd)
        self.wm.reset(self.rnd)
        self.cd.reset(self.rnd)
        self.ev.reset(self.rnd)
        self.fg.reset(self.rnd)
        self.tv.reset(self.rnd)
        self.lg.reset(self.rnd)

        return self._get_obs()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        x = range(self.T)
        xticks = range(0, self.T, 8)
        xticklabels = [str(i) for i in range(8,24,2)] + [str(i) for i in range(0,9,2)]

        n_subfigs = 7
        plt.close()
        plt.figure(figsize=(10,12))

        ax1 = plt.subplot(n_subfigs,1,1)
        ax1.step(x, self.monitor["Price"], label='price')
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(labels=xticklabels)
        ax1.set_xlim(0,self.T)
        ax1.legend()

        ax2 = plt.subplot(n_subfigs,1,2)
        ax2.step(x, self.monitor["P_st"], label='stove')
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(labels=xticklabels)
        ax2.axvline((self.st.alpha-self.init_time).seconds//self.dt.seconds, c='r')
        ax2.axvline((self.st.beta-self.init_time).seconds//self.dt.seconds, c='r')
        ax2.set_xlim(0,self.T)
        ax2.legend()

        ax3 = plt.subplot(n_subfigs,1,3)
        ax3.step(x, self.monitor["P_dw"], label='dishwasher')
        ax3.set_xticks(xticks)
        ax3.set_xticklabels(labels=xticklabels)
        ax3.axvline((self.dw.alpha-self.init_time).seconds//self.dt.seconds, c='r')
        ax3.axvline((self.dw.beta-self.init_time).seconds//self.dt.seconds, c='r')
        ax3.set_xlim(0,self.T)
        ax3.legend()

        ax4 = plt.subplot(n_subfigs,1,4)
        ax4.step(x, self.monitor["P_wm"], label='washing machine')
        ax4.set_xticks(xticks)
        ax4.set_xticklabels(labels=xticklabels)
        ax4.axvline((self.wm.alpha-self.init_time).seconds//self.dt.seconds, c='r')
        ax4.axvline((self.wm.beta-self.init_time).seconds//self.dt.seconds, c='r')
        ax4.set_xlim(0,self.T)
        ax4.legend()

        ax5 = plt.subplot(n_subfigs,1,5)
        ax5.step(x, self.monitor["P_cd"], label='clothes dryer')
        ax5.set_xticks(xticks)
        ax5.set_xticklabels(labels=xticklabels)
        ax5.axvline((self.cd.alpha-self.init_time).seconds//self.dt.seconds, c='r')
        ax5.axvline((self.cd.beta-self.init_time).seconds//self.dt.seconds, c='r')
        ax5.set_xlim(0,self.T)
        ax5.legend()

        ax6 = plt.subplot(n_subfigs,1,6)
        ax6.step(x, self.monitor["P_fg"], label='P_fg')
        ax6.step(x, self.monitor["P_tv"], label='P_tv')
        ax6.step(x, self.monitor["P_lg"], label='P_lg')
        ax6.set_xticks(xticks)
        ax6.set_xticklabels(labels=xticklabels)
        ax6.legend(ncol=6)
        ax6.set_xlim(0,self.T)

        ax7 = plt.subplot(n_subfigs,1,7)
        ax7.step(x, self.monitor["P_ev"], label='P_ev')
        ax7.set_xticks(xticks)
        ax7.set_xticklabels(labels=xticklabels)
        ax7.legend(ncol=6)
        ax7.set_xlim(0,self.T)
        ax71 = ax7.twinx()
        ax71.step(x, self.monitor["SoC"], color='r')

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.5)

    def _load_data(self, train):
        data_dir = '/home/lihepeng/Documents/Github/tmp/dr_isgt'
        if train:
            print('train')
            self.price = np.loadtxt(os.path.join(data_dir, 'price_train.txt'))
        else:
            print('test')
            self.price = np.loadtxt(os.path.join(data_dir, 'price_test.txt'))

    def _get_reward(self, action):
        # electricity cost
        power = action[0]*self.st.MAX_POWER + \
                action[1]*self.dw.MAX_POWER + \
                action[2]*self.wm.MAX_POWER + \
                action[3]*self.cd.MAX_POWER + \
                action[4] + \
                self.fg.state[0]*self.fg.MAX_POWER + \
                self.tv.state[0]*self.tv.MAX_POWER + \
                self.lg.state[0]*self.tv.MAX_POWER

        if power <= 6.5: # kW
            C_elec = power * self.price[self.t] * (self.dt.seconds//60/60)
        else:
            C_elec = 1.4423 * power * self.price[self.t] * (self.dt.seconds//60/60)

        # EV anxiety
        w1 = 0.02
        E_range, soc = 0, self.ev.state[1]
        if self.time == self.ev.beta:
            E_range += ((soc - 1.0) * self.ev.CAPCITY) ** 2
        elif self.time >= self.ev.alpha and self.time < self.ev.beta:
            E_range += (max(0, self.ev.MIN_SOC-soc) * self.ev.CAPCITY) ** 2 + \
                       (max(0, soc-self.ev.MAX_SOC) * self.ev.CAPCITY) ** 2
        E_range = w1 * E_range

        reward = - C_elec - E_range

        self.power = power

        return reward

    def _get_obs(self):
        self.ob = np.hstack([
            self.st.state[1:],
            self.dw.state[1:],
            self.wm.state[1:],
            self.cd.state[1:],
            self.ev.state[1:],
            self.price[self.t-self.T:self.t]*10,
        ]).astype(np.float32)

        return self.ob.ravel()

    def _feasible_action(self):
        fea_acts = np.array([
            self.st.feasible_action(),
            self.dw.feasible_action(),
            self.wm.feasible_action(),
            self.cd.feasible_action(),
            self.ev.feasible_action(),
        ], dtype=np.float32)

        return fea_acts[:,0], fea_acts[:,1]

    def _build_monitor(self):
        self._render_time = time.time()
        keys = ["Price","SoC",
                "P_st","P_dw","P_wm","P_cd","P_ev",
                "P_fg","P_tv","P_lg",
                "A_st","A_dw","A_wm","A_cd","A_ev",
                "B_st","B_dw","B_wm","B_cd","B_ev",
        ]
        values=[deque(maxlen=self.T) for _ in range(len(keys))]
        self.monitor = dict(zip(keys, values))

    def _update_monitor(self, action):
        self.monitor["Price"].append(self.price[self.t])
        self.monitor["SoC"].append(self.ev.state[1])
        self.monitor["P_st"].append(action[0]*self.st.MAX_POWER)
        self.monitor["P_dw"].append(action[1]*self.dw.MAX_POWER)
        self.monitor["P_wm"].append(action[2]*self.wm.MAX_POWER)
        self.monitor["P_cd"].append(action[3]*self.cd.MAX_POWER)
        self.monitor["P_ev"].append(action[4])
        self.monitor["P_fg"].append(self.fg.state[0]*self.fg.MAX_POWER)
        self.monitor["P_tv"].append(self.tv.state[0]*self.tv.MAX_POWER)
        self.monitor["P_lg"].append(self.lg.state[0]*self.lg.MAX_POWER)
        self.monitor["A_st"].append((self.st.alpha-self.init_time).seconds//self.dt.seconds)
        self.monitor["B_st"].append((self.st.beta-self.init_time).seconds//self.dt.seconds)
        self.monitor["A_dw"].append((self.dw.alpha-self.init_time).seconds//self.dt.seconds)
        self.monitor["B_dw"].append((self.dw.beta-self.init_time).seconds//self.dt.seconds)
        self.monitor["A_wm"].append((self.wm.alpha-self.init_time).seconds//self.dt.seconds)
        self.monitor["B_wm"].append((self.wm.beta-self.init_time).seconds//self.dt.seconds)
        self.monitor["A_cd"].append((self.cd.alpha-self.init_time).seconds//self.dt.seconds)
        self.monitor["B_cd"].append((self.cd.beta-self.init_time).seconds//self.dt.seconds)
        self.monitor["A_ev"].append((self.ev.alpha-self.init_time).seconds//self.dt.seconds)
        self.monitor["B_ev"].append((self.ev.beta-self.init_time).seconds//self.dt.seconds)