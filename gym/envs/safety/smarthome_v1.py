import os, time, pickle
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import gym
from collections import deque

""" Deferrable Appliances """
class Oven(object):
    def __init__(self, init_time, dt, rnd=np.random.RandomState()):

        self.state = [0.0, 0.0, 0.0] # status, progress, attribute
        self.action = 0.0

        self.MIN_POWER = 0.0 #kW
        self.MAX_POWER = 2.1 #kW
        self.DEMAND = 4 # time slots

        self.init_time = init_time
        self.dt = dt
        self._rnd = rnd
        self._truncnorm = lambda x: int(np.round(np.clip(self._rnd.normal(x[0],x[1]),x[2],x[3])))

        self.alpha_dist = dict(
            {"mean": (pd.datetime(2016,1,1,14,00) - self.init_time) // self.dt,
             "std": pd.Timedelta(minutes=30) // self.dt,
             "low": (pd.datetime(2016,1,1,13,30) - self.init_time) // self.dt,
             "high": (pd.datetime(2016,1,1,14,30) - self.init_time) // self.dt,
             })
        self.beta_dist = dict(
            {"mean": (pd.datetime(2016,1,1,17,00) - self.init_time) // self.dt,
             "std": pd.Timedelta(minutes=20) // self.dt,
             "low": (pd.datetime(2016,1,1,16,40) - self.init_time) // self.dt,
             "high": (pd.datetime(2016,1,1,17,20) - self.init_time) // self.dt,
             })

    def step(self, action, time):
        assert action in [0, 1]
        status, progress, remain_slots = self.state
        if time >= self.alpha and time < self.beta:
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
        self._truncnorm = lambda x: int(np.round(np.clip(self._rnd.normal(x[0],x[1]),x[2],x[3])))
        self.alpha = self.init_time + self.dt * self._truncnorm(list(self.alpha_dist.values()))
        self.beta = self.init_time + self.dt * self._truncnorm(list(self.beta_dist.values()))

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
        self.DEMAND = 3 # time slots

        self.init_time = init_time
        self.dt = dt
        self._rnd = rnd
        self._truncnorm = lambda x: int(np.round(np.clip(self._rnd.normal(x[0],x[1]),x[2],x[3])))

        self.alpha_dist = dict(
            {"mean": (pd.datetime(2016,1,1,18,30) - self.init_time) // self.dt,
             "std": pd.Timedelta(minutes=30) // self.dt,
             "low": (pd.datetime(2016,1,1,18,0) - self.init_time) // self.dt,
             "high": (pd.datetime(2016,1,1,19,0) - self.init_time) // self.dt,
             })
        self.beta_dist = dict(
            {"mean": (pd.datetime(2016,1,1,23,00) - self.init_time) // self.dt,
             "std": pd.Timedelta(minutes=30) // self.dt,
             "low": (pd.datetime(2016,1,1,22,30) - self.init_time) // self.dt,
             "high": (pd.datetime(2016,1,1,23,30) - self.init_time) // self.dt,
             })

    def step(self, action, time):
        assert action in [0, 1]
        status, progress, remain_slots = self.state
        if time >= self.alpha and time < self.beta:
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
        self._truncnorm = lambda x: int(np.round(np.clip(self._rnd.normal(x[0],x[1]),x[2],x[3])))
        self.alpha = self.init_time + self.dt * self._truncnorm(list(self.alpha_dist.values()))
        self.beta = self.init_time + self.dt * self._truncnorm(list(self.beta_dist.values()))

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
        self.MAX_POWER = 0.7 #kW
        self.DEMAND = 6 # time slots

        self.init_time = init_time
        self.dt = dt
        self._rnd = rnd
        self._truncnorm = lambda x: int(np.round(np.clip(self._rnd.normal(x[0],x[1]),x[2],x[3])))

        self.alpha_dist = dict(
            {"mean": (pd.datetime(2016,1,1,10,00) - self.init_time) // self.dt,
             "std": pd.Timedelta(minutes=60) // self.dt,
             "low": (pd.datetime(2016,1,1,9,30) - self.init_time) // self.dt,
             "high": (pd.datetime(2016,1,1,10,30) - self.init_time) // self.dt,
             })
        self.beta_dist = dict(
            {"mean": (pd.datetime(2016,1,1,20,00) - self.init_time) // self.dt,
             "std": pd.Timedelta(minutes=60) // self.dt,
             "low": (pd.datetime(2016,1,1,19,00) - self.init_time) // self.dt,
             "high": (pd.datetime(2016,1,1,21,00) - self.init_time) // self.dt,
             })

    def step(self, action, time):
        assert action in [0, 1]
        status, progress, remain_slots = self.state
        if time >= self.alpha and time < self.beta:
            status = 1.0
            progress += action
            remain_slots = (self.beta - time) / self.dt
        elif time == self.beta:
            status = 1.0
            progress += action
            remain_slots = 0.0
        else:
            status = 0.0
            progress = 0.0
            remain_slots = 0.0

        self.state = [status, progress, remain_slots]

    def reset(self, rnd):
        self.state = [0.0, 0.0, 0.0]
        self.action = 0.0

        self._rnd = rnd
        self._truncnorm = lambda x: int(np.round(np.clip(self._rnd.normal(x[0],x[1]),x[2],x[3])))
        self.alpha = self.init_time + self.dt * self._truncnorm(list(self.alpha_dist.values()))
        self.beta = self.init_time + self.dt * self._truncnorm(list(self.beta_dist.values()))

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
        self.DEMAND = 5 # time slots

        self.init_time = init_time
        self.dt = dt
        self._rnd = rnd
        self._truncnorm = lambda x: int(np.round(np.clip(self._rnd.normal(x[0],x[1]),x[2],x[3])))

        self.beta_dist = dict(
            {"mean": (pd.datetime(2016,1,2,7,30) - self.init_time) // self.dt,
             "std": pd.Timedelta(minutes=30) // self.dt,
             "low": (pd.datetime(2016,1,2,7,00) - self.init_time) // self.dt,
             "high": (pd.datetime(2016,1,2,8,00) - self.init_time) // self.dt,
             })

    def step(self, action, time, wm):
        assert action in [0, 1]
        status, progress, remain_slots = self.state

        # check if washer machine finishes it task
        if not hasattr(self, "alpha") and wm.state[1] == wm.DEMAND:
            self.alpha = time

        if hasattr(self, "alpha") and time < self.beta:
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
        self._truncnorm = lambda x: int(np.round(np.clip(self._rnd.normal(x[0],x[1]),x[2],x[3])))

        if hasattr(self, "alpha"):
            delattr(self, "alpha")
        self.beta = self.init_time + self.dt * self._truncnorm(list(self.beta_dist.values()))

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
            {"mean": (pd.datetime(2016,1,1,8,0) - self.init_time) // self.dt,
             "std": pd.Timedelta(minutes=0) // self.dt,
             "low": (pd.datetime(2016,1,1,8,0) - self.init_time) // self.dt,
             "high": (pd.datetime(2016,1,1,8,0) - self.init_time) // self.dt,
             })
        self.beta_dist = dict(
            {"mean": (pd.datetime(2016,1,2,8,0) - self.init_time) // self.dt,
             "std": pd.Timedelta(minutes=0) // self.dt,
             "low": (pd.datetime(2016,1,2,8,0) - self.init_time) // self.dt,
             "high": (pd.datetime(2016,1,2,8,0) - self.init_time) // self.dt,
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

class Vaccum(object):
    def __init__(self, init_time, dt, rnd=np.random.RandomState()):

        self.state = [0.0, 0.0, 0.0] # status, progress, attribute

        self.MIN_POWER = 0.0 #kW
        self.MAX_POWER = 1.5 #kW

        self.init_time = init_time
        self.dt = dt
        self._rnd = rnd
        self._truncnorm = lambda x: int(np.round(np.clip(self._rnd.normal(x[0],x[1]),x[2],x[3])))

        self.alpha_dist = dict(
            {"mean": (pd.datetime(2016,1,1,15,0) - self.init_time) // self.dt,
             "std": pd.Timedelta(minutes=60) // self.dt,
             "low": (pd.datetime(2016,1,1,14,0) - self.init_time) // self.dt,
             "high": (pd.datetime(2016,1,1,16,0) - self.init_time) // self.dt,
             })
        self.delta_dist = dict(
            {"mean": pd.Timedelta(minutes=40) // self.dt,
             "std": pd.Timedelta(minutes=20) // self.dt,
             "low": pd.Timedelta(minutes=20) // self.dt,
             "high": pd.Timedelta(minutes=60) // self.dt,
             })

    def step(self, time):
        status, progress, start_time = self.state
        if time >= self.alpha and time < self.beta:
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
        self._truncnorm = lambda x: int(np.round(np.clip(self._rnd.normal(x[0],x[1]),x[2],x[3])))
        self.alpha = self.init_time + self.dt * self._truncnorm(list(self.alpha_dist.values()))
        self.delta = self._truncnorm(list(self.delta_dist.values()))
        self.beta = self.alpha + self.dt * self.delta

class HairDryer(object):
    def __init__(self, init_time, dt, rnd=np.random.RandomState()):

        self.state = [0.0, 0.0, 0.0] # status, progress, attribute

        self.MIN_POWER = 0.0 #kW
        self.MAX_POWER = 1.0 #kW

        self.init_time = init_time
        self.dt = dt
        self._rnd = rnd
        self._truncnorm = lambda x: int(np.round(np.clip(self._rnd.normal(x[0],x[1]),x[2],x[3])))

        self.alpha_dist = dict(
            {"mean": (pd.datetime(2016,1,1,20,30) - self.init_time) // self.dt,
             "std": pd.Timedelta(minutes=30) // self.dt,
             "low": (pd.datetime(2016,1,1,20,0) - self.init_time) // self.dt,
             "high": (pd.datetime(2016,1,1,21,0) - self.init_time) // self.dt,
             })
        self.delta_dist = dict(
            {"mean": pd.Timedelta(minutes=10) // self.dt,
             "std": pd.Timedelta(minutes=0) // self.dt,
             "low": pd.Timedelta(minutes=10) // self.dt,
             "high": pd.Timedelta(minutes=10) // self.dt,
             })

    def step(self, time):
        status, progress, start_time = self.state
        if time >= self.alpha and time < self.beta:
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
        self._truncnorm = lambda x: int(np.round(np.clip(self._rnd.normal(x[0],x[1]),x[2],x[3])))
        self.alpha = self.init_time + self.dt * self._truncnorm(list(self.alpha_dist.values()))
        self.delta = self._truncnorm(list(self.delta_dist.values()))
        self.beta = self.alpha + self.dt * self.delta

class TV(object):
    def __init__(self, init_time, dt, rnd=np.random.RandomState()):

        self.state = [0.0, 0.0, 0.0] # status, progress, attribute

        self.MIN_POWER = 0.0 #kW
        self.MAX_POWER = 0.1 #kW

        self.init_time = init_time
        self.dt = dt
        self._rnd = rnd
        self._truncnorm = lambda x: int(np.round(np.clip(self._rnd.normal(x[0],x[1]),x[2],x[3])))

        self.alpha_dist = dict(
            {"mean": (pd.datetime(2016,1,1,18,30) - self.init_time) // self.dt,
             "std": pd.Timedelta(minutes=30) // self.dt,
             "low": (pd.datetime(2016,1,1,18,0) - self.init_time) // self.dt,
             "high": (pd.datetime(2016,1,1,19,0) - self.init_time) // self.dt,
             })
        self.delta_dist = dict(
            {"mean": pd.Timedelta(hours=4) // self.dt,
             "std": pd.Timedelta(minutes=40) // self.dt,
             "low": pd.Timedelta(hours=3, minutes=20) // self.dt,
             "high": pd.Timedelta(hours=4, minutes=40) // self.dt,
             })

    def step(self, time):
        status, progress, start_time = self.state
        if time >= self.alpha and time < self.beta:
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
        self._truncnorm = lambda x: int(np.round(np.clip(self._rnd.normal(x[0],x[1]),x[2],x[3])))
        self.alpha = self.init_time + self.dt * self._truncnorm(list(self.alpha_dist.values()))
        self.delta = self._truncnorm(list(self.delta_dist.values()))
        self.beta = self.alpha + self.dt * self.delta

class Laptop(object):
    def __init__(self, init_time, dt, rnd=np.random.RandomState()):

        self.state = [0.0, 0.0, 0.0] # status, progress, attribute

        self.MIN_POWER = 0.0 #kW
        self.MAX_POWER = 0.1 #kW

        self.init_time = init_time
        self.dt = dt
        self._rnd = rnd
        self._truncnorm = lambda x: int(np.round(np.clip(self._rnd.normal(x[0],x[1]),x[2],x[3])))

        self.alpha_dist = dict(
            {"mean": (pd.datetime(2016,1,1,20,0) - self.init_time) // self.dt,
             "std": pd.Timedelta(minutes=30) // self.dt,
             "low": (pd.datetime(2016,1,1,19,30) - self.init_time) // self.dt,
             "high": (pd.datetime(2016,1,1,20,30) - self.init_time) // self.dt,
             })
        self.delta_dist = dict(
            {"mean": pd.Timedelta(hours=3) // self.dt,
             "std": pd.Timedelta(minutes=40) // self.dt,
             "low": pd.Timedelta(hours=2, minutes=20) // self.dt,
             "high": pd.Timedelta(hours=3, minutes=40) // self.dt,
             })

    def step(self, time):
        status, progress, start_time = self.state
        if time >= self.alpha and time < self.beta:
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
        self._truncnorm = lambda x: int(np.round(np.clip(self._rnd.normal(x[0],x[1]),x[2],x[3])))
        self.alpha = self.init_time + self.dt * self._truncnorm(list(self.alpha_dist.values()))
        self.delta = self._truncnorm(list(self.delta_dist.values()))
        self.beta = self.alpha + self.dt * self.delta

class Lights(object):
    def __init__(self, init_time, dt, rnd=np.random.RandomState()):

        self.state = [0.0, 0.0, 0.0] # status, progress, attribute

        self.MIN_POWER = 0.0 #kW
        self.MAX_POWER = 0.2 #kW

        self.init_time = init_time
        self.dt = dt
        self._rnd = rnd
        self._truncnorm = lambda x: int(np.round(np.clip(self._rnd.normal(x[0],x[1]),x[2],x[3])))

        self.alpha_dist = dict(
            {"mean": (pd.datetime(2016,1,1,17,0) - self.init_time) // self.dt,
             "std": pd.Timedelta(hours=1) // self.dt,
             "low": (pd.datetime(2016,1,1,16,0) - self.init_time) // self.dt,
             "high": (pd.datetime(2016,1,1,18,0) - self.init_time) // self.dt,
             })
        self.delta_dist = dict(
            {"mean": pd.Timedelta(hours=5) // self.dt,
             "std": pd.Timedelta(minutes=30) // self.dt,
             "low": pd.Timedelta(hours=4, minutes=30) // self.dt,
             "high": pd.Timedelta(hours=5, minutes=30) // self.dt,
             })

    def step(self, time):
        status, progress, start_time = self.state
        if time >= self.alpha and time < self.beta:
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
        self._truncnorm = lambda x: int(np.round(np.clip(self._rnd.normal(x[0],x[1]),x[2],x[3])))
        self.alpha = self.init_time + self.dt * self._truncnorm(list(self.alpha_dist.values()))
        self.delta = self._truncnorm(list(self.delta_dist.values()))
        self.beta = self.alpha + self.dt * self.delta

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
        self._truncnorm = lambda x: int(np.round(np.clip(self._rnd.normal(x[0],x[1]),x[2],x[3])))

        self.alpha_dist = dict(
            {"mean": (pd.datetime(2016,1,1,18,00) - self.init_time) // self.dt,
             "std": pd.Timedelta(hours=1) // self.dt,
             "low": (pd.datetime(2016,1,1,15,0) - self.init_time) // self.dt,
             "high": (pd.datetime(2016,1,1,21,0) - self.init_time) // self.dt,
             })
        self.beta_dist = dict(
            {"mean": (pd.datetime(2016,1,2,7,30) - self.init_time) // self.dt,
             "std": pd.Timedelta(minutes=30) // self.dt,
             "low": (pd.datetime(2016,1,2,7,0) - self.init_time) // self.dt,
             "high": (pd.datetime(2016,1,2,8,0) - self.init_time) // self.dt,
             })

    def step(self, action, time):
        status, soc, slots = self.state
        if time == self.alpha:
            soc = self.init_soc

        if time >= self.alpha and time < self.beta:
            status = 1.0
            soc += (self.dt.seconds/3600) * action/self.CAPCITY
            slots = (time - self.init_time) / self.dt
        elif time == self.beta:
            status = 0.0
            soc += (self.dt.seconds/3600) * action/self.CAPCITY
            slots = 0.0
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
        self._truncnorm = lambda x: int(np.round(np.clip(self._rnd.normal(x[0],x[1]),x[2],x[3])))
        self.alpha = self.init_time + self.dt * self._truncnorm(list(self.alpha_dist.values()))
        self.beta = self.init_time + self.dt * self._truncnorm(list(self.beta_dist.values()))
        self.init_soc = np.clip(self._rnd.normal(0.5,0.1),0.2,0.8)

class HVAC(object):
    def __init__(self, init_time, dt, rnd=np.random.RandomState()):
        self.state = [0.0, 0.0, 0.0] # status, progress, attribute
        self.action = 0.0

        self.MIN_POWER = 0.0
        self.MAX_POWER = 2.352
        self.SETTING = 75.0
        self.MAX_DEV = 4.0
        self.MIN_TEMP = self.SETTING - self.MAX_DEV
        self.MAX_TEMP = self.SETTING + self.MAX_DEV
        self.init_time = init_time
        self.dt = dt
        self._rnd = rnd

    def step(self, action, T_out):
        Req = 3.1965e-6 * 1.8
        Ca = 1.01 / 1.8
        Ma = 1778.369
        COP = 2
        delta_t = self.dt.seconds/3600

        T_old = self.state[1] + self.SETTING
        T_new = (1-delta_t/(1000*Ma*Ca*Req))*T_old + delta_t/(1000*Ma*Ca*Req)*T_out - \
                (action*COP*delta_t)/(0.00027*Ma*Ca)

        self.state[1] = T_new - self.SETTING
        self.state[2] = self.SETTING

        self.action = action

    def reset(self, rnd):
        self._rnd = rnd
        self.state = [1.0, self._rnd.uniform(self.MIN_TEMP, self.MAX_TEMP)-self.SETTING, self.SETTING]
        self.action = 0.0

    def feasible_action(self):
        return [self.MIN_POWER, self.MAX_POWER]

class EWH(object):
    def __init__(self, init_time, dt, rnd=np.random.RandomState()):
        self.state = [0.0, 0.0, 0.0] # status, progress, attribute
        self.action = 0.0

        self.MIN_POWER = 0.0 #kW
        self.MAX_POWER = 4.5 #kW
        self.MIN_TEMP = 120
        self.MAX_TEMP = 130
        self.SET_POINT = 125
        self.MAX_DEV = 5

        self.init_time = init_time
        self.dt = dt
        self._rnd = rnd
        self._load_data()

    def _load_data(self):
        data_dir = '/home/lihepeng/Documents/Github/tmp/dr/data'
        csv_file = pd.read_csv(os.path.join(data_dir, 'water_usage_power.csv'))
        xp = np.arange(96)
        x = np.linspace(0, 96, 144)
        water_usage_power = np.interp(x, xp,csv_file['Curve1'].values)
        self.base_flow_profile = 5.85 * water_usage_power * self.dt.seconds/3600

    def step(self, action, time):
        t = (time-self.init_time)//self.dt
        flow = self.flow_profile[t]
        Tin = 60
        T_air = 75
        d = 8.34
        Cp = 1.0069
        volume = 40
        SA = 24.1
        R = 15
        Q = 3412.1
        C = volume * d * Cp
        G = SA / R
        B = d * flow * Cp
        R1 = 1/(G + B)
        coff = np.exp(-(self.dt.seconds/3600)/(R1*C))

        T_old = self.state[1] + self.SET_POINT
        T_new = coff*T_old+(1-coff)*(G*R1*T_air+B*R1*Tin+action*(self.dt.seconds/3600)*Q*R1)

        self.state[1] = T_new - self.SET_POINT
        self.state[2] = self.SET_POINT
        self.flow = flow

    def reset(self, rnd):
        self._rnd = rnd
        self.flow_profile = np.clip(self.base_flow_profile + rnd.normal(0,1.17,[144]) * self.dt.seconds/3600, 0, 1)

        self.state = [1.0, self._rnd.uniform(self.MIN_TEMP, self.MAX_TEMP)-self.SET_POINT, self.SET_POINT]
        self.action = 0.0

    def feasible_action(self):
        if self.state[0] == 0:
            action = [0.0, 0.0]
        else:
            action = [self.MIN_POWER, self.MAX_POWER]
        return action

class SmartHome_v1(gym.Env):
    """A smart home environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, seed=None, train=True):

        super(SmartHome_v1, self).__init__()
        self.seed(seed)

        self.t = 0
        self.T = 144
        self.dt = pd.Timedelta(minutes=10)
        self.init_time = pd.datetime(2016,1,1,8,0)
        self.one_day = pd.Timedelta(days=1)

        # Deferrable appliances
        self.ov = Oven(self.init_time, self.dt, rnd=self.rnd)
        self.dw = Dishwasher(self.init_time, self.dt, rnd=self.rnd)
        self.wm = WashingMachine(self.init_time, self.dt, rnd=self.rnd)
        self.cd = ClothDryer(self.init_time, self.dt, rnd=self.rnd)
        # Regulatable appliances
        self.ev = EV(self.init_time, self.dt, rnd=self.rnd)
        self.wh = EWH(self.init_time, self.dt, rnd=self.rnd)
        self.ac = HVAC(self.init_time, self.dt, rnd=self.rnd)
        # Critical appliances
        self.fg = Refrigerator(self.init_time, self.dt, rnd=self.rnd)
        self.vc = Vaccum(self.init_time, self.dt, rnd=self.rnd)
        self.hd = HairDryer(self.init_time, self.dt, rnd=self.rnd)
        self.tv = TV(self.init_time, self.dt, rnd=self.rnd)
        self.nb = Laptop(self.init_time, self.dt, rnd=self.rnd)
        self.lg = Lights(self.init_time, self.dt, rnd=self.rnd)

        self._load_data(train)
        self._build_monitor()

        self.reward_range = (-200.0, 200.0)

        self.action_space = gym.spaces.Tuple([
            gym.spaces.MultiBinary(4),
            gym.spaces.Box(
                low=np.array([self.ev.MIN_POWER, self.wh.MIN_POWER, self.ac.MIN_POWER]), 
                high=np.array([self.ev.MAX_POWER, self.wh.MAX_POWER, self.ac.MAX_POWER]), 
                dtype=np.float32
            ),
        ])

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(39+2*self.T,), dtype=np.float32)

    def seed(self, seed=None):
        self.rnd = np.random.RandomState(seed)
        return [seed]

    def step(self, action):
        action = action.ravel()

        # Step
        self.ov.step(action[0], self.time)
        self.dw.step(action[1], self.time)
        self.wm.step(action[2], self.time)
        self.cd.step(action[3], self.time, self.wm)
        self.ev.step(action[4], self.time)
        self.wh.step(action[5], self.time)
        self.ac.step(action[6], self.temp[self.t])
        self.fg.step(self.time)
        self.vc.step(self.time)
        self.hd.step(self.time)
        self.tv.step(self.time)
        self.nb.step(self.time)
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
        # render the env every 10 minutes
        if time.time() - self._render_time > 5 * 60:
            self.render()
            self._render_time = time.time()

        if seed is not None:
            self.seed(seed)

        if day is None:
            self.t = self.rnd.randint(1,60) * self.T
        else:
            self.t = day * self.T
        self.init_time = pd.datetime(2016,1,1,8,0)
        self.time = self.init_time

        # reset the appliances
        self.ov.reset(self.rnd)
        self.dw.reset(self.rnd)
        self.wm.reset(self.rnd)
        self.cd.reset(self.rnd)
        self.ev.reset(self.rnd)
        self.wh.reset(self.rnd)
        self.ac.reset(self.rnd)
        self.fg.reset(self.rnd)
        self.vc.reset(self.rnd)
        self.hd.reset(self.rnd)
        self.tv.reset(self.rnd)
        self.nb.reset(self.rnd)
        self.lg.reset(self.rnd)

        return self._get_obs()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        x = range(self.T)
        xticks = range(0, self.T, 12)
        xticklabels = [str(i) for i in range(8,24,2)] + [str(i) for i in range(0,9,2)]

        n_subfigs = 10
        plt.close()
        plt.figure(figsize=(10,15))

        ax1 = plt.subplot(n_subfigs,1,1)
        ax1.step(x, self.monitor["Price"])
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(labels=xticklabels)
        ax1.set_xlim(0,self.T)

        ax2 = plt.subplot(n_subfigs,1,2)
        ax2.step(x, self.monitor["P_ov"])
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(labels=xticklabels)
        ax2.axvline((self.ov.alpha-self.init_time)//self.dt, c='r')
        ax2.axvline((self.ov.beta-self.init_time)//self.dt, c='r')
        ax2.set_xlim(0,self.T)

        ax3 = plt.subplot(n_subfigs,1,3)
        ax3.step(x, self.monitor["P_dw"])
        ax3.set_xticks(xticks)
        ax3.set_xticklabels(labels=xticklabels)
        ax3.axvline((self.dw.alpha-self.init_time)//self.dt, c='r')
        ax3.axvline((self.dw.beta-self.init_time)//self.dt, c='r')
        ax3.set_xlim(0,self.T)

        ax4 = plt.subplot(n_subfigs,1,4)
        ax4.step(x, self.monitor["P_wm"])
        ax4.set_xticks(xticks)
        ax4.set_xticklabels(labels=xticklabels)
        ax4.axvline((self.wm.alpha-self.init_time)//self.dt, c='r')
        ax4.axvline((self.wm.beta-self.init_time)//self.dt, c='r')
        ax4.set_xlim(0,self.T)

        ax5 = plt.subplot(n_subfigs,1,5)
        ax5.step(x, self.monitor["P_cd"])
        ax5.set_xticks(xticks)
        ax5.set_xticklabels(labels=xticklabels)
        ax5.axvline((self.cd.alpha-self.init_time)//self.dt, c='r')
        ax5.axvline((self.cd.beta-self.init_time)//self.dt, c='r')
        ax5.set_xlim(0,self.T)

        ax6 = plt.subplot(n_subfigs,1,6)
        ax6.step(x, self.monitor["P_fg"], label='P_fg')
        ax6.step(x, self.monitor["P_vc"], label='P_vc')
        ax6.step(x, self.monitor["P_hd"], label='P_hd')
        ax6.step(x, self.monitor["P_tv"], label='P_tv')
        ax6.step(x, self.monitor["P_nb"], label='P_nb')
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

        ax8 = plt.subplot(n_subfigs,1,8)
        ax8.step(x, self.monitor["P_wh"], label='P_wh')
        ax8.set_xticks(xticks)
        ax8.set_xticklabels(labels=xticklabels)
        ax8.legend()
        ax8.set_xlim(0,self.T)
        ax81 = ax8.twinx()
        ax81.step(x, self.monitor["WtTemp"], color='r')
        ax81.axhline(self.wh.MIN_TEMP, c='c')
        ax81.axhline(self.wh.MAX_TEMP, c='c')

        ax9 = plt.subplot(n_subfigs,1,9)
        ax9.step(x, self.monitor["flow"])
        ax9.set_xticks(xticks)
        ax9.set_xticklabels(labels=xticklabels)
        ax9.set_xlim(0,self.T)
        ax91 = ax9.twinx()
        ax91.step(x, self.monitor["OdTemp"], color='r')

        ax10 = plt.subplot(n_subfigs,1,10)
        ax10.step(x, self.monitor["P_ac"], label='P_ac')
        ax10.set_xticks(xticks)
        ax10.set_xticklabels(labels=xticklabels)
        ax10.set_xlim(0,self.T)
        ax10.legend()
        ax101 = ax10.twinx()
        ax101.step(x, self.monitor["IdTemp"], color='r')
        ax101.axhline(self.ac.MIN_TEMP, c='c')
        ax101.axhline(self.ac.MAX_TEMP, c='c')

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.5)

    def _load_data(self, train):
        data_dir = '/home/lihepeng/Documents/Github/tmp/dr/data'
        if train:
            print('train')
            self.price = np.loadtxt(os.path.join(data_dir, 'price_train.txt'))
            self.temp = np.loadtxt(os.path.join(data_dir, 'temp_train.txt'))
        else:
            print('test')
            self.price = np.loadtxt(os.path.join(data_dir, 'price_test.txt'))
            self.temp = np.loadtxt(os.path.join(data_dir, 'temp_test.txt'))

    def _get_reward(self, action):
        # thermal comfort
        w1 = 0.01
        hvac_dev = min(0.0, self.ac.MAX_DEV - abs(self.ac.state[1]))
        ewh_dev = min(0.0, self.wh.MAX_DEV - abs(self.wh.state[1]))
        I_comf = w1 * (np.exp(hvac_dev) + np.exp(ewh_dev))

        # electricity cost
        power = action[0]*self.ov.MAX_POWER + \
                action[1]*self.dw.MAX_POWER + \
                action[2]*self.wm.MAX_POWER + \
                action[3]*self.cd.MAX_POWER + \
                action[4] + action[5] + action[6] + \
                self.fg.state[0]*self.fg.MAX_POWER + \
                self.vc.state[0]*self.vc.MAX_POWER + \
                self.hd.state[0]*self.hd.MAX_POWER + \
                self.tv.state[0]*self.tv.MAX_POWER + \
                self.nb.state[0]*self.nb.MAX_POWER + \
                self.lg.state[0]*self.lg.MAX_POWER
        if power <= 10: # kW
            C_elec = power * self.price[self.t] * (self.dt.seconds//60/60)
        else:
            C_elec = 1.4423 * power * self.price[self.t] * (self.dt.seconds//60/60)

        # EV anxiety
        w2 = 0.01
        E_range, soc = 0, self.ev.state[1]
        if self.time == self.ev.beta:
            E_range += 10 * ((soc - 1.0) * self.ev.CAPCITY) ** 2
        elif self.time >= self.ev.alpha and self.time < self.ev.beta:
            E_range += (max(0, self.ev.MIN_SOC-soc) * self.ev.CAPCITY) ** 2 + \
                       (max(0, soc-self.ev.MAX_SOC) * self.ev.CAPCITY) ** 2
        E_range = w2 * E_range

        reward = I_comf - C_elec - E_range
        # print(I_comf, C_elec, E_range)

        return reward

    def _get_obs(self):
        self.ob = np.hstack([
            self.ov.state,
            self.dw.state,
            self.wm.state,
            self.cd.state,
            self.ev.state,
            self.wh.state,
            self.ac.state,
            self.fg.state,
            self.vc.state,
            self.hd.state,
            self.tv.state,
            self.nb.state,
            self.lg.state,
            self.price[self.t-self.T:self.t],
            self.temp[self.t-self.T:self.t],
        ]).astype(np.float32)

        return self.ob.ravel()

    def _feasible_action(self):
        fea_acts = np.array([
            self.ov.feasible_action(),
            self.dw.feasible_action(),
            self.wm.feasible_action(),
            self.cd.feasible_action(),
            self.ev.feasible_action(),
            self.wh.feasible_action(),
            self.ac.feasible_action(),
        ], dtype=np.float32)

        return fea_acts[:,0], fea_acts[:,1]

    def _build_monitor(self):
        self._render_time = time.time()
        keys = ["Price","SoC","WtTemp","flow","IdTemp","OdTemp",
                "P_ov","P_dw","P_wm","P_cd",
                "P_ev","P_wh","P_ac",
                "P_fg","P_vc","P_hd","P_tv","P_nb","P_lg",
                "A_ov","A_dw","A_wm","A_ev",
                "B_ov","B_dw","B_wm","B_cd","B_ev",
        ]
        values=[deque(maxlen=self.T) for _ in range(len(keys))]
        self.monitor = dict(zip(keys, values))

    def _update_monitor(self, action):
        self.monitor["Price"].append(self.price[self.t])
        self.monitor["SoC"].append(self.ev.state[1])
        self.monitor["WtTemp"].append(sum(self.wh.state[1:]))
        self.monitor["flow"].append(self.wh.flow)
        self.monitor["IdTemp"].append(sum(self.ac.state[1:]))
        self.monitor["OdTemp"].append(self.temp[self.t])
        self.monitor["P_ov"].append(action[0]*self.ov.MAX_POWER)
        self.monitor["P_dw"].append(action[1]*self.dw.MAX_POWER)
        self.monitor["P_wm"].append(action[2]*self.wm.MAX_POWER)
        self.monitor["P_cd"].append(action[3]*self.cd.MAX_POWER)
        self.monitor["P_ev"].append(action[4])
        self.monitor["P_wh"].append(action[5])
        self.monitor["P_ac"].append(action[6])
        self.monitor["P_fg"].append(self.fg.state[0]*self.fg.MAX_POWER)
        self.monitor["P_vc"].append(self.vc.state[0]*self.vc.MAX_POWER)
        self.monitor["P_hd"].append(self.hd.state[0]*self.hd.MAX_POWER)
        self.monitor["P_tv"].append(self.tv.state[0]*self.tv.MAX_POWER)
        self.monitor["P_nb"].append(self.nb.state[0]*self.nb.MAX_POWER)
        self.monitor["P_lg"].append(self.lg.state[0]*self.lg.MAX_POWER)
        self.monitor["A_ov"].append((self.ov.alpha-self.init_time).seconds//self.dt.seconds)
        self.monitor["B_ov"].append((self.ov.beta-self.init_time).seconds//self.dt.seconds)
        self.monitor["A_dw"].append((self.dw.alpha-self.init_time).seconds//self.dt.seconds)
        self.monitor["B_dw"].append((self.dw.beta-self.init_time).seconds//self.dt.seconds)
        self.monitor["A_wm"].append((self.wm.alpha-self.init_time).seconds//self.dt.seconds)
        self.monitor["B_wm"].append((self.wm.beta-self.init_time).seconds//self.dt.seconds)
        self.monitor["B_cd"].append((self.cd.beta-self.init_time).seconds//self.dt.seconds)
        self.monitor["A_ev"].append((self.ev.alpha-self.init_time).seconds//self.dt.seconds)
        self.monitor["B_ev"].append((self.ev.beta-self.init_time).seconds//self.dt.seconds)