"""
June 2, 2019
Created by Hepeng Li

This is an emulator of electric vehicles.
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CAPACITY = 24
MAX_CHARGING_POWER = 6
MAX_DISCHARGING_POWER = -6
CHARGING_EFFICIENCY = 0.98
DISCHARGING_EFFICIENCY = 0.98
MAX_HORIZION = 24
DELTA_T = 1
MAX_SOC = 1.0
MIN_SOC = 0.1
TARGET_SOC = 1.0
RESCALE = 100

pricefile = '~/Documents/Github/tmp/data/price/RtpData2017.csv'
df_2017 = pd.read_csv(pricefile)
df_2017['PRICE'] = df_2017['PRICE'].astype('float32')
df_2017.index = pd.date_range('2017-01-01-00', '2017-12-31-23', freq='1H')

pricefile = '~/Documents/Github/tmp/data/price/RtpData2018.csv'
df_2018 = pd.read_csv(pricefile)
df_2018['PRICE'] = df_2018['PRICE'].astype('float32')
df_2018.index = pd.date_range('2018-01-01-00', '2018-12-31-23', freq='1H')

pricefile = '~/Documents/Github/tmp/data/price/RtpData2019.csv'
df_2019 = pd.read_csv(pricefile)
df_2019['PRICE'] = df_2019['PRICE'].astype('float32')
df_2019.index = pd.date_range('2019-01-01-00', '2019-08-31-23', freq='1H')

df_train = df_2017
df_test = pd.concat([df_2017[-24:], df_2018, df_2019[:24]])

price_data = {"train": df_train, "test": df_test}

class EVCharging(gym.Env):
    """A electric vehicle environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, seed=None, train=True):
        super(EVCharging, self).__init__()

        if train:
            self._price = price_data["train"]
        else:
            self._price = price_data["test"]

        self.reward_range = (MAX_DISCHARGING_POWER, MAX_CHARGING_POWER)

        # Actions are continuous
        self.action_space = spaces.Box(
            low=np.array([MAX_DISCHARGING_POWER]),
            high=np.array([MAX_CHARGING_POWER]), dtype=np.float32)

        # Obsercations include the past 24 hour prices and SOC of the EV battery
        # self.observation_space = spaces.Box(
        #     low=-1, high=1, shape=(24*2+2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(24+1,), dtype=np.float32)

    def seed(self, seed=None):
        self.rnd = np.random.RandomState(seed)
        return [seed]

    def step(self, action):
        action = float(action)

        # Execute one time step within the environment
        price = self._ep_prices[self._t+24]

        t = self._t + 1
        cur_time = self._cur_time + self._h(1)

        past_prices = self._ep_prices[t+1:t+25]
        diff_prices = past_prices - self._ep_prices[t:t+24]
        # past_prices = self._ep_prices[t:t+24]

        if action >= 0: # charging
            soc = float(self._soc + action * DELTA_T * CHARGING_EFFICIENCY / CAPACITY)
        else: # discharging
            soc = float(self._soc + action * DELTA_T / DISCHARGING_EFFICIENCY / CAPACITY)
        # ob = np.array(past_prices.tolist()+diff_prices.tolist()+[soc, cur_time.hour/24], dtype=np.float32)
        ob = np.array(past_prices.tolist()+[soc], dtype=np.float32)

        reward = float(- action * price)

        done = False if cur_time != self._dep_time else True

        safety = 0.0
        if done:
            safety += abs(TARGET_SOC - soc) * CAPACITY
        else:
            if soc > MAX_SOC:
                safety += (soc - MAX_SOC) * CAPACITY
            elif soc < MIN_SOC:
                safety += (MIN_SOC - soc) * CAPACITY

        self._t = t
        self._cur_time = cur_time
        self._soc = soc
        self._act = action
        self._ep_acts.append(action)
        self._ep_socs.append(soc)

        return ob, reward, done, {"s": safety}

    def reset(self, arr_date=None):
        # Reset the state of the environment to an initial state
        if arr_date is None:
            arr_date = self.rnd.choice(self._price['DATE'].unique()[1:-1])
        else:
            assert isinstance(arr_date, str)
        arr_hour = str(int(np.round(np.clip(self.rnd.normal(18,1),15,21)))).zfill(2)
        dep_hour = str(int(np.round(np.clip(self.rnd.normal(8,1),6,11)))).zfill(2)

        t = 0
        h = lambda x: pd.Timedelta(hours=x)
        arr_time = pd.to_datetime(arr_date+' '+arr_hour)
        dep_time = pd.to_datetime(arr_date+' '+dep_hour) + pd.Timedelta(days=1)
        cur_time = arr_time

        ep_prices = self._price.loc[arr_time-h(24):dep_time]["PRICE"].values
        past_prices = ep_prices[t+1:t+25]
        diff_prices = past_prices - ep_prices[t:t+24]
        # past_prices = ep_prices[t:t+24]

        soc = np.clip(self.rnd.normal(0.5, 0.1), 0.2, 0.8)
        # ob = np.array(past_prices.tolist()+diff_prices.tolist()+[soc, cur_time.hour/24], dtype=np.float32)
        ob = np.array(past_prices.tolist()+[soc], dtype=np.float32)

        self._t = t
        self._h = h
        self._cur_time = cur_time
        self._dep_time = dep_time
        self._soc = soc

        self._ep_acts = []
        self._ep_socs = [soc]
        self._ep_prices = ep_prices

        return ob

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        plt.close()

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ln1 = ax1.plot(self._ep_prices[24:], label='Price')

        ax2 = ax1.twinx()
        ln2 = ax2.plot(self._ep_socs, c='#ff7f0e', label='SOC')

        lns = ln1+ln2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=1, fontsize='x-large')

        plt.tight_layout(rect=[0,0,0.99,1.0])
        plt.show(block=False)
        plt.pause(.1)