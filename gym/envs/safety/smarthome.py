#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 12:13:40 2019

@author: lihepeng
"""

import os
import time
import pickle
import datetime
import numpy as np
import gym
import matplotlib.pyplot as plt
from pyscipopt import Model, quicksum
from collections import deque

class Stove(object):
    """
    Stove Model
    """
    def __init__(self, dt=1/12, rnd=np.random.RandomState()):

        self.state = [0.0, 0.0, 0.0] # status, progress, attribute
        self.action = 0.0
        self.max_power = 3.0 #kW
        self.dt = dt

        avg_cook_count = 12 # per week
        # statical information (please see '../data/recs/hc3.1.xlsx')
        number_homes = [16.8, 50.6, 23.9,13.4]
        usage_count = [range(1,4), range(4,8), range(8,15), range(15,20)]
        avg_usage_count = 0.0
        for i in range(number_homes.__len__()):
            avg_usage_count += np.mean(usage_count[i])*number_homes[i]
        avg_usage_count = avg_usage_count/np.sum(number_homes)

        self._usage_freq = avg_usage_count / avg_cook_count
        self._must_on_slots = max(1, int(np.ceil(3 * (1/12)/self.dt)))
        self._lamda = max(1, int(np.ceil(12 * (1/12)/self.dt)))
        self._waittime = 0
        self._rnd = rnd
        self._estimate_prob_off()
        self._estimate_prob_on()

    def _estimate_prob_on(self):
        # assume the stove must operate ON within n time slots after cooking 
        # starts, or will not operate in this cooking
        try:
            model = Model()
            p = model.addVar(vtype='C',name='p',lb=0,ub=1.0)
            cum_p = quicksum([(1-p)**n * p for n in range(self._must_on_slots)])
            model.addCons(cum_p == self._usage_freq, name='usage_freq')
            model.setObjective(p, 'maximize')
            model.hideOutput()
            model.setRealParam('limits/time', 180) # Maximal sovling time: 10 minutes 
            model.optimize()
            self._p_on = model.getVal(p)
        except:
            self._p_on = 0.2878379484943901

    def _estimate_prob_off(self):
        status = self.state[0]
        if status == 0.0:
            self._p_off = 0.0
            return

        K = int(self.state[1]) # Taks progress
        self._p_off = 0.0
        for k in range(K):
            self._p_off += np.exp(-self._lamda)*(self._lamda**k)/np.math.factorial(k)

    def step(self, activity, action, attribute):
        self.action = action
        if activity == 'cooking':
            self._waittime += 1
            status = self.state[0]
            if status == 1.0: # is carrying out a task
                self._estimate_prob_off()
                roll = self._rnd.rand()
                if roll < self._p_off:
                    self.state = [-1.0, 0.0, attribute]
                else:
                    self.state[1] += 1.0
                    self.state[2] = attribute
            elif status == 0.0: # is off
                roll = self._rnd.rand()
                if roll < self._p_on and self._waittime <= self._must_on_slots:
                    self.state = [1.0, 0.0, attribute]
                else:
                    self.state = [0.0, 0.0, attribute]
        else:
            self.state = [0.0, 0.0, attribute]
            self._waittime = 0

    def reset(self, rnd=np.random.RandomState()):
        self.state = [0.0, 0.0, 0.0]
        self.action = 0.0
        self._rnd = rnd

class Oven(object):
    """
    Oven Model
    """
    def __init__(self, rnd=np.random.RandomState(), dt=1/12):

        self.state = [0.0, 0.0, 0.0] # status, progress, attribute
        self.action = 0.0
        self.max_power = 3.3 #kW
        self.dt = dt

        avg_cook_count=12, # per week
        # statical information (please see '../data/recs/hc3.1.xlsx')
        number_homes = [56.8,28.3,3.7,1.1]
        usage_count = [range(1,4), range(4,8), range(8,15), range(15,20)]
        avg_usage_count = 0.0
        for i in range(number_homes.__len__()):
            avg_usage_count += np.mean(usage_count[i])*number_homes[i]
        avg_usage_count = avg_usage_count/np.sum(number_homes)

        self._usage_freq = avg_usage_count / avg_cook_count
        self._must_on_slots = max(1, int(np.ceil(5 * (1/12)/self.dt)))
        self._lamda = max(1, int(np.ceil(12 * (1/12)/self.dt)))
        self._waittime = 0
        self._rnd = rnd
        self._estimate_prob_off()
        self._estimate_prob_on()

    def _estimate_prob_on(self):
        # assume the oven must operate ON within n time slots after cooking 
        # starts, or will not operate in this cooking
        try:
            model = Model()
            p = model.addVar(vtype='C',name='p',lb=0,ub=1.0)
            cum_p = quicksum([(1-p)**n * p for n in range(self._must_on_slots)])
            model.addCons(cum_p == self._usage_freq, name='usage_probs')
            model.setObjective(p, 'maximize')
            model.hideOutput()
            model.setRealParam('limits/time', 180) # Maximal sovling time: 10 minutes 
            model.optimize()
            self._p_on = model.getVal(p)
        except:
            self._p_on = 0.07008892586079099

    def _estimate_prob_off(self):
        status = self.state[0]
        if status == 0.0:
            self.p_off = 0.0
            return

        K = int(self.state[1]) # Task progress
        self._p_off = 0.0
        for k in range(K):
            self._p_off += np.exp(-self._lamda)*(self._lamda**k)/np.math.factorial(k)

    def step(self, activity, action, attribute):
        self.action = action
        if activity == 'cooking':
            self._waittime += 1
            status = self.state[0]
            if status == 1.0:
                self._estimate_prob_off()
                roll = self._rnd.rand()
                if roll < self._p_off:
                    self.state = [-1.0, 0.0, attribute]
                else:
                    self.state[1] += 1.0
                    self.state[2] = attribute
            elif status == 0.0:
                roll = self._rnd.rand()
                if roll < self._p_on and self._waittime <= self._must_on_slots:
                    self.state = [1.0, 0.0, attribute]
                else:
                    self.state = [0.0, 0.0, attribute]
        else:
            self.state = [0.0, 0.0, attribute]
            self.activity = 0

    def reset(self, rnd=np.random.RandomState()):
        self.state = [0.0, 0.0, 0.0]
        self.action = 0.0
        self._rnd = rnd

class Microwave(object):
    """
    Microwave Model
    """
    def __init__(self, rnd=np.random.RandomState(), dt=1/12):

        self.state = [0.0, 0.0, 0.0] # status, progress, attribute
        self.action = 0.0
        self.max_power = 1.5 #kW
        self.dt = dt

        avg_cook_count=12 # per week
        # statical information (please see '../data/recs/hc3.1.xlsx')
        number_homes = [10.5,38.8,27.8,34.8]
        usage_count = [range(1,4), range(4,8), range(8,15), range(15,20)]
        avg_usage_count = 0.0
        for i in range(number_homes.__len__()):
            avg_usage_count += np.mean(usage_count[i])*number_homes[i]
        avg_usage_count = avg_usage_count/np.sum(number_homes)

        self._usage_freq = avg_usage_count / avg_cook_count
        self._must_on_slots = max(1, int(np.ceil(3 * (1/12)/self.dt)))
        self._lamda = avg_usage_time_per_cook = max(1, int(np.ceil(1 * (1/12)/self.dt)))
        self._waittime = 0
        self._rnd = rnd
        self._estimate_prob_off()
        self._estimate_prob_on()

    def _estimate_prob_on(self):
        # assume the microwave must operate ON within n time slots after cooking 
        # starts, or will not operate in this cooking
        try:
            model = Model()
            p = model.addVar(vtype='C',name='p',lb=0,ub=1.0)
            cum_p = quicksum([(1-p)**n * p for n in range(self._must_on_slots)])
            model.addCons(cum_p == self._usage_probs, name='usage_probs')
            model.setObjective(p, 'maximize')
            model.hideOutput()
            model.setRealParam('limits/time', 180) # Maximal sovling time: 10 minutes 
            model.optimize()
            self._p_on = model.getVal(p)
        except:
            self._p_on = 0.3093560297047645

    def _estimate_prob_off(self):
        status = self.state[0]
        if status == 0.0:
            self._p_off = 0.0
            return

        k = self.state[1]
        self._p_off = 0.0
        for k in range(int(self.state[1])):
            self._p_off += np.exp(-self._lamda)*(self._lamda**k)/np.math.factorial(k)

    def step(self, activity, action, attribute):
        self.action = action
        if activity == 'cooking':
            status = self.state[0]
            if status == 1.0:
                self._estimate_prob_off()
                roll = self._rnd.rand()
                if roll < self._p_off:
                    self.state = [0.0, 0.0, attribute]
                else:
                    self.state[1] += 1.0
                    self.state[2] = attribute
            elif status == 0.0:
                roll = self._rnd.rand()
                if roll < self._p_on:
                    self.state = [1.0, 0.0, attribute]
                else:
                    self.state = [0.0, 0.0, attribute]
        else:
            self.state = [0.0, 0.0, attribute]

    def reset(self, rnd):
        self.state = [0.0, 0.0, 0.0]
        self.action = 0.0
        self._rnd = rnd

class Vacuum(object):
    """
    Vaccum Model
    """
    def __init__(self, rnd=np.random.RandomState(), dt=1/12):

        self.state = [0.0, 0.0, 0.0] # status, progress, attribute
        self.action = 0.0
        self.max_power = 0.5 #kW
        self.dt = dt

        self._usage_freq = 0.9
        self._must_on_slots = 3
        self._lamda = avg_usage_time = 4
        self._estimate_prob_off()
        self._estimate_prob_on()
        self._rnd = rnd

    def _estimate_prob_on(self):
        try:
            model = Model()
            p = model.addVar(vtype='C',name='p',lb=0,ub=1.0)
            cum_p = quicksum([(1-p)**n * p for n in range(self._must_on_slots)])
            model.addCons(cum_p == self._usage_freq, name='usage_probs')
            model.setObjective(p, 'maximize')
            model.hideOutput()
            model.setRealParam('limits/time', 180) # Maximal sovling time: 10 minutes 
            model.optimize()
            self._p_on = model.getVal(p)
        except:
            self._p_on = 0.5358417249625212

    def _estimate_prob_off(self):
        status = self.state[0]
        if status == 0.0:
            self._p_off = 0.0
            return

        K = int(self.state[1]) # Task progress
        self._p_off = 0.0
        for k in range(K):
            self._p_off += np.exp(-self._lamda)*(self._lamda**k)/np.math.factorial(k)

    def step(self, activity, action, attribute):
        self.action = action
        if activity == 'interior cleaning':
            status = self.state[0]
            if status == 1.0:
                self._estimate_prob_off()
                roll = self._rnd.rand()
                if roll < self._p_off:
                    self.state = [-1.0, 0.0, attribute]
                else:
                    self.state[1] += 1.0
                    self.state[2] = attribute
            elif status == 0.0:
                roll = self._rnd.rand()
                if roll < self._p_on:
                    self.state = [1.0, 0.0, attribute]
                else:
                    self.state = [0.0, 0.0, attribute]
        else:
            self.state = [0.0, 0.0, attribute]

    def reset(self, rnd):
        self.state = [0.0, 0.0, 0.0]
        self.action = 0.0
        self._rnd = rnd

class Hairdryer(object):
    """
    Hairdryer Model
    """
    def __init__(self, rnd=np.random.RandomState(), dt=1/12):

        self.state = [0.0, 0.0, 0.0] # status, progress, attribute
        self.action = 0.0
        self.max_power = 1.5 #kW
        self.dt = dt

        # statical information (please see '../data/recs/hc3.1.xlsx')
        self._usage_freq = 0.75
        self._must_on_slots = max(1, int(np.ceil(5 * (1/12)/self.dt)))
        self._lamda = avg_usage_time = max(1, int(np.ceil(1 * (1/12)/self.dt)))
        self._estimate_prob_off()
        self._estimate_prob_on()
        self._waittime = 0
        self._rnd = rnd

    def _estimate_prob_on(self):
        try:
            model = Model()
            p = model.addVar(vtype='C',name='p',lb=0,ub=1.0)
            cum_p = quicksum([(1-p)**n * p for n in range(self._must_on_slots)])
            model.addCons(cum_p == self._usage_freq, name='usage_probs')
            model.setObjective(p, 'maximize')
            model.hideOutput()
            model.setRealParam('limits/time', 180) # Maximal sovling time: 10 minutes 
            model.optimize()
            self._p_on = model.getVal(p)
        except:
            self._p_on = 0.24214200388056806

    def _estimate_prob_off(self):
        status = self.state[0]
        if status == 0.0:
            self._p_off = 0.0
            return

        k = self.state[1]
        self._p_off = 0.0
        for k in range(int(self.state[1])):
            self._p_off += np.exp(-self._lamda)*(self._lamda**k)/np.math.factorial(k)

    def step(self, activity, action, attribute):
        self.action = action
        if activity == 'grooming':
            self._waittime += 1
            status = self.state[0]
            if status == 1.0:
                self._estimate_prob_off()
                roll = self._rnd.rand()
                if roll < self._p_off:
                    self.state = [-1.0, 0.0, attribute]
                else:
                    self.state[1] += 1.0
                    self.state[2] = attribute
            elif status == 0.0:
                roll = self._rnd.rand()
                if roll < self._p_on and self._waittime <= self._must_on_slots:
                    self.state = [1.0, 0.0, attribute]
                else:
                    self.state = [0.0, 0.0, attribute]
        else:
            self.state = [0.0, 0.0, attribute]
            self._waittime = 0

    def reset(self, rnd):
        self.state = [0.0, 0.0, 0.0]
        self.action = 0.0
        self._rnd = rnd

class TV(object):
    """
    TV Model
    """
    def __init__(self, dt=1/12):
        self.state = [0.0, 0.0, 0.0] # status, progress, attribute
        self.action = 0.0
        self.max_power = 0.21 #kW
        self.dt = dt

    def step(self, activity, action, attribute):
        if activity == 'Televisions':
            self.state[0] = 1.0
            self.state[1] += action
            self.state[2] = attribute
        else:
            self.state = [0.0, 0.0, attribute]

    def reset(self):
        self.state = [0.0, 0.0, 0.0]
        self.action = 0.0

class Computer(object):
    """
    Computer Model
    """
    def __init__(self, dt=1/12):
        self.state = [0.0, 0.0, 0.0] # status, progress, attribute
        self.action = 0.0
        self.max_power = 0.15 #kW
        self.dt = dt

    def step(self, activity, action, attribute):
        if activity == 'Computer':
            self.state[0] = 1.0
            self.state[1] += action
            self.state[2] = attribute
        else:
            self.state = [0.0, 0.0, attribute]

    def reset(self):
        self.state = [0.0, 0.0, 0.0]
        self.action = 0.0

class Light(object):
    """
    Light Model
    """
    def __init__(self, room=None, n=1, dt=1/12):
        self.state = [0.0, 0.0, 0.0] # status, progress, attribute
        self.action = 0.0
        self.max_power = 0.05 * n #kW
        self.room = room
        self.morning = datetime.datetime(year=2016,month=1,day=1,hour=8, minute=0)
        self.evening = datetime.datetime(year=2016,month=1,day=1,hour=18, minute=0)
        self.dt = dt

    def step(self, activity, action, attribute, time):
        assert isinstance(time, datetime.datetime)

        if time < self.morning or time > self.evening:
            self.action = action
            if activity == 'interior cleaning':
                if self.room in ['bedroom', 'livingroom', 'bathroom']:
                    self.state[0] = 1.0
                    self.state[1] += action
                    self.state[2] = attribute
                else:
                    self.state = [0.0, 0.0, attribute]
            elif activity in ['cooking', 'eating/drinking', 'kitchen leanup']:
                if self.room == 'kitchen':
                    self.state[0] = 1.0
                    self.state[1] += action
                    self.state[2] = attribute
                else:
                    self.state = [0.0, 0.0, attribute]
            elif activity in ['grooming', 'laundry']:
                if self.room == 'bathroom':
                    self.state[0] = 1.0
                    self.state[1] += action
                    self.state[2] = attribute
                else:
                    self.state = [0.0, 0.0, attribute]
            elif activity == 'Televisions':
                if self.room == 'living room':
                    self.state[0] = 1.0
                    self.state[1] += action
                    self.state[2] = attribute
                else:
                    self.state = [0.0, 0.0, attribute]
            elif activity in ['Computer', 'others']:
                if self.room in ['living room', 'bed room']:
                    self.state[0] = 1.0
                    self.state[1] += action
                    self.state[2] = attribute
                else:
                    self.state = [0.0, 0.0, attribute]
            else:
                self.state = [0.0, 0.0, attribute]
        else:
            self.state = [0.0, 0.0, attribute]

    def reset(self):
        self.state = [0.0, 0.0, 0.0]
        self.action = 0.0

class Dishwasher(object):
    """
    Dishwasher Model
    """
    def __init__(self, rnd=np.random.RandomState(), dt=1/12):

        self.state = [0.0, 0.0, 0.0] # status, progress, attribute
        self.action = 0.0
        self.min_power = 0.0 #kW
        self.max_power = 1.5 #kW
        self.dt = dt

        task_demand = max(1, int(np.ceil(12 * (1/12)/self.dt))) # 1 hour
        # statical information (please see '../data/recs/hc3.1.xlsx')
        self._usage_freq = 1.0
        self._must_on_slots = max(1, int(np.ceil(1 * (1/12)/self.dt)))
        self._task_demand = task_demand
        self._task_duration = task_demand
        self._deadline=[datetime.datetime(2016,1,1,17), datetime.datetime(2016,1,1,23)]
        self._estimate_prob_on()
        self._rnd = rnd

    def _estimate_prob_on(self):
        try:
            model = Model()
            p = model.addVar(vtype='C',name='p',lb=0,ub=1.0)
            cum_p = quicksum([(1-p)**n * p for n in range(self._must_on_slots)])
            model.addCons(cum_p == self._usage_freq, name='usage_probs')
            model.setObjective(p, 'maximize')
            model.hideOutput()
            model.setRealParam('limits/time', 180) # Maximal sovling time: 10 minutes 
            model.optimize()
            self._p_on = model.getVal(p)
        except:
            self._p_on = 1.0

    def step(self, activity, action, time):
        assert isinstance(time, datetime.datetime)

        if time == datetime.datetime(2016,1,2,8):
            self.state = [0.0, 0.0, 0.0]
        self.action = action
        status, progress, remain_slots = self.state
        if status == 1.0:
            progress += action
            remain_slots -= 1
            assert remain_slots >= 0
            if progress == self._task_demand:
                self.state = [-1.0, 0.0, 0]
            else:
                self.state = [status, progress, remain_slots]
        elif status == 0.0:
            if activity == 'kitchen leanup':
                roll = self._rnd.rand()
                if roll < self._p_on:
                    # if the time is earlier than the first deadline
                    if time < self._deadline[0]:
                        delta_t = self._deadline[0] - time
                        delta_slots = int(delta_t.total_seconds()/(3600*self.dt))
                        # if the delta_slots is not enough to fulfill the task
                        if delta_slots < self._task_demand:
                            delta_t = self._deadline[1] - time
                            delta_slots = int(delta_t.total_seconds()/(3600*self.dt))
                            self._task_duration = delta_slots
                        else:
                            self._task_duration = delta_slots
                    # if the time is earlier than the second deadline
                    elif time < self._deadline[1]:
                        delta_t = self._deadline[1] - time
                        delta_slots = int(delta_t.total_seconds()/(3600*self.dt))
                        if delta_slots < self._task_demand:
                            self._task_duration = self._task_demand
                        else:
                            self._task_duration = delta_slots
                    else:
                        return
                    self.state = [1.0, 0.0, self._task_duration]
                else:
                    pass
            else:
                pass
        else:
            if activity != 'kitchen leanup':
                self.state = [0.0, 0.0, 0]
            else:
                pass

    def feasible_action(self):
        status, progress, remain_slots = self.state
        if status == 1.0:
            if progress == 0.0:
                if remain_slots > self._task_demand:
                    action = np.array([0.0, 1.0], dtype=np.float32)
                else:
                    action = np.array([1.0, 1.0], dtype=np.float32)
            else:
                action = np.array([1.0, 1.0], dtype=np.float32)
        else:
            action = np.array([0.0, 0.0], dtype=np.float32)

        return action

    def reset(self, rnd):
        self.state = [0.0, 0.0, 0.0]
        self.action = 0.0
        self._rnd = rnd

class Clothwasher(object):
    """
    Clothwasher Model
    """
    def __init__(self, rnd=np.random.RandomState(), dt=1/12):

        self.state = [0.0, 0.0, 0.0] # status, progress, attribute
        self.action = 0.0
        self.min_power = 0.0 #kW
        self.max_power = 1.5 #kW
        self.dt = dt

        task_demand = max(1, int(np.ceil(12 * (1/12)/self.dt))) # 1 hour # 1 hour
        # statical information (please see '../data/recs/hc3.1.xlsx')
        self._usage_freq = 1.0
        self._must_on_slots = max(1, int(np.ceil(1 * (1/12)/self.dt))) # 1 hour
        self._task_demand = task_demand
        self._task_duration = task_demand
        self._deadline = datetime.datetime(2016,1,1,20)
        self._estimate_prob_on()
        self._rnd = rnd

    def _estimate_prob_on(self):
        try:
            model = Model()
            p = model.addVar(vtype='C',name='p',lb=0,ub=1.0)
            cum_p = quicksum([(1-p)**n * p for n in range(self._must_on_slots)])
            model.addCons(cum_p == self._usage_freqs, name='usage_probs')
            model.setObjective(p, 'maximize')
            model.hideOutput()
            model.setRealParam('limits/time', 180) # Maximal sovling time: 10 minutes 
            model.optimize()
            self._p_on = model.getVal(p)
        except:
            self._p_on = 1.0

    def step(self, activity, action, time):
        assert isinstance(time, datetime.datetime)
        if time == datetime.datetime(2016,1,2,8):
            self.state = [0.0, 0.0, 0.0]

        self.action = action
        status, progress, remain_slots = self.state
        if status == 1.0:
            progress += action
            remain_slots -= 1
            assert remain_slots >= 0
            if progress == self._task_demand:
                self.state = [-1.0, 0.0, 0]
            else:
                self.state = [status, progress, remain_slots]
        elif status == 0.0:
            if activity == 'laundry':
                roll = self._rnd.rand()
                if roll < self._p_on:
                    if time < self._deadline:
                        delta_t = self._deadline - time
                        delta_slots = int(delta_t.seconds/(3600*self.dt))
                        if delta_slots < self._task_demand:
                            self._task_duration = self._task_demand
                        else:
                            self._task_duration = delta_slots
                    else:
                        return
                    self.state = [1.0, 0.0, self._task_duration]
                else:
                    pass
            else:
                pass

    def feasible_action(self):
        status, progress, remain_slots = self.state
        if status == 1.0:
            if progress == 0.0:
                if remain_slots > self._task_demand:
                    action = np.array([0.0, 1.0], dtype=np.float32)
                else:
                    action = np.array([1.0, 1.0], dtype=np.float32)
            else:
                action = np.array([1.0, 1.0], dtype=np.float32)
        else:
            action = np.array([0.0, 0.0], dtype=np.float32)

        return action

    def reset(self, rnd):
        self.state = [0.0, 0.0, 0.0]
        self.action = 0.0
        self._rnd = rnd

class Clothdryer(object):
    """
    Clothdryer Model
    """
    def __init__(self, rnd=np.random.RandomState(), dt=1/12):

        self.state = [0.0, 0.0, 0.0] # status, progress, attribute
        self.action = 0.0
        self.min_power = 0.0 #kW
        self.max_power = 4.0 #kW
        self.dt = dt

        task_demand = max(1, int(np.ceil(12 * (1/12)/self.dt))) # 1 hour
        # statical information (please see '../data/recs/hc3.1.xlsx')
        self._usage_freq = 1.0
        self._must_on_slots = max(1, int(np.ceil(1 * (1/12)/self.dt)))
        self._task_demand = task_demand
        self._task_duration = task_demand
        self._deadline = datetime.datetime(2016,1,1,23)
        self._estimate_prob_on()
        self._rnd = rnd

    def _estimate_prob_on(self):
        try:
            model = Model()
            p = model.addVar(vtype='C',name='p',lb=0,ub=1.0)
            cum_p = quicksum([(1-p)**n * p for n in range(self._must_on_slots)])
            model.addCons(cum_p == self._usage_freq, name='usage_probs')
            model.setObjective(p, 'maximize')
            model.hideOutput()
            model.setRealParam('limits/time', 180) # Maximal sovling time: 10 minutes 
            model.optimize()
            self._p_on = model.getVal(p)
        except:
            self._p_on = 1.0

    def step(self, activity, action, time, clothwasher):
        assert isinstance(time, datetime.datetime)
        if time == datetime.datetime(2016,1,2,8):
            self.state = [0.0, 0.0, 0.0]

        self.action = action
        status, progress, remain_slots = self.state
        if status == 1.0:
            progress += action
            remain_slots -= 1
            assert remain_slots >= 0
            if progress == self._task_demand:
                self.state = [-1.0, 0.0, 0]
            else:
                self.state = [status, progress, remain_slots]
        elif status == 0.0:
            if clothwasher.state[0] == -1.0:
                roll = self._rnd.rand()
                if roll < self._p_on:
                    if time < self._deadline:
                        delta_t = self._deadline - time
                        delta_slots = int(delta_t.total_seconds()/(3600*self.dt))
                        if delta_slots < self._task_demand:
                            self._task_duration = self._task_demand
                        else:
                            self._task_duration = delta_slots
                    else:
                        return
                    self.state = [1.0, 0.0, self._task_duration]

    def feasible_action(self):
        status, progress, remain_slots = self.state
        if status == 1.0:
            if progress == 0.0:
                if remain_slots > self._task_duration:
                    action = np.array([0.0, 1.0], dtype=np.float32)
                else:
                    action = np.array([1.0, 1.0], dtype=np.float32)
            else:
                action = np.array([1.0, 1.0], dtype=np.float32)
        else:
            action = np.array([0.0, 0.0], dtype=np.float32)

        return action

    def reset(self, rnd):
        self.state = [0.0, 0.0, 0.0]
        self.action = 0.0
        self._rnd = rnd

class EV(object):
    """
    EV Model
    """
    def __init__(
            self,
            max_power=6, # kW,
            capacity=24, # kWh
            eta=1.0,
            min_soc=0.1,
            max_soc=1.0,
            soc_target=1.0,
            dt=1/12,
            rnd=np.random.RandomState(),
        ):
        self.state = [0.0, 0.0, 0.0] # status, progress, attribute
        self.action = 0.0
        self.min_power = -max_power
        self.max_power = max_power
        self.capacity = capacity
        self.eta = eta
        self.max_soc = max_soc
        self.min_soc = min_soc
        self.soc_target = soc_target
        self._init_soc = lambda : np.clip(self._rnd.normal(0.5,0.1),0.4,0.6)
        self._earliest_arrval_time = datetime.datetime(2016,1,1,17,0)
        self._latest_arrval_time = datetime.datetime(2016,1,1,22,0)
        self._deadline = datetime.datetime(2016,1,2,8,0)
        self._task_demand = int(0.62 * capacity / (max_power * dt))
        self._task_duration = self._task_demand
        self._dt = dt
        self._rnd = rnd

    def step(self, last_activity, activity, action, time):
        assert isinstance(time, datetime.datetime)
        self.action = action
        status, soc, remain_slots = self.state

        # EV arrival
        arrival = False
        if time >= self._earliest_arrval_time:
            # if the EV has not arrived yet
            if status == 0:
                # check if it is arriving now
                if last_activity=='out' and activity != 'out':
                    arrival = True

                # if EV does not arrive later than the latest_arrval_time
                if time >= self._latest_arrval_time and activity != 'out':
                    # the EV may arrive at any time with a high probability
                    if self._rnd.rand() < 0.25:
                        arrival = True

        # state transition
        if status == 1.0:
            soc += self._dt * action/self.capacity
            remain_slots -= 1
            assert remain_slots >= 0.0
            self.state = [status, soc, remain_slots]
            if remain_slots == 0.0:
                self.state = [0.0, 0.0, 0]
        elif status == 0.0:
            if arrival:
                delta_t = self._deadline - time
                delta_slots = int(delta_t.seconds * self._dt)
                if delta_slots >= self._task_demand:
                    self._task_duration = delta_slots
                    self.state = [1.0, self._init_soc(), delta_slots]
                else:
                    pass
            else:
                pass

    def feasible_action(self):
        if self.state[0] == 0:
            action = [0.0, 0.0]
        else:
            action = [self.min_power, self.max_power]

        return action

    def reset(self, rnd):
        self.state = [0.0, 0.0, 0.0]
        self.action = 0.0
        self._rnd = rnd

class EWH(object):
    """
    EWH Model
    """
    def __init__(
            self,
            min_power=0.0, # kW,
            max_power=4.0, # kW,
            setting=125.0,
            max_dev=5.0,
            dt=1/12,
            rnd=np.random.RandomState(),
        ):
        self.state = [1.0, 0.0, 0.0] # status, progress, attribute
        self.action = 0.0
        self.min_power = min_power
        self.max_power = max_power
        self.min_temp = setting - max_dev
        self.max_temp = setting + max_dev
        self.setting = setting
        self.max_dev = max_dev
        self.flow = 0.0
        self._dt = dt
        self._rnd = rnd

    def step(self, activity, action):
        if activity in ['grooming','kitchen leanup']:
            self.flow = np.clip(self._rnd.normal(2.0, 0.5), 1.0, 3.0)
        else:
            self.flow = 0.0

        Tin = 60
        T_air = 75
        d = 8.34
        Cp = 1.0069
        volume = 40
        SA = 24.1
        R = 15/3
        Q = 3412
        C = volume * d * Cp
        G = SA / R
        B = d * self.flow * Cp
        R1 = 1/(G + B)
        coff = np.exp(-self._dt/(R1*C))

        T_old = self.state[1] + self.setting
        T_new = coff*T_old+(1-coff)*(G*R1*T_air+B*R1*Tin+action*self._dt*Q*R1)

        self.state[0] = 1.0
        self.state[1] = T_new - self.setting
        self.state[2] = self.setting

    def reset(self, rnd):
        self.state = [0.0, 0.0, 0.0]
        self.action = 0.0
        self._rnd = rnd

    def feasible_action(self):
        return [self.min_power, self.max_power]

class HVAC(object):
    """
    HVAC Model
    """
    def __init__(
            self,
            min_power=0.0, # kW,
            max_power=5.0, # kW,
            setting=75.0,
            max_dev=4.0,
            dt=1/12,
        ):
        self.state = [0.0, 0.0, 0.0] # status, progress, attribute
        self.action = 0.0
        self.min_power = min_power
        self.max_power = max_power
        self.min_temp = setting - max_dev
        self.max_temp = setting + max_dev
        self.setting = setting
        self.max_dev = max_dev
        self._dt = dt

    def step(self, action, T_out):
        eps = 0.9683
        lam = 13.76
        A = 0.25

        T_old = self.state[1] + self.setting
        if T_old > self.max_temp:
            self.state[0] = 1.0
        elif T_out < self.max_temp:
            self.state[0] = 0.0

        if self.state[0] == 0.0:
            T_new = eps * T_old + (1 - eps) * (T_out - 0 * self._dt * lam/A)
        else:
            T_new = eps * T_old + (1 - eps) * (T_out - action * self._dt * lam/A)

        self.state[1] = T_new - self.setting
        self.state[2] = self.setting

        self.action = action

    def reset(self):
        self.state = [0.0, 0.0, 0.0]
        self.action = 0.0

    def feasible_action(self):
        return [self.min_power, self.max_power]

class SmartHome(gym.Env):
    """A smart home environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, seed=None, train=True):

        super(SmartHome, self).__init__()
        self.seed(seed)

        self.t = 0
        self.T = 288
        self.dt = datetime.timedelta(minutes=5)
        self.one_day = datetime.timedelta(days=1)
        self.time = datetime.datetime(2016,1,1,4,0)
        self.init_time = datetime.datetime(2016,1,1,4,0)
        self.start_time = datetime.datetime(2016,1,1,8,0)

        # Deferrable appliances
        self.dw = Dishwasher(rnd=self.rnd)
        self.cw = Clothwasher(rnd=self.rnd)
        self.cd = Clothdryer(rnd=self.rnd)
        # Energy storable appliances
        self.hvac = HVAC(dt=self.dt.seconds//60/60)
        self.ewh = EWH(dt=self.dt.seconds//60/60, rnd=self.rnd)
        self.ev = EV(dt=self.dt.seconds//60/60, rnd=self.rnd)
        # Critical appliances
        self.stove = Stove(rnd=self.rnd)
        self.oven = Oven(rnd=self.rnd)
        self.microwave = Microwave(rnd=self.rnd)
        self.vacuum = Vacuum(rnd=self.rnd)
        self.hairdryer = Hairdryer(rnd=self.rnd)
        self.tv = TV()
        self.computer = Computer()
        self.brlight = Light(room='bathroom')
        self.lvlight = Light(room='livingroom', n=3)
        self.bdlight = Light(room='bedroom', n=2)
        self.ktlight = Light(room='kitchen', n=2)

        self.reward_range = (-1000.0, 0.0)

        self.action_space = gym.spaces.Box(
            low=np.array([
                int(bool(self.dw.min_power)), 
                int(bool(self.cw.min_power)),
                int(bool(self.cd.min_power)),
                self.hvac.min_power,
                self.ewh.min_power,
                self.ev.min_power,
                ]),
            high=np.array([
                int(bool(self.dw.max_power)), 
                int(bool(self.cw.max_power)),
                int(bool(self.cd.max_power)),
                self.hvac.max_power,
                self.ewh.max_power,
                self.ev.max_power,
                ]), dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(51+self.T*2,), dtype=np.float32)

        self._load_data(train)
        self._build_monitor()

    def seed(self, seed=None):
        self.rnd = np.random.RandomState(seed)
        return [seed]

    def step(self, action):
        action = np.hstack([action.ravel(), self._gen_cri_actions()])

        # Get next activity and update Monitor
        self._next_activity()
        if self.time >= datetime.datetime(2016,1,1,8,0):
            self._update_monitor(action)

        # Step
        self.dw.step(self.activity, action[0], self.time)
        self.cw.step(self.activity, action[1], self.time)
        self.cd.step(self.activity, action[2], self.time, self.cw)
        self.hvac.step(action[3], self.temp[self.t])
        self.ewh.step(self.activity, action[4])
        self.ev.step(self.last_activity, self.activity, action[5], self.time)
        self.stove.step(self.activity, action[6], self.t)
        self.oven.step(self.activity, action[7], self.t)
        self.microwave.step(self.activity, action[8], self.t)
        self.vacuum.step(self.activity, action[9], self.t)
        self.hairdryer.step(self.activity, action[10], self.t)
        self.tv.step(self.activity, action[11], self.t)
        self.computer.step(self.activity, action[12], self.t)
        self.brlight.step(self.activity, action[13], self.t, self.time)
        self.lvlight.step(self.activity, action[14], self.t, self.time)
        self.bdlight.step(self.activity, action[15], self.t, self.time)
        self.ktlight.step(self.activity, action[16], self.t, self.time)

        # Get observation
        ob = self._get_obs()

        # Calculate the reward
        reward = self._get_reward(action)

        # Update time
        self.t += 1
        self.time += self.dt

        # Check done
        if self.time == self.start_time + self.one_day:
            done = True
        else:
            done = False

        return ob, reward, done, {}

    def reset(self, seed=None):
        self.seed(seed)

        # render the env every 10 minutes
        if time.time() - self._render_time > 1.0 * 60:
            self.render()
            # self._render_time = time.time()

        # reset the time to 4:00 AM
        day = 1 # self.rnd.choice(60)
        self.t = day * self.T
        self.time = datetime.datetime(2016,1,1,4,0)

        # reset the appliances
        self.dw.reset(self.rnd)
        self.cw.reset(self.rnd)
        self.cd.reset(self.rnd)
        self.hvac.reset()
        self.ewh.reset(self.rnd)
        self.ev.reset(self.rnd)
        self.stove.reset(self.rnd)
        self.oven.reset(self.rnd)
        self.microwave.reset(self.rnd)
        self.vacuum.reset(self.rnd)
        self.hairdryer.reset(self.rnd)
        self.tv.reset()
        self.computer.reset()
        self.brlight.reset()
        self.lvlight.reset()
        self.bdlight.reset()
        self.ktlight.reset()

        # simulate the smart home to 8:00 AM
        for _ in range(48):
            action = self._gen_dr_actions()
            ob, _, _, _ = self.step(action)
        # Randomly re-initialize the state of the HVAC and the EWH
        ob[10] = self.hvac.state[1] = self.rnd.uniform(-self.hvac.max_dev, self.hvac.max_dev)
        ob[13] = self.ewh.state[1] = self.rnd.uniform(-self.ewh.max_dev, self.ewh.max_dev)

        return ob

    def render(self, mode='human', close=False):
            # Render the environment to the screen
        act_labels = [
            'Others', 'Sleeping', 'Grooming', 'Laundry', 'Cooking', 
            'Eating/Drinking', 'Kitchen Cleanup', 'Interior Cleaning', 
            'Working/Relaxing', 'Out/Arriving'
        ]
        xticks = range(0, self.T, 24)
        xticklabels = [str(i) for i in range(8,24,2)] + [str(i) for i in range(0,8,2)]

        plt.close()
        plt.figure(figsize=(21,8))

        ax2 = plt.subplot(2,3,1)
        ax2.scatter(range(self.T), self.monitor["Activity"], label='Activity', c='y')
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(labels=xticklabels)
        ax2.set_yticks(range(10))
        ax2.set_yticklabels(act_labels, fontsize='large')
        ax2.grid(axis='y')

        ax3 = plt.subplot(2,3,2)
        l3 = ax3.plot(self.monitor["IndoorTemp"], label='Indoor Temp')
        l4 = ax3.plot(self.monitor["OutTemp"], label='Outdoor Temp')
        ax3.axhline(71.0, linewidth=2, color='r')
        ax3.axhline(79.0, linewidth=2, color='r')
        ax3.set_xticks(xticks)
        ax3.set_xticklabels(labels=xticklabels)
        ax4 = ax3.twinx()
        l5 = ax4.plot(self.monitor["Pac"], label='Power', c='c')
        lns = l3+l4+l5
        labs = [l.get_label() for l in lns]
        ax3.legend(lns, labs, loc=1, fontsize='x-large')

        ax5 = plt.subplot(2,3,3)
        l6 = ax5.plot(self.monitor["WaterTemp"], label='Water Temp')
        ax5.axhline(120.0, linewidth=2, color='r')
        ax5.axhline(130.0, linewidth=2, color='r')
        ax5.set_xticks(xticks)
        ax5.set_xticklabels(labels=xticklabels)
        ax6 = ax5.twinx()
        l7 = ax6.plot(self.monitor["FlowRate"], label='Flow rate', c='y')
        l8 = ax6.plot(self.monitor["Pewh"], label='Power', c='c')
        lns = l6+l7+l8
        labs = [l.get_label() for l in lns]
        ax5.legend(lns, labs, loc=1, fontsize='x-large')

        ax7 = plt.subplot(2,3,4)
        l9 = ax7.plot(self.monitor["SoC"], label='SOC')
        ax7.set_xticks(xticks)
        ax7.set_xticklabels(labels=xticklabels)
        ax8 = ax7.twinx()
        l10 = ax8.plot(self.monitor["Pev"], label='Power', c='c')
        lns = l9+l10
        labs = [l.get_label() for l in lns]
        ax7.legend(lns, labs, loc=1, fontsize='x-large')

        ax9 = plt.subplot(2,3,5)
        l10 = ax9.plot(self.monitor["Pdw"], label='DW')
        l11 = ax9.plot(self.monitor["Pcw"], label='CW')
        l12 = ax9.plot(self.monitor["Pcd"], label='CD')
        ax9.set_xticks(xticks)
        ax9.set_xticklabels(labels=xticklabels)
        lns = l10+l11+l12
        labs = [l.get_label() for l in lns]
        ax9.legend(lns, labs, loc=1, fontsize='x-large')

        ax10 = plt.subplot(2,3,6)
        l13 = ax10.plot(self.monitor["P"])
        ax10.set_xticks(xticks)
        ax10.set_xticklabels(labels=xticklabels)
        ax1 = ax10.twinx()
        l1 = ax1.plot(self.monitor["Price"], label='Price', c='#ff7f0e')
        lns = l13+l1
        labs = [l.get_label() for l in lns]
        ax10.legend(fontsize='x-large')

        plt.tight_layout()
        plt.show(block=True)
        # plt.pause(1.0)

    def _load_data(self, train):
        data_dir = '/home/lihepeng/Documents/Github/tmp/dr/data'
        pkl_file = open(os.path.join(data_dir, 'init_probs.pkl'), 'rb')
        self.init_probs = pickle.load(pkl_file)
        pkl_file.close()

        pkl_file = open(os.path.join(data_dir, 'trans_probs.pkl'), 'rb')
        self.trans_probs = pickle.load(pkl_file)
        pkl_file.close()

        if train:
            print('train')
            self.price = np.loadtxt(os.path.join(data_dir, 'price_train.txt'))
            self.temp = np.loadtxt(os.path.join(data_dir, 'temp_train.txt'))
        else:
            print('test')
            self.price = np.loadtxt(os.path.join(data_dir, 'price_test.txt'))
            self.temp = np.loadtxt(os.path.join(data_dir, 'temp_test.txt'))

    def _roulette(self, probs):
        # Inputs: probs is a dictionary. 
        # The keys are actitivities， the values are the corresponding probs.

        roll = self.rnd.rand()
        roulette = np.cumsum(list(probs.values()))

        act_index = sum(roll >= roulette)
        act = list(probs.keys())[act_index]

        return act

    def _init_activity(self):
        return self._roulette(self.init_probs)

    def _next_activity(self):
        if self.time.hour == 4 and self.time.minute == 0:
            self.activity = self._init_activity()
            self.last_activity = self.activity
        else:
            self.last_activity = self.activity
            if self.time.hour < 4:
                time = datetime.datetime(2016,1,2,self.time.hour,self.time.minute)
            else:
                time = datetime.datetime(2016,1,1,self.time.hour,self.time.minute)
            trans_prob = self.trans_probs[time.__str__()].loc[self.activity].to_dict()
            self.activity = self._roulette(trans_prob)

            # Laundry
            if len(self.monitor["ep_acts"])>0 and self.cw.state[0] == 0:
                if self.time.hour in [8,9,10,16,17,18,19]:
                    if self.activity != self.monitor["ep_acts"][-1]:
                        if self.rnd.rand() > 0.5:
                            self.activity = 'laundry'

    def _get_reward(self, action):
        # thermal comfort
        w1 = 0.01
        w2 = 0.01
        hvac_dev = min(0.0, self.hvac.max_dev - abs(self.hvac.state[1]))
        ewh_dev = min(0.0, self.ewh.max_dev - abs(self.ewh.state[1]))
        I_comf = w1 * np.exp(hvac_dev) + w2 * np.exp(ewh_dev)

        # electricity cost
        power = action[0]*self.dw.max_power + \
                action[1]*self.cw.max_power + \
                action[2]*self.cd.max_power + \
                action[3] + action[4] + action[5] +\
                action[6]*self.stove.max_power + \
                action[7]*self.oven.max_power + \
                action[8]*self.microwave.max_power + \
                action[9]*self.vacuum.max_power + \
                action[10]*self.hairdryer.max_power + \
                action[11]*self.tv.max_power + \
                action[12]*self.computer.max_power + \
                action[13]*self.brlight.max_power + \
                action[14]*self.lvlight.max_power + \
                action[15]*self.bdlight.max_power + \
                action[16]*self.ktlight.max_power
        C_elec = power * self.price[self.t] * (self.dt.seconds//60/60)

        # EV anxiety
        w3 = 0.02
        E_range, soc = 0, self.ev.state[1]
        if self.time.hour == 8 and self.time.minute == 0:
            E_range += ((soc - self.ev.soc_target) * self.ev.capacity) ** 2
        else:
            E_range += np.log(max(0, self.ev.min_soc-soc) + \
                              max(0, soc-self.ev.max_soc) + 0.01) - np.log(0.01)
        E_range = w3 * E_range

        reward = I_comf - C_elec - E_range

        return reward

    def _get_obs(self):
        self.ob = np.hstack([
            self.dw.state,
            self.cw.state,
            self.cd.state,
            self.hvac.state,
            self.ewh.state,
            self.ev.state,
            self.stove.state,
            self.oven.state,
            self.microwave.state,
            self.vacuum.state,
            self.hairdryer.state,
            self.tv.state,
            self.computer.state,
            self.brlight.state,
            self.lvlight.state,
            self.bdlight.state,
            self.ktlight.state,
            self.price[self.t-self.T:self.t],
            self.temp[self.t-self.T:self.t],
        ]).astype(np.float32)

        return self.ob.ravel()

    def _gen_cri_actions(self):
        action = np.zeros(11, 'float32')
        action[0] = 1.0 if self.stove.state[0]==1 else 0.0
        action[1] = 1.0 if self.oven.state[0]==1 else 0.0
        action[2] = 1.0 if self.microwave.state[0]==1 else 0.0
        action[3] = 1.0 if self.vacuum.state[0]==1 else 0.0
        action[4] = 1.0 if self.hairdryer.state[0]==1 else 0.0
        action[5] = 1.0 if self.tv.state[0]==1 else 0.0
        action[6] = 1.0 if self.computer.state[0]==1 else 0.0
        action[7] = 1.0 if self.brlight.state[0]==1 else 0.0
        action[8] = 1.0 if self.lvlight.state[0]==1 else 0.0
        action[9] = 1.0 if self.bdlight.state[0]==1 else 0.0
        action[10] = 1.0 if self.ktlight.state[0]==1 else 0.0

        return action

    def _gen_dr_actions(self):
        action = self.action_space.sample()
        low, high = self._feasible_action()
        action[:3] = np.clip(np.round(action[:3]), low[:3], high[:3])
        action[3:] = np.clip(action[3:], low[3:], high[3:])

        return action

    def _feasible_action(self):
        fea_acts = np.array([
            self.dw.feasible_action(),
            self.cw.feasible_action(),
            self.cd.feasible_action(),
            self.hvac.feasible_action(),
            self.ewh.feasible_action(),
            self.ev.feasible_action(),
        ], dtype=np.float32)

        return fea_acts[:,0], fea_acts[:,1]

    def _build_monitor(self):
        self.act_dict = {
            'others':0,
            'sleeping':1,
            'grooming':2,
            'laundry':3,
            'cooking':4,
            'eating/drinking':5,
            'kitchen leanup':6,
            'interior cleaning':7,
            'Televisions':8,
            'Computer':8,
            'out':9,
        }
        self._render_time = time.time()
        keys = ["Activity","Price","IndoorTemp","OutTemp","WaterTemp","FlowRate",
                "SoC","Pev","Pac","Pewh","Pdw","Pcw","Pcd","P","ep_acts"]
        values=[deque(maxlen=self.T) for _ in range(16)]
        self.monitor = dict(zip(keys, values))

    def _update_monitor(self, action):
        self.monitor["Activity"].append(self.act_dict[self.activity])
        self.monitor["Price"].append(self.price[self.t])
        self.monitor["IndoorTemp"].append(self.hvac.state[1]+self.hvac.setting)
        self.monitor["OutTemp"].append(self.temp[self.t])
        self.monitor["WaterTemp"].append(self.ewh.state[1]+self.ewh.setting)
        self.monitor["FlowRate"].append(self.ewh.flow)
        self.monitor["SoC"].append(self.ev.state[1])
        self.monitor["Pev"].append(action[5])
        self.monitor["Pac"].append(action[3])
        self.monitor["Pewh"].append(action[4])
        self.monitor["Pdw"].append(action[0])
        self.monitor["Pcw"].append(action[1])
        self.monitor["Pcd"].append(action[2])
        self.monitor["P"].append(action[6:].sum())
        self.monitor["ep_acts"].append(self.activity)