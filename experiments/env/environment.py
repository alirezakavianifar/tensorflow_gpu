import gym
from gym import spaces
import numpy as np
import math
import sys
# sys.path.append(r'D:\projects\tensorflow_gpu\experiments')
sys.path.append(r'D:\projects\tensorflow_gpu\experiments')
from utils import utility, load_data

N_STATES = 216
N_ACTIONS = 216


class DeltaIotEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DeltaIotEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_ACTIONS)
        # Example for using image as input:
        self.observation_space = spaces.Discrete(N_STATES)
        # Load the data
        self.data = load_data()
        self.ut = math.inf
        self.done = False
        self.time_steps = N_ACTIONS
        self.reward = 0
        self.info = {}

    def step(self, action):
        observation = action
        energy_consumption = self.df.iloc[action:action +
                                          1, 2:3].values.tolist()[0][0]
        packet_loss = self.df.iloc[action:action+1, 3:4].values.tolist()[0][0]
        ut = utility(0.8, 0.2, energy_consumption, packet_loss)

        if ut <= self.ut:
            self.reward = 1
            self.ut = ut
        else:
            self.reward = -0.02

        self.time_steps -= 1
        if self.time_steps == 0:
            self.done = True

        return self.obs, self.reward, self.done, self.info

        # Execute one time step within the environment

    def reset(self):
        # Reset the state of the environment to an initial state
        self.obs = np.random.randint(N_STATES)
        self.df = self.data[self.obs]
        self.ut = math.inf
        self.done = False
        self.time_steps = N_ACTIONS
        self.reward = 0
        self.info = {}
        return self.obs

        ...

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        ...
