import pandas as pd
from utils import load_data, utility, plotRunningAverage, epsilon_dec, create_qtable
import numpy as np
from sarsa import sarsa
from qlearning import qlearning
from double_qlearning import double_qlearning
from env.environment import DeltaIotEnv

N_STATES = 216
N_ACTIONS = 216


if __name__ == '__main__':
    env = DeltaIotEnv()
    # model hyperparameters
    ALPHA = 0.1
    GAMMA = 0.99
    EPS = 1.0
    numGames = 3000

    totalRewards1, eps_end_sarsa = sarsa(
        env, numGames, N_STATES, N_ACTIONS, EPS, ALPHA, GAMMA)
    totalRewards2, eps_end_qlearning = qlearning(
        env, numGames, N_STATES, N_ACTIONS, EPS, ALPHA, GAMMA)
    totalRewards3, eps_end_doubleqlearning = double_qlearning(
        env, numGames, N_STATES, N_ACTIONS, EPS, ALPHA, GAMMA)
    totalRewards = {}
    totalRewards['sarsa'] = totalRewards1
    totalRewards['qlearning'] = totalRewards2
    totalRewards['double_qlearning'] = totalRewards3

    plotRunningAverage(totalRewards, color=['r', 'g', 'b'],
                       info='episodes = %s and eps_dec_sarsa = %s and eps_end_qlearning = %s' % (numGames, eps_end_sarsa, eps_end_qlearning))
