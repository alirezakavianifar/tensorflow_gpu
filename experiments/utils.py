from cProfile import label
import os
from turtle import color
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import json

PATH = r'D:\projects\papers\Deep Learning for Effective and Efficient  Reduction of Large Adaptation Spaces in Self-Adaptive Systems\DLASeR_plus_online_material\dlaser_plus\raw\DeltaIoTv1'


def plotRunningAverage(totalrewards, color, info=None):
    i = 0
    for key, value in totalrewards.items():
        N = len(value)
        running_avg = np.empty(N)
        for t in range(N):
            running_avg[t] = np.mean(value[max(0, t - 100):(t + 1)])
        plt.plot(running_avg, label=key, color=color[i])
        i += 1
    # plt.plot(running_avg1, 'g*', running_avg2, 'ro', label=['sarsa', 'qlearning'])
    plt.xlabel('timesteps')
    plt.ylabel('average reward')
    plt.title("Running Average")
    plt.legend()
    plt.savefig('sarsa-vs-qlearning-vs-double_qlearning-%s.png' % info)
    plt.show()


def load_data(path=PATH, load_all=False):
    features = []
    packetLoss = []
    latency = []
    energyConsumption = []
    version = 'DeltaIoTv1'

    for i in range(300):
        with open(os.path.join(PATH, version,  f'dataset_with_all_features{i + 1}.json')) as f:
            data = json.load(f)

            features.extend(data['features'])
            packetLoss.extend(data['packetloss'])
            latency.extend(data['latency'])
            energyConsumption.extend(data['energyconsumption'])

            del data
    print("data is loaded successfuly")
    return features, (packetLoss, latency, energyConsumption)


def load_dataV1(path=PATH, load_all=False):

    json_files = glob.glob(os.path.join(path, "*.json"))

    json_lst = []

    for f in json_files:
        df = pd.read_json(f)
        json_lst.append(df)

    if load_all:
        # Merge all dataframes into one
        final_df = pd.concat(json_lst)
        return final_df

    return json_lst


def choose_randomly(lst):
    index = np.random.randint(len(lst))
    return lst[index]


def utility(imp1, imp2, energy_consumption, packet_loss):

    return -(imp1 * energy_consumption + imp2 * packet_loss)


def epsilon_dec(eps, eps_dec=0.001, eps_end=0.01):

    if eps > eps_end:
        eps = eps - eps_dec
    else:
        eps = eps_end
    return eps, eps_end


def create_qtable(N_STATES, N_ACTIONS):
    Q = {}

    for i in range(N_STATES):
        for j in range(N_ACTIONS):
            Q[i, j] = 0

    return Q


def max_action(Q, state, n_actions):
    values = np.array([Q[state, a] for a in range(n_actions)])
    action = np.argmax(values)
    return action


def max_action_v2(Q1, Q2, state, n_actions):
    values = np.array([Q1[state, a] + Q2[state, a] for a in range(n_actions)])
    action = np.argmax(values)
    return action


def packetloss_threshold(a):
    if a > 10:
        return 0
    else:
        return 1


def energyconsumption_threshold(a):
    if a > 12.70:
        return 0
    else:
        return 1
