import numpy as np
from utils import create_qtable, max_action_v2, epsilon_dec


def double_qlearning(env, n_games, n_states, n_actions, eps, alpha, gamma):
    Q1 = create_qtable(n_states, n_actions)
    Q2 = create_qtable(n_states, n_actions)
    totalRewards = np.zeros(n_games)
    for i in range(n_games):
        if i % 100 == 0:
            print('starting game', i)
        done = False
        epRewards = 0
        observation = env.reset()
        while not done:
            rand = np.random.random()
            action = max_action_v2(Q1, Q2, observation, n_actions) if rand < (
                1-eps) else env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            epRewards += reward
            rand = np.random.random()
            if rand <= 0.5:
                action_ = max_action_v2(Q1, Q1, observation_, n_actions)
                Q1[observation, action] = Q1[observation, action] + alpha * \
                    (reward + gamma*Q2[observation_, action_] -
                     Q1[observation, action])
            elif rand > 0.5:
                action_ = max_action_v2(Q2, Q2, observation_, n_actions)
                Q2[observation, action] = Q2[observation, action] + alpha * \
                    (reward + gamma*Q1[observation_, action_] -
                     Q2[observation, action])
            observation = observation_
        eps, eps_end = epsilon_dec(eps=eps)
        totalRewards[i] = epRewards
        if i % 100 == 0:
            print('total reward = %s and eps = %s' %
                  (totalRewards[i], eps))
    return totalRewards, eps_end
