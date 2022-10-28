import numpy as np
from utils import create_qtable, max_action, epsilon_dec


def sarsa(env, n_games, n_states, n_actions, eps, alpha, gamma):
    Q = create_qtable(n_states, n_actions)
    totalRewards = np.zeros(n_games)
    for i in range(n_games):
        if i % 100 == 0:
            print('starting game', i)
        observation = env.reset()
        rand = np.random.random()
        action = max_action(Q, observation, n_actions) if rand < (
            1 - eps) else env.action_space.sample()
        done = False
        epRewards = 0
        while not done:
            # env.render()
            observation_, reward, done, info = env.step(action)
            epRewards += reward
            rand = np.random.random()
            action_ = max_action(Q, observation_, n_actions) if rand < (
                1 - eps) else env.action_space.sample()
            Q[observation, action] = Q[observation, action] + alpha * \
                (reward + gamma * Q[observation_,
                                    action_] - Q[observation, action])
            observation, action = observation_, action_
        eps, eps_end = epsilon_dec(eps=eps)
        totalRewards[i] = epRewards
        if i % 100 == 0:
            print('total reward = %s and eps = %s' %
                  (totalRewards[i], eps))
    return totalRewards, eps_end
