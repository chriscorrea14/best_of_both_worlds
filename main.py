import numpy as np
from abc import ABCMeta, abstractmethod
from random import choice
from math import sqrt
import argparse

class Env(object):
    __metaclass__ = ABCMeta

    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.arm_avgs = np.random.random_sample(n_arms)

    @abstractmethod
    def reward(self):
        pass

class StochasticEnv(Env):
    def __init__(self, n_arms, reward_sigma):
        Env.__init__(self, n_arms)
        self.reward_sigma = reward_sigma # each reward is normally distributed with parameters (arm_avgs[i], reward_sigma)

    def reward(self, action):
        return np.clip(np.random.normal(loc=self.arm_avgs[action], scale=self.reward_sigma), 0, 1) 

class AdversarialEnv(Env):
    def __init__(self, n_arms, p_adversarial, policy): 
        Env.__init__(n_arms)
        self.p_adversarial = p_adversarial # if 1, this is complete adversarial env, if 0, complete stochastic
        self.policy = policy # different adversaries for different policies

    def reward(self):
        raise NotImplementedError()




class Policy(object):
    __metaclass__ = ABCMeta

    def __init__(self, n_arms=10):
        self.n_arms = n_arms

    @abstractmethod
    def action(self):
        pass

    @abstractmethod
    def record_reward(self, reward):
        pass

class RandomPolicy(Policy):
    def action(self):
        return choice(range(self.n_arms))

    def record_reward(self, reward):
        pass

class Exp3Policy(Policy):
    def __init__(self, n_arms, T):
        Policy.__init__(self, n_arms)
        self.loss = np.zeros(n_arms) # \tilde{L}_{i,t}
        self.learning_rate = sqrt(np.log(n_arms) / (T * n_arms)) # ?????
        self.recent_action = -1

    def action(self):
        exp = np.exp(-self.learning_rate * self.loss)
        self.probabilities = exp / np.sum(exp)
        self.recent_action = np.random.choice(range(self.n_arms), p=self.probabilities)
        return self.recent_action

    def record_reward(self, reward):
        action = self.recent_action
        self.loss[action] += (1 - reward) * self.probabilities[action] # Is this correct ?????

class BOBPolicy(Policy):
    def __init__(self, n_arms):
        Policy.__init__(self, n_arms)

    def action(self):
        raise NotImplementedError()

    def record_reward(self, reward):
        raise NotImplementedError()

class SAOPolicy(Policy):
    def __init__(self, n_arms, T):
        if self.n_arms = 
        Policy.__init__(n_arms)
        self.T = T
        self.C = 12*np.log(T)
        self.exploration_phase = True

    def action(self):
        if self.exploration_phase:
            self.recent_action = if np.random.random() < 0.5


def run_bandit(policy, env, T):
    total_reward = 0
    for _ in range(T):
        action = policy.action()
        reward = env.reward(action)
            # should we be calculating this ^ differently? 
        policy.record_reward(reward)
        total_reward += reward

    best_in_hindsight = T*np.max(env.arm_avgs)
    regret = best_in_hindsight - total_reward
    print 'Regret: {} - {} = {}\n'.format(best_in_hindsight, total_reward, regret)
    print 'Final Loss: {}\n'.format(policy.loss)
    print 'Arm Averages: {}\n'.format(env.arm_avgs)
    # print 'Regret is sublinear!' if regret / T < 1 else 'Regret is not sublinear :('

if __name__ == "__main__":
    # How to use: python main.py --T 1000000 --policy Random
    # All arguments are optional
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='Exp3', help='which policy to run')
    parser.add_argument('--env', type=str, default='Sto', help='which env to use')
    parser.add_argument('--n_arms', type=int, default=10, help='how many arms')
    parser.add_argument('--p_adversarial', type=float, default=1, help='probability env is adversarial')
    parser.add_argument('--T', type=int, default=10000, help='time horizon')
    parser.add_argument('--reward_sigma', type=float, default=.1666, help='')
        # I picked sigma kinda at random.  I said if mu=.5, 3 stddev should cover [0,1]
    args = parser.parse_args()

    if args.policy == 'Random':
        policy = RandomPolicy(args.n_arms)
    elif args.policy == 'Exp3':
        policy = Exp3Policy(args.n_arms, args.T)
    elif args.policy == 'BOB':
        policy = BOBPolicy(args.n_arms)
    else:        
        raise Exception('Policy not recognized')

    if args.env == 'Sto':
        env = StochasticEnv(args.n_arms, args.reward_sigma)
    elif args.env == 'Adv':
        env = AdversarialEnv(args.n_arms, args.p_adversarial, policy)
    else:
        raise Exception('Env not recognized')
    run_bandit(policy, env, args.T)