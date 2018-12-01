import numpy as np
from abc import ABCMeta, abstractmethod
from random import choice
from math import sqrt
from sets import Set
import argparse

class Env(object):
    __metaclass__ = ABCMeta

    def __init__(self, K):
        self.K = K
        self.arm_avgs = np.random.random_sample(K)

    @abstractmethod
    def reward(self):
        pass

class StochasticEnv(Env):
    def __init__(self, K, reward_sigma):
        Env.__init__(self, K)
        self.reward_sigma = reward_sigma # each reward is normally distributed with parameters (arm_avgs[i], reward_sigma)

    def reward(self, action):
        return np.clip(np.random.normal(loc=self.arm_avgs[action], scale=self.reward_sigma), 0, 1) 
        # should we be calculating this ^ differently? 

class AdversarialEnv(Env):
    def __init__(self, K, p_adversarial, policy): 
        Env.__init__(K)
        self.p_adversarial = p_adversarial # if 1, this is complete adversarial env, if 0, complete stochastic
        self.policy = policy # different adversaries for different policies

    def reward(self):
        raise NotImplementedError()




class Policy(object):
    __metaclass__ = ABCMeta

    def __init__(self, K=10):
        self.K = K

    @abstractmethod
    def action(self):
        pass

    @abstractmethod
    def record_reward(self, action, reward):
        pass

class RandomPolicy(Policy):
    def action(self):
        return choice(range(self.K))

    def record_reward(self, action, reward):
        pass

class Exp3Policy(Policy):
    def __init__(self, K, T):
        Policy.__init__(self, K)
        self.loss = np.zeros(K) # \tilde{L}_{i,t}
        self.learning_rate = sqrt(np.log(K) / (T * K)) # ?????

    def action(self):
        exp = np.exp(-self.learning_rate * self.loss)
        self.p = exp / np.sum(exp)
        return np.random.choice(range(self.K), p=self.p)

    def record_reward(self, action, reward):
        self.loss[action] += (1 - reward) * self.p[action] # Is this correct ?????

class BOBPolicy(Policy):
    def __init__(self, K):
        Policy.__init__(self, K)

    def action(self):
        raise NotImplementedError()

    def record_reward(self, action, reward):
        raise NotImplementedError()

class SAOPolicy(Policy):
    # Page 42.17 from http://proceedings.mlr.press/v23/bubeck12b/bubeck12b.pdf
    def __init__(self, K, T):
        Policy.__init__(K)
        # self.T = T
        self.t = 0
        self.beta = np.sqrt(12*np.log(T)) # bottom of 42.14  Not sure if this is correct
        self.predicted_adversarial = False
        self.exp3_policy = Exp3Policy(K, T)
        self.p = np.ones(K) / float(K) # probability of picking arm i
        self.q = np.ones(K) / float(K) # probability of arm i when it's deactivated
        self.tau = np.ones(K) * T
        self.A, self.B = Set(range(K)), Set() # set of arms to be considered, set to not be considered

        self.g_tilde = np.zeros(K) # middle of page 42.5
        self.g_hat = np.zeros(K)
        self.T = np.zeros(K)

    def action(self):
        if self.predicted_adversarial:
            self.recent_action = exp3_policy.action()
        else:
            self.recent_action = np.random.choice(range(self.K), p=self.p)
        return self.recent_action
        
    def record_reward(self, action, reward):
        exp3_policy.record_reward(action, reward)
        if self.predicted_adversarial:
            return

        K = self.K
        self.g_tilde[action] += reward * self.p[action] # Is this correct ?????
        self.g_hat[action] += reward

        log_b = np.log(self.beta)
        max_h_tilde = np.max(g_tilde[list(A)]) / t
        rhs_17 = 6*sqrt(
            (4*K*log_b / t) + \
            5 * (K * log_b / t)**2
        )
        for i in range(K):
            if i in self.A and (max_h_tilde - g_tilde[i] / t > rhs_17): # (17)
                self.B.add(self.A.remove(i)) # Deactivate arm i
                self.tau[i] = t
                self.q[i] = self.p[i]

            t_star = min(self.tau[i], t)
            
            condition_18 = np.abs(self.g_tilde - self.g_hat) / t > sqrt(2*log_b / self.T[i]) + \
                sqrt(4*( K*t_star/self.t**2 + (t-t_star)/(self.q[i]*self.tau[i]*self.t) )*log_b + 5*(K*log_b/t_star)**2)
            condition_19 = i not in A and \
                max_h_tilde - self.g_tilde[i] / t > 10*sqrt(4*K*log_b/(self.tau[i] - 1) + 5*(K*log_b/(self.tau[i] - 1))**2)
            condition_20 = i not in A and \
                max_h_tilde - self.g_tilde[i] / t > 2 *sqrt(4*K*log_b/self.tau[i] + 5*(K*log_b/self.tau[i])**2)
            if condition_18 or condition_19 or condition_20: # use Exp3.P
                self.predicted_adversarial = True
                break

            # Update Probabilities 
            if i in self.A:
                all_others = np.sum([self.q[j] * self.tau[j] / (self.t + 1) for j in B])
                p[i] = (1 - all_others) / len(A) # (21)
            else:
                p[i] = self.q[i] * self.tau[i] / (self.t + 1)
        self.t += 1


def run_bandit(policy, env, T):
    total_reward = 0
    for _ in range(T):
        action = policy.action()
        reward = env.reward(action)
        policy.record_reward(action, reward)
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
    parser.add_argument('--K', type=int, default=10, help='how many arms')
    parser.add_argument('--p_adversarial', type=float, default=1, help='probability env is adversarial')
    parser.add_argument('--T', type=int, default=10000, help='time horizon')
    parser.add_argument('--reward_sigma', type=float, default=.1666, help='')
        # I picked sigma kinda at random.  I said if mu=.5, 3 stddev should cover [0,1]
    args = parser.parse_args()

    if args.policy == 'Random':
        policy = RandomPolicy(args.K)
    elif args.policy == 'Exp3':
        policy = Exp3Policy(args.K, args.T)
    elif args.policy == 'BOB':
        policy = BOBPolicy(args.K)
    elif args.policy == 'SAO':
        policy = SAOPolicy(args.K, args.T)
    else:        
        raise Exception('Policy not recognized')

    if args.env == 'Sto':
        env = StochasticEnv(args.K, args.reward_sigma)
    elif args.env == 'Adv':
        env = AdversarialEnv(args.K, args.p_adversarial, policy)
    else:
        raise Exception('Env not recognized')
    run_bandit(policy, env, args.T)