import numpy as np
from abc import ABCMeta, abstractmethod
from random import choice
from math import sqrt
from sets import Set
import argparse
from tqdm import tqdm

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

    def reward(self):
        return np.clip(np.random.normal(loc=self.arm_avgs, scale=self.reward_sigma), 0, 1) 
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

    @abstractmethod
    def __str__(self):
        pass

class RandomPolicy(Policy):
    def action(self):
        return choice(range(self.K))

    def record_reward(self, action, reward):
        pass

    def __str__(self):
        return 'Random Policy'

class Exp3Policy(Policy):
    def __init__(self, K, T):
        Policy.__init__(self, K)
        self.loss = np.zeros(K) # \tilde{L}_{i,t}
        self.learning_rate = sqrt(np.log(K) / (T * K)) # ?????
        self.p = np.ones(K) / float(K)

    def action(self):
        exp = np.exp(-self.learning_rate * self.loss)
        self.p = exp / np.sum(exp)
        return np.random.choice(range(self.K), p=self.p)

    def record_reward(self, action, reward):
        self.loss[action] += (1 - reward) / self.p[action] # Is this correct ?????

    def __str__(self):
        return 'Exp3 Policy'

class BOBPolicy(Policy):
    def __init__(self, K):
        Policy.__init__(self, K)

    def action(self):
        raise NotImplementedError()

    def record_reward(self, action, reward):
        raise NotImplementedError()

    def __str__(self):
        return 'BOB Policy'

class SAOPolicy(Policy):
    # Page 42.17 from http://proceedings.mlr.press/v23/bubeck12b/bubeck12b.pdf
    def __init__(self, K, T):
        Policy.__init__(self, K)
        self.t = 1
        self.beta = np.sqrt(12*np.log(T)) # bottom of 42.14  Not sure if this is correct
        self.predicted_adversarial = False
        self.exp3_policy = Exp3Policy(K, T) # In case env is adversarial
        self.p = np.ones(K) / float(K) # probability of picking arm i
        self.q = np.ones(K) / float(K) # probability of arm i at the time that it's deactivated
        self.tau = np.ones(K) * T
        self.A, self.B = Set(range(K)), Set() # set of arms to be considered, set to not be considered

        self.g_tilde = np.zeros(K) # middle of page 42.5.  E[g_tilde] = g
        self.g_hat = np.zeros(K) # middle of page 42.5.  arm's cumulative reward for arm i
        self.T = np.zeros(K) # number of times you pulled arm i

    def action(self):
        if self.predicted_adversarial:
            self.recent_action = self.exp3_policy.action()
        else:
            p = self.p / np.sum(self.p)
            # print 'prob', p
            self.recent_action = np.random.choice(range(self.K), p=p)
        self.T[self.recent_action] += 1
        return self.recent_action
        
    def record_reward(self, action, reward):
        self.exp3_policy.record_reward(action, reward)
        if self.predicted_adversarial:
            return

        K = self.K
        t = self.t
        self.g_tilde[action] += reward / self.p[action] # Is this correct ?????
        self.g_hat[action] += reward

        log_b = np.log(self.beta)
        max_h_tilde = np.max(self.g_tilde[list(self.A)]) / t

        rhs_17 = 6*sqrt(
            (4*K*log_b / t) + \
            5 * (K * log_b / t)**2
        )
        for i in range(K):
            h_tilde = self.g_tilde[i] / t
            h_hat = self.g_hat[i] / self.T[i]
            if i in self.A and (max_h_tilde - h_tilde > rhs_17): # (17)
                self.B.add(i)
                self.A.remove(i) # Deactivate arm i
                self.tau[i] = t
                self.q[i] = self.p[i]

            t_star = min(self.tau[i], t)
            
            condition_18 = np.abs(h_tilde - h_hat) > sqrt(2*log_b / (self.T[i] + 1e-7)) + \
                sqrt(4*( K*t_star/t**2 + (t-t_star)/(self.q[i]*self.tau[i]*t) )*log_b + 5*(K*log_b/t_star)**2)
            condition_19 = i not in self.A and \
                max_h_tilde - h_tilde > 10*sqrt(4*K*log_b/(self.tau[i] - 1) + 5*(K*log_b/(self.tau[i] - 1))**2)
            condition_20 = i not in self.A and \
                max_h_tilde - h_tilde <= 2 *sqrt(4*K*log_b/self.tau[i] + 5*(K*log_b/self.tau[i])**2)

            if condition_18 or condition_19 or condition_20: # use Exp3.P
                self.predicted_adversarial = True
                print h_tilde, h_hat
                print 'PREDICTED ADVERSARIAL.  CONDITIONS: ', condition_18, condition_19, condition_20
                break

            # print self.q[i]
            # print self.A, self.B
            # Update Probabilities 
            if i in self.A:
                self.q[i] = self.p[i]
                all_others = np.sum([self.q[j] * self.tau[j] / (t + 1) for j in self.B])
                self.p[i] = (1 - all_others) / len(self.A) # (21)
            else:
                self.p[i] = self.q[i] * self.tau[i] / (t + 1)
        self.t += 1

    def __str__(self):
        return 'SAO Policy'

def run_bandit(policies, env, T):
    total_reward = [0] * len(policies)
    for _ in tqdm(range(T)):
        reward = env.reward()
        for i, policy in enumerate(policies):
            action = policy.action()
            policy.record_reward(action, reward[action])
            total_reward[i] += reward[action]

    best_in_hindsight = T*np.max(env.arm_avgs)
    print 'Arm Averages: {}\n'.format(env.arm_avgs)
    
    for policy, reward in zip(policies, total_reward):
        regret = best_in_hindsight - reward
        print 'Policy: {}'.format(policy)
        print 'Regret: {} - {} = {}\n'.format(best_in_hindsight, reward, regret)
        # print 'Final Loss: {}\n'.format(policy.loss)
        # print 'Regret is sublinear!' if regret / T < 1 else 'Regret is not sublinear :('

    print 'g_tilde', policies[1].g_tilde / T
    print 'p', policies[1].p

if __name__ == "__main__":
    # How to use: python main.py --T 1000000 --policy Random SAO
    # All arguments are optional
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', '--list', default=['Exp3', 'SAO'], help='which policy to run')
    parser.add_argument('--env', type=str, default='Sto', help='which env to use')
    parser.add_argument('--K', type=int, default=10, help='how many arms')
    parser.add_argument('--p_adversarial', type=float, default=1, help='probability env is adversarial')
    parser.add_argument('--T', type=int, default=100000, help='time horizon')
    parser.add_argument('--reward_sigma', type=float, default=.1666, help='')
        # I picked sigma kinda at random.  I said if mu=.5, 3 stddev should cover [0,1]
    args = parser.parse_args()

    policies = []
    for policy in args.policy:
        if policy == 'Random':
            policies.append(RandomPolicy(args.K))
        elif policy == 'Exp3':
            policies.append(Exp3Policy(args.K, args.T))
        elif policy == 'BOB':
            policies.append(BOBPolicy(args.K))
        elif policy == 'SAO':
            policies.append(SAOPolicy(args.K, args.T))
        else:        
            raise Exception('Policy {} not recognized'.format(policy))

    if args.env == 'Sto':
        env = StochasticEnv(args.K, args.reward_sigma)
    elif args.env == 'Adv':
        env = AdversarialEnv(args.K, args.p_adversarial, policy)
    else:
        raise Exception('Env not recognized')
    run_bandit(policies, env, args.T)