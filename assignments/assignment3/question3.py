from dataclasses import dataclass
from typing import Tuple, Dict, Mapping
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from rl.distribution import Categorical
from scipy.stats import poisson

@dataclass(frozen=True)
class JobState:
    current: int
    next: int #0: not lose job. 1,2,..., n: lose and offered job i

JobActionMapping = Mapping[
    JobState,
    Mapping[int, Categorical[Tuple[JobState, float]]]
]

class JobHoppingMRPFinite(FiniteMarkovDecisionProcess[JobState, int]):

    def __init__(
        self,
        num_job,
        alpha,
        job_offer_prob,
        wages,
    ):
        self.num_job: int = num_job
        self.alpha: int = alpha
        self.p: list = job_offer_prob #n+1
        self.wages: list = wages #n+1

        super().__init__(self.get_action_transition_reward_map())

    #part e) create transition reward map
    def get_action_transition_reward_map(self) -> JobActionMapping:
        d: Dict[JobState, Dict[int, Categorical[Tuple[JobState, float]]]] = {}
        for i in range(self.num_job+1):
            for j in range(self.num_job+1):
                if i == 0 and j == 0:
                    continue
                state = JobState(i, j)

                d1: Dict[int, Categorical[Tuple[JobState, float]]] = {}
                a1_sr_prob: Dict[Tuple[JobState, float], float] = {}
                a2_sr_prob: Dict[Tuple[JobState, float], float] = {}
                if i == 0: #currently unemployed
                    if j != 0:
                        # offered with job j and accept, next day lose and offered with job k
                        a1_sr_prob = {(JobState(j, k), self.wages[j]): self.p[j] * self.alpha * self.p[k] for k in range(1, self.num_job+1)}
                        # offered with job j and accept, next day does not lose job j
                        a1_sr_prob[(JobState(j, 0), self.wages[j])] = self.p[j]*(1-self.alpha)
                        # offered with job j and decline, next day offered with job k
                        a2_sr_prob = {(JobState(0, k), self.wages[0]): self.p[j] * self.p[k] for k in range(1, self.num_job+1)}
                else: #currently employed with job i
                    if j == 0: #does not lose job i today
                        #does not lose job i the next day
                        a1_sr_prob[(JobState(i, 0), self.wages[i])] = (1-self.alpha)**2
                        #lose job i and offered new job k the next day
                        for k in range(1, self.num_job+1):
                            a1_sr_prob[(JobState(i, k), self.wages[i])] = (1-self.alpha) * self.alpha * self.p[k]
                    else: #lose job i today and offered job j
                        #accept job j and does not lose job j the next day
                        a1_sr_prob[(JobState(j, 0), self.wages[j])] = self.alpha * self.p[j]* (1-self.alpha)
                        for k in range(1, self.num_job+1):
                            # accept job j, lose job j and offered job k the next day
                            a1_sr_prob[(JobState(j, k), self.wages[j])] = self.alpha * self.p[j] * self.alpha * self.p[k]
                            #decline job j, and offered a new job k next day
                            a2_sr_prob[(JobState(0, k), self.wages[0])] = self.alpha * self.p[j] * self.p[k]
                d1[1] = Categorical(a1_sr_prob)
                d1[0] = Categorical(a2_sr_prob)

                d[state] = d1
        return d

if __name__ == '__main__':
    from pprint import pprint

    n_job = 5
    alpha = 0.4
    job_offer_prob = [0, 0.2, 0.1, 0.3, 0.25, 0.15]
    wages = [10, 15, 20, 8, 16, 18]

    user_gamma = 0.9

    si_mdp: FiniteMarkovDecisionProcess[JobState, int] =\
        JobHoppingMRPFinite(n_job, alpha, job_offer_prob, wages)

    print("MDP Transition Map")
    print("------------------")
    print(si_mdp)

    from rl.dynamic_programming import value_iteration_result
    print("MDP Value Iteration Optimal Value Function and Optimal Policy")
    print("--------------")
    opt_vf_vi, opt_policy_vi = value_iteration_result(si_mdp, gamma=user_gamma)
    pprint(opt_vf_vi)
    print(opt_policy_vi)
    print()