from dataclasses import dataclass
from typing import Tuple, Dict, Mapping
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy
from rl.markov_process import FiniteMarkovRewardProcess
from rl.distribution import Categorical, Choose
from scipy.stats import poisson

from assignments.assignment4.question1 import policy_iteration, fa_policy_iteration_result
from rl.function_approx import Tabular

@dataclass(frozen=True)
class InventoryState:
    on_hand: int
    on_order: int

    def inventory_position(self) -> int:
        return self.on_hand + self.on_order


InvOrderMapping = Mapping[
    InventoryState,
    Mapping[int, Categorical[Tuple[InventoryState, float]]]
]

# We test our approximate policy iteration and the original policy iteration on this simple inventory mdp.
# The result shows that the result from approximate policy iteration using Tabular function approximation
# is the same as the result from the original policy iteration.

class SimpleInventoryMDPCap(FiniteMarkovDecisionProcess[InventoryState, int]):

    def __init__(
        self,
        capacity: int,
        poisson_lambda: float,
        holding_cost: float,
        stockout_cost: float
    ):
        self.capacity: int = capacity
        self.poisson_lambda: float = poisson_lambda
        self.holding_cost: float = holding_cost
        self.stockout_cost: float = stockout_cost

        self.poisson_distr = poisson(poisson_lambda)
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> InvOrderMapping:
        d: Dict[InventoryState, Dict[int, Categorical[Tuple[InventoryState,
                                                            float]]]] = {}

        for alpha in range(self.capacity + 1):
            for beta in range(self.capacity + 1 - alpha):
                state: InventoryState = InventoryState(alpha, beta)
                ip: int = state.inventory_position()
                base_reward: float = - self.holding_cost * alpha
                d1: Dict[int, Categorical[Tuple[InventoryState, float]]] = {}

                for order in range(self.capacity - ip + 1):
                    sr_probs_dict: Dict[Tuple[InventoryState, float], float] =\
                        {(InventoryState(ip - i, order), base_reward):
                         self.poisson_distr.pmf(i) for i in range(ip)}

                    probability: float = 1 - self.poisson_distr.cdf(ip - 1)
                    reward: float = base_reward - self.stockout_cost *\
                        (probability * (self.poisson_lambda - ip) +
                         ip * self.poisson_distr.pmf(ip))
                    sr_probs_dict[(InventoryState(0, order), reward)] = \
                        probability
                    d1[order] = Categorical(sr_probs_dict)

                d[state] = d1
        return d


if __name__ == '__main__':
    from pprint import pprint

    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    user_gamma = 0.9

    si_mdp: FiniteMarkovDecisionProcess[InventoryState, int] =\
        SimpleInventoryMDPCap(
            capacity=user_capacity,
            poisson_lambda=user_poisson_lambda,
            holding_cost=user_holding_cost,
            stockout_cost=user_stockout_cost
        )

    print("MDP Transition Map")
    print("------------------")
    print(si_mdp)

    fdp: FiniteDeterministicPolicy[InventoryState, int] = \
        FiniteDeterministicPolicy(
            {InventoryState(alpha, beta): user_capacity - (alpha + beta)
             for alpha in range(user_capacity + 1)
             for beta in range(user_capacity + 1 - alpha)}
    )

    implied_mrp: FiniteMarkovRewardProcess[InventoryState] =\
        si_mdp.apply_finite_policy(fdp)

    from rl.dynamic_programming import policy_iteration_result

    print("MDP Policy Iteration Optimal Value Function and Optimal Policy")
    print("--------------")
    opt_vf_pi, opt_policy_pi = policy_iteration_result(
        si_mdp,
        gamma=user_gamma
    )
    pprint(opt_vf_pi)
    print(opt_policy_pi)
    print()

    print("MDP Approximate Policy Iteration Optimal Value Function and Optimal Policy")
    print("--------------")
    tabular = Tabular({s: 0.0 for s in implied_mrp.non_terminal_states}, count_to_weight_func=lambda _: 1)
    non_terminal_distribution = Choose(implied_mrp.non_terminal_states)
    fa_policy_vf, fa_policy_pi = fa_policy_iteration_result(si_mdp, user_gamma, tabular, non_terminal_distribution, 10)
    pprint(fa_policy_vf)
    print(fa_policy_pi)
    print()


