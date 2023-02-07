from typing import Sequence, Mapping
import unittest

import numpy as np

from rl.approximate_dynamic_programming import (value_iteration_finite, value_iteration)
from rl.dynamic_programming import value_iteration_result
from rl.distribution import Choose
from rl.function_approx import Dynamic
import rl.iterate as iterate
from rl.iterate import converged

from rl.markov_process import FiniteMarkovRewardProcess, NonTerminal
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy

from rl.chapter3.simple_inventory_mdp_cap import (InventoryState,
                                                  SimpleInventoryMDPCap)

from assignments.assignment4.question1 import policy_iteration, almost_equal_vf_pis

#By running the test, we observe that the optimal value function the value iteraion converges to and
#the final value function the policy iteration converges to are the same.
class TestEvaluate(unittest.TestCase):
    def setUp(self):
        user_capacity = 2
        user_poisson_lambda = 1.0
        user_holding_cost = 1.0
        user_stockout_cost = 10.0

        self.gamma = 0.9

        self.si_mdp: FiniteMarkovDecisionProcess[InventoryState, int] =\
            SimpleInventoryMDPCap(
                capacity=user_capacity,
                poisson_lambda=user_poisson_lambda,
                holding_cost=user_holding_cost,
                stockout_cost=user_stockout_cost
            )

        self.fdp: FiniteDeterministicPolicy[InventoryState, int] = \
            FiniteDeterministicPolicy(
                {InventoryState(alpha, beta): user_capacity - (alpha + beta)
                 for alpha in range(user_capacity + 1)
                 for beta in range(user_capacity + 1 - alpha)}
        )

        self.implied_mrp: FiniteMarkovRewardProcess[InventoryState] =\
            self.si_mdp.apply_finite_policy(self.fdp)

        self.states: Sequence[NonTerminal[InventoryState]] = \
            self.implied_mrp.non_terminal_states


    def test_value_iteration(self):
        mdp_map: Mapping[NonTerminal[InventoryState], float] = value_iteration_result(
            self.si_mdp,
            self.gamma
        )[0]
        # print(mdp_map)
        mdp_vf1: np.ndarray = np.array([mdp_map[s] for s in self.states])

        fa = Dynamic({s: 0.0 for s in self.states})
        mdp_finite_fa = iterate.converged(
            value_iteration_finite(
                self.si_mdp,
                self.gamma,
                fa
            ),
            done=lambda a, b: a.within(b, 1e-5)
        )
        # print(mdp_finite_fa.values_map)
        mdp_vf2: np.ndarray = mdp_finite_fa.evaluate(self.states)
        # print(mdp_vf2)
        self.assertLess(max(abs(mdp_vf1 - mdp_vf2)), 0.01)

        mdp_fa = iterate.converged(
            value_iteration(
                self.si_mdp,
                self.gamma,
                fa,
                Choose(self.states),
                num_state_samples=30
            ),
            done=lambda a, b: a.within(b, 1e-5)
        )
        print(mdp_fa.values_map)
        mdp_vf3: np.ndarray = mdp_fa.evaluate(self.states)
        # print(mdp_vf3)
        self.assertLess(max(abs(mdp_vf1 - mdp_vf3)), 0.01)


    def test_policy_iteration(self):
        mdp_map: Mapping[NonTerminal[InventoryState], float] = value_iteration_result(
            self.si_mdp,
            self.gamma
        )[0]
        fa = Dynamic({s: 0.0 for s in self.states})
        mdp_policy = policy_iteration(
                self.si_mdp,
                self.gamma,
                fa,
                Choose(self.states),
                num_state_samples=10
            )
        opt_policy_vf, opt_policy_pi = converged(mdp_policy, done=almost_equal_vf_pis)
        print(mdp_map)
        print(opt_policy_vf.values_map)

