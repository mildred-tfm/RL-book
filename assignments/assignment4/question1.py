from typing import Iterator, Tuple, TypeVar, Sequence, List, Dict
from operator import itemgetter
import numpy as np

from rl.approximate_dynamic_programming import evaluate_mrp, extended_vf
from rl.distribution import Distribution
from rl.function_approx import FunctionApprox
from rl.iterate import iterate
from rl.markov_process import (FiniteMarkovRewardProcess, MarkovRewardProcess,
                               RewardTransition, NonTerminal, State)
from rl.markov_decision_process import (FiniteMarkovDecisionProcess,
                                        MarkovDecisionProcess,
                                        StateActionMapping)
from rl.policy import FinitePolicy, FiniteDeterministicPolicy

S = TypeVar('S')
A = TypeVar('A')


ValueFunctionApprox = FunctionApprox[NonTerminal[S]]
QValueFunctionApprox = FunctionApprox[Tuple[NonTerminal[S], A]]
NTStateDistribution = Distribution[NonTerminal[S]]


def policy_iteration(
        mdp: MarkovDecisionProcess[S, A],
        γ: float,
        approx_0: ValueFunctionApprox[S],
        non_terminal_states_distribution: NTStateDistribution[S],
        num_state_samples: int
)-> Iterator[ValueFunctionApprox[S]]:
    def update(vf_policy: Tuple[ValueFunctionApprox[S], FinitePolicy[S, A]]) -> Tuple[ValueFunctionApprox[S], FiniteDeterministicPolicy[S,A]]:
        nt_states: Sequence[NonTerminal[S]] = \
            non_terminal_states_distribution.sample_n(num_state_samples)
        vf, pi = vf_policy
        mrp: FiniteMarkovRewardProcess[S] = mdp.apply_finite_policy(pi)
        mrp_fa = iterate.converged(
            evaluate_mrp(
                mrp,
                γ,
                approx_0,
                non_terminal_states_distribution,
                num_state_samples
            ),
            done=lambda a, b: a.within(b, 1e-4)
        )
        policy_vf: ValueFunctionApprox[S] = mrp_fa.evaluate(nt_states)
        greedy_policy_dict: Dict[S, A] = {}
        for s in nt_states:
            q_values: Iterator[Tuple[A, float]] = \
                ((a, mdp.mapping[s][a].expectation(lambda s_r: s_r[1]+γ*extended_vf(policy_vf, s_r[0]))) for a in mdp.actions(s))
            greedy_policy_dict[s.state] = \
                max(q_values, key=itemgetter(1))[0]
        improved_pi: FiniteDeterministicPolicy[S, A] = FiniteDeterministicPolicy(greedy_policy_dict)
        return policy_vf, improved_pi
    return iterate(update, approx_0)
