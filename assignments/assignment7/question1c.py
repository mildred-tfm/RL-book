from assignments.assignment7.question1 import td_lambda_fa, td_lambda_tabular
from assignments.assignment6.question3 import MC_prediction_comparison, TD_prediction_comparison

from typing import Iterable, Iterator, Callable, Mapping, TypeVar
from rl.distribution import Choose
from rl.function_approx import Tabular, learning_rate_schedule
import rl.iterate as iterate
from rl.markov_process import TransitionStep
from rl.markov_decision_process import NonTerminal

import itertools
from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite, InventoryState
from rl.approximate_dynamic_programming import NTStateDistribution, ValueFunctionApprox
from pprint import pprint

S = TypeVar('S')



def TD_lmda_comparison(fmrp, user_gamma, initial_vf_dict, lr_param, num_episodes, td_episode_length, lambd, tabular):
    initial_learning_rate, half_life, exponent = lr_param
    learning_rate_func: Callable[[int], float] = learning_rate_schedule(
        initial_learning_rate=initial_learning_rate,
        half_life=half_life,
        exponent=exponent
    )
    states = fmrp.non_terminal_states
    start_state_distribution: NTStateDistribution[S] = Choose(states)
    traces: Iterable[Iterable[TransitionStep[S]]] = \
        fmrp.reward_traces(start_state_distribution)
    if tabular:
        td_lmda_vfs: Iterator[Mapping[NonTerminal[S], float]] = \
            td_lambda_tabular(values_map=initial_vf_dict,
                              counts_map={},
                              traces=traces,
                              learning_rate=learning_rate_func,
                              gamma=user_gamma,
                              lambd=lambd)
        final_tab_vf = iterate.last(itertools.islice(td_lmda_vfs, td_episode_length * num_episodes))
        pprint(final_tab_vf)
    else:
        curtailed_episodes: Iterable[Iterable[TransitionStep[S]]] = \
            (itertools.islice(episode, td_episode_length) for episode in traces)
        td_lmda_vfs: Iterator[ValueFunctionApprox[S]] = td_lambda_fa(
            traces=curtailed_episodes,
            approx_0=Tabular(
                values_map=initial_vf_dict,
                count_to_weight_func=learning_rate_func
            ),
            gamma=user_gamma,
            lambd=lambd)
        final_td_vf = iterate.last(itertools.islice(td_lmda_vfs, td_episode_length * num_episodes))
        pprint({s: final_td_vf(s) for s in fmrp.non_terminal_states})

user_capacity = 2
user_poisson_lambda = 1.0
user_holding_cost = 1.0
user_stockout_cost = 10.0

user_gamma = 0.9

si_mrp = SimpleInventoryMRPFinite(
    capacity=user_capacity,
    poisson_lambda=user_poisson_lambda,
    holding_cost=user_holding_cost,
    stockout_cost=user_stockout_cost
)


print("---DP Value Function---------")
si_mrp.display_value_function(gamma=user_gamma)
print()

initial_vf_dict: Mapping[NonTerminal[InventoryState], float] = \
        {s: 0. for s in si_mrp.non_terminal_states}

num_episodes: int = 8000

#MC prediction using Function Approximation
print("---MC Prediction using Function Approximation---------")
MC_prediction_comparison(fmrp=si_mrp,
                        gamma=user_gamma,
                        num_episodes=num_episodes,
                        initial_vf_dict=initial_vf_dict,
                        tabular=False)


td_episode_length: int = 100
initial_learning_rate: float = 0.03
half_life: float = 1000.0
exponent: float = 0.5

#TD prediction using Function Approximation
print("---TD Prediction using Function Approximation---------")
TD_prediction_comparison(fmrp=si_mrp,
                        gamma=user_gamma,
                        episode_length=td_episode_length,
                        num_episodes=num_episodes,
                        initial_learning_rate=initial_learning_rate,
                        half_life=half_life,
                        exponent=exponent,
                        initial_vf_dict=initial_vf_dict,
                        tabular=False)

print("---TD Lambda Prediction using Tabular---------")
TD_lmda_comparison(fmrp=si_mrp,
                   user_gamma=user_gamma,
                   initial_vf_dict=initial_vf_dict,
                   lr_param=(0.03, 1e5, 0.5),
                   num_episodes=num_episodes,
                   td_episode_length=td_episode_length,
                   lambd=0.5,
                   tabular=True
                   )

print("---TD Lambda Prediction using Function Approximation---------")
TD_lmda_comparison(fmrp=si_mrp,
                   user_gamma=user_gamma,
                   initial_vf_dict=initial_vf_dict,
                   lr_param=(0.03, 1e5, 0.5),
                   num_episodes=num_episodes,
                   td_episode_length=td_episode_length,
                   lambd=0.5,
                   tabular=False
                   )