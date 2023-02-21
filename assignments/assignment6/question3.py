from typing import Iterable, Iterator, Callable, Mapping, TypeVar
from rl.distribution import Choose
from rl.function_approx import Tabular, learning_rate_schedule
import rl.markov_process as mp
import rl.td as td
from rl.markov_process import TransitionStep, FiniteMarkovRewardProcess
from rl.markov_decision_process import NonTerminal
from rl.returns import returns
import rl.iterate as iterate
import itertools
from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite, InventoryState
from rl.approximate_dynamic_programming import NTStateDistribution, ValueFunctionApprox
import rl.monte_carlo as mc
from rl.chapter10.prediction_utils import unit_experiences_from_episodes, fmrp_episodes_stream
from pprint import pprint

S = TypeVar('S')

def mc_prediction_tabular(
    values_map: Mapping[S, float],
    counts_map: Mapping[S, int],
    traces: Iterable[Iterable[mp.TransitionStep[S]]],
    count_to_weight_func: Callable[[int], float],
    γ: float,
    episode_length_tolerance: float = 1e-6
) -> Iterator[Mapping[S, float]]:
    episodes: Iterator[Iterator[mp.ReturnStep[S]]] = \
        (returns(trace, γ, episode_length_tolerance) for trace in traces)
    for episode in episodes:
        for step in episode:
            s = step.state
            y = step.return_
            counts_map[s] = counts_map.get(s, 0) + 1
            weight: float = count_to_weight_func(counts_map[s])
            values_map[s] = weight * y + (1 - weight) * values_map.get(s, 0.)
        yield values_map

def td_prediction_tabular(
    values_map: Mapping[NonTerminal[S], float],
    counts_map: Mapping[NonTerminal[S], int],
    learning_rate: Callable[[int], float],
    transitions: Iterable[mp.TransitionStep[S]],
    γ: float
) -> Iterator[Mapping[NonTerminal[S], float]]:
    for step in transitions:
        s = step.state
        r = step.reward
        values_map[s] += learning_rate(counts_map.get(s, 0))*(r + γ * values_map.get(step.next_state, 0.) - values_map.get(s, 0.))
        yield values_map


def MC_prediction_comparison(
    fmrp: FiniteMarkovRewardProcess[S],
    gamma: float,
    num_episodes: int,
    initial_vf_dict: Mapping[NonTerminal[S], float],
    tabular: bool
):
    start_state_distribution: NTStateDistribution[S] = Choose(fmrp.non_terminal_states)
    traces: Iterable[Iterable[TransitionStep[S]]] = \
        fmrp.reward_traces(start_state_distribution)
    if tabular:
        mc_tab_vfs: Iterator[Mapping[NonTerminal[S], float]] = \
            mc_prediction_tabular(values_map=initial_vf_dict,
                                  counts_map={},
                                  traces=traces,
                                  count_to_weight_func=lambda n: 1.0/n,
                                  γ=gamma)
        final_tab_vf: Mapping[NonTerminal[S], float] = \
            iterate.last(itertools.islice(mc_tab_vfs, num_episodes))
        pprint(final_tab_vf)
    else:
        mc_vfs: Iterator[ValueFunctionApprox[S]] = \
            mc.mc_prediction(traces=traces, γ=gamma, approx_0=Tabular())
        final_vf: ValueFunctionApprox[S] = \
            iterate.last(itertools.islice(mc_vfs, num_episodes))
        pprint({s: final_vf(s) for s in fmrp.non_terminal_states})

def TD_prediction_comparison(
    fmrp: FiniteMarkovRewardProcess[S],
    gamma: float,
    episode_length: int,
    num_episodes: int,
    initial_learning_rate: float,
    half_life: float,
    exponent: float,
    initial_vf_dict: Mapping[NonTerminal[S], float],
    tabular: bool
) -> None:
    episodes: Iterable[Iterable[TransitionStep[S]]] = \
        fmrp_episodes_stream(fmrp)
    td_experiences: Iterable[TransitionStep[S]] = \
        unit_experiences_from_episodes(
            episodes,
            episode_length
        )
    learning_rate_func: Callable[[int], float] = learning_rate_schedule(
        initial_learning_rate=initial_learning_rate,
        half_life=half_life,
        exponent=exponent
    )
    if tabular:
        td_tab_vfs: Iterator[Mapping[NonTerminal[S], float]] = \
            td_prediction_tabular(values_map=initial_vf_dict,
                              counts_map={},
                              learning_rate=learning_rate_func,
                              transitions=td_experiences,
                              γ=gamma
                              )
        final_tab_vf: Mapping[NonTerminal[S], float]= \
            iterate.last(itertools.islice(td_tab_vfs, episode_length * num_episodes))
        pprint(final_tab_vf)
    else:
        td_vfs: Iterator[ValueFunctionApprox[S]] = td.td_prediction(
            transitions=td_experiences,
            approx_0=Tabular(
                values_map=initial_vf_dict,
                count_to_weight_func=learning_rate_func
            ),
            γ=gamma)
        final_td_vf: ValueFunctionApprox[S] = \
            iterate.last(itertools.islice(td_vfs, episode_length * num_episodes))

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

#MC_prediction using Tabular implementation
print("---MC Prediction using Tabular implementation---------")
MC_prediction_comparison(fmrp=si_mrp,
                        gamma=user_gamma,
                        num_episodes=num_episodes,
                        initial_vf_dict=initial_vf_dict,
                        tabular=True)


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

#TD prediction with Tabular
print("---TD Prediction using Tabular Implementation---------")
TD_prediction_comparison(fmrp=si_mrp,
                        gamma=user_gamma,
                        episode_length=td_episode_length,
                        num_episodes=num_episodes,
                        initial_learning_rate=initial_learning_rate,
                        half_life=half_life,
                        exponent=exponent,
                        initial_vf_dict=initial_vf_dict,
                        tabular=True)
