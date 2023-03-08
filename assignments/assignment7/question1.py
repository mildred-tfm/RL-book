from typing import Iterable, Iterator, Callable, Mapping, TypeVar
from rl.distribution import Choose
from rl.function_approx import Tabular, learning_rate_schedule
import rl.markov_process as mp

from rl.markov_process import TransitionStep
from rl.markov_decision_process import NonTerminal

import itertools
from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite, InventoryState
from rl.approximate_dynamic_programming import NTStateDistribution, ValueFunctionApprox

import numpy as np
import matplotlib.pyplot as plt

from rl.approximate_dynamic_programming import extended_vf
from rl.function_approx import Gradient

S = TypeVar('S')


def td_lambda_tabular(
    values_map: Mapping[NonTerminal[S], float],
    counts_map: Mapping[NonTerminal[S], int],
    traces: Iterable[Iterable[mp.TransitionStep[S]]],
    learning_rate: Callable[[int], float],
    gamma: float,
    lambd: float
) -> Iterator[Mapping[NonTerminal[S], float]]:
    values_map = values_map
    yield values_map

    for trace in traces:
        eltr_map: Mapping[NonTerminal[S], float] = {}
        for step in trace:
            s = step.state
            r = step.reward
            for k, v in eltr_map.items():
                eltr_map[k] = gamma * lambd * v
            eltr_map[s] = eltr_map.get(s, 0) + 1
            values_map[s] += learning_rate(counts_map.get(s, 0)) * (
                        r + gamma * values_map.get(step.next_state, 0.) - values_map.get(s, 0.)) * eltr_map.get(s, 0)
            yield values_map

def td_lambda_fa(
    traces: Iterable[Iterable[mp.TransitionStep[S]]],
    approx_0: ValueFunctionApprox[S],
    gamma: float,
    lambd: float
) -> Iterator[ValueFunctionApprox[S]]:
    func_approx: ValueFunctionApprox[S] = approx_0
    yield func_approx

    for trace in traces:
        el_tr: Gradient[ValueFunctionApprox[S]] = Gradient(func_approx).zero()
        for step in trace:
            s = step.state
            r = step.reward
            y: float = r + gamma * \
                       extended_vf(func_approx, step.next_state)
            el_tr = el_tr * (gamma * lambd) + func_approx.objective_gradient(
                xy_vals_seq=[(s, y)],
                obj_deriv_out_fun=lambda x1, y1: np.ones(len(x1))
            )
            func_approx = func_approx.update_with_gradient(
                el_tr * (func_approx(s) - y)
            )
            yield func_approx


def plot_td_lmda_convergence(fmrp, lambdas, lr_param, num_episodes, td_episode_length, tabular, plot_batch=7, plot_start=0):
    initial_learning_rate, half_life, exponent = lr_param
    learning_rate_func: Callable[[int], float] = learning_rate_schedule(
            initial_learning_rate=initial_learning_rate,
            half_life=half_life,
            exponent=exponent
        )
    true_vf: np.ndarray = fmrp.get_value_function_vec(user_gamma)
    states = fmrp.non_terminal_states
    for user_lambda in lambdas:

        initial_vf_dict: Mapping[NonTerminal[InventoryState], float] = \
            {s: 0. for s in states}
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
                                              lambd=user_lambda)
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
                lambd=user_lambda)

        td_errors = []
        transitions_batch = plot_batch * td_episode_length
        batch_td_errs = []

        for i, tdlmda_vf in enumerate(
            itertools.islice(td_lmda_vfs, num_episodes * td_episode_length)
        ):
            if tabular:
                batch_td_errs.append(np.sqrt(sum(
                    (tdlmda_vf[s] - true_vf[j]) ** 2 for j, s in enumerate(states)
                ) / len(states)))
            else:
                batch_td_errs.append(np.sqrt(sum(
                    (tdlmda_vf(s) - true_vf[j]) ** 2 for j, s in enumerate(states)
                ) / len(states)))
            if i % transitions_batch == transitions_batch - 1:
                td_errors.append(sum(batch_td_errs) / transitions_batch)
                batch_td_errs = []
        td_plot = td_errors[plot_start:]

        plt.plot(
            range(len(td_plot)),
            td_plot,
            label=f"lambda = {user_lambda:.2f}"
        )
    if tabular:
        plt.title(f"VF RMSE for Tabular Case plotting starts from batch {plot_start}")
    else:
        plt.title(f"VF RMSE for Func Approx plotting starts from batch {plot_start}")
    plt.ylabel("Value Function RMSE")
    plt.legend()
    plt.show()

if __name__ == "__main__":
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

    print("Value Function")
    print("--------------")
    si_mrp.display_value_function(gamma=user_gamma)
    print()

    plot_td_lmda_convergence(si_mrp, np.arange(0, 1.1, 0.25), (0.03, 1e5, 0.5), 1000, 100, tabular=True, plot_start=3)
    plot_td_lmda_convergence(si_mrp, np.arange(0, 1.1, 0.25), (0.03, 1e5, 0.5), 1000, 100, tabular=False, plot_start=3)

