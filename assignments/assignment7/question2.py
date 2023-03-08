from typing import Iterable, Tuple, Callable, TypeVar, Sequence, Iterator, Mapping
from rl.function_approx import Tabular, DNNSpec, AdamGradient, DNNApprox, learning_rate_schedule
from rl.chapter3.simple_inventory_mdp_cap import InventoryState
from rl.markov_decision_process import NonTerminal, MarkovDecisionProcess, Policy, TransitionStep
from rl.approximate_dynamic_programming import (ValueFunctionApprox,
                                                QValueFunctionApprox,
                                                NTStateDistribution)
from rl.policy import DeterministicPolicy, RandomPolicy, UniformPolicy
from rl.distribution import Categorical, Choose, Gaussian
from operator import itemgetter
from rl.chapter11.control_utils import glie_mc_finite_learning_rate_correctness
from rl.chapter11.control_utils import \
    q_learning_finite_learning_rate_correctness
from rl.chapter11.control_utils import \
    glie_sarsa_finite_learning_rate_correctness
from rl.chapter11.control_utils import compare_mc_sarsa_ql
from rl.chapter3.simple_inventory_mdp_cap import SimpleInventoryMDPCap
from rl.dynamic_programming import value_iteration_result
from rl.monte_carlo import glie_mc_control
from rl.td import glie_sarsa
from rl.chapter11.control_utils import get_vf_and_policy_from_qvf
from rl.chapter7.asset_alloc_discrete import AssetAllocDiscrete
import itertools
import rl.iterate as iterate
from pprint import pprint
import numpy as np
import random
S = TypeVar('S')
A = TypeVar('A')

#Simple Inventory mdp
capacity: int = 2
poisson_lambda: float = 1.0
holding_cost: float = 1.0
stockout_cost: float = 10.0
gamma: float = 0.9

si_mdp: SimpleInventoryMDPCap = SimpleInventoryMDPCap(
    capacity=capacity,
    poisson_lambda=poisson_lambda,
    holding_cost=holding_cost,
    stockout_cost=stockout_cost
)

true_opt_vf, true_opt_policy = value_iteration_result(si_mdp, gamma=gamma)
print("True Optimal Value Function")
pprint(true_opt_vf)
print("True Optimal Policy")
print(true_opt_policy)

##Asset Allocation mdp
steps: int = 4
μ: float = 0.13
σ: float = 0.2
r: float = 0.07
a: float = 1.0
init_wealth: float = 1.0
init_wealth_stdev: float = 0.1

excess: float = μ - r
var: float = σ * σ
base_alloc: float = excess / (a * var)

risky_ret: Sequence[Gaussian] = [Gaussian(μ=μ, σ=σ) for _ in range(steps)]
riskless_ret: Sequence[float] = [r for _ in range(steps)]
utility_function: Callable[[float], float] = lambda x: - np.exp(-a * x) / a
alloc_choices: Sequence[float] = np.linspace(
    2 / 3 * base_alloc,
    4 / 3 * base_alloc,
    11
)
feature_funcs: Sequence[Callable[[Tuple[float, float]], float]] = \
    [
        lambda _: 1.,
        lambda w_x: w_x[0],
        lambda w_x: w_x[1],
        lambda w_x: w_x[1] * w_x[1]
    ]
dnn: DNNSpec = DNNSpec(
    neurons=[],
    bias=False,
    hidden_activation=lambda x: x,
    hidden_activation_deriv=lambda y: np.ones_like(y),
    output_activation=lambda x: - np.sign(a) * np.exp(-x),
    output_activation_deriv=lambda y: -y
)
init_wealth_distr: Gaussian = Gaussian(μ=init_wealth, σ=init_wealth_stdev)

aad: AssetAllocDiscrete = AssetAllocDiscrete(
    risky_return_distributions=risky_ret,
    riskless_returns=riskless_ret,
    utility_func=utility_function,
    risky_alloc_choices=alloc_choices,
    feature_functions=feature_funcs,
    dnn_spec=dnn,
    initial_wealth_distribution=init_wealth_distr
)

it_qvf: Iterator[QValueFunctionApprox[float, float]] = \
    aad.backward_induction_qvf()

print("Backward Induction on Q-Value Function")
print("--------------------------------------")
print()
for t, q in enumerate(it_qvf):
    print(f"Time {t:d}")
    print()
    opt_alloc: float = max(
        ((q((NonTerminal(init_wealth), ac)), ac) for ac in alloc_choices),
        key=itemgetter(0)
    )[1]
    val: float = max(q((NonTerminal(init_wealth), ac))
                     for ac in alloc_choices)
    print(f"Opt Risky Allocation = {opt_alloc:.3f}, Opt Val = {val:.3f}")
    print("Optimal Weights below:")
    for wts in q.weights:
        pprint(wts.weights)
    print()

print("Analytical Solution")
print("-------------------")
print()

for t in range(steps):
    print(f"Time {t:d}")
    print()
    left: int = steps - t
    growth: float = (1 + r) ** (left - 1)
    alloc: float = base_alloc / growth
    vval: float = - np.exp(- excess * excess * left / (2 * var)
                           - a * growth * (1 + r) * init_wealth) / a
    bias_wt: float = excess * excess * (left - 1) / (2 * var) + \
                     np.log(np.abs(a))
    w_t_wt: float = a * growth * (1 + r)
    x_t_wt: float = a * excess * growth
    x_t2_wt: float = - var * (a * growth) ** 2 / 2

    print(f"Opt Risky Allocation = {alloc:.3f}, Opt Val = {vval:.3f}")
    print(f"Bias Weight = {bias_wt:.3f}")
    print(f"W_t Weight = {w_t_wt:.3f}")
    print(f"x_t Weight = {x_t_wt:.3f}")
    print(f"x_t^2 Weight = {x_t2_wt:.3f}")
    print()

def simdp_MC_test():

    mc_episode_length_tol: float = 1e-5
    num_episodes = 10000
    epsilon_as_func_of_episodes: Callable[[int], float] = lambda k: 1/k

    initial_qvf_dict: Mapping[Tuple[NonTerminal[InventoryState], int], float] = {
        (s, a): 0. for s in si_mdp.non_terminal_states for a in si_mdp.actions(s)
    }
    approx_0 = Tabular(values_map=initial_qvf_dict, count_to_weight_func=lambda n: 1.0/n)
    qvfs = glie_mc_control(
        mdp=si_mdp,
        states=Choose(si_mdp.non_terminal_states),
        approx_0=approx_0,
        γ=gamma,
        ϵ_as_func_of_episodes=epsilon_as_func_of_episodes,
        episode_length_tolerance=mc_episode_length_tol
    )

    final_qvf = iterate.last(itertools.islice(qvfs, num_episodes))

    opt_vf, opt_policy = get_vf_and_policy_from_qvf(
        mdp=si_mdp,
        qvf=final_qvf
    )

    print(f"GLIE MC Optimal Value Function with {num_episodes:d} episodes")
    pprint(opt_vf)
    print(f"GLIE MC Optimal Policy with {num_episodes:d} episodes")
    print(opt_policy)

def aad_MC_test():
    aad: AssetAllocDiscrete = AssetAllocDiscrete(
        risky_return_distributions=risky_ret,
        riskless_returns=riskless_ret,
        utility_func=utility_function,
        risky_alloc_choices=alloc_choices,
        feature_functions=feature_funcs,
        dnn_spec=dnn,
        initial_wealth_distribution=init_wealth_distr
    )

    mc_episode_length_tol: float = 1e-5
    num_episodes = 5000
    epsilon_as_func_of_episodes: Callable[[int], float] = lambda k: 1 / k


    for i in range(steps):
        print(f"Time {i:d}")
        print()
        mdp_i = aad.get_mdp(i)
        qvfs = glie_mc_control(
            mdp=mdp_i,
            states=aad.get_states_distribution(i),
            approx_0=aad.get_qvf_func_approx(),
            γ=gamma,
            ϵ_as_func_of_episodes=epsilon_as_func_of_episodes,
            episode_length_tolerance=mc_episode_length_tol
        )

        q = iterate.last(itertools.islice(qvfs, num_episodes))
        print(q)
        opt_alloc: float = max(
            ((q((NonTerminal(init_wealth), ac)), ac) for ac in alloc_choices),
            key=itemgetter(0)
        )[1]
        val: float = max(q((NonTerminal(init_wealth), ac))
                         for ac in alloc_choices)

        print(f"GLIE MC Opt Risky Allocation = {opt_alloc:.3f}, Opt Val = {val:.3f}")
        for wts in q.weights:
            pprint(wts.weights)
        print()

def simdp_SARSA_test():
    num_episodes = 1000
    max_episode_length: int = 100
    epsilon_as_func_of_episodes: Callable[[int], float] = lambda k: k ** -0.5
    initial_learning_rate: float = 0.1
    half_life: float = 10000.0
    exponent: float = 1.0
    gamma: float = 0.9
    initial_qvf_dict: Mapping[Tuple[NonTerminal[InventoryState], int], float] = {
        (s, a): 0. for s in si_mdp.non_terminal_states for a in si_mdp.actions(s)
    }
    learning_rate_func: Callable[[int], float] = learning_rate_schedule(
        initial_learning_rate=initial_learning_rate,
        half_life=half_life,
        exponent=exponent
    )
    qvfs: Iterator[QValueFunctionApprox[InventoryState, int]] = glie_sarsa(
        mdp=si_mdp,
        states=Choose(si_mdp.non_terminal_states),
        approx_0=Tabular(
            values_map=initial_qvf_dict,
            count_to_weight_func=learning_rate_func
        ),
        γ=gamma,
        ϵ_as_func_of_episodes=epsilon_as_func_of_episodes,
        max_episode_length=max_episode_length
    )

    num_updates = num_episodes * max_episode_length
    final_qvf: QValueFunctionApprox[InventoryState, int] = \
        iterate.last(itertools.islice(qvfs, num_updates))
    opt_vf, opt_policy = get_vf_and_policy_from_qvf(
        mdp=si_mdp,
        qvf=final_qvf
    )
    print(f"GLIE SARSA Optimal Value Function with {num_updates: d} updates")
    pprint(opt_vf)
    print(f"GLIE SARSA Optimal Policy with {num_updates: d} updates")
    print(opt_policy)


def aad_SARSA_test():
    aad: AssetAllocDiscrete = AssetAllocDiscrete(
        risky_return_distributions=risky_ret,
        riskless_returns=riskless_ret,
        utility_func=utility_function,
        risky_alloc_choices=alloc_choices,
        feature_functions=feature_funcs,
        dnn_spec=dnn,
        initial_wealth_distribution=init_wealth_distr
    )

    num_episodes = 1000
    epsilon_as_func_of_episodes: Callable[[int], float] = lambda k: k ** -0.5
    max_episode_length: int = 100
    for i in range(steps):
        print(f"Time {i:d}")
        print()
        mdp_i = aad.get_mdp(i)
        qvfs = glie_sarsa(
            mdp=mdp_i,
            states=aad.get_states_distribution(i),
            approx_0=aad.get_qvf_func_approx(),
            γ=gamma,
            ϵ_as_func_of_episodes=epsilon_as_func_of_episodes,
            max_episode_length=max_episode_length
        )

        fa = iterate.last(itertools.islice(qvfs, num_episodes))
        print(fa)
        opt_alloc: float = max(
            ((fa((NonTerminal(init_wealth), ac)), ac) for ac in alloc_choices),
            key=itemgetter(0)
        )[1]
        val: float = max(fa((NonTerminal(init_wealth), ac))
                         for ac in alloc_choices)

        print(f"GLIE SARSA Opt Risky Allocation = {opt_alloc:.3f}, Opt Val = {val:.3f}")
        for wts in fa.weights:
            pprint(wts.weights)
        print()

if __name__ == "__main__":
    simdp_MC_test()
    simdp_SARSA_test()
    aad_MC_test()
    aad_SARSA_test()