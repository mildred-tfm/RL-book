from typing import Callable, Sequence, Tuple, List
import numpy as np
import random
from scipy.stats import norm
from rl.function_approx import DNNApprox, LinearFunctionApprox, \
    FunctionApprox, DNNSpec, AdamGradient, Weights
from random import randrange
from numpy.polynomial.laguerre import lagval
from rl.chapter8.optimal_exercise_bin_tree import OptimalExerciseBinTree
from rl.markov_process import NonTerminal
from rl.gen_utils.plot_funcs import plot_list_of_curves
from rl.chapter12.optimal_exercise_rl import training_sim_data, scoring_sim_data
TrainingDataType = Tuple[int, float, float]


def fitted_lspi_option(
    type: str,
    expiry: float,
    num_steps: int,
    num_paths: int,
    spot_price: float,
    spot_price_frac: float,
    rate: float,
    vol: float,
    strike: float,
    training_iters: int
) -> LinearFunctionApprox[Tuple[float, float]]:

    num_laguerre: int = 4
    epsilon: float = 1e-3

    ident: np.ndarray = np.eye(num_laguerre)
    features: List[Callable[[Tuple[float, float]], float]] = [lambda _: 1.]
    features += [(lambda t_s, i=i: np.exp(-t_s[1] / (2 * strike)) *
                  lagval(t_s[1] / strike, ident[i]))
                 for i in range(num_laguerre)]
    features += [
        lambda t_s: np.cos(-t_s[0] * np.pi / (2 * expiry)),
        lambda t_s: np.log(expiry - t_s[0]) if t_s[0] != expiry else 0.,
        lambda t_s: (t_s[0] / expiry) ** 2
    ]

    training_data: Sequence[TrainingDataType] = training_sim_data(
        expiry=expiry,
        num_steps=num_steps,
        num_paths=num_paths,
        spot_price=spot_price,
        spot_price_frac=spot_price_frac,
        rate=rate,
        vol=vol
    )

    dt: float = expiry / num_steps
    gamma: float = np.exp(-rate * dt)
    num_features: int = len(features)
    states: Sequence[Tuple[float, float]] = [(i * dt, s) for
                                             i, s, _ in training_data]
    next_states: Sequence[Tuple[float, float]] = \
        [((i + 1) * dt, s1) for i, _, s1 in training_data]
    feature_vals: np.ndarray = np.array([[f(x) for f in features]
                                         for x in states])
    next_feature_vals: np.ndarray = np.array([[f(x) for f in features]
                                              for x in next_states])
    non_terminal: np.ndarray = np.array(
        [i < num_steps - 1 for i, _, _ in training_data]
    )
    if type == "put":
        exer: np.ndarray = np.array([max(strike - s1, 0)
                                     for _, s1 in next_states])
    elif type == "call":
        exer: np.ndarray = np.array([max(s1 - strike, 0)
                                     for _, s1 in next_states])
    else:
        assert type == "put" or type == "call"
    wts: np.ndarray = np.zeros(num_features)
    for _ in range(training_iters):
        a_inv: np.ndarray = np.eye(num_features) / epsilon
        b_vec: np.ndarray = np.zeros(num_features)
        cont: np.ndarray = np.dot(next_feature_vals, wts)
        cont_cond: np.ndarray = non_terminal * (cont > exer)
        for i in range(len(training_data)):
            phi1: np.ndarray = feature_vals[i]
            phi2: np.ndarray = phi1 - \
                cont_cond[i] * gamma * next_feature_vals[i]
            temp: np.ndarray = a_inv.T.dot(phi2)
            a_inv -= np.outer(a_inv.dot(phi1), temp) / (1 + phi1.dot(temp))
            b_vec += phi1 * (1 - cont_cond[i]) * exer[i] * gamma
        wts = a_inv.dot(b_vec)

    return LinearFunctionApprox.create(
        feature_functions=features,
        weights=Weights.create(wts)
    )

def fitted_dql_option(
    type: str,
    expiry: float,
    num_steps: int,
    num_paths: int,
    spot_price: float,
    spot_price_frac: float,
    rate: float,
    vol: float,
    strike: float,
    training_iters: int
) -> DNNApprox[Tuple[float, float]]:

    reg_coeff: float = 1e-2
    neurons: Sequence[int] = [6]

    num_laguerre: int = 2
    ident: np.ndarray = np.eye(num_laguerre)
    features: List[Callable[[Tuple[float, float]], float]] = [lambda _: 1.]
    features += [(lambda t_s, i=i: np.exp(-t_s[1] / (2 * strike)) *
                  lagval(t_s[1] / strike, ident[i]))
                 for i in range(num_laguerre)]
    features += [
        lambda t_s: np.cos(-t_s[0] * np.pi / (2 * expiry)),
        lambda t_s: np.log(expiry - t_s[0]) if t_s[0] != expiry else 0.,
        lambda t_s: (t_s[0] / expiry) ** 2
    ]

    ds: DNNSpec = DNNSpec(
        neurons=neurons,
        bias=True,
        hidden_activation=lambda x: np.log(1 + np.exp(-x)),
        hidden_activation_deriv=lambda y: np.exp(-y) - 1,
        output_activation=lambda x: x,
        output_activation_deriv=lambda y: np.ones_like(y)
    )

    fa: DNNApprox[Tuple[float, float]] = DNNApprox.create(
        feature_functions=features,
        dnn_spec=ds,
        adam_gradient=AdamGradient(
            learning_rate=0.1,
            decay1=0.9,
            decay2=0.999
        ),
        regularization_coeff=reg_coeff
    )

    dt: float = expiry / num_steps
    gamma: float = np.exp(-rate * dt)
    training_data: Sequence[TrainingDataType] = training_sim_data(
        expiry=expiry,
        num_steps=num_steps,
        num_paths=num_paths,
        spot_price=spot_price,
        spot_price_frac=spot_price_frac,
        rate=rate,
        vol=vol
    )
    for _ in range(training_iters):
        t_ind, s, s1 = training_data[randrange(len(training_data))]
        t = t_ind * dt
        x_val: Tuple[float, float] = (t, s)
        if type == "put":
            val: float = max(strike - s1, 0)
        elif type == "call":
            val: float = max(s1 - strike, 0)
        else:
            assert type == "put" or type == "call"
        if t_ind < num_steps - 1:
            val = max(val, fa.evaluate([(t + dt, s1)])[0])
        y_val: float = gamma * val
        fa = fa.update([(x_val, y_val)])
        # for w in fa.weights:
        #     pprint(w.weights)
    return fa

def option_price(
    type: str,
    scoring_data: np.ndarray,
    func: FunctionApprox[Tuple[float, float]],
    expiry: float,
    rate: float,
    strike: float
) -> float:
    num_paths: int = scoring_data.shape[0]
    num_steps: int = scoring_data.shape[1] - 1
    prices: np.ndarray = np.zeros(num_paths)
    dt: float = expiry / num_steps

    for i, path in enumerate(scoring_data):
        step: int = 0
        while step <= num_steps:
            t: float = step * dt
            if type == "put":
                exercise_price: float = max(strike - path[step], 0)
            elif type == "call":
                exercise_price: float = max(path[step] - strike, 0)
            else:
                assert type == "put" or type == "call"
            continue_price: float = func.evaluate([(t, path[step])])[0] \
                if step < num_steps else 0.
            step += 1
            if exercise_price >= continue_price:
                prices[i] = np.exp(-rate * t) * exercise_price
                step = num_steps + 1

    return np.average(prices)



if __name__ == '__main__':

    spot_price_val: float = 100.0
    strike_val: float = 100.0
    expiry_val: float = 1.0
    rate_val: float = 0.05
    vol_val: float = 0.25
    num_scoring_paths: int = 10000
    num_steps_scoring: int = 100

    num_steps_lspi: int = 20
    num_training_paths_lspi: int = 1000
    spot_price_frac_lspi: float = 0.3
    training_iters_lspi: int = 8

    num_steps_dql: int = 20
    num_training_paths_dql: int = 1000
    spot_price_frac_dql: float = 0.02
    training_iters_dql: int = 100000

    random.seed(100)
    np.random.seed(100)

    for type in ["put", "call"]:
        payoff_f = lambda _, x: max(strike_val - x, 0) if type == "put" else max(x - strike_val, 0)

        flspi: LinearFunctionApprox[Tuple[float, float]] = fitted_lspi_option(
            type=type,
            expiry=expiry_val,
            num_steps=num_steps_lspi,
            num_paths=num_training_paths_lspi,
            spot_price=spot_price_val,
            spot_price_frac=spot_price_frac_lspi,
            rate=rate_val,
            vol=vol_val,
            strike=strike_val,
            training_iters=training_iters_lspi
        )

        fdql: DNNApprox[Tuple[float, float]] = fitted_dql_option(
            type=type,
            expiry=expiry_val,
            num_steps=num_steps_dql,
            num_paths=num_training_paths_dql,
            spot_price=spot_price_val,
            spot_price_frac=spot_price_frac_dql,
            rate=rate_val,
            vol=vol_val,
            strike=strike_val,
            training_iters=training_iters_dql
        )

        opt_ex_bin_tree: OptimalExerciseBinTree = OptimalExerciseBinTree(
            spot_price=spot_price_val,
            payoff=payoff_f,
            expiry=expiry_val,
            rate=rate_val,
            vol=vol_val,
            num_steps=100
        )

        vf_seq, policy_seq = zip(*opt_ex_bin_tree.get_opt_vf_and_policy())
        bin_tree_price: float = vf_seq[0][NonTerminal(0)]
        print("Binary Tree American %s option price: %.3f" % (type, bin_tree_price))

        scoring_data: np.ndarray = scoring_sim_data(
            expiry=expiry_val,
            num_steps=num_steps_scoring,
            num_paths=num_scoring_paths,
            spot_price=spot_price_val,
            rate=rate_val,
            vol=vol_val
        )

        lspi_opt_price: float = option_price(
            type=type,
            scoring_data=scoring_data,
            func=flspi,
            expiry=expiry_val,
            rate=rate_val,
            strike=strike_val,
        )
        print(f"LSPI American {type} Option Price = {lspi_opt_price:.3f}")

        dql_opt_price: float = option_price(
            type=type,
            scoring_data=scoring_data,
            func=fdql,
            expiry=expiry_val,
            rate=rate_val,
            strike=strike_val,
        )
        print(f"DQL American {type} Option Price = {dql_opt_price:.3f}")
