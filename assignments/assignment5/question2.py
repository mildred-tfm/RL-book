from dataclasses import dataclass
from typing import Callable, Iterator, Tuple, Sequence, List
import numpy as np
from rl.policy import DeterministicPolicy
from rl.distribution import Gaussian,  SampledDistribution, Distribution
from rl.markov_decision_process import MarkovDecisionProcess
from rl.markov_process import MarkovProcess, NonTerminal, State, Terminal
from rl.function_approx import LinearFunctionApprox, FunctionApprox
from numpy.polynomial.laguerre import lagval
from rl.approximate_dynamic_programming import back_opt_vf_and_policy


@dataclass(frozen = True)
class ExerciseAmericanOption:
    spot_price: float
    payoff: Callable[[float], float]
    expiry: float
    alpha: float
    sigma: float
    num_steps: int
    spot_price_frac: float

    def get_mdp(self, t: int) -> MarkovDecisionProcess[float, bool]:
        dt: float = self.expiry/self.num_steps
        exer_payoff: Callable[[float], float] = self.payoff
        r: float = self.alpha
        s: float = self.sigma

        class AmericanOptionMDP(MarkovDecisionProcess[float, bool]):

            def step(self,
                     price: NonTerminal[float],
                     action: bool
                     ) -> SampledDistribution[Tuple[State[float], float]]:

                def sample_next_state_reward(state=price, action=action) -> Tuple[State[float], float]:
                    if action:
                        # if exercise the option
                        reward = exer_payoff(state.state)
                        return Terminal(0.), reward
                    else:
                        # if continue, assume price follow log-normal
                        next_s: float = np.exp(np.random.normal(
                            np.log(price.state) + (r - s * s / 2) * dt,
                            s * np.sqrt(dt)))
                        reward = 0.0
                        return NonTerminal(next_s), reward

                return SampledDistribution(sampler=sample_next_state_reward,
                                           expectation_samples=200)

            def actions(self, price: NonTerminal[float]) -> Sequence[bool]:
                return [True, False]

        return AmericanOptionMDP()

    def get_states_distribution(
            self,
            t: int
    ) -> SampledDistribution[NonTerminal[float]]:
        spot_mean2: float = self.spot_price * self.spot_price
        spot_var: float = spot_mean2 * \
                          self.spot_price_frac * self.spot_price_frac
        log_mean: float = np.log(spot_mean2 / np.sqrt(spot_var + spot_mean2))
        log_stdev: float = np.sqrt(np.log(spot_var / spot_mean2 + 1))

        time: float = t * self.expiry / self.num_steps

        def states_sampler_func() -> NonTerminal[float]:
            start: float = np.random.lognormal(log_mean, log_stdev)
            price = np.exp(np.random.normal(
                np.log(start) + (self.alpha - self.sigma * self.sigma / 2) * time,
                self.sigma * np.sqrt(time)
            ))
            return NonTerminal(price)

        return SampledDistribution(states_sampler_func)

    def get_vf_func_approx(
            self,
            t: int,
            features: Sequence[Callable[[NonTerminal[float]], float]],
            reg_coeff: float
    ) -> LinearFunctionApprox[NonTerminal[float]]:
        return LinearFunctionApprox.create(
            feature_functions=features,
            regularization_coeff=reg_coeff,
            direct_solve=True
        )

    def backward_induction_vf_and_pi(
            self,
            features: Sequence[Callable[[NonTerminal[float]], float]],
            reg_coeff: float
    ) -> Iterator[
        Tuple[FunctionApprox[NonTerminal[float]],
              DeterministicPolicy[float, bool]]
    ]:

        mdp_f0_mu_triples: Sequence[Tuple[
            MarkovDecisionProcess[float, bool],
            FunctionApprox[NonTerminal[float]],
            SampledDistribution[NonTerminal[float]]
        ]] = [(
            self.get_mdp(t=i),
            self.get_vf_func_approx(
                t=i,
                features=features,
                reg_coeff=reg_coeff
            ),
            self.get_states_distribution(t=i)
        ) for i in range(self.num_steps + 1)]

        num_state_samples: int = 1000

        return back_opt_vf_and_policy(
            mdp_f0_mu_triples=mdp_f0_mu_triples,
            γ=np.exp(-self.alpha * self.expiry / self.num_steps),
            num_state_samples=num_state_samples,
            error_tolerance=1e-8
        )

if __name__ == '__main__':
    spot_price_val: float = 100.0
    strike: float = 100.0
    expiry_val: float = 1.0
    alpha_val: float = 0.05
    sigma_val: float = 0.25
    num_steps_val: int = 100
    spot_price_frac_val: float = 0.02
    is_call: bool = True

    if is_call:
        opt_payoff = lambda x: max(x - strike, 0)
    else:
        opt_payoff = lambda x: max(strike - x, 0)

    opt_ex_bi: ExerciseAmericanOption = ExerciseAmericanOption(
        spot_price=spot_price_val,
        payoff=opt_payoff,
        expiry=expiry_val,
        alpha=alpha_val,
        sigma=sigma_val,
        num_steps=num_steps_val,
        spot_price_frac=spot_price_frac_val
    )

    num_laguerre: int = 4
    reglr_coeff: float = 0.001

    ident: np.ndarray = np.eye(num_laguerre)
    ffs: List[Callable[[NonTerminal[float]], float]] = [lambda _: 1.]
    ffs += [(lambda s, i=i: np.log(1 + np.exp(-s.state / (2 * strike))) *
                            lagval(s.state / strike, ident[i]))
            for i in range(num_laguerre)]
    it_vf = opt_ex_bi.backward_induction_vf_and_pi(
        features=ffs,
        reg_coeff=reglr_coeff
    )
    print(it_vf)
    all_funcs: List[FunctionApprox[NonTerminal[float]]] = []
    for t, (v, p) in enumerate(it_vf):
        print(f"Time {t:d}")
        print()

        all_funcs.append(v)

        opt_alloc: float = p.action_for(spot_price_val)
        val: float = v(NonTerminal(spot_price_val))
        print(f"Opt Action = {opt_alloc}, Opt Val = {val:.3f}")
        print()

    print(all_funcs)