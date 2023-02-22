import numpy as np
import math
from dataclasses import dataclass
from typing import Sequence, Tuple
from rl.markov_decision_process import MarkovRewardProcess, \
    NonTerminal, State, Terminal
from rl.distribution import SampledDistribution, Constant
import itertools
import matplotlib.pyplot as plt
@dataclass(frozen=True)
class OrderBookState:
    time: float
    Pb: float
    Pa: float
    OBMid: float
    pnl: float
    inventory: int
    nhit: int
    nlifts: int
    spread: float

OrderBook = Sequence[OrderBookState]

@dataclass(frozen=False)
class OptimalMarketMaking(MarkovRewardProcess[OrderBookState]):
    def __init__(self,
                Tm: float,
                dt: float,
                gamma: float,
                sigma: float,
                k: float,
                c: float,
                naive: bool,
                average_bid_ask: float
                ):

        self.Tm = Tm
        self.dt = dt
        self.gamma = gamma
        self.sigma = sigma
        self.k = k
        self.c =c
        self.naive = naive
        self.average_bid_ask = average_bid_ask
    def optimal_bid(self, It: int, t: float) -> float:
        return ((2*It+1)*self.gamma*self.sigma*self.sigma*(self.Tm-t))/2+(1/self.gamma)*np.log(1+self.gamma/self.k)

    def optimal_ask(self, It: int, t: float) -> float:
        return ((1-2*It)*self.gamma*self.sigma*self.sigma*(self.Tm-t))/2+(1/self.gamma)*np.log(1+self.gamma/self.k)

    def naive_policy(self) -> float:
        return self.average_bid_ask/2

    def transition_reward(self, state: NonTerminal[OrderBookState]) -> SampledDistribution[Tuple[State[OrderBookState], float]]:
        def sample_next_state_reward(state=state) -> [Tuple[State[OrderBookState], float]]:

            # Observe State
            St = state.state.OBMid
            Wt = state.state.pnl
            It = state.state.inventory
            t = state.state.time
            hit = state.state.nhit
            lift = state.state.nlifts

            # perform action
            if self.naive:
                # if perform the naive policy
                db = self.naive_policy()
                da = self.naive_policy()
            else:
                # if perform the optimal policy
                db = self.optimal_bid(It, t)
                da = self.optimal_ask(It, t)
            Pbt = St-db
            Pat = St+da
            up = min(1, self.c*np.exp(-self.k*db)*self.dt)
            down = min(1, self.c*np.exp(-self.k*da)*self.dt)

            # update Inventory, Wealth,
            if np.random.choice(2, p=[1-up, up]):
                It += 1
                hit += 1
                Wt -= Pbt
            if np.random.choice(2, p=[1-down, down]):
                It -= 1
                lift += 1
                Wt += Pat
            # update OB Mid Price
            if np.random.random() > 0.5:
                St += self.sigma*np.sqrt(self.dt)
            else:
                St -= self.sigma*np.sqrt(self.dt)
            next_state: OrderBookState = OrderBookState(
                t+self.dt,
                Pbt,
                Pat,
                St,
                Wt,
                It,
                hit,
                lift,
                Pat-Pbt
            )
            if math.isclose(t+self.dt, self.Tm):
                # If terminate
                reward = -np.exp(-self.gamma*(Wt+It*St))
                return Terminal(next_state), reward
            else:
                reward = 0
                return NonTerminal(next_state), reward
        return SampledDistribution(sample_next_state_reward)

def MRP_reward_traces(
        Tm: float,
        dt: float,
        gamma: float,
        sigma: float,
        k: float,
        c: float,
        naive: bool,
        average_bid_ask: float,
        S0: float,
        W0: float,
        I0 : int,
        num_traces: int,
        output: str
) -> np.ndarray:
    process = OptimalMarketMaking(
            Tm = Tm,
            dt = dt,
            gamma = gamma,
            sigma = sigma,
            k = k,
            c = c,
            naive = naive,
            average_bid_ask =average_bid_ask)
    start_state_distribution = Constant(NonTerminal(OrderBookState(0.0, 0.0, 0.0, S0, W0, I0, 0, 0, 0.0)))
    # (state, next state, reward)
    if output == "spread":
        return np.vstack([np.fromiter((s.next_state.state.spread for s in
                         itertools.islice(
                         process.simulate_reward(start_state_distribution=start_state_distribution),
                         int(Tm / dt) + 1)), float) for _ in range(num_traces)])
    #0:pnl, 1: Mid, 2: Pa 3: Pb  4:nhit  5:nlift  6: inventory 7: reward  8:spread
    return np.stack([np.fromiter((itertools.chain.from_iterable(
        (s.next_state.state.pnl, s.next_state.state.OBMid, s.next_state.state.Pa,
         s.next_state.state.Pb, s.next_state.state.nhit,s.next_state.state.nlifts,
         s.next_state.state.inventory, s.next_state.state.spread) for s in
                      itertools.islice(
                      process.simulate_reward(start_state_distribution=start_state_distribution),
                      int(Tm / dt) + 1))), float) for _ in range(num_traces)]).reshape((num_traces, int(Tm / dt), -1))


if __name__ == '__main__':
    S0 = 100
    Tm = 1.0
    dt = 0.005
    gamma = 0.1
    sigma = 2
    I0 = 0
    W0 = 0
    k = 1.5
    c = 140
    num_traces = 10000
    average_bid_ask = 0

    si_mrp_spread = MRP_reward_traces(Tm = Tm, dt = dt, gamma = gamma, sigma = sigma, k = k, c = c, naive = False,
                                      average_bid_ask = average_bid_ask, S0 = S0, W0 = W0, I0 = I0,
                                      num_traces = num_traces, output = 'spread')
    # first find out the average Bid-Ask Spread across all time steps across all simulation traces
    average_bid_ask = np.mean(si_mrp_spread)
    print("Average Bid-Ask Spread across all time steps across all simulation traces")
    print("---------------------")
    print(average_bid_ask)

    # Optimal Policy PnL
    si_mrp_traces = MRP_reward_traces(Tm = Tm, dt = dt, gamma = gamma, sigma = sigma, k = k, c = c, naive = False,
                                      average_bid_ask = average_bid_ask, S0 = S0, W0 = W0, I0 = I0,
                                      num_traces = num_traces, output=None)


    # Naive Policy PnL
    si_mrp_naive = MRP_reward_traces(Tm = Tm, dt = dt, gamma = gamma, sigma = sigma, k = k, c = c, naive = True,
                                      average_bid_ask = average_bid_ask, S0 = S0, W0 = W0, I0 = I0,
                                      num_traces = num_traces, output=None)

    # Plot graphs for a single simulation trace
    times = np.arange(0, Tm, dt)

    labels = ["PnL", "Mid Price", "Ask Price", "Bid Price", "Number of Hits", "Number of Lifts", "Inventory", "spread"]
    for i, label in enumerate(labels):
        # compute the average at T across all simulated traces
        print("Average %s at T (optimal): %.5f" % (label, np.mean(si_mrp_traces[:, -1, i], axis=0)))
        print("Average %s at T (naive): %.5f" % (label, np.mean(si_mrp_naive[:, -1, i], axis=0)))
        plt.figure()
        plt.plot(times, np.mean(si_mrp_naive[:,:,i], axis=0), label = 'naive ' + label)
        plt.plot(times, np.mean(si_mrp_traces[:,:,i], axis=0), label = 'Optimal ' + label)
        plt.legend()
        plt.show()

    plt.figure()
    plt.hist(si_mrp_naive[:, -1, 0], label="Naive Pnl at T")
    plt.hist(si_mrp_traces[:, -1, 0], label="Optimal Pnl at T")
    plt.legend()
    plt.title("Histogram of PnL at T")
    plt.show()


