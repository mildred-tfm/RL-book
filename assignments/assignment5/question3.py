from dataclasses import dataclass
from typing import Mapping, Dict, Iterator
import itertools
from rl.distribution import Categorical, Constant, Gaussian, Poisson, Choose
from rl.markov_process import MarkovProcess, NonTerminal, State
import scipy.stats as ss
from scipy.stats import norm
from rl.chapter9.order_book import OrderBook, PriceSizePairs, DollarsAndShares
from rl.approximate_dynamic_programming import back_opt_vf_and_policy
import numpy as np

@dataclass
class OrderBookMP(MarkovProcess[OrderBook]):
    price_mean: float
    price_sigma: float
    size_lambda: float

    def transition(self, state: NonTerminal[OrderBook]) -> Choose[State[OrderBook]]:
        #assume the order price follows normal distribution
        price = norm.rvs(self.price_mean, self.price_sigma, 1)[0]
        #assume order size follows a poisson distribution
        share = ss.poisson.rvs(self.size_lambda)
        #assume it is equal likely that the arrival is a limit buy/limit sell/market buy/market sell order
        _, next_limit_sell = state.state.sell_limit_order(price, share)
        _, next_limit_buy = state.state.buy_limit_order(price, share)
        _, next_market_sell = state.state.sell_market_order(share)
        _, next_market_buy = state.state.buy_market_order(share)

        return Choose([NonTerminal(next_limit_sell), NonTerminal(next_limit_buy), NonTerminal(next_market_sell), NonTerminal(next_market_buy)])



def simulate_OrderBook(initial_bid: PriceSizePairs,
                       initial_ask: PriceSizePairs,
                       price_mu: float,
                       price_sigma: float,
                       size_lambda: float,
                       time_steps: int,
                       num_traces: int)-> Iterator[OrderBook]:
    mp = OrderBookMP(price_mean=price_mu, price_sigma=price_sigma, size_lambda=size_lambda)
    start_state_distribution = \
        Constant(NonTerminal(OrderBook(descending_bids=initial_bid, ascending_asks=initial_ask)))

    return [[s.state for s in itertools.islice(
            mp.simulate(start_state_distribution),
            time_steps + 1
        )] for _ in range(num_traces)]


if __name__ == '__main__':
    from numpy.random import poisson

    bids: PriceSizePairs = [DollarsAndShares(
        dollars=x,
        shares=poisson(100. - (100 - x) * 10)
    ) for x in range(100, 90, -1)]
    asks: PriceSizePairs = [DollarsAndShares(
        dollars=x,
        shares=poisson(100. - (x - 105) * 10)
    ) for x in range(105, 115, 1)]

    price_mu: float = (100 + 105) / 2
    price_sigma: float = 5
    size_lambda: float = 50
    num_traces: int = 10
    num_timestep: int = 29

    OrderBook_dynamtics = simulate_OrderBook(bids, asks, price_mu, price_sigma, size_lambda, num_timestep, num_traces)

    # display order book evolution for a sampled trace
    # If the mean of price distribution is close to the mid price and variance is not too big, and the order size is also
    # about the average number of size for different price, the order book evolves realistically. Lots of buy/sell order will
    # happen in the middle
    for orderbook in OrderBook_dynamtics[0]:
        orderbook.pretty_print_order_book()
        orderbook.display_order_book()

    #expriment with different transition model parameters
    #If price distribution is not close to the initial order book distribution, the dynamtic can be exotic. For example, if the parameter
    #for the poisson distribution is too large, there can be a large order that eat up the whole bid depth/ask depth.
    OrderBook_dynamtics2 = simulate_OrderBook(bids, asks, price_mu+5, price_sigma+5, size_lambda+10, num_timestep, num_traces)
    for orderbook in OrderBook_dynamtics2[0]:
        orderbook.pretty_print_order_book()
        orderbook.display_order_book()


