from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
from typing import Mapping, Dict, Tuple
from rl.distribution import Constant, Choose, FiniteDistribution
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess, NonTerminal
import random
from scipy.stats import poisson


@dataclass(frozen=True)
class SnakeLadderState:
    position: int


class SnakesAndLaddersMPFinite(FiniteMarkovProcess[SnakeLadderState]):

    def __init__(
        self,
        start=0,
        end=100,
    ):
        self.start: int = start
        self.end: int = end
        self.snake_ladder_dict: Dict[int, int] = \
            {1: 38, 4: 14, 8: 30, 21: 42, 28: 76, 50: 67, 71: 92, 80: 99, 97: 78, 95: 56, 88: 24, 62: 18, 48: 26, 36: 6,
             32: 10, 101: 100, 102: 100, 103: 100, 104: 100, 105: 100}
        super().__init__(self.get_transition_map())

    #part c) create transition map according to part b)
    def get_transition_map(self) -> \
            Mapping[SnakeLadderState, FiniteDistribution[SnakeLadderState]]:
        d: Dict[SnakeLadderState, Choose[SnakeLadderState]] = {}
        for i in range(self.start, self.end):
            state = SnakeLadderState(i)
            next_states = \
                [SnakeLadderState(i+j) if i+j not in self.snake_ladder_dict else
                 SnakeLadderState(self.snake_ladder_dict[i+j]) for j in range(1, 7)]
            d[state] = Choose(next_states)
        return d


class SnakeAndLadderMRPFinite(FiniteMarkovRewardProcess[SnakeLadderState]):

    def __init__(
        self,
        start=0,
        end=100
    ):
        self.start: int = start
        self.end: int = end
        self.snake_ladder_dict: Dict[int, int] = \
            {1: 38, 4: 14, 8: 30, 21: 42, 28: 76, 50: 67, 71: 92, 80: 99, 97: 78, 95: 56, 88: 24, 62: 18, 48: 26, 36: 6,
             32: 10, 101: 100, 102: 100, 103: 100, 104: 100, 105: 100}
        super().__init__(self.get_transition_reward_map())

    #part e) create transition reward map
    def get_transition_reward_map(self) -> \
            Mapping[
                SnakeLadderState,
                FiniteDistribution[Tuple[SnakeLadderState, float]]
            ]:
        d: Dict[SnakeLadderState, Choose[Tuple[SnakeLadderState, float]]] = {}
        #reward = 1 for each possible next state
        for i in range(self.start, self.end):
            state = SnakeLadderState(i)
            next_state_rewards = \
                [(SnakeLadderState(i + j), 1.0) if i + j not in self.snake_ladder_dict else
                 (SnakeLadderState(self.snake_ladder_dict[i + j]), 1.0) for j in range(1, 7)]
            d[state] = Choose(next_state_rewards)
        return d

#part c) function to sample traces
def snake_ladder_traces(
    start_position: int,
    end_position: int,
    time_steps: int,
    num_traces: int
) -> list:
    mp = SnakesAndLaddersMPFinite(start_position, end_position)
    start_state_distribution = Constant(
        NonTerminal(SnakeLadderState(0))
    )
    return [
        np.fromiter((s.state.position for s in itertools.islice(
            mp.simulate(start_state_distribution),
            time_steps + 1
        )), float) for _ in range(num_traces)]

#part d) function to plot [num] number of sampling traces
def plot_traces(traces, num):
    assert num <= len(traces), "Not enough samples to draw."
    samples = random.sample(traces, num)
    for trace in samples:
        plt.plot(trace)
    plt.title("%s Sample Traces of the Snakes and Ladders Game" % num)
    plt.xlabel("time steps")
    plt.ylabel("position")
    plt.savefig('sampling_traces.png')
    plt.show()

#part d) function to plot distribution of time steps to finish the game
def plot_timestep_distribution(traces):
    time_steps = [len(trace) for trace in traces]
    g = sns.displot(data=time_steps, kde=True)
    g.set(xlabel="time steps to finish the game", title="Distribution of Time Steps to Finish the Game")
    plt.savefig('timestep_distribution.png')
    plt.show()


if __name__ == '__main__':
    sl_mp = SnakesAndLaddersMPFinite()
    sl_mrp = SnakeAndLadderMRPFinite()

    print("Transition Map")
    print("--------------")
    print(sl_mp)

    print("Transition Reward Map")
    print("---------------------")
    print(sl_mrp)

    time_steps: int = 200
    num_traces: int = 200
    user_gamma: float = 1

    #part c) sample traces
    simulated_traces: list = snake_ladder_traces(
        start_position=0,
        end_position=100,
        time_steps=time_steps,
        num_traces=num_traces
    )

    #print some sampled traces
    print(simulated_traces[:10])
    #part d) plot sample traces
    plot_traces(simulated_traces, 10)
    #part d) plot distribution of time steps
    plot_timestep_distribution(simulated_traces)

    print("Value Function")
    print("--------------")
    sl_mrp.display_value_function(gamma=user_gamma)
    print()

    #part e) expected number of rolls to finish the game
    value_func = sl_mrp.get_value_function_vec(gamma=user_gamma)
    print(value_func)
    print("Expected number of rolls to finish the game: %d" % value_func[0])
