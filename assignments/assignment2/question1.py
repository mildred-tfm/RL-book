from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
from typing import Mapping, Dict
from rl.distribution import Constant, Choose, FiniteDistribution
from rl.markov_process import FiniteMarkovProcess, NonTerminal
import random
from scipy.stats import poisson


@dataclass(frozen=True)
class SnakeLadderState:
    position: int
    # on_order: int

    # def inventory_position(self) -> int:
    #     return self.on_hand + self.on_order


class SnakesAndLaddersMPFinite(FiniteMarkovProcess[int]):

    def __init__(
        self,
        start=0,
        end=100,
    ):
        self.start: int = start
        self.end: int = end
        self.snake_ladder_dict = \
            {1:38, 4:14, 8:30, 21:42, 28:76, 50:67, 71:92, 80:99, 97:78, 95:56, 88:24, 62:18, 48:26, 36:6, 32:10}

        super().__init__(self.get_transition_map())

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

def simulation(process, start_state):
    state = start_state
    while True:
        yield state
        state = process.next_state(state)



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

def plot_traces(traces, num):
    assert num <= len(traces), "Not enough samples to draw."
    samples = random.sample(traces, num)
    for trace in samples:
        plt.plot(trace)
    plt.title("%s Sample Traces of the Snakes and Ladders Game" % (num))
    plt.xlabel("time steps")
    plt.ylabel("position")
    plt.show()

def plot_timestep_distribution(traces):
    time_steps = [len(trace) for trace in traces]
    g = sns.displot(data=time_steps, kde=True)
    g.set(xlabel="time steps to finish the game", title="Distribution of Time Steps to Finish the Game")
    plt.show()


if __name__ == '__main__':


    si_mp = SnakesAndLaddersMPFinite()

    print("Transition Map")
    print("--------------")
    print(si_mp)

    time_steps: int = 200
    num_traces: int = 100

    simulated_traces: list = snake_ladder_traces(
        start_position=0,
        end_position=100,
        time_steps=time_steps,
        num_traces=num_traces
    )

    print(simulated_traces)
    plot_traces(simulated_traces, 10)
    plot_timestep_distribution(simulated_traces)
