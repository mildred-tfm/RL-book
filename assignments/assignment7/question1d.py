from typing import Mapping, Dict, Tuple
from rl.distribution import Categorical, Choose
from rl.markov_process import FiniteMarkovRewardProcess


class RandomWalkMRP(FiniteMarkovRewardProcess[int]):
    '''
    This MRP's states are {0, 1, 2,...,self.barrier}
    with 0 and self.barrier as the terminal states.
    At each time step, we go from state i to state
    i+1 with probability self.p or to state i-1 with
    probability 1-self.p, for all 0 < i < self.barrier.
    The reward is 0 if we transition to a non-terminal
    state or to terminal state 0, and the reward is 1
    if we transition to terminal state self.barrier
    '''
    barrier: int
    p: float

    def __init__(
        self,
        barrier: int,
        p: float
    ):
        self.barrier = barrier
        self.p = p
        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> \
            Mapping[int, Categorical[Tuple[int, float]]]:
        d: Dict[int, Categorical[Tuple[int, float]]] = {
            i: Categorical({
                (i + 1, 0. if i < self.barrier - 1 else 1.): self.p,
                (i - 1, 0.): 1 - self.p
            }) for i in range(1, self.barrier)
        }
        return d

class RandomWalk2DMRP(FiniteMarkovRewardProcess[Tuple[int, int]]):
    barrier1: int
    barrier2: int
    def __init__(self, barrier1, barrier2):
        self.barrier1 = barrier1
        self.barrier2 = barrier2
        super().__init__(self.get_transition_map())


    def get_transition_map(self) -> Mapping[Tuple[int, int], Choose[Tuple[Tuple[int, int], float]]]:
        d: Dict[Tuple[int, int], Choose[Tuple[Tuple[int, int], float]]] ={
            (i, j): Categorical({
                ((i + 1, j), 0. if j < self.barrier2 - 1 else 1.): 0.25,
                ((i - 1, j), 0. ): 0.25,
                ((i, j+1), 0 if i < self.barrier1 - 1 else 1.): 0.25,
                ((i, j-1), 0): 0.25
            }) for i in range(1, self.barrier1) for j in range(1, self.barrier2)
        }
        return d

if __name__ == '__main__':
    from rl.chapter10.prediction_utils import compare_td_and_mc

    b1: int = 10
    b2: int = 10
    random_walk: RandomWalk2DMRP = RandomWalk2DMRP(
        barrier1=b1,
        barrier2=b2
    )
    compare_td_and_mc(
        fmrp=random_walk,
        gamma=1.0,
        mc_episode_length_tol=1e-6,
        num_episodes=2000,
        learning_rates=[(0.01, 1e8, 0.5), (0.05, 1e8, 0.5)],
        initial_vf_dict={s: 0.5 for s in random_walk.non_terminal_states},
        plot_batch=7,
        plot_start=0
    )