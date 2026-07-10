"""Environment verification (curriculum phase P0, report section 3.5)."""
import numpy as np

from drlqn.env.games import make, REGISTRY
from drlqn.eval.baselines import optimal_actions, SOLVERS


def test_determinism():
    for eid in REGISTRY:
        a, b = make(eid), make(eid)
        a.reset(); b.reset()
        rng = np.random.default_rng(0)
        for _ in range(50):
            act = int(rng.choice(list(a.action_space) or [6]))
            x, y = (int(rng.integers(64)), int(rng.integers(64)))
            ra = a.step(act, x, y)
            rb = b.step(act, x, y)
            assert np.array_equal(ra.frame.grid, rb.frame.grid)
            if a.state != "NOT_FINISHED":
                break


def test_all_levels_scripted_solvable():
    for eid in REGISTRY:
        counts = optimal_actions(eid)
        assert len(counts) == 6, f"{eid}: {counts}"


def test_random_agent_does_not_win_final_level():
    """The final level must resist uninformed random play (500 actions).

    Known limitation vs the report's 1-in-10,000 criterion: rt02's cyclic
    state space (<=256 states) can fall to long random runs; other
    environments resist thousands of random actions.
    """
    from drlqn.eval.baselines import SOLVERS
    for eid in REGISTRY:
        env = make(eid)
        env.reset()
        while env.level < 6:
            for (a, x, y) in SOLVERS[eid](env):
                env.step(a, x, y)
        rng = np.random.default_rng(1)
        for _ in range(500):
            if env.state != "NOT_FINISHED":
                break
            acts = list(env.action_space) or [6]
            r = env.step(int(rng.choice(acts)), int(rng.integers(64)),
                         int(rng.integers(64)))
            assert not r.level_won, f"{eid} final level fell to random play"


def test_frames_are_valid():
    for eid in REGISTRY:
        env = make(eid)
        f = env.reset()
        assert f.grid.shape == (64, 64) and f.grid.max() < 16


if __name__ == "__main__":
    test_determinism()
    test_all_levels_scripted_solvable()
    test_random_agent_does_not_win_final_level()
    test_frames_are_valid()
    print("env checks OK")
