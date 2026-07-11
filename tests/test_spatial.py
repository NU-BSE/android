"""Regression tests for Duck-inspired spatial action proposals."""
import numpy as np

from drlqn.planning.spatial import generate_click_candidates, infer_tile_scale
from drlqn.state import extract_objects
from drlqn.config import drlqn_xs
from drlqn.train.hybrid_solver import HybridDRLQNSolver
from drlqn.env.games import make


def _render_logical(logical: np.ndarray, scale: int = 8) -> np.ndarray:
    return np.repeat(np.repeat(logical, scale, axis=0), scale, axis=1)


def test_infers_rendered_tile_scale():
    logical = np.array([
        [5, 5, 5, 5, 5, 5, 5, 5],
        [5, 2, 2, 5, 3, 3, 5, 5],
        [5, 2, 2, 5, 3, 3, 5, 5],
        [5, 5, 5, 5, 5, 5, 5, 5],
        [5, 4, 5, 4, 5, 4, 5, 4],
        [5, 5, 5, 5, 5, 5, 5, 5],
        [5, 5, 5, 5, 5, 5, 5, 5],
        [5, 5, 5, 5, 5, 5, 5, 5],
    ], dtype=np.uint8)
    grid = _render_logical(logical)
    assert infer_tile_scale(grid) == 8


def test_proposals_cover_individual_cells_inside_merged_region():
    # Four adjacent logical goal cells render as one connected component. The
    # legacy object-centroid policy can click only the merged object's centre;
    # the spatial proposal policy must expose each logical cell centre.
    logical = np.full((8, 8), 5, dtype=np.uint8)
    logical[2:4, 2:4] = 2
    grid = _render_logical(logical)
    objs = extract_objects(grid)
    proposals = generate_click_candidates(grid, objs, max_candidates=24)
    coords = {(p.x, p.y) for p in proposals}
    expected = {(20, 20), (28, 20), (20, 28), (28, 28)}
    assert expected.issubset(coords), (expected, coords)


def test_changed_region_gets_high_priority_anchor():
    before = np.zeros((64, 64), dtype=np.uint8)
    after = before.copy()
    after[30:34, 42:46] = 11
    proposals = generate_click_candidates(after, extract_objects(after), 24, before)
    assert proposals
    assert any(p.kind == "changed" and abs(p.x - 44) <= 1 and abs(p.y - 32) <= 1
               for p in proposals[:8])


def test_hybrid_solver_smoke_on_local_environment():
    env = make("xg01")
    solver = HybridDRLQNSolver(drlqn_xs(), seed=0)
    frame = env.reset()
    node = solver.perceive(env, frame.grid)
    assert node.valid.shape == (128,)
    assert node.lam_feats.shape[0] == 128
    assert len(node.objs) <= 24
