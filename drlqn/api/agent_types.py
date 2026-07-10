from dataclasses import dataclass

import numpy as np


@dataclass
class SimpleFrame:
    grid: np.ndarray
    state: str
    level: int
    score: int


@dataclass
class SimpleStep:
    frame: SimpleFrame
    level_won: bool = False
    env_won: bool = False
