"""sq05 "Chroma": click environment about ordered composition.

A cue strip shows a sequence of colored squares. Below sit four colored
buttons. Press the buttons in cue order; a progress strip fills as you go and
a wrong press clears progress. Later levels: longer sequences, the cue hides
after the first press (working memory), and a legend that permutes cue colors
to button colors (relational indirection: cue color X means "press the button
paired with X in the legend").
Colors: buttons/cues use 3,4,6,9; progress 14; legend frame 5; hidden cue 1.
"""
from __future__ import annotations

import numpy as np

from ..core import ArcEnv, canvas, rect, border

PALETTE = [3, 4, 6, 9]
BTN = 10           # button size in px
BTN_Y = 46


class Chroma(ArcEnv):
    env_id = "sq05"
    action_space = ()
    uses_click = True

    _SPECS = {
        # level: (sequence indices, hide_after_first, legend permutation or None)
        1: ([0], False, None),
        2: ([2, 0], False, None),
        3: ([1, 3, 0], False, None),
        4: ([0, 2, 1, 3], True, None),
        5: ([3, 1, 0], False, [2, 3, 1, 0]),
        6: ([1, 0, 3, 2, 1], True, [1, 2, 3, 0]),
    }

    def _load_level(self, level: int) -> None:
        seq, hide, legend = self._SPECS[level]
        self.seq = list(seq)
        self.hide = hide
        self.legend = legend            # legend[i]: cue color i pairs with button color legend[i]
        self.progress = 0
        self.pressed_once = False

    def _button_at(self, px, py):
        for i in range(4):
            x0 = 4 + i * (BTN + 5)
            if x0 <= px < x0 + BTN and BTN_Y <= py < BTN_Y + BTN:
                return i
        return None

    def _expected_button(self):
        cue = self.seq[self.progress]
        return cue if self.legend is None else self.legend[cue]

    def _apply(self, action, x, y):
        b = self._button_at(x, y)
        if b is None:
            return False
        self.pressed_once = True
        if b == self._expected_button():
            self.progress += 1
            return self.progress == len(self.seq)
        self.progress = 0
        return False

    def _render(self):
        g = canvas(0)
        hide = self.hide and self.pressed_once
        for i, cue in enumerate(self.seq):                # cue strip
            c = 1 if hide else PALETTE[cue]
            rect(g, 4, 4 + i * 8, 6, 6, c)
        for i in range(len(self.seq)):                    # progress strip
            c = 14 if i < self.progress else 1
            rect(g, 14, 4 + i * 8, 3, 6, c)
        if self.legend is not None:                       # legend pairs
            for i in range(4):
                x0 = 4 + i * 14
                border(g, 24, x0, 8, 12, 5)
                rect(g, 26, x0 + 2, 4, 4, PALETTE[i])
                rect(g, 26, x0 + 7, 4, 4, PALETTE[self.legend[i]])
        for i in range(4):                                # buttons
            x0 = 4 + i * (BTN + 5)
            rect(g, BTN_Y, x0, BTN, BTN, PALETTE[i])
            border(g, BTN_Y, x0, BTN, BTN, 5)
        return g
