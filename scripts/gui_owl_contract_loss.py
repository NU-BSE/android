"""Token weighting helpers for the GUI-Owl Action + tool-call contract."""
from __future__ import annotations

from typing import Sequence


def _token_ids(tokenizer, text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in tokenizer.encode(text, add_special_tokens=False))


def _find(ids: Sequence[int], pattern: Sequence[int], start: int = 0) -> int | None:
    if not pattern:
        return None
    stop = len(ids) - len(pattern) + 1
    for index in range(max(0, start), max(0, stop)):
        if tuple(ids[index : index + len(pattern)]) == tuple(pattern):
            return index
    return None


def _find_any(ids: Sequence[int], patterns: Sequence[Sequence[int]], start: int = 0):
    matches = []
    for pattern in patterns:
        index = _find(ids, pattern, start)
        if index is not None:
            matches.append((index, tuple(pattern)))
    return min(matches, key=lambda value: value[0]) if matches else None


class ContractTokenWeighter:
    """Build per-token CE/KL weights for completions containing a tool call."""

    def __init__(
        self,
        tokenizer,
        *,
        prefix_weight: float = 1.0,
        suffix_weight: float = 1.0,
        focus_weight: float = 1.0,
        first_token_weight: float = 1.0,
        require_marker: bool = False,
    ) -> None:
        values = (prefix_weight, suffix_weight, focus_weight, first_token_weight)
        if min(values) <= 0:
            raise ValueError("contract token weights must be positive")
        if prefix_weight > suffix_weight or suffix_weight > focus_weight:
            raise ValueError("expected prefix_weight <= suffix_weight <= focus_weight")
        self.prefix_weight = float(prefix_weight)
        self.suffix_weight = float(suffix_weight)
        self.focus_weight = float(focus_weight)
        self.first_token_weight = float(first_token_weight)
        self.require_marker = bool(require_marker)
        self.open_patterns = tuple(
            pattern
            for pattern in (_token_ids(tokenizer, "\n<tool_call>"), _token_ids(tokenizer, "<tool_call>"))
            if pattern
        )
        self.close_patterns = tuple(
            pattern
            for pattern in (_token_ids(tokenizer, "</tool_call>"), _token_ids(tokenizer, "</tool_call>\n"))
            if pattern
        )
        self.focus_patterns = tuple(
            pattern
            for pattern in (
                _token_ids(tokenizer, "<tool_call>"),
                _token_ids(tokenizer, '\"name\":\"mobile.'),
                _token_ids(tokenizer, '\"arguments\":'),
                _token_ids(tokenizer, '\"coordinate\":['),
                _token_ids(tokenizer, '\"coordinate2\":['),
                _token_ids(tokenizer, "</tool_call>"),
            )
            if pattern
        )

    def weights(self, targets, torch_module):
        result = targets.new_ones(targets.shape, dtype=torch_module.float32)
        for row in range(int(targets.shape[0])):
            ids = [int(value) for value in targets[row].tolist()]
            opening = _find_any(ids, self.open_patterns)
            if opening is None:
                if self.require_marker:
                    raise RuntimeError("GUI contract bootstrap target lacks <tool_call> marker")
                if ids and self.first_token_weight != 1.0:
                    result[row, 0] = self.first_token_weight
                continue

            open_index, open_pattern = opening
            result[row, :] = self.prefix_weight
            closing = _find_any(ids, self.close_patterns, open_index + len(open_pattern))
            suffix_end = len(ids)
            if closing is not None:
                close_index, close_pattern = closing
                suffix_end = min(len(ids), close_index + len(close_pattern))
            result[row, open_index:suffix_end] = self.suffix_weight

            for pattern in self.focus_patterns:
                search = open_index
                while search < suffix_end:
                    found = _find(ids, pattern, search)
                    if found is None or found >= suffix_end:
                        break
                    end = min(suffix_end, found + len(pattern))
                    result[row, found:end] = self.focus_weight
                    search = max(found + 1, end)

            # This is the exact autoregressive transition that failed in the
            # observed depth-24 students, so weight the opening marker highest.
            result[row, open_index : min(suffix_end, open_index + len(open_pattern))] = self.focus_weight
            if ids and self.first_token_weight != 1.0:
                result[row, 0] = max(float(result[row, 0]), self.first_token_weight)
        return result


__all__ = ["ContractTokenWeighter"]
