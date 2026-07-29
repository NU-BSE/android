"""Semantic Android window graph and BFS planner."""
from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from .core import AndroidState, GestureAction, tree_fingerprint


class SubgoalType(str, Enum):
    OPEN_MENU = "OPEN_MENU"
    FIND_SETTING = "FIND_SETTING"
    ENTER_VALUE = "ENTER_VALUE"
    CONFIRM = "CONFIRM"
    TAP_TARGET = "TAP_TARGET"
    SCROLL = "SCROLL"
    DRAG_SLIDER = "DRAG_SLIDER"
    BACK = "BACK"
    WAIT = "WAIT"
    DONE = "DONE"


@dataclass(frozen=True)
class PlannerSubgoal:
    kind: SubgoalType
    argument: str | None = None

    def serialize(self):
        return self.kind.value if self.argument is None else f'{self.kind.value}("{self.argument}")'


@dataclass(frozen=True)
class WindowSignature:
    package: str
    activity: str
    window_id: int
    tree_hash: str

    @classmethod
    def from_state(cls, state: AndroidState):
        return cls(state.package, state.activity, state.accessibility.active_window_id,
                   tree_fingerprint(state.accessibility))

    @property
    def key(self):
        raw=f"{self.package}\0{self.activity}\0{self.window_id}\0{self.tree_hash}".encode()
        return hashlib.sha1(raw).hexdigest()[:20]


@dataclass
class WindowNode:
    signature: WindowSignature
    visits: int = 0
    labels: set[str] = field(default_factory=set)


@dataclass
class WindowEdge:
    source: str
    target: str
    subgoal: PlannerSubgoal
    action: GestureAction | None = None
    attempts: int = 0
    successes: int = 0
    mean_cost: float = 1.0

    @property
    def success_rate(self):
        return self.successes/max(1,self.attempts)

    def observe(self, success, cost):
        self.attempts += 1
        self.successes += int(success)
        self.mean_cost += (float(cost)-self.mean_cost)/self.attempts


class WindowGraph:
    """BFS reasons over semantic edges; coordinates remain demonstrations."""
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.outgoing = {}

    def observe_state(self, state):
        signature=WindowSignature.from_state(state); key=signature.key
        node=self.nodes.setdefault(key,WindowNode(signature)); node.visits+=1
        node.labels.update(n.label.lower() for n in state.accessibility.nodes if n.label and n.visible)
        node.labels.update(c.text.lower() for c in state.captions if c.text)
        return key

    def observe_transition(self,before,after,subgoal,action=None,success=True,cost=1.0):
        source,target=self.observe_state(before),self.observe_state(after)
        identity=(source,target,subgoal.serialize())
        edge=self.edges.get(identity)
        if edge is None:
            edge=WindowEdge(source,target,subgoal,action)
            self.edges[identity]=edge; self.outgoing.setdefault(source,[]).append(edge)
        elif success and action is not None:
            edge.action=action
        edge.observe(success,cost)
        return edge

    def shortest_plan(self,start,goal,min_success_rate=.25,max_depth=32):
        start_key=self.observe_state(start) if isinstance(start,AndroidState) else start
        if start_key not in self.nodes: return None
        if callable(goal): predicate=goal
        else:
            needle=goal.lower()
            predicate=lambda n: needle in n.signature.package.lower() or needle in n.signature.activity.lower() or any(needle in x for x in n.labels)
        if predicate(self.nodes[start_key]): return []
        queue=deque([(start_key,[])]); visited={start_key}
        while queue:
            current,plan=queue.popleft()
            if len(plan)>=max_depth: continue
            edges=sorted(self.outgoing.get(current,()),key=lambda e:(-e.success_rate,e.mean_cost,e.subgoal.serialize()))
            for edge in edges:
                if edge.attempts and edge.success_rate<min_success_rate: continue
                if edge.target in visited or edge.target not in self.nodes: continue
                candidate=plan+[edge.subgoal]
                if predicate(self.nodes[edge.target]): return candidate
                visited.add(edge.target); queue.append((edge.target,candidate))
        return None
