"""Minimal one-step chat advisor example."""
import numpy as np

from android_drlqn import (
    AccessibilityNode,
    AccessibilitySnapshot,
    AndroidState,
    Bounds,
    Caption,
    PlannerAdvisor,
    PlannerSubgoal,
    ScreenshotFrame,
    SubgoalType,
    ViewportMetadata,
)

state = AndroidState(
    screenshot=ScreenshotFrame(1000, np.zeros((64, 64, 3), dtype=np.uint8)),
    accessibility=AccessibilitySnapshot(
        1000,
        (
            AccessibilityNode("1", text="Network & internet", bounds=Bounds(0, 0, 100, 20)),
            AccessibilityNode("2", text="Bluetooth", bounds=Bounds(0, 20, 100, 40)),
        ),
    ),
    captions=(Caption(1000, "Settings"),),
    package="com.android.settings",
    activity=".Settings",
    viewport=ViewportMetadata(1080, 2400),
)
plan = (PlannerSubgoal(SubgoalType.FIND_SETTING, "Bluetooth"),)
reply = PlannerAdvisor().advise(plan, state, "Включить Bluetooth")
print(reply.step.instruction)
print(reply.step.confirmation_question)
