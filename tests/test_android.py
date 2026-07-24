import numpy as np
import pytest

from drlqn.android.core import (
    AccessibilityNode, AccessibilitySnapshot, AlignmentError, AndroidBatchConfig,
    AndroidState, Bounds, Caption, ScreenshotFrame, TimestampSynchronizer,
    VerificationExpectation, ViewportMetadata, normalize_screenshot,
    prepare_model_inputs, verify_transition,
)
from drlqn.android.window_graph import PlannerSubgoal, SubgoalType, WindowGraph
from drlqn.model.android_drlqn import AndroidDRLQN, AndroidDRLQNConfig
from drlqn.train.android_losses import peng_q_lambda_targets


def state(ts=1000,text="Home",activity=".Main",window=1):
    node=AccessibilityNode("n1",text=text,bounds=Bounds(0,0,100,50),clickable=True,window_id=window)
    return AndroidState(ScreenshotFrame(ts,np.zeros((128,256,3),dtype=np.uint8)),
        AccessibilitySnapshot(ts,(node,),window), (Caption(ts,text),),
        "org.example",activity,ViewportMetadata(256,128))


def test_timestamp_alignment_rejects_stale_tree():
    sync=TimestampSynchronizer(max_tree_skew_ms=10)
    with pytest.raises(AlignmentError):
        sync.synchronize(ScreenshotFrame(100,np.zeros((2,2,3))),
            [AccessibilitySnapshot(150,())],[],"p","a",ViewportMetadata(2,2))


def test_letterbox_and_batch_shapes():
    image,transform=normalize_screenshot(np.zeros((100,200,3),dtype=np.uint8))
    assert image.shape==(64,64,3) and transform.pad_y>0
    batch=prepare_model_inputs([state()],"open settings",AndroidBatchConfig(history=3,max_nodes=8,max_text_tokens=8,max_events=4))
    assert batch["screenshots"].shape==(1,3,64,64,3)
    assert batch["history_mask"].tolist()==[[False,False,True]]


def test_bfs_and_verifier():
    before=state(text="Home",activity=".Main",window=1)
    after=state(ts=1100,text="Bluetooth",activity=".Settings",window=2)
    graph=WindowGraph(); graph.observe_transition(before,after,PlannerSubgoal(SubgoalType.FIND_SETTING,"Bluetooth"),success=True)
    plan=graph.shortest_plan(before,"Bluetooth")
    assert plan and plan[0].kind is SubgoalType.FIND_SETTING
    result=verify_transition(before,after,VerificationExpectation(expected_text="Bluetooth",require_window_change=True))
    assert result.success


def test_peng_q_lambda_targets():
    target=peng_q_lambda_targets([[1,1]],[[.9,.9]],[[2,3]],trace_lambda=1.0)
    assert target.shape==(1,2)
    assert np.allclose(target,[[1+.9*(1+.9*3),1+.9*3]])


def test_android_model_head_shapes():
    cfg=AndroidDRLQNConfig(d_model=32,heads=2,fusion_blocks=1,history=3,max_nodes=8,heatmap_size=8)
    batch=prepare_model_inputs([state()],"open settings",AndroidBatchConfig(history=3,max_nodes=8,max_text_tokens=8,max_events=4))
    outputs=AndroidDRLQN(cfg,seed=0)(batch)
    assert outputs["planner_logits"].shape==(1,len(SubgoalType))
    assert outputs["ground_node_logits"].shape==(1,8)
    assert outputs["ground_heatmap"].shape==(1,8,8)
    assert outputs["control_points"].shape==(1,4,4,2)
