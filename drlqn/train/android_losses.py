"""Q(lambda) and multitask losses for Android trajectories."""
from __future__ import annotations
import numpy as np
from ..autodiff import T


def peng_q_lambda_targets(rewards, discounts, bootstrap_q, trace_lambda=.9, terminals=None):
    """Truncated Peng Q(lambda) returns for arrays shaped (B,T)."""
    rewards=np.asarray(rewards,float); discounts=np.asarray(discounts,float)
    bootstrap_q=np.asarray(bootstrap_q,float)
    if rewards.shape!=discounts.shape or rewards.shape!=bootstrap_q.shape:
        raise ValueError("rewards, discounts and bootstrap_q must have same shape")
    terminals=np.zeros_like(rewards,bool) if terminals is None else np.asarray(terminals,bool)
    out=np.zeros_like(rewards); carry=bootstrap_q[:,-1]
    lam=float(trace_lambda)
    for t in range(rewards.shape[1]-1,-1,-1):
        gamma=discounts[:,t]*(~terminals[:,t])
        one_step=rewards[:,t]+gamma*bootstrap_q[:,t]
        carry=(1-lam)*one_step+lam*(rewards[:,t]+gamma*carry)
        out[:,t]=carry
    return out


def huber(error, delta=1.0):
    absolute=(error*error+1e-12)**.5
    quadratic=(absolute.data<=delta).astype(float)
    return T(quadratic)*.5*error*error+T(1-quadratic)*delta*(absolute-.5*delta)


def residual_q_loss(q_selected, target_values, q_next_online=None, q_next_target=None,
                    residual_mix=.1):
    """Stable semi-gradient target loss plus a small Bellman-residual term."""
    target=T(np.asarray(target_values,float))
    semi=huber(q_selected-target).mean()
    if q_next_online is None or q_next_target is None or residual_mix<=0:
        return semi
    correction=huber(q_next_online-q_next_target.detach()).mean()
    return (1-residual_mix)*semi+residual_mix*correction


def binary_cross_entropy_with_logits(logits, labels):
    labels=T(np.asarray(labels,float)); probabilities=logits.sigmoid()
    eps=1e-7
    return -(labels*(probabilities+eps).log()+(1-labels)*(1-probabilities+eps).log()).mean()


def android_multitask_loss(outputs, targets, weights=None):
    """Compose grounding, planner, gesture, verifier and world-model objectives."""
    weights=weights or {}; losses={}
    def ce(logits, labels):
        labels=np.asarray(labels,int); probs=logits.softmax(axis=-1)
        rows=np.arange(labels.shape[0]); return -(probs[rows,labels]+1e-9).log().mean()
    for name,key in (("planner","planner_logits"),("node","ground_node_logits"),
                     ("gesture","gesture_type_logits"),("fingers","finger_count_logits")):
        target=targets.get(name)
        if target is not None: losses[name]=ce(outputs[key],target)
    if "box" in targets:
        error=outputs["ground_box"]-T(targets["box"]); losses["box"]=(error*error).mean()
    if "heatmap" in targets:
        error=outputs["ground_heatmap"]-T(targets["heatmap"]); losses["heatmap"]=(error*error).mean()
    if "control_points" in targets:
        error=outputs["control_points"]-T(targets["control_points"]); losses["control_points"]=(error*error).mean()
    if "verifier" in targets and "verifier_logits" in outputs:
        losses["verifier"]=binary_cross_entropy_with_logits(outputs["verifier_logits"],targets["verifier"])
    if "next_state" in targets:
        error=outputs["next_state_prediction"]-T(targets["next_state"]); losses["world_model"]=(error*error).mean()
    total=T(0.0)
    for name,value in losses.items(): total=total+weights.get(name,1.0)*value
    return total,losses
