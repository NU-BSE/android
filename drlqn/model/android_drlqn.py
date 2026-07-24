"""Multimodal DRLQN heads for Android window trajectories.

Planner emits semantic subgoals; grounder and gesture decoder own coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..autodiff import T, Module, Linear, MLP, Param
from ..android.core import ACCESS_NODE_DIM, TEXT_TOKEN_DIM, META_DIM, ACTION_HISTORY_DIM, EVENT_DIM
from ..android.window_graph import SubgoalType
from .relational_encoder import EncoderBlock, MultiHeadAttention


@dataclass
class AndroidDRLQNConfig:
    d_model: int = 128
    heads: int = 4
    fusion_blocks: int = 2
    history: int = 8
    max_nodes: int = 96
    heatmap_size: int = 16
    planner_args: int = 32
    gesture_types: int = 9
    max_fingers: int = 4
    control_points: int = 4
    verifier_labels: int = 6

    def __post_init__(self):
        if not 3 <= self.history <= 8:
            raise ValueError("history must be in [3, 8]")


class AndroidDRLQN(Module):
    """Small dependency-free multimodal model using the repo numpy autodiff.

    Inputs are produced by ``prepare_model_inputs`` and shaped (B,H,...).
    The latest frame's accessibility tokens are preserved for pointer grounding.
    """
    def __init__(self, cfg: AndroidDRLQNConfig = AndroidDRLQNConfig(), seed: int = 0):
        rng=np.random.default_rng(seed); d=cfg.d_model; self.cfg=cfg
        self.vision_proj=Linear(3,d,rng)
        self.node_proj=Linear(ACCESS_NODE_DIM,d,rng)
        self.text_proj=Linear(TEXT_TOKEN_DIM,d,rng)
        self.event_proj=Linear(EVENT_DIM,d,rng)
        self.meta_proj=Linear(META_DIM+ACTION_HISTORY_DIM,d,rng)
        self.modality_tokens=Param(rng.normal(0,.15,size=(5,d)))
        self.fusion=[EncoderBlock(d,cfg.heads,rng) for _ in range(cfg.fusion_blocks)]
        self.frame_query=Param(rng.normal(0,.15,size=(1,d)))
        self.frame_pool=MultiHeadAttention(d,cfg.heads,rng)
        self.temporal_in=Linear(d,3*d,rng)
        self.temporal_h=Linear(d,3*d,rng,bias=False)
        self.planner=MLP([d,d,len(SubgoalType)],rng)
        self.planner_arg=Linear(d,cfg.planner_args,rng)
        self.node_key=Linear(d,d,rng)
        self.node_query=Linear(d,d,rng)
        self.box_head=MLP([2*d,d,4],rng)
        self.heatmap=MLP([d,d,cfg.heatmap_size*cfg.heatmap_size],rng)
        gesture_param=cfg.max_fingers*cfg.control_points*2
        self.gesture_type=Linear(d,cfg.gesture_types,rng)
        self.finger_count=Linear(d,cfg.max_fingers,rng)
        self.control_points=MLP([d,d,gesture_param],rng)
        self.duration=MLP([d,d,1],rng)
        self.velocity=MLP([d,d,3],rng)
        self.verifier=MLP([2*d,d,cfg.verifier_labels],rng)
        self.world_model=MLP([2*d,d,d],rng)
        self.lambda_gate=Linear(2*d,d,rng)
        self.lambda_delta=MLP([2*d,d,d],rng)

    @staticmethod
    def _masked_mean(x,mask):
        weights=mask.astype(np.float64)[...,None]
        denom=np.maximum(weights.sum(axis=-2),1.0)
        return (x*T(weights)).sum(axis=-2)/T(denom)

    def _vision_tokens(self,images):
        # 64x64 -> 8x8 patch means, retaining a small vision token grid.
        B,H=images.shape[:2]
        patches=images.reshape(B,H,8,8,8,8,3).mean(axis=(3,5))
        return self.vision_proj(T(patches.reshape(B*H,64,3)))

    def _encode_frames(self,batch):
        cfg=self.cfg; d=cfg.d_model
        images=batch["screenshots"]; B,H=images.shape[:2]; BH=B*H
        vision=self._vision_tokens(images)
        nodes=self.node_proj(T(batch["node_features"].reshape(BH,-1,ACCESS_NODE_DIM)))
        texts=self.text_proj(T(batch["text_features"].reshape(BH,-1,TEXT_TOKEN_DIM)))
        events=self.event_proj(T(batch["events"].reshape(BH,-1,EVENT_DIM)))
        meta=np.concatenate([batch["metadata"],batch["previous_actions"]],axis=-1)
        meta=self.meta_proj(T(meta.reshape(BH,1,-1)))
        tokens=T.concat([vision,nodes,texts,events,meta],axis=1)
        valid=np.concatenate([
            np.ones((BH,vision.shape[1]),bool),
            batch["node_mask"].reshape(BH,-1),
            batch["text_mask"].reshape(BH,-1),
            batch["event_mask"].reshape(BH,-1),
            np.ones((BH,1),bool),
        ],axis=1)
        offsets=[0,vision.shape[1],vision.shape[1]+nodes.shape[1],vision.shape[1]+nodes.shape[1]+texts.shape[1],tokens.shape[1]-1]
        for i,offset in enumerate(offsets):
            bias=np.zeros((BH,tokens.shape[1],d)); bias[:,offset,:]=self.modality_tokens[i].data
            tokens=tokens+T(bias)
        for block in self.fusion:
            tokens=block(tokens,valid)
        query=self.frame_query.reshape(1,1,d)+T(np.zeros((BH,1,1)))
        frames=self.frame_pool(query,tokens,valid).reshape(B,H,d)
        latest_nodes=nodes.reshape(B,H,-1,d)[:,-1]
        latest_mask=batch["node_mask"][:,-1]
        return frames,latest_nodes,latest_mask

    def _temporal(self,frames,history_mask):
        B,H,d=frames.shape; hidden=T(np.zeros((B,d)))
        for step in range(H):
            gates=self.temporal_in(frames[:,step])+self.temporal_h(hidden)
            update=gates[:,:d].sigmoid(); reset=gates[:,d:2*d].sigmoid()
            candidate=(gates[:,2*d:]+reset*hidden).tanh()
            next_hidden=(1-update)*hidden+update*candidate
            valid=T(history_mask[:,step:step+1].astype(float))
            hidden=valid*next_hidden+(1-valid)*hidden
        return hidden

    def lambda_update(self,state_latent,action_embedding):
        joined=T.concat([state_latent,action_embedding],axis=-1)
        return state_latent+self.lambda_gate(joined).sigmoid()*self.lambda_delta(joined).tanh()

    def __call__(self,batch,next_state_latent=None):
        frames,nodes,node_mask=self._encode_frames(batch)
        latent=self._temporal(frames,batch["history_mask"])
        query=self.node_query(latent).reshape(latent.shape[0],1,self.cfg.d_model)
        keys=self.node_key(nodes)
        node_logits=(keys*query).sum(axis=-1)*(1/np.sqrt(self.cfg.d_model))
        node_logits=node_logits+T(np.where(node_mask,0.0,-1e9))
        node_weights=node_logits.softmax(axis=-1).reshape(latent.shape[0],-1,1)
        grounded=(nodes*node_weights).sum(axis=1)
        combined=T.concat([latent,grounded],axis=-1)
        outputs={
            "state_latent":latent,
            "planner_logits":self.planner(latent),
            "planner_argument":self.planner_arg(latent),
            "ground_node_logits":node_logits,
            "ground_box":self.box_head(combined).sigmoid(),
            "ground_heatmap":self.heatmap(latent).reshape(latent.shape[0],self.cfg.heatmap_size,self.cfg.heatmap_size),
            "gesture_type_logits":self.gesture_type(latent),
            "finger_count_logits":self.finger_count(latent),
            "control_points":self.control_points(latent).sigmoid().reshape(latent.shape[0],self.cfg.max_fingers,self.cfg.control_points,2),
            "duration_ms":self.duration(latent).exp(),
            "velocity_profile":self.velocity(latent),
            "next_state_prediction":self.world_model(combined),
        }
        if next_state_latent is not None:
            outputs["verifier_logits"]=self.verifier(T.concat([latent,next_state_latent],axis=-1))
        return outputs
