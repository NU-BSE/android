"""Optional PyTorch/PEFT student used for real VLM and LLM weight updates."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .distillation import DistillationSample, OnPolicyConfig, ProposalKind


@dataclass(frozen=True)
class JointOutputNames:
    action_logits: str = "action_logits"
    box_prediction: str = "box_prediction"
    language_logits: str = "language_logits"
    reference_action_logits: str = "reference_action_logits"


class JointTorchVlmLlmStudent:
    """Model-specific preprocessing with shared, reviewable loss semantics.

    The wrapped VLM and LLM may be Hugging Face, custom torch modules, or PEFT
    adapters. The batch encoder must create tensors from rollout samples and the
    models must return the configured output keys.
    """

    def __init__(self, vlm: object, llm: object, batch_encoder: object, names: JointOutputNames | None = None):
        self.vlm = vlm
        self.llm = llm
        self.batch_encoder = batch_encoder
        self.names = names or JointOutputNames()

    def parameters(self):
        yield from self.vlm.parameters()
        yield from self.llm.parameters()

    def train(self, mode: bool = True):
        self.vlm.train(mode)
        self.llm.train(mode)
        return self

    def loss(self, samples: Sequence[DistillationSample], config: OnPolicyConfig):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            raise RuntimeError("install android-drlqn[train] to update torch weights") from exc

        batch = self.batch_encoder(samples)
        vlm_out = self.vlm(**batch["vlm_inputs"])
        llm_out = self.llm(**batch["llm_inputs"])
        action_logits = _get(vlm_out, self.names.action_logits)
        targets = torch.tensor(
            [list(ProposalKind).index(sample.judgement.target.preferred_kind) for sample in samples],
            device=action_logits.device,
            dtype=torch.long,
        )
        supervised = F.cross_entropy(action_logits, targets)
        log_probs = F.log_softmax(action_logits, dim=-1)
        chosen = torch.tensor(
            [list(ProposalKind).index(sample.proposal.kind) for sample in samples],
            device=action_logits.device,
            dtype=torch.long,
        )
        advantages = torch.tensor([sample.advantage for sample in samples], device=action_logits.device)
        policy = -(log_probs.gather(1, chosen[:, None]).squeeze(1) * advantages).mean()

        grounding = action_logits.new_zeros(())
        predicted_boxes = _optional(vlm_out, self.names.box_prediction)
        box_rows = [sample.judgement.target.bbox_norm for sample in samples]
        valid = [index for index, value in enumerate(box_rows) if value is not None]
        if predicted_boxes is not None and valid:
            indexes = torch.tensor(valid, device=predicted_boxes.device)
            targets_box = torch.tensor([box_rows[index] for index in valid], device=predicted_boxes.device, dtype=predicted_boxes.dtype)
            grounding = F.smooth_l1_loss(predicted_boxes.index_select(0, indexes), targets_box)

        language_logits = _get(llm_out, self.names.language_logits)
        labels = batch["language_labels"].to(language_logits.device)
        language = F.cross_entropy(language_logits.reshape(-1, language_logits.shape[-1]), labels.reshape(-1), ignore_index=-100)

        kl = action_logits.new_zeros(())
        reference = _optional(vlm_out, self.names.reference_action_logits)
        if reference is not None:
            kl = F.kl_div(F.log_softmax(action_logits, dim=-1), F.softmax(reference.detach(), dim=-1), reduction="batchmean")

        total = config.supervised_weight * supervised + config.policy_weight * policy + config.grounding_weight * grounding + config.language_weight * language + config.kl_weight * kl
        return total, {"supervised": supervised.detach(), "policy": policy.detach(), "grounding": grounding.detach(), "language": language.detach(), "kl": kl.detach()}

    def checkpoint(self, directory: str | Path, step: int) -> str:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("install android-drlqn[train]") from exc
        directory = Path(directory) / f"step-{step:08d}"
        directory.mkdir(parents=True, exist_ok=True)
        vlm_path = directory / "vlm.pt"
        llm_path = directory / "llm.pt"
        torch.save(self.vlm.state_dict(), vlm_path)
        torch.save(self.llm.state_dict(), llm_path)
        import hashlib
        digest = hashlib.sha256(vlm_path.read_bytes() + llm_path.read_bytes()).hexdigest()[:24]
        (directory / "checkpoint.id").write_text(digest, encoding="utf-8")
        return digest


def apply_lora(model: object, target_modules: Sequence[str], rank: int = 16, alpha: int = 32, dropout: float = 0.05):
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("install android-drlqn[train] for PEFT/LoRA") from exc
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(target_modules), bias="none")
    return get_peft_model(model, config)


def make_joint_loss(student: JointTorchVlmLlmStudent):
    def loss_fn(_module: object, samples: Sequence[DistillationSample], config: OnPolicyConfig):
        return student.loss(samples, config)[0]
    return loss_fn


def _get(output: object, name: str):
    value = _optional(output, name)
    if value is None:
        raise KeyError(f"model output is missing {name!r}")
    return value


def _optional(output: object, name: str):
    if isinstance(output, Mapping):
        return output.get(name)
    return getattr(output, name, None)
