"""Lädt eine im C++-Training gespeicherte Policy und rechnet identisch zu RLGymPPO_CPP.

Der C++-Learner speichert `PPO_POLICY.lt` mit torch::save. Python kann das mit
torch.jit.load lesen und die Gewichte übernehmen. Der Rechenweg (Sequential mit ReLU,
Softmax, Clamp, argmax) entspricht RLGymPPO_CPP DiscretePolicy; geprüft in
tests/test_policy_parity.py gegen dump_policy_actions.exe.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

# RLGymPPO_CPP DiscretePolicy::ACTION_MIN_PROB
ACTION_MIN_PROB = 1e-11


@dataclass
class PolicyMeta:
    obs_size: int
    action_count: int
    layer_sizes: list[int]
    source: str = ""
    timesteps: int = 0


def _build_sequential(obs_size: int, action_count: int, layer_sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(obs_size, layer_sizes[0]), nn.ReLU()]
    for prev, size in zip(layer_sizes, layer_sizes[1:]):
        layers += [nn.Linear(prev, size), nn.ReLU()]
    layers.append(nn.Linear(layer_sizes[-1], action_count))
    return nn.Sequential(*layers)


class Policy:
    """Deterministische Inferenz für das Deployment."""

    def __init__(self, seq: nn.Sequential, meta: PolicyMeta, device: str = "cpu"):
        self.seq = seq.to(device).eval()
        self.meta = meta
        self.device = device
        for p in self.seq.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        single = obs.ndim == 1
        x = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32, device=self.device)
        probs = torch.softmax(self.seq(x), dim=-1).clamp(ACTION_MIN_PROB, 1.0)
        out = probs.cpu().numpy()
        return out[0] if single else out

    def act(self, obs: np.ndarray, deterministic: bool = True) -> int | np.ndarray:
        probs = self.action_probs(obs)
        if deterministic:
            return int(np.argmax(probs)) if probs.ndim == 1 else np.argmax(probs, axis=-1)
        if probs.ndim == 1:
            return int(np.random.choice(len(probs), p=probs / probs.sum()))
        return np.array([np.random.choice(p.size, p=p / p.sum()) for p in probs])


def _state_dict_from_lt(path: Path) -> dict[str, torch.Tensor]:
    """torch::save-Archive aus C++ lassen sich in Python über torch.jit.load öffnen."""
    module = torch.jit.load(str(path), map_location="cpu")
    return {k: v.clone() for k, v in module.state_dict().items()}


def _meta_from_state_dict(sd: dict[str, torch.Tensor]) -> PolicyMeta:
    # Schlüssel sind "<index>.weight"/"<index>.bias" in Reihenfolge der Sequential-Module
    weights = [v for k, v in sorted(sd.items(), key=lambda kv: int(kv[0].split(".")[0]))
               if k.endswith("weight")]
    if not weights:
        raise ValueError("Checkpoint enthält keine Gewichte")
    return PolicyMeta(
        obs_size=weights[0].shape[1],
        action_count=weights[-1].shape[0],
        layer_sizes=[w.shape[0] for w in weights[:-1]],
    )


def load_policy(path: str | Path, device: str = "cpu") -> Policy:
    """Lädt aus einem Checkpoint-Ordner, einer PPO_POLICY.lt oder einer exportierten .pt."""
    path = Path(path)
    if path.is_dir():
        candidate = path / "PPO_POLICY.lt"
        if not candidate.exists():
            raise FileNotFoundError(f"PPO_POLICY.lt fehlt in {path}")
        path = candidate

    if path.suffix == ".lt":
        sd = _state_dict_from_lt(path)
        meta = _meta_from_state_dict(sd)
        meta.source = str(path)
        try:
            meta.timesteps = int(path.parent.name)
        except ValueError:
            pass
    else:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        sd = blob["state_dict"]
        meta = PolicyMeta(**blob["meta"])

    seq = _build_sequential(meta.obs_size, meta.action_count, meta.layer_sizes)
    seq.load_state_dict(sd)
    return Policy(seq, meta, device)


def export_policy(checkpoint: str | Path, out_path: str | Path) -> PolicyMeta:
    """Schreibt Gewichte plus Metadaten als eine .pt-Datei für den RLBot-Ordner."""
    policy = load_policy(checkpoint)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.seq.state_dict(), "meta": policy.meta.__dict__}, out_path)
    (out_path.with_suffix(".json")).write_text(
        json.dumps(policy.meta.__dict__, indent=2), encoding="utf-8")
    return policy.meta
