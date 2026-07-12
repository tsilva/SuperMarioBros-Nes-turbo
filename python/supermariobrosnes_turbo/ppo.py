from __future__ import annotations

import io
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class NatureCNN(nn.Module):
    def __init__(self, input_channels: int = 4) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations.float() / 255.0))


class PlainPPOPolicy(nn.Module):
    """Nature-CNN actor critic used by the standalone PPO trainer."""

    def __init__(self, input_channels: int = 4, action_count: int = 7) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.action_count = int(action_count)
        self.features_extractor = NatureCNN(self.input_channels)
        self.action_net = nn.Linear(512, self.action_count)
        self.value_net = nn.Linear(512, 1)
        self.apply(self._orthogonal_init)
        nn.init.orthogonal_(self.action_net.weight, gain=0.01)
        nn.init.orthogonal_(self.value_net.weight, gain=1.0)

    @staticmethod
    def _orthogonal_init(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def distribution_and_value(
        self, observations: torch.Tensor
    ) -> tuple[Categorical, torch.Tensor]:
        features = self.features_extractor(observations)
        return Categorical(logits=self.action_net(features)), self.value_net(features).flatten()

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, values = self.distribution_and_value(observations)
        return distribution.log_prob(actions), distribution.entropy(), values

    def predict(
        self, observations: np.ndarray, *, deterministic: bool = False
    ) -> tuple[np.ndarray, None]:
        device = next(self.parameters()).device
        with torch.no_grad():
            distribution, _values = self.distribution_and_value(
                torch.as_tensor(observations, device=device)
            )
            actions = torch.argmax(distribution.logits, dim=1) if deterministic else distribution.sample()
        return actions.cpu().numpy(), None


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def save_policy_checkpoint(
    path: str | Path,
    policy: PlainPPOPolicy,
    *,
    timesteps: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "plain-ppo-v1",
            "timesteps": int(timesteps),
            "input_channels": policy.input_channels,
            "action_count": policy.action_count,
            "model_state_dict": _cpu_state_dict(policy),
            "metadata": dict(metadata or {}),
        },
        target,
    )
    return target


def _load_legacy_sb3_zip(path: Path) -> tuple[dict[str, torch.Tensor], int, int]:
    with zipfile.ZipFile(path) as archive:
        try:
            raw_state = torch.load(
                io.BytesIO(archive.read("policy.pth")),
                map_location="cpu",
                weights_only=True,
            )
        except KeyError as exc:
            raise ValueError(f"{path} is not a supported PPO checkpoint") from exc
    state = {
        name: value
        for name, value in raw_state.items()
        if name.startswith("features_extractor.")
        or name.startswith("action_net.")
        or name.startswith("value_net.")
    }
    input_channels = int(state["features_extractor.cnn.0.weight"].shape[1])
    action_count = int(state["action_net.weight"].shape[0])
    return state, input_channels, action_count


def load_policy_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> PlainPPOPolicy:
    source = Path(path)
    if source.suffix == ".zip":
        state, input_channels, action_count = _load_legacy_sb3_zip(source)
    else:
        payload = torch.load(source, map_location="cpu", weights_only=True)
        if payload.get("format") != "plain-ppo-v1":
            raise ValueError(f"{source} is not a plain PPO checkpoint")
        state = payload["model_state_dict"]
        input_channels = int(payload["input_channels"])
        action_count = int(payload["action_count"])
    policy = PlainPPOPolicy(input_channels=input_channels, action_count=action_count)
    policy.load_state_dict(state, strict=True)
    if str(device) == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    policy.to(device)
    policy.eval()
    return policy
