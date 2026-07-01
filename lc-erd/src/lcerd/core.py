from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class StepTrace:
    """One step-agent decision in a reasoning trajectory."""

    state: str
    action: int
    base_logprob: float
    base_logits: Tensor


@dataclass(frozen=True)
class Trajectory:
    """A sampled reasoning path, optionally accepted by a terminal verifier."""

    query: str
    steps: tuple[StepTrace, ...]
    text: str = ""
    verified: bool = False

    @property
    def mean_logprob(self) -> float:
        if not self.steps:
            return float("-inf")
        return sum(step.base_logprob for step in self.steps) / len(self.steps)


def _verified(trajectories: list[Trajectory]) -> list[Trajectory]:
    accepted = [trajectory for trajectory in trajectories if trajectory.verified and trajectory.steps]
    if not accepted:
        raise ValueError("LLE discovery needs at least one verified trajectory.")
    return accepted


def confidence_weights(trajectories: list[Trajectory], temperature: float = 1.0) -> Tensor:
    """Eq. 4: normalized confidence weights from average base log probability."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    accepted = _verified(trajectories)
    scores = torch.tensor([trajectory.mean_logprob / temperature for trajectory in accepted])
    return torch.softmax(scores, dim=0)


def weighted_consensus_logits(
    trajectories: list[Trajectory],
    weights: Tensor | None = None,
    *,
    top_k: int | None = None,
) -> list[Tensor]:
    """Eq. 5: confidence-weighted LLE logits for each reasoning step."""

    accepted = _verified(trajectories)
    if weights is None:
        weights = confidence_weights(accepted)
    if weights.numel() != len(accepted):
        raise ValueError("weights must have one value per verified trajectory")

    max_horizon = max(len(trajectory.steps) for trajectory in accepted)
    consensus: list[Tensor] = []

    for step_idx in range(max_horizon):
        active_ids = [
            traj_idx for traj_idx, trajectory in enumerate(accepted) if step_idx < len(trajectory.steps)
        ]
        active_weights = weights[active_ids]
        active_weights = active_weights / active_weights.sum().clamp_min(1e-12)
        logits = torch.stack(
            [accepted[traj_idx].steps[step_idx].base_logits for traj_idx in active_ids],
            dim=0,
        )
        step_logits = torch.sum(active_weights.to(logits.device).unsqueeze(-1) * logits, dim=0)
        consensus.append(_keep_top_k(step_logits, top_k))
    return consensus


def _keep_top_k(logits: Tensor, top_k: int | None) -> Tensor:
    if top_k is None or top_k >= logits.numel():
        return logits
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    values, indices = torch.topk(logits, k=top_k)
    sparse = torch.full_like(logits, torch.finfo(logits.dtype).min)
    sparse.scatter_(0, indices, values)
    return sparse


def kl_from_logits(policy_logits: Tensor, target_logits: Tensor) -> Tensor:
    """KL(policy || target) for unnormalized logits."""

    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    target_log_probs = F.log_softmax(target_logits, dim=-1)
    policy_probs = policy_log_probs.exp()
    return torch.sum(policy_probs * (policy_log_probs - target_log_probs), dim=-1)


def variational_potential(
    value: Tensor | float,
    policy_logits: Tensor,
    lle_logits: Tensor,
    beta: float,
) -> Tensor:
    """Eq. 6: Phi_LLE(s) = V(s) - beta * KL(pi_theta || pi_LLE)."""

    value_tensor = torch.as_tensor(value, dtype=policy_logits.dtype, device=policy_logits.device)
    return value_tensor - beta * kl_from_logits(policy_logits, lle_logits)


def lcerd_rewards(
    base_logprobs: Tensor,
    potentials: Tensor,
    *,
    gamma: float = 0.95,
    alpha: float = 1.0,
) -> Tensor:
    """Eq. 8: R = alpha * log pi_base(a|s) + gamma * Phi(s') - Phi(s)."""

    if potentials.numel() != base_logprobs.numel() + 1:
        raise ValueError("potentials must contain H + 1 values for H rewards")
    return alpha * base_logprobs + gamma * potentials[1:] - potentials[:-1]


def temporal_difference_loss(
    q_values: Tensor,
    target_next_q_values: Tensor,
    rewards: Tensor,
    actions: Tensor,
    *,
    gamma: float = 0.95,
) -> Tensor:
    """Eq. 12 TD loss with max over target-network next actions."""

    chosen_q = q_values.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
    next_q = target_next_q_values.max(dim=-1).values
    targets = rewards + gamma * next_q
    return F.mse_loss(chosen_q, targets.detach())


def individual_consistency_loss(policy_logits: Tensor, lle_logits: Tensor) -> Tensor:
    """Eq. 13 logit consistency loss weighted by the LLE distribution."""

    lle_probs = F.softmax(lle_logits, dim=-1)
    sq_error = (policy_logits - lle_logits).pow(2)
    return torch.sum(lle_probs * sq_error, dim=-1).mean()


def lcerd_losses(
    policy_logits: Tensor,
    q_values: Tensor,
    target_next_q_values: Tensor,
    lle_logits: Tensor,
    base_logprobs: Tensor,
    values: Tensor,
    actions: Tensor,
    *,
    beta: float = 0.1,
    gamma: float = 0.95,
    alpha: float = 1.0,
    lambda_indiv: float = 1.0,
) -> dict[str, Tensor]:
    """Compute LC-ERD rewards and the joint optimization objective."""

    if values.numel() != base_logprobs.numel() + 1:
        raise ValueError("values must contain H + 1 entries")

    step_count = base_logprobs.numel()
    potentials = []
    for step_idx in range(step_count):
        potentials.append(
            variational_potential(
                values[step_idx],
                policy_logits[step_idx],
                lle_logits[step_idx],
                beta,
            )
        )
    potentials.append(values[-1].to(dtype=policy_logits.dtype, device=policy_logits.device))
    potential_tensor = torch.stack(potentials)

    rewards = lcerd_rewards(base_logprobs, potential_tensor, gamma=gamma, alpha=alpha)
    td = temporal_difference_loss(q_values, target_next_q_values, rewards, actions, gamma=gamma)
    indiv = individual_consistency_loss(policy_logits, lle_logits)
    total = td + lambda_indiv * indiv
    return {
        "total": total,
        "td": td,
        "individual": indiv,
        "rewards": rewards.detach(),
        "potentials": potential_tensor.detach(),
    }
