"""LC-ERD reproduction primitives."""

from .core import (
    StepTrace,
    Trajectory,
    confidence_weights,
    individual_consistency_loss,
    kl_from_logits,
    lcerd_losses,
    lcerd_rewards,
    temporal_difference_loss,
    variational_potential,
    weighted_consensus_logits,
)

__all__ = [
    "StepTrace",
    "Trajectory",
    "confidence_weights",
    "individual_consistency_loss",
    "kl_from_logits",
    "lcerd_losses",
    "lcerd_rewards",
    "temporal_difference_loss",
    "variational_potential",
    "weighted_consensus_logits",
]
