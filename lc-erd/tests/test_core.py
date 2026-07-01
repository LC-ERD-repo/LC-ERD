import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lcerd.cache import TopKLogitCache
from lcerd.core import (
    StepTrace,
    Trajectory,
    confidence_weights,
    kl_from_logits,
    lcerd_rewards,
    weighted_consensus_logits,
)
from lcerd.verifiers import NumericAnswerVerifier, extract_last_number


def _trajectory(mean_logprob: float, logits: torch.Tensor, verified: bool = True) -> Trajectory:
    return Trajectory(
        query="q",
        steps=(StepTrace(state="s", action=0, base_logprob=mean_logprob, base_logits=logits),),
        verified=verified,
    )


def test_confidence_weights_prefer_higher_probability_verified_paths() -> None:
    trajectories = [
        _trajectory(-0.1, torch.tensor([2.0, 0.0])),
        _trajectory(-3.0, torch.tensor([0.0, 2.0])),
        _trajectory(0.0, torch.tensor([9.0, 9.0]), verified=False),
    ]
    weights = confidence_weights(trajectories)
    assert weights.shape == (2,)
    assert weights[0] > weights[1]
    assert torch.isclose(weights.sum(), torch.tensor(1.0))


def test_weighted_consensus_averages_verified_logits() -> None:
    trajectories = [
        _trajectory(-0.1, torch.tensor([4.0, 0.0])),
        _trajectory(-0.1, torch.tensor([0.0, 2.0])),
    ]
    consensus = weighted_consensus_logits(trajectories, torch.tensor([0.25, 0.75]))
    assert len(consensus) == 1
    assert torch.allclose(consensus[0], torch.tensor([1.0, 1.5]))


def test_kl_is_zero_for_identical_logits() -> None:
    logits = torch.tensor([1.0, 2.0, -1.0])
    assert torch.isclose(kl_from_logits(logits, logits), torch.tensor(0.0))


def test_lcerd_rewards_use_potential_difference() -> None:
    rewards = lcerd_rewards(
        torch.tensor([-0.5, -0.25]),
        torch.tensor([1.0, 0.5, 0.0]),
        gamma=0.9,
    )
    assert torch.allclose(rewards, torch.tensor([-1.05, -0.75]))


def test_topk_cache_round_trips_shape() -> None:
    cache = TopKLogitCache(k=2)
    packed = cache.pack(torch.tensor([[1.0, 4.0, 2.0, 3.0]]))
    dense = packed.dense()
    assert dense.shape == (1, 4)
    assert torch.isfinite(dense[0, 1])
    assert torch.isfinite(dense[0, 3])


def test_numeric_verifier() -> None:
    assert extract_last_number("answer is 3/2") == 1.5
    assert NumericAnswerVerifier()("therefore 42", "#### 42")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
