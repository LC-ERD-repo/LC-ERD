from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lcerd.core import StepTrace, Trajectory, confidence_weights, lcerd_losses, weighted_consensus_logits


VOCAB = ("add", "subtract", "multiply", "divide", "copy")


def sample_trajectories(width: int, seed: int) -> tuple[list[Trajectory], tuple[int, ...]]:
    random.seed(seed)
    torch.manual_seed(seed)
    correct_actions = (0, 2, 0, 4)
    trajectories: list[Trajectory] = []

    for sample_idx in range(width):
        steps = []
        chosen = []
        for h, gold in enumerate(correct_actions):
            logits = torch.randn(len(VOCAB)) * 0.9
            logits[gold] += 1.2
            if random.random() < 0.25:
                logits[random.randrange(len(VOCAB))] += 2.0
            probs = torch.softmax(logits, dim=-1)
            action = int(torch.multinomial(probs, 1).item())
            chosen.append(action)
            steps.append(
                StepTrace(
                    state=f"step-{h}",
                    action=action,
                    base_logprob=float(torch.log(probs[action])),
                    base_logits=logits,
                )
            )
        verified = tuple(chosen) == correct_actions
        trajectories.append(
            Trajectory(
                query="Toy arithmetic chain: add, multiply, add, copy.",
                steps=tuple(steps),
                text=" -> ".join(VOCAB[idx] for idx in chosen),
                verified=verified,
            )
        )

    if not any(t.verified for t in trajectories):
        return sample_trajectories(width=width, seed=seed + 1)
    return trajectories, correct_actions


class TinyLCERDPolicy(nn.Module):
    def __init__(self, horizon: int, vocab_size: int):
        super().__init__()
        self.policy_logits = nn.Parameter(torch.zeros(horizon, vocab_size))
        self.q_values = nn.Parameter(torch.zeros(horizon, vocab_size))

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy_logits, self.q_values


def run(args: argparse.Namespace) -> None:
    trajectories, correct_actions = sample_trajectories(args.width, args.seed)
    accepted = [trajectory for trajectory in trajectories if trajectory.verified]
    weights = confidence_weights(trajectories, temperature=args.confidence_temperature)
    lle_logits = torch.stack(weighted_consensus_logits(trajectories, weights, top_k=args.top_k))

    reference = accepted[0]
    base_logprobs = torch.tensor([step.base_logprob for step in reference.steps])
    actions = torch.tensor([step.action for step in reference.steps], dtype=torch.long)
    values = torch.zeros(len(reference.steps) + 1)

    model = TinyLCERDPolicy(horizon=len(reference.steps), vocab_size=len(VOCAB))
    target_next_q = torch.zeros_like(model.q_values)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"verified={len(accepted)}/{len(trajectories)}")
    print("lle_top1=" + " ".join(VOCAB[int(i)] for i in lle_logits.argmax(dim=-1)))
    print("gold=" + " ".join(VOCAB[i] for i in correct_actions))

    for step in range(1, args.steps + 1):
        policy_logits, q_values = model()
        losses = lcerd_losses(
            policy_logits=policy_logits,
            q_values=q_values,
            target_next_q_values=target_next_q,
            lle_logits=lle_logits,
            base_logprobs=base_logprobs,
            values=values,
            actions=actions,
            beta=args.beta,
            gamma=args.gamma,
            lambda_indiv=args.lambda_indiv,
        )
        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        target_next_q.mul_(1 - args.target_tau).add_(args.target_tau * q_values.detach())

        if step == 1 or step % args.report_every == 0 or step == args.steps:
            with torch.no_grad():
                top1 = model.policy_logits.argmax(dim=-1)
                matches = (top1 == lle_logits.argmax(dim=-1)).float().mean().item()
                print(
                    f"step={step:04d} "
                    f"loss={losses['total'].item():.4f} "
                    f"td={losses['td'].item():.4f} "
                    f"indiv={losses['individual'].item():.4f} "
                    f"lcs={matches:.3f} "
                    f"policy_top1={' '.join(VOCAB[int(i)] for i in top1)}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lambda-indiv", type=float, default=1.0)
    parser.add_argument("--confidence-temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--target-tau", type=float, default=0.05)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
