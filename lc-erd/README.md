# LC-ERD reproduction scaffold

This directory implements the core of:

LC-ERD: Mining Latent Logic for Self-Evolving Reasoning via
Consistency-Regulated Reward Decomposition, arXiv:2605.24005.

The paper's official GitHub URL currently does not return refs from
`git ls-remote`, so this folder contains an independent implementation of the
algorithmic pieces described in the paper:

- Weighted Latent Logic Expertise (LLE) discovery from verified trajectories.
- Variational Logic Potential (VLP) shaping.
- LC-ERD step reward reconstruction.
- Temporal-difference value loss and individual logit-consistency loss.
- Top-k logit cache utilities for a memory-efficient expert manifold.

The large-scale paper setting uses Qwen2.5-72B, `K=64`, `beta=0.1`,
`gamma=0.95`, and 8 H100 GPUs. This scaffold separates that expensive path
from a local smoke test that runs on CPU/GPU with only PyTorch.

## Quick smoke test

```bash
cd /root/autodl-tmp/lc-erd
python examples/toy_lcerd.py --steps 80 --width 64 --beta 0.1 --gamma 0.95
```

Expected behavior:

- verified trajectories are mined from sampled reasoning paths;
- policy top-1 actions move toward the LLE consensus;
- total loss and logic KL generally decrease.

## Paper-to-code map

| Paper item | Code |
| --- | --- |
| Eq. 4 confidence weight | `lcerd.core.confidence_weights` |
| Eq. 5 weighted LLE consensus | `lcerd.core.weighted_consensus_logits` |
| Eq. 6 VLP | `lcerd.core.variational_potential` |
| Eq. 8/9 LC-ERD reward | `lcerd.core.lcerd_rewards` |
| Eq. 11-13 joint losses | `lcerd.core.lcerd_losses` |
| Eq. 16 top-k cache | `lcerd.cache.TopKLogitCache` |

## Real-model reproduction outline

1. Install optional dependencies:

   ```bash
   pip install -e ".[hf]"
   ```

2. Use a reasoning backbone such as Qwen2.5-Math or Qwen2.5-Instruct.
   The full paper used Qwen2.5-72B; for a single GPU start with 1.5B/7B.

3. For each query, sample `K=64` trajectories with temperature > 0, parse
   intermediate steps, and verify terminal answers with a task verifier
   (`lcerd.verifiers.NumericAnswerVerifier` is included for GSM8K/MATH-style
   numeric answers).

4. Cache per-step logits from verified trajectories, build the LLE consensus,
   and train with:

   ```text
   total_loss = TD_loss + lambda_indiv * logit_consistency_loss
   ```

5. Match the paper's default hyperparameters first:

   ```text
   K=64
   beta=0.1
   gamma=0.95
   lambda_indiv=1.0
   confidence_temperature=1.0
   top_k_cache=64
   ```

## Notes

This is a faithful formula-level scaffold, not a claim that the KDD-scale
numbers can be reproduced on this machine without the same data, model size,
and distributed training setup.
